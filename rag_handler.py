# rag_handler.py
import os
import re
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta

from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_google_genai import GoogleGenerativeAI
from langchain_groq import ChatGroq

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SessionManager:
    """Quản lý sessions với automatic cleanup"""
    
    def __init__(self, max_idle_minutes: int = 30):
        self.sessions: Dict[str, Dict] = {}
        self.max_idle_minutes = max_idle_minutes
    
    def get_or_create(self, session_id: str, factory):
        """Get existing session hoặc tạo mới"""
        now = datetime.now()
        self._cleanup_old_sessions(now)
        if session_id not in self.sessions:
            logger.info(f"Creating new session: {session_id}")
            self.sessions[session_id] = {
                "chain": factory(),
                "last_access": now,
                "created_at": now,
            }
        else:
            self.sessions[session_id]["last_access"] = now
        return self.sessions[session_id]["chain"]
    
    def _cleanup_old_sessions(self, now: datetime):
        expired = []
        for sid, session in self.sessions.items():
            idle_time = now - session["last_access"]
            if idle_time > timedelta(minutes=self.max_idle_minutes):
                expired.append(sid)
        for sid in expired:
            logger.info(f"Cleaning up expired session: {sid}")
            del self.sessions[sid]
    
    def clear_session(self, session_id: str):
        if session_id in self.sessions:
            logger.info(f"Clearing session: {session_id}")
            del self.sessions[session_id]
    
    def get_stats(self) -> Dict:
        return {
            "active_sessions": len(self.sessions),
            "sessions": {
                sid: {
                    "created_at": s["created_at"].isoformat(),
                    "last_access": s["last_access"].isoformat(),
                }
                for sid, s in self.sessions.items()
            }
        }


def _require_groq_key() -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY in environment.")
    return api_key


def _require_gemini_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY in environment.")
    return api_key


def create_llm(
    provider: str = "auto",
    temperature: float = 0.2,
    max_tokens: int = 1024,
):
    def _build_groq():
        api_key = _require_groq_key()
        return ChatGroq(
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )

    def _build_gemini():
        api_key = _require_gemini_key()
        return GoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=temperature,
            max_output_tokens=max_tokens,
            google_api_key=api_key,
        )

    if provider == "groq":
        return _build_groq()
    if provider == "gemini":
        return _build_gemini()

    try:
        return _build_groq()
    except Exception as e:
        logger.warning(f"Groq unavailable, falling back to Gemini: {e}")
        return _build_gemini()


def create_chain(
    vectorstore: Chroma,
    retrieval_k: int = 6,
    memory_window: int = 5,
    temperature: float = 0.2,
    provider: str = "auto"
) -> ConversationalRetrievalChain:
    llm = create_llm(provider=provider, temperature=temperature)
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=memory_window,
        output_key="answer",
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            """Bạn là trợ lý AI chuyên phân tích tin tức tài chính Việt Nam.

Nguyên tắc:
- Trả lời tối đa 512 từ
- Tóm tắt thông tin từ ngữ cảnh một cách đầy đủ và ngắn gọn nhất có thể
- Trích dẫn chính xác các con số (ví dụ: "tăng 6.3%", "giảm xuống 5.2 triệu")
- Phân tích và đưa ra dự đoán nếu được yêu cầu (nêu rõ đây là phân tích, không phải lời khuyên đầu tư)
- Chỉ dùng thông tin có trong ngữ cảnh, không tự thêm thông tin
- Nếu không có thông tin: "Xin lỗi, tôi không tìm thấy thông tin phù hợp trong dữ liệu hiện có."

Trả lời bằng tiếng Việt, giọng điệu chuyên nghiệp và khách quan."""
        ),
        HumanMessagePromptTemplate.from_template(
            """Câu hỏi: {question}

Ngữ cảnh: {context}"""
        ),
    ])

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": retrieval_k}
        ),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        return_source_documents=True,
        output_key="answer",
    )


def _detect_existing_collection(persist_dir: str) -> Optional[str]:
    """Check nếu có collection Chroma tồn tại trong persist_dir"""
    if not os.path.exists(persist_dir):
        return None
    for folder in os.listdir(persist_dir):
        full_path = os.path.join(persist_dir, folder)
        if os.path.isdir(full_path):
            if any(f.endswith(".sqlite") for f in os.listdir(full_path)):
                return folder
    return None


def load_vectorstore(
    collection_name: Optional[str] = None,
    persist_dir: str = "chroma_store",
) -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name='BAAI/bge-m3',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    if collection_name is None:
        detected = _detect_existing_collection(persist_dir)
        if detected:
            collection_name = detected
            logger.info(f"Detected existing Chroma collection: {collection_name}")
        else:
            collection_name = "vnexpress_kinhdoanh"
            logger.info(f"No existing collection found. Will create new: {collection_name}")

    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )
    logger.info(f"Loaded vectorstore with {vectorstore._collection.count()} documents")
    return vectorstore

def format_answer(text: str, max_line_length: int = 100) -> str:
    import textwrap
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(^|\n)\s*\*\s+", r"\1- ", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"([.!?])\s+(?=[A-ZĐÀÁẢÃẠ])", r"\1\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    wrapped_lines = []
    for line in text.split("\n"):
        if len(line) > max_line_length:
            sublines = textwrap.wrap(line, width=max_line_length, break_long_words=False, break_on_hyphens=False)
            wrapped_lines.extend(sublines)
        else:
            wrapped_lines.append(line)
    return "\n".join(wrapped_lines).strip()

class RAGHandler:
    def __init__(
        self,
        collection_name: str = "vnexpress_kinhdoanh",
        persist_dir: str = "chroma_store",
        retrieval_k: int = 6,
        memory_window: int = 5,
        session_timeout: int = 30,
        provider="auto"
    ):
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.retrieval_k = retrieval_k
        self.memory_window = memory_window
        self.provider = provider
        
        self.vectorstore: Optional[Chroma] = None
        self.session_manager = SessionManager(max_idle_minutes=session_timeout)
    
    def initialize(self):
        try:
            logger.info("Initializing RAG handler...")
            self.vectorstore = load_vectorstore(
                collection_name=self.collection_name,
                persist_dir=self.persist_dir,
            )
            logger.info("RAG handler initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG handler: {e}")
            raise
    
    def _create_chain(self) -> ConversationalRetrievalChain:
        if self.vectorstore is None:
            raise RuntimeError("Vectorstore not initialized. Call initialize() first.")
        return create_chain(
            vectorstore=self.vectorstore,
            retrieval_k=self.retrieval_k,
            memory_window=self.memory_window,
            provider=self.provider,
        )
    
    def get_chain(self, session_id: str) -> ConversationalRetrievalChain:
        return self.session_manager.get_or_create(
            session_id=session_id,
            factory=self._create_chain,
        )
    
    def process_rag_query(
        self,
        session_id: str,
        message: str,
        return_sources: bool = False,
    ) -> Dict:
        try:
            logger.info(f"Processing query for session {session_id}: {message[:100]}")
            chain = self.get_chain(session_id)
            result = chain.invoke({"question": message})
            raw_answer = result.get("answer") or result.get("result") or ""
            formatted_answer = format_answer(raw_answer)
            response = {"answer": formatted_answer, "success": True}
            if return_sources and "source_documents" in result:
                sources = []
                for doc in result["source_documents"]:
                    sources.append({
                        "title": doc.metadata.get("title", ""),
                        "url": doc.metadata.get("url", ""),
                        "date": doc.metadata.get("date", ""),
                        "content_preview": doc.page_content[:200] + "...",
                    })
                response["sources"] = sources
            logger.info(f"Query processed successfully for session {session_id}")
            return response
        except Exception as e:
            logger.error(f"Error processing query for session {session_id}: {e}", exc_info=True)
            return {
                "answer": "Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi của bạn. Vui lòng thử lại.",
                "success": False,
                "error": str(e),
            }
    
    def clear_session(self, session_id: str):
        self.session_manager.clear_session(session_id)
    
    def get_stats(self) -> Dict:
        return self.session_manager.get_stats()
    
    def health_check(self) -> Dict:
        try:
            if self.vectorstore is None:
                return {"status": "unhealthy", "reason": "vectorstore not initialized"}
            count = self.vectorstore._collection.count()
            stats = self.get_stats()
            return {
                "status": "healthy",
                "vectorstore_documents": count,
                "active_sessions": stats["active_sessions"],
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "reason": str(e)}


_rag_handler: Optional[RAGHandler] = None

def get_rag_handler() -> RAGHandler:
    global _rag_handler
    if _rag_handler is None:
        _rag_handler = RAGHandler()
        _rag_handler.initialize()
    return _rag_handler