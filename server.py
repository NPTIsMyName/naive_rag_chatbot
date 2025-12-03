#server.py
import os
import logging
from contextlib import asynccontextmanager
from typing import Optional, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Import từ modules improved
from rag_handler import get_rag_handler, RAGHandler
from build_data import NewsVectorStoreBuilder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


# ==================== Request/Response Models ====================

class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=100)
    message: str = Field(..., min_length=1, max_length=2000)
    return_sources: bool = Field(default=False)
    
    @validator('session_id')
    def validate_session_id(cls, v):
        # Only allow alphanumeric, underscore, hyphen
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('session_id must be alphanumeric with _ or -')
        return v
    
    @validator('message')
    def validate_message(cls, v):
        # Strip whitespace
        v = v.strip()
        if not v:
            raise ValueError('message cannot be empty')
        return v


class SourceDocument(BaseModel):
    title: str
    url: str
    date: str
    content_preview: str


class ChatResponse(BaseModel):
    success: bool
    answer: str
    sources: Optional[List[SourceDocument]] = None
    session_id: str
    timestamp: str


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    vectorstore_documents: Optional[int] = None
    active_sessions: Optional[int] = None


class SessionStatsResponse(BaseModel):
    active_sessions: int
    sessions: dict


class DatabaseStatsResponse(BaseModel):
    total_documents: int
    processed_urls: int
    status: str
    persist_dir: Optional[str] = None
    collection_name: Optional[str] = None


class UpdateDatabaseRequest(BaseModel):
    mode: str = Field(default="incremental", pattern="^(incremental|full)$")
    json_path: Optional[str] = Field(default="vnexpress_kinhdoanh.json")


class UpdateDatabaseResponse(BaseModel):
    success: bool
    message: str
    stats: dict
    mode: str


# ==================== Lifespan Management ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown events"""
    # Startup
    logger.info("Starting up application...")
    try:
        rag_handler = get_rag_handler()
        app.state.rag_handler = rag_handler
        logger.info("RAG handler initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG handler: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    # Cleanup if needed


# ==================== App Factory ====================

def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    load_dotenv()
    
    # Get CORS origins from env
    cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
    
    app = FastAPI(
        title="Financial News Chatbot API",
        description="RAG-based chatbot for Vietnamese financial news analysis",
        version="1.0.0",
        lifespan=lifespan,
    )
    
    # Add rate limiter
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )
    
    # Static files for demo UI
    if os.path.isdir("static"):
        app.mount("/static", StaticFiles(directory="static"), name="static")
        
        @app.get("/", include_in_schema=False)
        def index():
            """Serve demo UI"""
            return FileResponse("static/index.html")
    
    # ==================== API Endpoints ====================
    
    @app.get("/health", response_model=HealthResponse, tags=["System"])
    def health_check():
        """Check system health status"""
        try:
            handler: RAGHandler = app.state.rag_handler
            health = handler.health_check()
            return HealthResponse(**health)
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "unhealthy", "error": str(e)}
            )
    
    @app.get("/api/admin/database-stats", response_model=DatabaseStatsResponse, tags=["Admin"])
    def get_database_stats():
        """Get database statistics"""
        try:
            builder = NewsVectorStoreBuilder(
                json_path="vnexpress_kinhdoanh.json",
                persist_dir="chroma_store",
            )
            
            stats = builder.get_stats()
            return DatabaseStatsResponse(**stats)
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )
    
    @app.post("/api/admin/update-database", response_model=UpdateDatabaseResponse, tags=["Admin"])
    async def update_database(body: UpdateDatabaseRequest, background_tasks: BackgroundTasks):
        """
        Update vector database with new articles
        
        - **mode**: "incremental" (add new only) or "full" (rebuild all)
        - **json_path**: Path to JSON file with news articles
        
        Note: This is a long-running operation. The update runs in background.
        """
        try:
            logger.info(f"Starting database update in {body.mode} mode")
            
            builder = NewsVectorStoreBuilder(
                json_path=body.json_path,
                collection_name="vnexpress_kinhdoanh",
                persist_dir="chroma_store",
            )
            
            # Run update in background to avoid timeout
            def do_update():
                try:
                    builder.build_vectorstore(mode=body.mode)
                    # Reload handler's vectorstore after update
                    logger.info("Reloading RAG handler after database update")
                    app.state.rag_handler.initialize()
                except Exception as e:
                    logger.error(f"Background database update failed: {e}")
            
            background_tasks.add_task(do_update)
            
            # Get current stats
            stats = builder.get_stats()
            
            return UpdateDatabaseResponse(
                success=True,
                message=f"Database update started in {body.mode} mode. Check logs for progress.",
                stats=stats,
                mode=body.mode,
            )
            
        except Exception as e:
            logger.error(f"Database update failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Update failed: {str(e)}"
            )

    @app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
    @limiter.limit("30/minute")  # Rate limit: 30 requests per minute
    def chat(request: Request, body: ChatRequest):
        """
        Process chat message with RAG
        
        - **session_id**: Unique session identifier (alphanumeric, _, -)
        - **message**: User query (1-2000 chars)
        - **return_sources**: Whether to return source documents
        """
        try:
            handler: RAGHandler = app.state.rag_handler
            
            # Check vectorstore initialized
            if handler.vectorstore is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Vector store not initialized"
                )
            
            logger.info(f"Processing chat for session {body.session_id}")
            
            # Process query
            result = handler.process_rag_query(
                session_id=body.session_id,
                message=body.message,
                return_sources=body.return_sources,
            )
            
            # Check if processing was successful
            if not result.get("success", False):
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=result.get("error", "Unknown error")
                )
            
            # Build response
            from datetime import datetime
            response = ChatResponse(
                success=True,
                answer=result["answer"],
                sources=[SourceDocument(**src) for src in result.get("sources", [])] if body.return_sources else None,
                session_id=body.session_id,
                timestamp=datetime.now().isoformat(),
            )
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in chat endpoint: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error. Please try again later."
            )
    
    @app.delete("/api/session/{session_id}", tags=["Session"])
    def clear_session(session_id: str):
        """
        Clear a specific session's conversation history
        
        - **session_id**: Session to clear
        """
        try:
            handler: RAGHandler = app.state.rag_handler
            handler.clear_session(session_id)
            logger.info(f"Cleared session: {session_id}")
            return {"success": True, "message": f"Session {session_id} cleared"}
        except Exception as e:
            logger.error(f"Error clearing session: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )
    
    @app.get("/api/sessions/stats", response_model=SessionStatsResponse, tags=["Session"])
    def get_session_stats():
        """Get statistics about active sessions"""
        try:
            handler: RAGHandler = app.state.rag_handler
            stats = handler.get_stats()
            return SessionStatsResponse(**stats)
        except Exception as e:
            logger.error(f"Error getting session stats: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )
    
    # ==================== Error Handlers ====================
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions"""
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=exc.detail,
                detail=f"Request to {request.url.path} failed"
            ).dict()
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions"""
        logger.error(f"Unexpected error: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="Internal server error",
                detail="An unexpected error occurred"
            ).dict()
        )
    
    return app


# ==================== App Instance ====================

app = create_app()


# ==================== CLI for local testing ====================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        "server:app",  # Fixed: use correct module name
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info",
    )