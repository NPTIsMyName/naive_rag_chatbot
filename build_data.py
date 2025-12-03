import os
import re
import json
import logging
import hashlib
from datetime import datetime
from typing import List, Dict, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ============================================================
# Setup logging
# ============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# Helpers
# ============================================================
def clean_date(raw_date: str) -> Optional[datetime]:
    """
    Normalize VNExpress-style date to datetime.
    Example: "Thứ tư, 26/11/2025, 00:15 (GMT+7)"
    """
    if not raw_date:
        return None

    # Remove weekday + comma + optional spaces at start
    raw = re.sub(r"^thứ [a-záàảãạêơôúùýíìụủ]+,\s*", "", raw_date.lower())
    # Remove timezone like (GMT+7)
    raw = re.sub(r"\(gmt[^\)]*\)", "", raw)
    raw = raw.strip()

    formats = [
        "%d/%m/%Y, %H:%M",
        "%d/%m/%Y - %H:%M",
        "%d/%m/%Y",
        "%d/%m/%y, %H:%M",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue

    logger.warning(f"Cannot parse date: {raw_date}")
    return None


def generate_chunk_id(content: str) -> str:
    """Stable, content-based ID."""
    return hashlib.md5(content.encode("utf-8")).hexdigest()

# ============================================================
# Main class
# ============================================================
class NewsVectorStoreBuilder:
    def __init__(
        self,
        collection_name="vnexpress_kinhdoanh",
        persist_dir="chroma_store",
    ):
        self.collection_name = collection_name
        self.persist_dir = persist_dir

        os.makedirs(persist_dir, exist_ok=True)

        # Embedding
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cpu"},
            encode_kwargs={'normalize_embeddings': True}  # ← PHẢI KHỚP

        )
        logger.info("Embedding model loaded.")

        # Vectorstore (used for adding docs)
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
        )

    # ------------------------------------------------------------
    def load_json_file(self, file_path: str) -> List[Dict]:
        """Load JSON array of articles."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []

    # ------------------------------------------------------------
    def filter_new_items(self, items: List[Dict]) -> List[Dict]:
        """Return only articles that have never been processed."""
        processed_file = os.path.join(self.persist_dir, "processed_urls.txt")
        processed_urls = set()
        if os.path.exists(processed_file):
            with open(processed_file, "r", encoding="utf-8") as f:
                processed_urls = {line.strip() for line in f}

        new_items = []
        for item in items:
            url = item.get("url") or ""
            if not url:
                title = item.get("title", "")
                url = "no_url_" + hashlib.md5(title.encode("utf-8")).hexdigest()
            if url not in processed_urls:
                item["url"] = url
                new_items.append(item)

        logger.info(f"Found {len(new_items)} new items.")
        return new_items

    # ------------------------------------------------------------
    def process_json_file(self, file_path: str) -> None:
        logger.info(f"Processing file: {file_path}")
        items = self.load_json_file(file_path)
        items = self.filter_new_items(items)

        if not items:
            logger.info("No new items to add.")
            return

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = []
        ids = []

        for item in items:
            title = item.get("title", "").strip()
            content = item.get("content", "").strip()
            url = item.get("url") or ""
            date_parsed = clean_date(item.get("date", ""))

            if not title and not content:
                logger.warning("Skipping item with no title + no content.")
                continue

            full_text = title + "\n\n" + content
            chunks = splitter.create_documents([full_text])

            for chunk in chunks:
                chunk.metadata = {
                    "url": url,
                    "title": title or "",
                    "date": date_parsed.isoformat() if date_parsed else "",
                }
                doc_id = generate_chunk_id(chunk.page_content)
                docs.append(chunk)
                ids.append(doc_id)

        # Save to vectorstore
        if docs:
            logger.info(f"Adding {len(docs)} chunks to ChromaDB...")
            self.vectorstore.add_documents(docs, ids=ids)
            self.vectorstore.persist()
            logger.info("Persisted to disk.")

        # Mark processed URLs
        new_urls = [item["url"] for item in items]
        with open(os.path.join(self.persist_dir, "processed_urls.txt"), "a", encoding="utf-8") as f:
            for u in new_urls:
                f.write(u + "\n")

        logger.info(f"Finished processing {len(items)} articles.")

    # ------------------------------------------------------------
    def get_stats(self):
        """Return vectorstore count without loading embeddings again."""
        vs = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_dir,
            embedding_function=None,
        )
        return vs._collection.count()

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    builder = NewsVectorStoreBuilder(
        collection_name="vnexpress_kinhdoanh",
        persist_dir="chroma_store",
    )

    # Chỉ load 1 file JSON duy nhất
    builder.process_json_file("vnexpress_kinhdoanh.json")

    logger.info(f"Total documents in DB: {builder.get_stats()}")
