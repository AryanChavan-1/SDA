"""
seaf_agent.rag — ChromaDB Semantic Retrieval & Experience Memory.
VRAM Footprint:
- nomic-embed-text:latest (~274MB VRAM)
- ChromaDB: Disk-based SQLite, 0MB VRAM
Enforces hard token budget to prevent VRAM overflow.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from seaf_agent.config import (
    OLLAMA_BASE_URL,
    EMBEDDING_MODEL,
    CHROMA_DIR,
    BASE_CONTEXT_WINDOW,
)

log = logging.getLogger("seaf.rag")

SUPPORTED_EXTENSIONS = {".py", ".md", ".txt", ".json", ".rst"}


class SEAFRAGEngine:
    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        embed_model: str = EMBEDDING_MODEL,
        ollama_base: str = OLLAMA_BASE_URL,
    ):
        self.persist_dir = str(persist_dir or CHROMA_DIR)
        self.embed_model = embed_model
        self.ollama_base = ollama_base
        self._embeddings: Optional[OllamaEmbeddings] = None

    def _get_embeddings(self) -> OllamaEmbeddings:
        if self._embeddings is None:
            self._embeddings = OllamaEmbeddings(
                model=self.embed_model,
                base_url=self.ollama_base,
            )
        return self._embeddings

    def _get_vectorstore(self, collection_name: str = "seaf_kb") -> Chroma:
        return Chroma(
            collection_name=collection_name,
            embedding_function=self._get_embeddings(),
            persist_directory=self.persist_dir,
        )

    def ingest_directory(self, dir_path: str, collection_name: str = "seaf_kb") -> Dict[str, Any]:
        """Splits and indexes files into the specified ChromaDB collection."""
        path = Path(dir_path)
        if not path.exists():
            return {"status": "ERROR", "message": f"Directory not found: {dir_path}"}

        docs: List[Document] = []
        for file in path.rglob("*"):
            if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS:
                try:
                    text = file.read_text(encoding="utf-8", errors="replace")
                    if text.strip():
                        docs.append(Document(
                            page_content=text,
                            metadata={"source": str(file.relative_to(path)), "suffix": file.suffix}
                        ))
                except Exception as exc:
                    log.warning("Could not read %s: %s", file, exc)

        if not docs:
            return {"status": "SUCCESS", "ingested_chunks": 0, "message": "No documents found to index"}

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        store = self._get_vectorstore(collection_name)
        store.add_documents(chunks)
        return {
            "status": "SUCCESS",
            "files_scanned": len(docs),
            "ingested_chunks": len(chunks),
        }

    def query(self, query_text: str, top_k: int = 4, max_chars: int = 3000, collection_name: str = "seaf_kb") -> str:
        """Queries vector database and caps returned context characters."""
        try:
            store = self._get_vectorstore(collection_name)
            results = store.similarity_search(query_text, k=top_k)
            if not results:
                return ""

            parts = []
            total_len = 0
            for doc in results:
                content = doc.page_content.strip()
                src = doc.metadata.get("source", "knowledge_base")
                block = f"--- [Source: {src}] ---\n{content}\n"
                if total_len + len(block) > max_chars:
                    parts.append(block[: max_chars - total_len] + "\n[Context Truncated]")
                    break
                parts.append(block)
                total_len += len(block)

            return "\n".join(parts)
        except Exception as exc:
            log.warning("RAG query failed: %s", exc)
            return ""

    def store_experience(self, problem: str, solution: str) -> None:
        """Stores a resolved error and working patch into experience memory."""
        try:
            content = f"PROBLEM:\n{problem}\n\nSOLUTION:\n```python\n{solution}\n```"
            doc_id = hashlib.md5(content.encode("utf-8")).hexdigest()[:16]
            store = self._get_vectorstore("seaf_experience")
            doc = Document(
                page_content=content,
                metadata={"doc_id": doc_id, "type": "experience"}
            )
            store.add_documents([doc], ids=[doc_id])
        except Exception as exc:
            log.warning("Failed to store experience: %s", exc)

    def query_experience(self, error_desc: str, top_k: int = 2) -> str:
        """Retrieves similar past bugs and fixes."""
        return self.query(error_desc, top_k=top_k, max_chars=2000, collection_name="seaf_experience")


# Global singleton
rag_engine = SEAFRAGEngine()
