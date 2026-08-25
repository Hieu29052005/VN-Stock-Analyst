"""ChromaDB vector store wrapper."""
import logging
from pathlib import Path

import chromadb

from src.config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """Wrapper around ChromaDB for vector storage and retrieval."""

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        collection_name: str | None = None,
    ):
        self.persist_dir = str(persist_dir or settings.CHROMA_DIR)
        self.collection_name = collection_name or settings.CHROMA_COLLECTION

        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"VectorStore: collection={self.collection_name}, "
            f"count={self.collection.count()}"
        )

    def add_documents(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict] | None = None,
        batch_size: int = 100,
    ) -> int:
        """Add documents to the vector store in batches."""
        total_added = 0
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_docs = documents[i : i + batch_size]
            batch_meta = metadatas[i : i + batch_size] if metadatas else None

            try:
                self.collection.upsert(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_meta,
                )
                total_added += len(batch_ids)
            except Exception as e:
                logger.error(f"Error adding batch {i}: {e}")
                continue

        logger.info(f"Added {total_added} documents to vector store")
        return total_added

    def query(
        self,
        query_text: str,
        n_results: int = 20,
        where: dict | None = None,
    ) -> dict:
        """
        Query the vector store for similar documents.

        Returns dict with 'ids', 'documents', 'metadatas', 'distances'.
        """
        kwargs = {
            "query_texts": [query_text],
            "n_results": min(n_results, self.collection.count() or n_results),
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)
        return results

    def query_with_filter(
        self,
        query_text: str,
        ticker: str | None = None,
        doc_type: str | None = None,
        n_results: int = 20,
    ) -> dict:
        """Query with metadata filters."""
        where_conditions = {}
        if ticker:
            where_conditions["ticker"] = ticker
        if doc_type:
            where_conditions["doc_type"] = doc_type

        where = where_conditions if len(where_conditions) > 1 else (
            where_conditions if where_conditions else None
        )
        return self.query(query_text, n_results, where)

    def delete_all(self) -> None:
        """Delete all documents from the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Deleted all documents from vector store")

    @property
    def count(self) -> int:
        return self.collection.count()
