"""BM25 sparse index for hybrid retrieval."""
import logging
import pickle
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from src.config import settings

logger = logging.getLogger(__name__)


class BM25Store:
    """BM25 sparse retrieval index using rank_bm25."""

    def __init__(self, persist_dir: str | Path | None = None):
        self.persist_dir = Path(persist_dir or settings.DATA_DIR / "bm25_index")
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.persist_dir / "bm25_index.pkl"

        self.corpus: list[str] = []
        self.doc_ids: list[str] = []
        self.metadata: list[dict] = []
        self._index: BM25Okapi | None = None

        if self.index_path.exists():
            self._load()

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace + lowercasing tokenizer for Vietnamese."""
        import re
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        tokens = text.split()
        return [t for t in tokens if len(t) > 1]

    def _build_index(self) -> None:
        """Build the BM25 index from corpus."""
        if not self.corpus:
            self._index = None
            return
        tokenized = [self._tokenize(doc) for doc in self.corpus]
        self._index = BM25Okapi(tokenized)
        logger.info(f"Built BM25 index with {len(self.corpus)} documents")

    def add_documents(
        self,
        documents: list[str],
        doc_ids: list[str],
        metadatas: list[dict] | None = None,
    ) -> None:
        """Add documents to the BM25 index."""
        self.corpus.extend(documents)
        self.doc_ids.extend(doc_ids)
        self.metadata.extend(metadatas or [{}] * len(documents))
        self._build_index()

    def query(self, query: str, top_k: int = 20) -> list[dict]:
        """
        Query the BM25 index.

        Returns list of dicts with 'doc_id', 'score', 'text', 'metadata'.
        """
        if not self._index or not self.corpus:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._index.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "doc_id": self.doc_ids[idx],
                    "score": float(scores[idx]),
                    "text": self.corpus[idx],
                    "metadata": self.metadata[idx],
                })

        return results

    def save(self) -> None:
        """Save the index to disk."""
        data = {
            "corpus": self.corpus,
            "doc_ids": self.doc_ids,
            "metadata": self.metadata,
        }
        with open(self.index_path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Saved BM25 index: {len(self.corpus)} docs")

    def _load(self) -> None:
        """Load the index from disk."""
        try:
            with open(self.index_path, "rb") as f:
                data = pickle.load(f)
            self.corpus = data["corpus"]
            self.doc_ids = data["doc_ids"]
            self.metadata = data["metadata"]
            self._build_index()
            logger.info(f"Loaded BM25 index: {len(self.corpus)} docs")
        except Exception as e:
            logger.warning(f"Failed to load BM25 index: {e}")

    @property
    def count(self) -> int:
        return len(self.corpus)
