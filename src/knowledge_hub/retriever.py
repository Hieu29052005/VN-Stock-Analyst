"""Hybrid retriever combining dense (vector) + sparse (BM25) with RRF fusion."""
import logging
from dataclasses import dataclass, field

from src.config import settings
from src.knowledge_hub.bm25_store import BM25Store
from src.knowledge_hub.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieval result."""

    doc_id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)
    retrieval_method: str = ""


class HybridRetriever:
    """
    Hybrid retriever: BM25 (sparse) + Dense embeddings, fused via
    Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        bm25_store: BM25Store | None = None,
        vector_weight: float | None = None,
        bm25_weight: float | None = None,
        rrf_k: int | None = None,
    ):
        self.vector_store = vector_store or VectorStore()
        self.bm25_store = bm25_store or BM25Store()
        self.vector_weight = vector_weight or settings.VECTOR_WEIGHT
        self.bm25_weight = bm25_weight or settings.BM25_WEIGHT
        self.rrf_k = rrf_k or settings.RRF_K
        self.k = settings.RETRIEVAL_K

    def retrieve(
        self,
        query: str,
        n_results: int | None = None,
        ticker_filter: str | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve using hybrid search with RRF fusion.

        Args:
            query: Search query
            n_results: Number of final results
            ticker_filter: Optional ticker to filter results

        Returns:
            List of RetrievalResult sorted by fused score
        """
        k = n_results or self.k

        # Dense retrieval
        dense_results = self._dense_retrieve(query, k * 2, ticker_filter)

        # Sparse retrieval
        sparse_results = self._sparse_retrieve(query, k * 2)

        # Fuse with RRF
        fused = self._rrf_fuse(dense_results, sparse_results)

        return fused[:k]

    def _dense_retrieve(
        self, query: str, k: int, ticker_filter: str | None = None
    ) -> list[RetrievalResult]:
        """Dense vector retrieval."""
        try:
            if ticker_filter:
                results = self.vector_store.query_with_filter(
                    query, ticker=ticker_filter, n_results=k
                )
            else:
                results = self.vector_store.query(query, n_results=k)

            dense_results = []
            for i, doc_id in enumerate(results.get("ids", [[]])[0]):
                distances = results.get("distances", [[]])[0]
                score = 1.0 - distances[i] if distances else 0.0
                dense_results.append(RetrievalResult(
                    doc_id=doc_id,
                    text=results["documents"][0][i],
                    score=score,
                    metadata=results["metadatas"][0][i] if results.get("metadatas") else {},
                    retrieval_method="dense",
                ))
            return dense_results
        except Exception as e:
            logger.warning(f"Dense retrieval failed: {e}")
            return []

    def _sparse_retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        """BM25 sparse retrieval."""
        try:
            results = self.bm25_store.query(query, top_k=k)
            return [
                RetrievalResult(
                    doc_id=r["doc_id"],
                    text=r["text"],
                    score=r["score"],
                    metadata=r["metadata"],
                    retrieval_method="sparse",
                )
                for r in results
            ]
        except Exception as e:
            logger.warning(f"Sparse retrieval failed: {e}")
            return []

    def _rrf_fuse(
        self,
        dense_results: list[RetrievalResult],
        sparse_results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """
        Reciprocal Rank Fusion.
        score(d) = sum(weight_i * 1/(k + rank_i))
        """
        doc_scores: dict[str, float] = {}
        doc_map: dict[str, RetrievalResult] = {}

        for weight, results in [
            (self.vector_weight, dense_results),
            (self.bm25_weight, sparse_results),
        ]:
            for rank, result in enumerate(results, start=1):
                rrf_score = weight * (1 / (self.rrf_k + rank))
                if result.doc_id in doc_scores:
                    doc_scores[result.doc_id] += rrf_score
                else:
                    doc_scores[result.doc_id] = rrf_score
                doc_map[result.doc_id] = result

        sorted_ids = sorted(doc_scores, key=doc_scores.get, reverse=True)
        return [
            RetrievalResult(
                doc_id=doc_map[doc_id].doc_id,
                text=doc_map[doc_id].text,
                score=doc_scores[doc_id],
                metadata=doc_map[doc_id].metadata,
                retrieval_method="hybrid_rrf",
            )
            for doc_id in sorted_ids
        ]
