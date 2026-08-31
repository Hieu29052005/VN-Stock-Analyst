"""Cross-encoder reranker for retrieved documents."""
import logging
from dataclasses import dataclass, field

from src.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RerankedResult:
    """A reranked retrieval result."""

    doc_id: str
    text: str
    score: float
    original_score: float = 0.0
    metadata: dict = field(default_factory=dict)


class CrossEncoderReranker:
    """Rerank retrieved documents using a cross-encoder model."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.RERANKER_MODEL
        self._model = None

    def _get_model(self):
        """Lazy-load the cross-encoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading reranker model: {self.model_name}")
            self._model = CrossEncoder(self.model_name, max_length=512)
        return self._model

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int | None = None,
    ) -> list[RerankedResult]:
        """
        Rerank documents using cross-encoder scoring.

        Args:
            query: The search query
            documents: List of dicts with 'text', 'doc_id', 'metadata', 'score'
            top_k: Number of top results to return

        Returns:
            List of RerankedResult sorted by reranked score
        """
        top_k = top_k or settings.RERANK_TOP_K

        if not documents:
            return []

        if len(documents) <= top_k:
            docs_to_rerank = documents
        else:
            docs_to_rerank = documents[:top_k * 3]

        model = self._get_model()

        pairs = [(query, doc["text"]) for doc in docs_to_rerank]
        scores = model.predict(pairs)

        ranked = sorted(
            zip(docs_to_rerank, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        results = []
        for doc, score in ranked[:top_k]:
            results.append(RerankedResult(
                doc_id=doc.get("doc_id", ""),
                text=doc["text"],
                score=float(score),
                original_score=doc.get("score", 0.0),
                metadata=doc.get("metadata", {}),
            ))

        logger.debug(
            f"Reranked {len(documents)} docs → top {top_k}, "
            f"score range: {results[0].score:.3f} - {results[-1].score:.3f}"
        )
        return results
