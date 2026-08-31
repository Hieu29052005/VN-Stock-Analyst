"""Full RAG pipeline orchestrator."""
import logging
import time
from dataclasses import dataclass, field

from src.config import settings
from src.knowledge_hub.retriever import HybridRetriever, RetrievalResult
from src.rag_pipeline.context_builder import FinancialContextBuilder
from src.rag_pipeline.generator import LLMGenerator
from src.rag_pipeline.guardrails import CitationGuardrails
from src.rag_pipeline.query_classifier import QueryClassifier, QueryIntent
from src.rag_pipeline.query_transform.hyde import HyDETransform
from src.rag_pipeline.query_transform.multi_query import MultiQueryTransform
from src.rag_pipeline.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Response from the RAG pipeline."""

    answer: str
    sources: list[dict] = field(default_factory=list)
    query_classification: str = "general"
    confidence: float = 0.0
    latency_ms: float = 0.0
    retrieval_count: int = 0


class StockRAGPipeline:
    """
    End-to-end RAG pipeline for Vietnamese stock market Q&A.

    Flow: Classify → Transform → Retrieve → Rerank → Build Context → Generate → Guard
    """

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        reranker: CrossEncoderReranker | None = None,
        generator: LLMGenerator | None = None,
        use_hyde: bool = True,
        use_multi_query: bool = True,
    ):
        self.classifier = QueryClassifier()
        self.retriever = retriever or HybridRetriever()
        self.reranker = reranker or CrossEncoderReranker()
        self.context_builder = FinancialContextBuilder()
        self.generator = generator or LLMGenerator()
        self.guardrails = CitationGuardrails()

        self.hyde = HyDETransform(use_llm=use_hyde)
        self.multi_query = MultiQueryTransform(use_llm=use_multi_query)

        logger.info("StockRAGPipeline initialized")

    async def query(
        self,
        question: str,
        ticker_filter: str | None = None,
    ) -> RAGResponse:
        """
        Process a user question through the full RAG pipeline.

        Args:
            question: User's question in Vietnamese
            ticker_filter: Optional ticker to narrow results

        Returns:
            RAGResponse with answer, sources, and metadata
        """
        start = time.time()

        # 1. Classify intent
        intent = self.classifier.classify(question)
        logger.info(f"Query intent: {intent.value}")

        # 2. Transform queries based on intent
        queries = self._transform_query(question, intent)

        # 3. Retrieve (hybrid)
        all_results = []
        for q in queries:
            results = self.retriever.retrieve(
                q,
                n_results=settings.RERANK_TOP_K * 3,
                ticker_filter=ticker_filter,
            )
            all_results.extend(results)

        # Deduplicate by doc_id
        seen = set()
        unique_results = []
        for r in all_results:
            if r.doc_id not in seen:
                seen.add(r.doc_id)
                unique_results.append(r)

        logger.info(
            f"Retrieved {len(unique_results)} unique docs "
            f"from {len(queries)} query variations"
        )

        # 4. Rerank
        docs_for_rerank = [
            {
                "doc_id": r.doc_id,
                "text": r.text,
                "score": r.score,
                "metadata": r.metadata,
            }
            for r in unique_results
        ]
        reranked = self.reranker.rerank(question, docs_for_rerank)

        # 5. Build context
        context = self.context_builder.build(question, reranked, intent=intent.value)

        # 6. Generate
        answer = await self.generator.generate(context)

        # 7. Guardrails
        answer = self.guardrails.verify(answer, context, question)

        # 8. Build sources list
        sources = [
            {
                "text": r.text[:200] + "..." if len(r.text) > 200 else r.text,
                "score": round(r.score, 4),
                "metadata": r.metadata,
            }
            for r in reranked
        ]

        latency = (time.time() - start) * 1000
        confidence = self._calculate_confidence(reranked)

        return RAGResponse(
            answer=answer,
            sources=sources,
            query_classification=intent.value,
            confidence=confidence,
            latency_ms=latency,
            retrieval_count=len(reranked),
        )

    def _transform_query(
        self, query: str, intent: QueryIntent
    ) -> list[str]:
        """Transform query based on intent."""
        if intent == QueryIntent.COMPARISON:
            return self.multi_query.transform(query, n=5)
        elif intent in (QueryIntent.FUNDAMENTAL, QueryIntent.ANALYSIS):
            hyde_query = self.hyde.transform(query)
            return [query, hyde_query]
        else:
            return [query]

    def _calculate_confidence(self, reranked: list) -> float:
        """Calculate confidence score from reranked results."""
        if not reranked:
            return 0.0
        top_score = reranked[0].score
        avg_score = sum(r.score for r in reranked) / len(reranked)
        return min(1.0, (top_score * 0.6 + avg_score * 0.4))
