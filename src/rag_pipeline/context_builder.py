"""Build structured prompt context from retrieved chunks."""
import logging
from pathlib import Path

from src.config import settings
from src.rag_pipeline.reranker import RerankedResult

logger = logging.getLogger(__name__)


class FinancialContextBuilder:
    """Build formatted context for LLM from retrieved chunks."""

    def __init__(self):
        self.prompts_dir = Path(__file__).parent / "prompts"
        self._templates: dict[str, str] = {}
        self._load_templates()

    def _load_templates(self) -> None:
        """Load prompt templates from files."""
        for prompt_file in self.prompts_dir.glob("*.txt"):
            self._templates[prompt_file.stem] = prompt_file.read_text(
                encoding="utf-8"
            )

    def build(
        self,
        query: str,
        reranked_results: list[RerankedResult],
        intent: str = "general",
    ) -> str:
        """
        Build the full context string for the LLM.

        Args:
            query: User's question
            reranked_results: Reranked retrieval results
            intent: Query intent for template selection

        Returns:
            Formatted context string
        """
        sources_text = self._format_sources(reranked_results)
        template = self._get_template(intent)

        context = template.format(
            context=sources_text,
            query=query,
        )
        return context

    def _format_sources(self, results: list[RerankedResult]) -> str:
        """Format retrieval results into source blocks."""
        sources = []
        for i, result in enumerate(results, 1):
            ticker = result.metadata.get("ticker", "")
            source = result.metadata.get("source", "")
            doc_type = result.metadata.get("doc_type", "")
            date = result.metadata.get("date", "")

            header_parts = [f"[Source {i}]"]
            if ticker:
                header_parts.append(f"Cổ phiếu: {ticker}")
            if doc_type:
                header_parts.append(f"Loại: {doc_type}")
            if source:
                header_parts.append(f"Nguồn: {source}")
            if date:
                header_parts.append(f"Ngày: {date}")

            header = " | ".join(header_parts)
            sources.append(f"{header}\n{result.text.strip()}")

        return "\n\n---\n\n".join(sources)

    def _get_template(self, intent: str) -> str:
        """Get the appropriate prompt template for the intent."""
        template_map = {
            "comparison": "comparison",
            "fundamental": "financial_qa",
            "analysis": "financial_qa",
            "price_lookup": "financial_qa",
            "general": "financial_qa",
        }
        template_name = template_map.get(intent, "financial_qa")
        return self._templates.get(
            template_name,
            self._templates.get("financial_qa", "{context}\n\n{query}")
        )

    def build_sources_citation(self, results: list[RerankedResult]) -> str:
        """Build a formatted sources list for citation."""
        lines = ["Nguồn tham khảo:"]
        for i, result in enumerate(results, 1):
            parts = [f"[{i}]"]
            if result.metadata.get("ticker"):
                parts.append(result.metadata["ticker"])
            if result.metadata.get("source"):
                parts.append(result.metadata["source"])
            if result.metadata.get("date"):
                parts.append(result.metadata["date"])
            lines.append(" ".join(parts))
        return "\n".join(lines)
