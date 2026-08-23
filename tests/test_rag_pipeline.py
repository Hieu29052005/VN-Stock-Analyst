"""Tests for RAG pipeline components."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_pipeline.context_builder import FinancialContextBuilder
from src.rag_pipeline.guardrails import CitationGuardrails
from src.rag_pipeline.reranker import RerankedResult


def test_guardrails_softens_claims():
    guardrails = CitationGuardrails()
    answer = "Giá VIC sẽ tăng mạnh trong tuần tới"
    result = guardrails.verify(answer)
    assert "có xu hướng" in result.lower()


def test_guardrails_no_change_on_clean():
    guardrails = CitationGuardrails()
    answer = "Theo [Source 1], P/E của FPT là 25x."
    result = guardrails.verify(answer)
    assert "25x" in result


def test_context_builder_formats_sources():
    builder = FinancialContextBuilder()
    results = [
        RerankedResult(
            doc_id="1",
            text="FPT có P/E 25x",
            score=0.9,
            metadata={"ticker": "FPT", "source": "vnstock", "doc_type": "financial_data"},
        )
    ]
    context = builder.build("P/E FPT", results, intent="fundamental")
    assert "FPT" in context
    assert "Source 1" in context
