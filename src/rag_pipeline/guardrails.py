"""Guardrails for citation enforcement and hallucination checks."""
import logging
import re

logger = logging.getLogger(__name__)


class CitationGuardrails:
    """Post-generation guardrails for financial RAG responses."""

    HALLUCINATION_PATTERNS = [
        (r"giá\s+\w*\s+sẽ\s+(tăng|giảm)", "claims about future prices"),
        (r"chắc chắn (là|sẽ)", "overly certain claims"),
        (r"tôi (tin|rõ|đảm bảo|khẳng định)", "personal opinion as fact"),
        (r"khuyến nghị (mua|bán|giữ)", "direct trading recommendations"),
    ]

    def __init__(self):
        self._compiled_patterns = [
            (re.compile(p, re.IGNORECASE), desc)
            for p, desc in self.HALLUCINATION_PATTERNS
        ]

    def verify(self, answer: str, context: str = "", query: str = "") -> str:
        """
        Apply guardrails to the generated answer.

        Returns the (potentially modified) answer.
        """
        answer = self._check_citations(answer)
        answer = self._soften_claims(answer)
        answer = self._add_disclaimer_if_needed(answer)
        return answer

    def _check_citations(self, answer: str) -> str:
        """Check if answer has source citations."""
        has_citation = bool(
            re.search(r"\[Source\s*\d+\]", answer)
            or re.search(r"\[Nguồn\s*\d+\]", answer)
        )
        if not has_citation and len(answer) > 100:
            logger.debug("No citations found in answer")
        return answer

    def _soften_claims(self, answer: str) -> str:
        """Soften overly certain or predictive claims."""
        for pattern, desc in self._compiled_patterns:
            if pattern.search(answer):
                logger.debug(f"Detected potential issue: {desc}")

        answer = re.sub(
            r"(giá\s+\w*\s+)sẽ\s+(tăng|giảm)",
            r"\1có xu hướng \2",
            answer,
            flags=re.IGNORECASE,
        )
        return answer

    def _add_disclaimer_if_needed(self, answer: str) -> str:
        """Add disclaimer for investment-related questions."""
        disclaimer_keywords = [
            "mua", "bán", "đầu tư", "nên", "khuyến nghị",
            "recommendation",
        ]
        needs_disclaimer = any(
            kw in answer.lower() for kw in disclaimer_keywords
        )
        if needs_disclaimer and "Lưu ý" not in answer:
            answer += (
                "\n\n*Lưu ý: Đây chỉ là phân tích tham khảo, "
                "không phải lời khuyên đầu tư. Hãy tham khảo ý kiến "
                "chuyên gia tài chính trước khi đưa ra quyết định.*"
            )
        return answer
