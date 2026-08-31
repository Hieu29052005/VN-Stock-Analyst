"""Multi-Query retrieval transform."""
import logging

from src.config import settings

logger = logging.getLogger(__name__)

MULTI_QUERY_PROMPT = """Bạn là chuyên gia tài chính Việt Nam.
Hãy tạo {n} câu hỏi khác nhau từ câu hỏi gốc sau,
mỗi câu hỏi khía cạnh khác nhau để cải thiện khả năng tìm kiếm:

Câu hỏi gốc: {query}

Trả lời (mỗi câu trên một dòng, bắt đầu bằng số):
1."""


class MultiQueryTransform:
    """
    Generate multiple query variations and retrieve for each,
    then merge results with RRF.
    """

    def __init__(self, n_variations: int = 3, use_llm: bool = True):
        self.n_variations = n_variations
        self.use_llm = use_llm
        self._client = None

    def transform(self, query: str, n: int | None = None) -> list[str]:
        """
        Generate multiple query variations.
        Returns list of queries including the original.
        """
        n = n or self.n_variations
        queries = [query]

        if not self.use_llm:
            return queries

        try:
            variations = self._generate_variations(query, n)
            queries.extend(variations)
        except Exception as e:
            logger.warning(f"Multi-query generation failed: {e}")

        return queries

    def _generate_variations(self, query: str, n: int) -> list[str]:
        """Generate query variations using LLM."""
        from openai import OpenAI

        if self._client is None:
            self._client = OpenAI(api_key=settings.OPENAI_API_KEY)

        prompt = MULTI_QUERY_PROMPT.format(n=n, query=query)
        response = self._client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.4,
        )

        text = response.choices[0].message.content.strip()
        variations = []
        for line in text.split("\n"):
            line = line.strip()
            # Remove numbering: "1. ", "2. ", etc.
            cleaned = line.lstrip("0123456789. ") .strip()
            if cleaned and len(cleaned) > 10:
                variations.append(cleaned)

        return variations[:n]
