"""HyDE - Hypothetical Document Embeddings transform."""
import logging

from src.config import settings

logger = logging.getLogger(__name__)

HYPOTHESIS_PROMPT = """Bạn là chuyên gia tài chính Việt Nam.
Hãy viết một đoạn trả lời ngắn gọn (100-200 từ) cho câu hỏi sau,
bao gồm các số liệu cụ thể và phân tích:

Câu hỏi: {query}

Trả lời:"""


class HyDETransform:
    """
    Hypothetical Document Embeddings.
    Generates a hypothetical answer, then uses it as the query for retrieval.
    """

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self._client = None

    def transform(self, query: str) -> str:
        """
        Transform query using HyDE.
        Returns the original query if LLM generation fails.
        """
        if not self.use_llm:
            return query

        try:
            return self._generate_hypothesis(query)
        except Exception as e:
            logger.warning(f"HyDE generation failed, using original query: {e}")
            return query

    def _generate_hypothesis(self, query: str) -> str:
        """Generate hypothetical answer using LLM."""
        from openai import OpenAI

        if self._client is None:
            self._client = OpenAI(api_key=settings.OPENAI_API_KEY)

        prompt = HYPOTHESIS_PROMPT.format(query=query)
        response = self._client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.3,
        )

        hypothesis = response.choices[0].message.content.strip()
        logger.debug(f"HyDE: '{query}' → '{hypothesis[:100]}...'")
        return hypothesis
