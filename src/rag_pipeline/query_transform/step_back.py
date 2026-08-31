"""Step-back prompting transform."""
import logging

from src.config import settings

logger = logging.getLogger(__name__)

STEP_BACK_PROMPT = """Bạn là chuyên gia tài chính.
Từ câu hỏi cụ thể sau, hãy tạo một câu hỏi tổng quát hơn
để tìm kiếm bối cảnh rộng hơn:

Câu hỏi cụ thể: {query}

Câu hỏi tổng quát hơn:"""


class StepBackTransform:
    """
    Step-back prompting: generate a more general query
    to retrieve broader context.
    """

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self._client = None

    def transform(self, query: str) -> str:
        """Generate a step-back (more general) query."""
        if not self.use_llm:
            return query

        try:
            return self._generate_step_back(query)
        except Exception as e:
            logger.warning(f"Step-back generation failed: {e}")
            return query

    def _generate_step_back(self, query: str) -> str:
        from openai import OpenAI

        if self._client is None:
            self._client = OpenAI(api_key=settings.OPENAI_API_KEY)

        prompt = STEP_BACK_PROMPT.format(query=query)
        response = self._client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3,
        )

        step_back = response.choices[0].message.content.strip()
        logger.debug(f"Step-back: '{query}' → '{step_back}'")
        return step_back
