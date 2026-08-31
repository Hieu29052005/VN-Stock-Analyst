"""LLM generator for RAG pipeline responses."""
import logging

from src.config import settings

logger = logging.getLogger(__name__)


class LLMGenerator:
    """Generate responses using LLM with the built context."""

    def __init__(self, model: str | None = None, temperature: float | None = None):
        self.model = model or settings.LLM_MODEL
        self.temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        self.max_tokens = settings.LLM_MAX_TOKENS
        self._client = None

    def _get_client(self):
        """Lazy-load OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=settings.OPENAI_API_KEY)
        return self._client

    async def generate(
        self,
        prompt: str,
        system_message: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: The full prompt with context and query
            system_message: Optional system message
            temperature: Override temperature
            max_tokens: Override max tokens

        Returns:
            Generated text response
        """
        client = self._get_client()

        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
            )
            answer = response.choices[0].message.content.strip()
            logger.debug(f"LLM response: {len(answer)} chars")
            return answer

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return (
                "Xin lỗi, tôi gặp lỗi kỹ thuật khi xử lý câu hỏi của bạn. "
                "Vui lòng thử lại sau."
            )

    def generate_sync(self, prompt: str, system_message: str = "") -> str:
        """Synchronous version of generate."""
        client = self._get_client()

        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return (
                "Xin lỗi, tôi gặp lỗi kỹ thuật khi xử lý câu hỏi. "
                "Vui lòng thử lại sau."
            )
