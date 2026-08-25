"""Embedding model wrapper for Vietnamese financial text."""
import logging
from functools import lru_cache

from src.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedding_model(model_name: str | None = None):
    """
    Get or create a cached embedding model instance.

    Uses Vietnamese-optimized BGE-M3 by default for best accuracy.
    """
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    model_name = model_name or settings.EMBEDDING_MODEL
    logger.info(f"Loading embedding model: {model_name}")

    model = HuggingFaceEmbedding(
        model_name=model_name,
        max_length=2048,
        trust_remote_code=True,
    )

    logger.info(f"Embedding model loaded. Dim={settings.EMBEDDING_DIM}")
    return model


def get_embedding_dimension(model_name: str | None = None) -> int:
    """Get the embedding dimension for the model."""
    return settings.EMBEDDING_DIM
