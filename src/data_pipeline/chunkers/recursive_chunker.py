"""Recursive text splitter for chunking documents."""
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A text chunk with metadata."""

    text: str
    metadata: dict = field(default_factory=dict)
    chunk_id: str = ""
    start_idx: int = 0
    end_idx: int = 0


class RecursiveChunker:
    """
    Recursively split text into chunks, trying to preserve
    semantic boundaries (paragraphs, sentences, words).
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 78,
        separators: list[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ".", " ", ""]

    def chunk(
        self,
        text: str,
        metadata: dict | None = None,
        chunk_id_prefix: str = "",
    ) -> list[Chunk]:
        """
        Split text into chunks recursively.

        Args:
            text: Input text to chunk
            metadata: Optional metadata to attach to each chunk
            chunk_id_prefix: Prefix for chunk IDs

        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            return []

        metadata = metadata or {}
        raw_chunks = self._recursive_split(text, self.separators)

        # Merge small chunks and handle overlap
        chunks = self._merge_chunks(raw_chunks)

        # Create Chunk objects
        result = []
        offset = 0
        for i, chunk_text in enumerate(chunks):
            start = text.find(chunk_text[:50], max(0, offset - 10))
            if start == -1:
                start = offset
            end = start + len(chunk_text)

            result.append(Chunk(
                text=chunk_text,
                metadata={**metadata, "chunk_index": i, "total_chunks": len(chunks)},
                chunk_id=f"{chunk_id_prefix}_chunk_{i}" if chunk_id_prefix else f"chunk_{i}",
                start_idx=start,
                end_idx=end,
            ))
            offset = end - self.chunk_overlap

        return result

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using separators in priority order."""
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        if not separators:
            return self._split_by_length(text)

        sep = separators[0]
        remaining_separators = separators[1:] if len(separators) > 1 else [""]

        if sep == "":
            return self._split_by_length(text)

        parts = text.split(sep)
        chunks = []
        current = ""

        for part in parts:
            test = current + sep + part if current else part
            if len(test) <= self.chunk_size:
                current = test
            else:
                if current:
                    chunks.extend(self._recursive_split(current, remaining_separators))
                current = part

        if current:
            if len(current) <= self.chunk_size:
                chunks.append(current.strip())
            else:
                chunks.extend(self._recursive_split(current, remaining_separators))

        return [c for c in chunks if c.strip()]

    def _split_by_length(self, text: str) -> list[str]:
        """Split text by character count as a last resort."""
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i : i + self.chunk_size].strip()
            if chunk:
                chunks.append(chunk)
        return chunks

    def _merge_chunks(self, raw_chunks: list[str]) -> list[str]:
        """Merge small chunks respecting chunk_size and overlap."""
        if not raw_chunks:
            return []

        merged = []
        current = ""

        for chunk in raw_chunks:
            if not chunk.strip():
                continue
            if len(current) + len(chunk) + 1 <= self.chunk_size:
                current = current + "\n" + chunk if current else chunk
            else:
                if current:
                    merged.append(current.strip())
                current = chunk

        if current.strip():
            merged.append(current.strip())

        return merged
