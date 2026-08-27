"""Table-aware chunking that preserves tables as whole chunks."""
import re
from dataclasses import dataclass, field

from src.data_pipeline.chunkers.recursive_chunker import Chunk, RecursiveChunker
from src.data_pipeline.processors.table_extractor import TableExtractor


class TableAwareChunker:
    """
    Chunks documents while preserving tables as whole units.
    Tables are kept intact even if they exceed chunk_size.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 78,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.recursive_chunker = RecursiveChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.table_extractor = TableExtractor()

    def chunk(
        self,
        text: str,
        metadata: dict | None = None,
        chunk_id_prefix: str = "",
    ) -> list[Chunk]:
        """
        Chunk text with table preservation.

        Strategy:
        1. Split text into segments (table vs non-table)
        2. Tables become individual chunks (preserved whole)
        3. Non-table text is chunked recursively
        """
        if not text or not text.strip():
            return []

        metadata = metadata or {}

        # Split into table and non-table segments
        segments = self._split_segments(text)

        chunks = []
        chunk_idx = 0

        for segment_type, segment_text in segments:
            if segment_type == "table":
                # Preserve table as whole chunk
                table_text = self._format_table_segment(segment_text)
                if table_text.strip():
                    chunks.append(Chunk(
                        text=table_text,
                        metadata={
                            **metadata,
                            "is_table": True,
                            "chunk_index": chunk_idx,
                        },
                        chunk_id=f"{chunk_id_prefix}_table_{chunk_idx}" if chunk_id_prefix else f"table_{chunk_idx}",
                    ))
                    chunk_idx += 1
            else:
                # Chunk non-table text normally
                text_chunks = self.recursive_chunker.chunk(
                    segment_text,
                    metadata={**metadata, "is_table": False},
                    chunk_id_prefix=chunk_id_prefix,
                )
                for tc in text_chunks:
                    tc.metadata["chunk_index"] = chunk_idx
                    chunks.append(tc)
                    chunk_idx += 1

        # Update total_chunks
        for chunk in chunks:
            chunk.metadata["total_chunks"] = len(chunks)

        return chunks

    def _split_segments(self, text: str) -> list[tuple[str, str]]:
        """
        Split text into alternating table/non-table segments.
        Returns list of (type, text) tuples.
        """
        segments = []
        lines = text.split("\n")
        current_lines = []
        current_type = "text"
        table_buffer = []

        for line in lines:
            stripped = line.strip()
            is_table_line = stripped.startswith("|") and stripped.endswith("|")

            if is_table_line:
                if current_type == "text" and current_lines:
                    segments.append(("text", "\n".join(current_lines)))
                    current_lines = []
                current_type = "table"
                table_buffer.append(line)
            else:
                if current_type == "table" and table_buffer:
                    segments.append(("table", "\n".join(table_buffer)))
                    table_buffer = []
                current_type = "text"
                current_lines.append(line)

        # Flush remaining
        if current_type == "table" and table_buffer:
            segments.append(("table", "\n".join(table_buffer)))
        elif current_lines:
            segments.append(("text", "\n".join(current_lines)))

        return segments

    def _format_table_segment(self, table_text: str) -> str:
        """Format a table segment, merging if multiple tables are adjacent."""
        tables = self.table_extractor.extract_from_text(table_text)
        if not tables:
            return table_text

        # Merge consecutive similar tables
        merged = self.table_extractor.merge_consecutive_tables(tables)

        formatted = []
        for table in merged:
            formatted.append(self.table_extractor.format_as_text(table))

        return "\n\n".join(formatted)
