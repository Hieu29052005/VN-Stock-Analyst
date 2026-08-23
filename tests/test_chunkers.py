"""Tests for chunkers."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.chunkers.recursive_chunker import RecursiveChunker
from src.data_pipeline.chunkers.table_aware import TableAwareChunker


def test_recursive_chunker_basic():
    chunker = RecursiveChunker(chunk_size=200, chunk_overlap=30)
    text = "Đây là đoạn văn bản测试. " * 50
    chunks = chunker.chunk(text)
    assert len(chunks) > 0
    assert all(c.text.strip() for c in chunks)


def test_recursive_chunker_empty():
    chunker = RecursiveChunker()
    assert chunker.chunk("") == []
    assert chunker.chunk("   ") == []


def test_recursive_chunker_short_text():
    chunker = RecursiveChunker(chunk_size=500)
    text = "Short text that fits in one chunk."
    chunks = chunker.chunk(text)
    assert len(chunks) == 1


def test_table_aware_preserves_tables():
    chunker = TableAwareChunker(chunk_size=200)
    text = """Văn bản trước bảng.

| Tên | Giá |
|-----|-----|
| VIC | 100 |
| FPT | 200 |

Văn bản sau bảng."""
    chunks = chunker.chunk(text)
    table_chunks = [c for c in chunks if c.metadata.get("is_table")]
    text_chunks = [c for c in chunks if not c.metadata.get("is_table")]
    assert len(chunks) >= 2
    assert any(c.metadata.get("is_table") for c in chunks)


def test_chunk_metadata():
    chunker = RecursiveChunker(chunk_size=200, chunk_overlap=30)
    text = "Word " * 100
    chunks = chunker.chunk(text, metadata={"ticker": "FPT"})
    assert all(c.metadata["ticker"] == "FPT" for c in chunks)
    assert chunks[0].metadata["total_chunks"] == len(chunks)
