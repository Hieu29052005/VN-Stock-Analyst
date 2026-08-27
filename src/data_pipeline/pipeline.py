"""Data pipeline orchestrator: collect → process → chunk → embed → store."""
import json
import logging
from pathlib import Path

from src.config import settings
from src.data_pipeline.collectors.base import Document
from src.data_pipeline.chunkers.recursive_chunker import RecursiveChunker
from src.data_pipeline.chunkers.table_aware import TableAwareChunker
from src.data_pipeline.metadata import enrich_metadata, generate_chunk_id

logger = logging.getLogger(__name__)


class DataPipeline:
    """Orchestrates the full data ingestion pipeline."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.chunker = TableAwareChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def run_collectors(self, collectors: list) -> list[Document]:
        """Run a list of collectors and aggregate results."""
        all_docs = []
        for collector in collectors:
            try:
                docs = collector.collect()
                all_docs.extend(docs)
                logger.info(f"{collector.__class__.__name__}: collected {len(docs)} docs")
            except Exception as e:
                logger.error(f"{collector.__class__.__name__} failed: {e}")
        return all_docs

    def process_documents(self, documents: list[Document]) -> list[Document]:
        """Clean and normalize document content."""
        processed = []
        for doc in documents:
            content = doc.content.strip()
            if not content or len(content) < 20:
                continue
            doc.content = content
            processed.append(doc)
        logger.info(f"Processed {len(processed)}/{len(documents)} documents")
        return processed

    def chunk_documents(self, documents: list[Document]) -> list[dict]:
        """Chunk documents and enrich metadata."""
        all_chunks = []
        for doc in documents:
            chunks = self.chunker.chunk(
                text=doc.content,
                metadata=doc.metadata,
                chunk_id_prefix=doc.doc_id,
            )
            for chunk in chunks:
                enriched_meta = enrich_metadata(
                    chunk.text, chunk.metadata, source=doc.source
                )
                chunk.chunk_id = generate_chunk_id(
                    chunk.text, prefix=enriched_meta.get("ticker", "doc")
                )
                all_chunks.append({
                    "id": chunk.chunk_id,
                    "text": chunk.text,
                    "metadata": enriched_meta,
                })
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks

    def save_chunks(self, chunks: list[dict], filename: str = "chunks.json") -> Path:
        """Save processed chunks to disk."""
        output_path = settings.PROCESSED_DIR / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(chunks)} chunks to {output_path}")
        return output_path

    def load_chunks(self, filename: str = "chunks.json") -> list[dict]:
        """Load previously processed chunks."""
        path = settings.PROCESSED_DIR / filename
        if not path.exists():
            return []
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def run(self, collectors: list) -> list[dict]:
        """Execute full pipeline: collect → process → chunk → save."""
        docs = self.run_collectors(collectors)
        docs = self.process_documents(docs)
        chunks = self.chunk_documents(docs)
        self.save_chunks(chunks)
        return chunks
