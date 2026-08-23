"""Script to run evaluation on the RAG pipeline."""
import asyncio
import json
import logging
import sys
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import settings
from src.evaluation.evaluator import Evaluator
from src.knowledge_hub.bm25_store import BM25Store
from src.knowledge_hub.retriever import HybridRetriever
from src.knowledge_hub.vector_store import VectorStore
from src.rag_pipeline.pipeline import StockRAGPipeline
from src.rag_pipeline.reranker import CrossEncoderReranker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("VN Stock Analyst - Evaluation")
    logger.info("=" * 60)

    # Initialize pipeline
    vector_store = VectorStore()
    if vector_store.count == 0:
        logger.error("Vector store is empty. Run 'make ingest' first.")
        return

    bm25_store = BM25Store()
    retriever = HybridRetriever(vector_store, bm25_store)

    try:
        reranker = CrossEncoderReranker()
    except Exception:
        logger.warning("Reranker unavailable, proceeding without")
        reranker = None

    pipeline = StockRAGPipeline(
        retriever=retriever,
        reranker=reranker,
        use_hyde=True,
        use_multi_query=True,
    )

    # Run evaluation
    evaluator = Evaluator()
    results = evaluator.run(pipeline, experiment_name="full_pipeline_v1")

    # Print results
    logger.info("\nEvaluation Results:")
    logger.info("-" * 40)
    if "metrics" in results:
        for metric, value in results["metrics"].items():
            if isinstance(value, float):
                logger.info(f"  {metric}: {value:.4f}")
            else:
                logger.info(f"  {metric}: {value}")
    else:
        logger.info(f"  Results: {json.dumps(results, indent=2)}")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
