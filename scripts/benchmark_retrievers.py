"""Benchmark different retriever configurations."""
import json
import logging
import sys
import time
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import settings
from src.knowledge_hub.bm25_store import BM25Store
from src.knowledge_hub.retriever import HybridRetriever, RetrievalResult
from src.knowledge_hub.vector_store import VectorStore
from src.rag_pipeline.reranker import CrossEncoderReranker

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Benchmark queries
BENCHMARK_QUERIES = [
    "P/E của FPT là bao nhiêu?",
    "So sánh VHM và NVL",
    "Giá cổ phiếu VIC hiện tại",
    "Triển vọng ngành ngân hàng 2025",
    "Tỷ suất lợi nhuận gộp của HPG",
    "Lãi suất cho vay mua nhà ngân hàng nào thấp nhất?",
    "Doanh thu của MWG trong quý gần nhất",
    "ROE của TCB so với BID",
]


def benchmark_retriever(name, retriever, query, k=10):
    """Benchmark a single retriever."""
    start = time.time()
    results = retriever.retrieve(query, n_results=k)
    latency = (time.time() - start) * 1000
    return {
        "results": len(results),
        "latency_ms": round(latency, 2),
        "top_score": results[0].score if results else 0,
    }


def main():
    logger.info("=" * 60)
    logger.info("Retriever Benchmark")
    logger.info("=" * 60)

    vector_store = VectorStore()
    if vector_store.count == 0:
        logger.error("Vector store is empty. Run 'make ingest' first.")
        return

    bm25_store = BM25Store()

    # Config A: Dense only
    dense_retriever = HybridRetriever(vector_store, bm25_store)
    dense_retriever.vector_weight = 1.0
    dense_retriever.bm25_weight = 0.0

    # Config B: Sparse only
    sparse_retriever = HybridRetriever(vector_store, bm25_store)
    sparse_retriever.vector_weight = 0.0
    sparse_retriever.bm25_weight = 1.0

    # Config C: Hybrid (default)
    hybrid_retriever = HybridRetriever(vector_store, bm25_store)

    configs = {
        "Dense Only": dense_retriever,
        "BM25 Only": sparse_retriever,
        "Hybrid (RRF)": hybrid_retriever,
    }

    all_results = {}
    for config_name, retriever in configs.items():
        logger.info(f"\nBenchmarking: {config_name}")
        config_results = []
        for query in BENCHMARK_QUERIES:
            result = benchmark_retriever(config_name, retriever, query)
            config_results.append(result)
            logger.info(
                f"  Q: {query[:40]}... → {result['results']} results, "
                f"{result['latency_ms']}ms, score={result['top_score']:.4f}"
            )

        avg_latency = sum(r["latency_ms"] for r in config_results) / len(config_results)
        avg_results = sum(r["results"] for r in config_results) / len(config_results)
        all_results[config_name] = {
            "avg_latency_ms": round(avg_latency, 2),
            "avg_results": round(avg_results, 1),
            "queries": len(config_results),
        }

    logger.info("\n" + "=" * 60)
    logger.info("Summary:")
    for config, metrics in all_results.items():
        logger.info(
            f"  {config}: latency={metrics['avg_latency_ms']}ms, "
            f"results={metrics['avg_results']}"
        )

    output_path = settings.DATA_DIR / "evaluation" / "benchmark_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
