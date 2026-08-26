"""Evaluation runner for the RAG pipeline."""
import json
import logging
from pathlib import Path

from src.config import settings
from src.evaluation.metrics.ragas_metrics import RAGASEvaluator

logger = logging.getLogger(__name__)


class Evaluator:
    """Run evaluation suite on the RAG pipeline."""

    def __init__(self, judge_model: str | None = None):
        self.dataset_path = settings.DATA_DIR / "evaluation" / "questions.jsonl"
        self.results_dir = settings.DATA_DIR / "evaluation" / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.ragas_eval = RAGASEvaluator(
            judge_model=judge_model or settings.JUDGE_MODEL,
        )

    def load_dataset(self) -> list[dict]:
        """Load the evaluation dataset."""
        if not self.dataset_path.exists():
            logger.warning(f"Dataset not found: {self.dataset_path}")
            return []

        data = []
        with open(self.dataset_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    def run(
        self,
        pipeline,
        experiment_name: str = "default",
    ) -> dict:
        """
        Run evaluation on a RAG pipeline.

        Args:
            pipeline: StockRAGPipeline instance
            experiment_name: Name for this experiment

        Returns:
            Evaluation results dict
        """
        dataset = self.load_dataset()
        if not dataset:
            return {"error": "No dataset found"}

        logger.info(f"Running evaluation: {len(dataset)} questions")

        def pipeline_fn(question: str) -> dict:
            import asyncio
            loop = asyncio.new_event_loop()
            try:
                response = loop.run_until_complete(pipeline.query(question))
                return {
                    "answer": response.answer,
                    "contexts": [s["text"] for s in response.sources],
                }
            finally:
                loop.close()

        results = self.ragas_eval.evaluate(dataset, pipeline_fn)

        # Save results
        output = {
            "experiment": experiment_name,
            "dataset_size": len(dataset),
            "metrics": results,
        }
        output_path = self.results_dir / f"{experiment_name}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        logger.info(f"Results saved to {output_path}")
        return output
