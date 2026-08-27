"""Abstract base collector for data pipeline."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Document:
    """Represents a raw document collected from a source."""

    content: str
    metadata: dict = field(default_factory=dict)
    source: str = ""
    doc_id: str = ""


class BaseCollector(ABC):
    """Abstract base class for all data collectors."""

    def __init__(self, output_dir: Path, max_items: int = 500):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_items = max_items

    @abstractmethod
    def collect(self) -> list[Document]:
        """Collect documents from the source."""
        ...

    def save(self, documents: list[Document], filename: str) -> Path:
        """Save collected documents to JSON."""
        import json

        output_path = self.output_dir / filename
        data = []
        for doc in documents:
            data.append({
                "content": doc.content,
                "metadata": doc.metadata,
                "source": doc.source,
                "doc_id": doc.doc_id,
            })
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return output_path
