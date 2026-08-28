"""Conversation memory for agent."""
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MemoryMessage:
    """A single message in memory."""

    role: str  # "user" or "assistant"
    content: str
    metadata: dict = field(default_factory=dict)


class ConversationMemory:
    """Simple conversation memory with persistence."""

    def __init__(self, max_messages: int = 20, persist_path: Path | None = None):
        self.max_messages = max_messages
        self.persist_path = persist_path
        self.messages: list[MemoryMessage] = []
        if persist_path and persist_path.exists():
            self._load()

    def add(self, role: str, content: str, **metadata) -> None:
        """Add a message to memory."""
        self.messages.append(MemoryMessage(
            role=role,
            content=content,
            metadata=metadata,
        ))
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
        if self.persist_path:
            self._save()

    def get_context(self, n_messages: int = 5) -> str:
        """Get recent conversation context as a string."""
        recent = self.messages[-n_messages:]
        lines = []
        for msg in recent:
            prefix = "Người dùng" if msg.role == "user" else "Ensa"
            lines.append(f"{prefix}: {msg.content[:500]}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all messages."""
        self.messages.clear()
        if self.persist_path and self.persist_path.exists():
            self.persist_path.unlink()

    def _save(self) -> None:
        data = [
            {"role": m.role, "content": m.content, "metadata": m.metadata}
            for m in self.messages
        ]
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.persist_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load(self) -> None:
        try:
            with open(self.persist_path, encoding="utf-8") as f:
                data = json.load(f)
            self.messages = [
                MemoryMessage(**m) for m in data
            ]
        except Exception:
            self.messages = []
