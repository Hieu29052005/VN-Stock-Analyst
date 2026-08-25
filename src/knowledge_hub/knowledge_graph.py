"""Simple knowledge graph for company-sector relationships."""
import json
import logging
from pathlib import Path

import networkx as nx

from src.config import settings

logger = logging.getLogger(__name__)

SECTOR_DATA = {
    "Banking": [
        "BID", "CTG", "MBB", "STB", "TCB", "TPB", "VPB", "ACB",
        "HDB", "VIB", "EIB", "LPB", "SHB",
    ],
    "Real Estate": ["VIC", "VHM", "VRE", "NVL", "KDH"],
    "Technology": ["FPT"],
    "Energy": ["GAS", "POW", "PGV", "NT2", "PPC"],
    "Consumer": ["MSN", "MWG", "SAB", "VNM", "SBT"],
    "Manufacturing": ["HPG", "GVR", "BCM", "PHR"],
    "Securities": ["SSI", "VND"],
}


class KnowledgeGraph:
    """Simple graph for company-sector-industry relationships."""

    def __init__(self, persist_dir: str | Path | None = None):
        self.persist_dir = Path(persist_dir or settings.DATA_DIR / "knowledge_graph")
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.graph_path = self.persist_dir / "kg.json"

        self.graph = nx.DiGraph()
        self._load_or_build()

    def _load_or_build(self) -> None:
        """Load existing graph or build from default sector data."""
        if self.graph_path.exists():
            self._load()
        else:
            self._build_default()
            self.save()

    def _build_default(self) -> None:
        """Build default knowledge graph from sector data."""
        for sector, tickers in SECTOR_DATA.items():
            self.graph.add_node(sector, type="sector")
            for ticker in tickers:
                self.graph.add_node(ticker, type="company")
                self.graph.add_edge(ticker, sector, relation="belongs_to")

    def add_company(self, ticker: str, sector: str, **attrs) -> None:
        """Add a company node and its sector edge."""
        self.graph.add_node(ticker, type="company", **attrs)
        if not self.graph.has_node(sector):
            self.graph.add_node(sector, type="sector")
        self.graph.add_edge(ticker, sector, relation="belongs_to")

    def get_sector(self, ticker: str) -> str:
        """Get the sector for a given ticker."""
        for neighbor in self.graph.successors(ticker):
            data = self.graph[ticker][neighbor]
            if data.get("relation") == "belongs_to":
                return neighbor
        return "Unknown"

    def get_tickers_in_sector(self, sector: str) -> list[str]:
        """Get all tickers in a sector."""
        tickers = []
        for node in self.graph.predecessors(sector):
            if self.graph.nodes[node].get("type") == "company":
                tickers.append(node)
        return tickers

    def get_related_companies(self, ticker: str) -> list[str]:
        """Get companies in the same sector."""
        sector = self.get_sector(ticker)
        if sector == "Unknown":
            return []
        return [t for t in self.get_tickers_in_sector(sector) if t != ticker]

    def save(self) -> None:
        """Save the graph to disk."""
        data = nx.node_link_data(self.graph)
        with open(self.graph_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load(self) -> None:
        """Load graph from disk."""
        try:
            with open(self.graph_path, encoding="utf-8") as f:
                data = json.load(f)
            self.graph = nx.node_link_graph(data, directed=True)
        except Exception as e:
            logger.warning(f"Failed to load knowledge graph: {e}")
            self._build_default()
