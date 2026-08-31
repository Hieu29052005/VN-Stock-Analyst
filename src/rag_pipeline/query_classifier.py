"""Query intent classifier for routing queries to appropriate retrieval strategy."""
import re
from enum import Enum


class QueryIntent(str, Enum):
    """Classification of user query intent."""

    PRICE_LOOKUP = "price_lookup"
    FUNDAMENTAL = "fundamental"
    COMPARISON = "comparison"
    ANALYSIS = "analysis"
    GENERAL = "general"


class QueryClassifier:
    """Classify user queries by intent using rule-based heuristics."""

    COMPARISON_KEYWORDS = [
        "so sánh", "compare", "vs", "với", "hơn", "kém",
        "giữa", "among", "trong số", "đối chiếu",
    ]
    PRICE_KEYWORDS = [
        "price", "trading", "khop lenh",
        "cao nhất", "thấp nhất", "sàn", "khớp lệnh", "current",
    ]
    PRICE_PATTERNS = [
        re.compile(r"\bgiá\b", re.IGNORECASE),
        re.compile(r"\bbao nhiêu\b", re.IGNORECASE),
    ]
    FUNDAMENTAL_KEYWORDS = [
        "p/e", "p/b", "roe", "roa", "eps", "pe", "pb",
        "tỷ suất", "lợi nhuận", "doanh thu", "vốn hóa",
        "dividend", "cổ tức", "tài chính", "báo cáo",
        "balance sheet", "income", "cash flow", "vay",
        "tỷ lệ", "ratio", "margin", "profit",
    ]
    ANALYSIS_KEYWORDS = [
        "phân tích", "analysis", "đánh giá", "triển vọng",
        "nhận định", "dự báo", "forecast", "tương lai",
        "chiến lược", "strategy", "nên mua", "nên bán",
        "recommendation", "gợi ý", "khuyến nghị",
    ]

    def classify(self, query: str) -> QueryIntent:
        """Classify the query intent."""
        query_lower = query.lower()

        # Check comparison first (strong signal)
        if any(kw in query_lower for kw in self.COMPARISON_KEYWORDS):
            if self._has_multiple_tickers(query):
                return QueryIntent.COMPARISON

        # Check fundamental (strong signal - specific financial terms)
        if any(kw in query_lower for kw in self.FUNDAMENTAL_KEYWORDS):
            return QueryIntent.FUNDAMENTAL

        # Check analysis (before price to avoid "đánh giá" → price match)
        if any(kw in query_lower for kw in self.ANALYSIS_KEYWORDS):
            return QueryIntent.ANALYSIS

        # Check price lookup (use word-boundary patterns)
        if any(p.search(query) for p in self.PRICE_PATTERNS):
            return QueryIntent.PRICE_LOOKUP
        if any(kw in query_lower for kw in self.PRICE_KEYWORDS):
            return QueryIntent.PRICE_LOOKUP

        # Check for comparison even with single ticker
        if any(kw in query_lower for kw in self.COMPARISON_KEYWORDS):
            return QueryIntent.COMPARISON

        return QueryIntent.GENERAL

    def _has_multiple_tickers(self, query: str) -> bool:
        """Check if query mentions multiple stock tickers."""
        known_tickers = {
            "ACB", "BCM", "BID", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
            "MBB", "MSN", "MWG", "PGV", "PHR", "POW", "SAB", "SBT", "SSI",
            "STB", "TCB", "TPB", "VIB", "VIC", "VHM", "VNM", "VPB", "VRE",
            "EIB", "LPB", "SHB", "VND", "PVD", "PLX",
        }
        words = re.findall(r"\b([A-Z]{2,5})\b", query.upper())
        found = [w for w in words if w in known_tickers]
        return len(found) >= 2
