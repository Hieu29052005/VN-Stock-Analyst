"""LangGraph agent with tools, memory, and RAG integration."""
import logging
from dataclasses import dataclass, field

from src.agents.memory import ConversationMemory
from src.agents.tools.financial_ratio_tool import FinancialRatioTool
from src.agents.tools.news_search_tool import NewsSearchTool
from src.agents.tools.stock_price_tool import StockPriceTool
from src.config import settings

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Response from the agent."""

    answer: str
    tool_used: str = ""
    sources: list[dict] = field(default_factory=list)


class StockAgent:
    """
    AI agent for stock research with tools.
    Routes between RAG pipeline and tools based on query.
    """

    def __init__(self, rag_pipeline=None):
        self.rag_pipeline = rag_pipeline
        self.price_tool = StockPriceTool()
        self.ratio_tool = FinancialRatioTool()
        self.news_tool = NewsSearchTool()
        self.memory = ConversationMemory(max_messages=20)

        self.tools = {
            "stock_price": self.price_tool,
            "financial_ratio": self.ratio_tool,
            "news_search": self.news_tool,
        }

        self.tool_descriptions = [
            self.price_tool.get_tool_description(),
            self.ratio_tool.get_tool_description(),
            self.news_tool.get_tool_description(),
        ]

    async def query(self, question: str) -> AgentResponse:
        """
        Process a question: route to tool or RAG pipeline.
        """
        self.memory.add("user", question)

        intent = self._classify_tool_need(question)

        if intent == "price":
            return await self._handle_price_query(question)
        elif intent == "ratio":
            return await self._handle_ratio_query(question)
        elif intent == "news":
            return await self._handle_news_query(question)
        else:
            return await self._handle_rag_query(question)

    def _classify_tool_need(self, question: str) -> str:
        """Simple rule-based routing."""
        q = question.lower()
        price_kw = ["giá", "price", "bao nhiêu tiền", "đang trading", "khop lenh"]
        ratio_kw = ["p/e", "p/b", "roe", "roa", "tỷ số", "tỷ lệ", "ratio"]
        news_kw = ["tin tức", "news", "báo mới", "mới nhất"]

        if any(kw in q for kw in price_kw):
            return "price"
        if any(kw in q for kw in ratio_kw):
            return "ratio"
        if any(kw in q for kw in news_kw):
            return "news"
        return "rag"

    async def _handle_price_query(self, question: str) -> AgentResponse:
        """Handle price lookup queries."""
        import re
        known_tickers = {
            "ACB", "BCM", "BID", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
            "MBB", "MSN", "MWG", "PGV", "PHR", "POW", "SAB", "SBT", "SSI",
            "STB", "TCB", "TPB", "VIB", "VIC", "VHM", "VNM", "VPB", "VRE",
        }
        words = re.findall(r"\b([A-Z]{2,5})\b", question.upper())
        ticker = next((w for w in words if w in known_tickers), None)

        if ticker:
            result = self.price_tool.get_price(ticker)
            self.memory.add("assistant", str(result), tool="stock_price")
            return AgentResponse(
                answer=f"Giá cổ phiếu {ticker}: {result}",
                tool_used="stock_price",
            )

        return AgentResponse(
            answer="Vui lòng cung cấp mã cổ phiếu cụ thể để tra cứu giá."
        )

    async def _handle_ratio_query(self, question: str) -> AgentResponse:
        """Handle financial ratio queries."""
        import re
        known_tickers = {
            "ACB", "BID", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
            "MBB", "MSN", "MWG", "SSI", "STB", "TCB", "TPB", "VIC", "VHM", "VNM", "VPB", "VRE",
        }
        words = re.findall(r"\b([A-Z]{2,5})\b", question.upper())
        ticker = next((w for w in words if w in known_tickers), None)

        if ticker:
            result = self.ratio_tool.calculate(ticker)
            self.memory.add("assistant", str(result), tool="financial_ratio")
            return AgentResponse(
                answer=f"Tỷ số tài chính {ticker}: {result}",
                tool_used="financial_ratio",
            )

        return await self._handle_rag_query(question)

    async def _handle_news_query(self, question: str) -> AgentResponse:
        """Handle news search queries."""
        result = self.news_tool.search(question)
        self.memory.add("assistant", str(result), tool="news_search")
        return AgentResponse(
            answer=f"Tin tức liên quan: {result}",
            tool_used="news_search",
        )

    async def _handle_rag_query(self, question: str) -> AgentResponse:
        """Handle queries using RAG pipeline."""
        if self.rag_pipeline is None:
            return AgentResponse(
                answer="RAG pipeline chưa được khởi tạo. Vui lòng chạy ingest_data trước."
            )

        response = await self.rag_pipeline.query(question)
        self.memory.add("assistant", response.answer)
        return AgentResponse(
            answer=response.answer,
            sources=response.sources,
        )

    def get_tools_prompt(self) -> str:
        """Get tool descriptions for prompt."""
        lines = ["Công cụ có sẵn:"]
        for tool in self.tool_descriptions:
            lines.append(f"- {tool['name']}: {tool['description']}")
        return "\n".join(lines)
