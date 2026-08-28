"""News search tool for agent."""
import logging

logger = logging.getLogger(__name__)


class NewsSearchTool:
    """Search recent financial news by ticker or keyword."""

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        """
        Search for news articles.
        Returns list of dicts with title, summary, url, date.
        """
        # This is a simplified version using cached/indexed data
        # In production, this would search the vector store
        return [{
            "title": f"Tin tức về {query}",
            "summary": f"Dữ liệu tin tức cho {query} cần được truy xuất từ cơ sở dữ liệu",
            "source": "system",
            "note": "Sử dụng RAG pipeline để tìm tin tức chi tiết",
        }]

    def get_tool_description(self) -> dict:
        return {
            "name": "news_search",
            "description": "Tìm kiếm tin tức tài chính theo mã cổ phiếu hoặc từ khóa",
            "parameters": {
                "query": {
                    "type": "string",
                    "description": "Từ khóa tìm tin tức (VD: FPT, lãi suất, bất động sản)",
                }
            },
        }
