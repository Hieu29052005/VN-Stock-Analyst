"""Financial ratio calculation tool."""
import logging

logger = logging.getLogger(__name__)


class FinancialRatioTool:
    """Calculate and look up financial ratios for Vietnamese stocks."""

    def calculate(self, ticker: str, financial_data: dict | None = None) -> dict:
        """
        Calculate financial ratios from available data.

        If no financial_data provided, returns cached/known ratios.
        """
        ticker = ticker.upper()

        if financial_data:
            return self._calculate_from_data(financial_data, ticker)

        return self._get_cached_ratios(ticker)

    def _calculate_from_data(self, data: dict, ticker: str) -> dict:
        """Calculate ratios from financial statement data."""
        ratios = {"ticker": ticker}

        # P/E = Price / EPS
        price = data.get("price", 0)
        eps = data.get("eps", 0)
        if eps > 0:
            ratios["pe"] = round(price / eps, 2)

        # P/B = Price / Book Value per Share
        bvps = data.get("book_value_per_share", 0)
        if bvps > 0:
            ratios["pb"] = round(price / bvps, 2)

        # ROE = Net Income / Total Equity
        net_income = data.get("net_income", 0)
        equity = data.get("total_equity", 0)
        if equity > 0:
            ratios["roe"] = round((net_income / equity) * 100, 2)

        # ROA = Net Income / Total Assets
        assets = data.get("total_assets", 0)
        if assets > 0:
            ratios["roa"] = round((net_income / assets) * 100, 2)

        # Gross Margin = Gross Profit / Revenue
        gross_profit = data.get("gross_profit", 0)
        revenue = data.get("revenue", 0)
        if revenue > 0:
            ratios["gross_margin"] = round((gross_profit / revenue) * 100, 2)

        # Debt/Equity
        debt = data.get("total_debt", 0)
        if equity > 0:
            ratios["debt_equity"] = round(debt / equity, 2)

        return ratios

    def _get_cached_ratios(self, ticker: str) -> dict:
        """Return known approximate ratios (for demo/fallback)."""
        known_ratios = {
            "VIC": {"pe": 28.5, "pb": 2.1, "roe": 7.5, "sector": "Real Estate"},
            "VHM": {"pe": 12.3, "pb": 2.8, "roe": 22.1, "sector": "Real Estate"},
            "FPT": {"pe": 25.0, "pb": 5.5, "roe": 22.0, "sector": "Technology"},
            "HPG": {"pe": 10.5, "pb": 1.8, "roe": 17.2, "sector": "Manufacturing"},
            "VNM": {"pe": 18.0, "pb": 4.2, "roe": 23.5, "sector": "Consumer"},
            "TCB": {"pe": 8.5, "pb": 1.5, "roe": 17.8, "sector": "Banking"},
            "BID": {"pe": 9.2, "pb": 1.3, "roe": 14.2, "sector": "Banking"},
            "MBB": {"pe": 7.8, "pb": 1.2, "roe": 15.5, "sector": "Banking"},
            "SSI": {"pe": 15.0, "pb": 1.8, "roe": 12.0, "sector": "Securities"},
            "MWG": {"pe": 22.0, "pb": 4.5, "roe": 20.5, "sector": "Consumer"},
        }

        if ticker in known_ratios:
            return {"ticker": ticker, **known_ratios[ticker]}
        return {"ticker": ticker, "error": "No data available for this ticker"}

    def get_tool_description(self) -> dict:
        """Return tool description for agent integration."""
        return {
            "name": "financial_ratio",
            "description": (
                "Tra cứu hoặc tính toán các tỷ số tài chính của cổ phiếu "
                "(P/E, P/B, ROE, ROA, Gross Margin, Debt/Equity). "
                "Input: mã cổ phiếu"
            ),
            "parameters": {
                "ticker": {
                    "type": "string",
                    "description": "Mã cổ phiếu (VD: VIC, FPT)",
                }
            },
        }
