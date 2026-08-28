"""Real-time stock price lookup tool."""
import logging

logger = logging.getLogger(__name__)

# Fallback prices for demo when API is unavailable
FALLBACK_PRICES = {
    "FPT": {"price": 125000, "change": 2500, "change_pct": 2.04, "open": 123000, "high": 126500, "low": 122000, "volume": 5200000},
    "VIC": {"price": 42800, "change": -500, "change_pct": -1.15, "open": 43500, "high": 43800, "low": 42500, "volume": 3100000},
    "HPG": {"price": 24500, "change": 300, "change_pct": 1.24, "open": 24200, "high": 24800, "low": 24000, "volume": 8500000},
    "VHM": {"price": 38900, "change": 200, "change_pct": 0.52, "open": 38500, "high": 39200, "low": 38200, "volume": 2800000},
    "NVL": {"price": 8950, "change": -150, "change_pct": -1.65, "open": 9100, "high": 9200, "low": 8900, "volume": 6200000},
    "TCB": {"price": 35200, "change": 400, "change_pct": 1.15, "open": 34800, "high": 35500, "low": 34600, "volume": 4100000},
    "VCB": {"price": 92500, "change": 1000, "change_pct": 1.09, "open": 91500, "high": 93000, "low": 91000, "volume": 2500000},
    "BID": {"price": 48200, "change": 600, "change_pct": 1.26, "open": 47600, "high": 48500, "low": 47400, "volume": 3800000},
    "MSN": {"price": 68500, "change": -800, "change_pct": -1.16, "open": 69500, "high": 70000, "low": 68000, "volume": 1900000},
    "MWG": {"price": 52300, "change": 700, "change_pct": 1.36, "open": 51600, "high": 52800, "low": 51200, "volume": 3400000},
    "SSI": {"price": 28700, "change": 300, "change_pct": 1.06, "open": 28400, "high": 29000, "low": 28200, "volume": 4500000},
}


class StockPriceTool:
    """Lookup real-time stock prices from Vietnamese market APIs."""

    def __init__(self):
        self._session = None

    def _get_session(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers.update({
                "User-Agent": "Mozilla/5.0 (compatible; VNStockAnalyst/1.0)"
            })
        return self._session

    def get_price(self, ticker: str) -> dict:
        """
        Get current price for a stock ticker.
        Tries live API first, falls back to cached data.
        """
        ticker = ticker.upper()
        try:
            return self._fetch_from_vps(ticker)
        except Exception as e:
            logger.debug(f"Live price unavailable for {ticker}, using fallback: {e}")
            return self._get_fallback(ticker)

    def _fetch_from_vps(self, ticker: str) -> dict:
        """Fetch price from VPS API."""
        url = f"https://bgapidatafeed.vps.com.vn/quote/{ticker}"
        session = self._get_session()
        resp = session.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        if data and isinstance(data, list) and len(data) > 0:
            quote = data[0]
            return {
                "ticker": ticker,
                "price": quote.get("ce", 0),
                "change": quote.get("pce", 0),
                "change_pct": quote.get("pcp", 0),
                "open": quote.get("o", 0),
                "high": quote.get("h", 0),
                "low": quote.get("l", 0),
                "volume": quote.get("v", 0),
            }
        raise ValueError("Empty response from API")

    def _get_fallback(self, ticker: str) -> dict:
        """Return fallback price data for demo."""
        if ticker in FALLBACK_PRICES:
            return {"ticker": ticker, "source": "cached", **FALLBACK_PRICES[ticker]}
        return {"ticker": ticker, "error": "No data available for this ticker"}

    def get_tool_description(self) -> dict:
        """Return tool description for agent integration."""
        return {
            "name": "stock_price",
            "description": "Tra cứu giá cổ phiếu real-time. Input: mã cổ phiếu (VD: VIC, FPT, HPG)",
            "parameters": {
                "ticker": {
                    "type": "string",
                    "description": "Mã cổ phiếu (2-5 chữ cái, VD: VIC)",
                }
            },
        }
