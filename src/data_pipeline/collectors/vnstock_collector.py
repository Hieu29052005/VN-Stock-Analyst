"""Collector for Vietnamese stock financial data using vnstock library."""
import json
import logging
import time
from pathlib import Path

from src.config import settings
from src.data_pipeline.collectors.base import BaseCollector, Document

logger = logging.getLogger(__name__)


class VNStockCollector(BaseCollector):
    """Collects financial statements and company profiles from vnstock."""

    def __init__(
        self,
        output_dir: Path | None = None,
        tickers: list[str] | None = None,
        max_items: int = 500,
    ):
        output_dir = output_dir or settings.RAW_DIR / "vnstock"
        super().__init__(output_dir, max_items)
        self.tickers = tickers or settings.VN30_TICKERS[: settings.MAX_STOCKS]

    def collect(self) -> list[Document]:
        """Collect financial data for all tickers."""
        from vnstock import Vnstock

        all_docs = []
        for ticker in self.tickers:
            try:
                docs = self._collect_ticker(Vnstock, ticker)
                all_docs.extend(docs)
                logger.info(f"Collected {len(docs)} docs for {ticker}")
                time.sleep(1)  # Rate limiting
            except Exception as e:
                logger.warning(f"Error collecting {ticker}: {e}")
                continue

        self.save(all_docs, "vnstock_financial.json")
        return all_docs

    def _collect_ticker(self, vnstock_cls, ticker: str) -> list[Document]:
        """Collect data for a single ticker."""
        docs = []
        stock = vnstock_cls().stock(symbol=ticker, source="VPS")

        # Company profile
        try:
            profile = stock.company_profile()
            if profile is not None and not profile.empty:
                profile_text = self._profile_to_text(profile, ticker)
                docs.append(Document(
                    content=profile_text,
                    metadata={
                        "ticker": ticker,
                        "doc_type": "company_profile",
                        "source": "vnstock",
                    },
                    source="vnstock",
                    doc_id=f"{ticker}_profile",
                ))
        except Exception as e:
            logger.debug(f"Profile error for {ticker}: {e}")

        # Income statement
        try:
            income = stock.finance.income_statement(period="quarter", lang="vi")
            if income is not None and not income.empty:
                income_text = self._df_to_text(income, ticker, "Báo cáo kết quả kinh doanh")
                docs.append(Document(
                    content=income_text,
                    metadata={
                        "ticker": ticker,
                        "doc_type": "income_statement",
                        "source": "vnstock",
                    },
                    source="vnstock",
                    doc_id=f"{ticker}_income",
                ))
        except Exception as e:
            logger.debug(f"Income error for {ticker}: {e}")

        # Balance sheet
        try:
            balance = stock.finance.balance_sheet(period="quarter", lang="vi")
            if balance is not None and not balance.empty:
                balance_text = self._df_to_text(balance, ticker, "Báo cáo cân đối kế toán")
                docs.append(Document(
                    content=balance_text,
                    metadata={
                        "ticker": ticker,
                        "doc_type": "balance_sheet",
                        "source": "vnstock",
                    },
                    source="vnstock",
                    doc_id=f"{ticker}_balance",
                ))
        except Exception as e:
            logger.debug(f"Balance error for {ticker}: {e}")

        # Cash flow
        try:
            cashflow = stock.finance.cash_flow(period="quarter", lang="vi")
            if cashflow is not None and not cashflow.empty:
                cf_text = self._df_to_text(cashflow, ticker, "Báo cáo lưu chuyển tiền tệ")
                docs.append(Document(
                    content=cf_text,
                    metadata={
                        "ticker": ticker,
                        "doc_type": "cash_flow",
                        "source": "vnstock",
                    },
                    source="vnstock",
                    doc_id=f"{ticker}_cashflow",
                ))
        except Exception as e:
            logger.debug(f"Cash flow error for {ticker}: {e}")

        # Ratio analysis
        try:
            ratio = stock.finance.ratio(period="quarter", lang="vi")
            if ratio is not None and not ratio.empty:
                ratio_text = self._df_to_text(ratio, ticker, "Phân tích tỷ số tài chính")
                docs.append(Document(
                    content=ratio_text,
                    metadata={
                        "ticker": ticker,
                        "doc_type": "financial_ratio",
                        "source": "vnstock",
                    },
                    source="vnstock",
                    doc_id=f"{ticker}_ratio",
                ))
        except Exception as e:
            logger.debug(f"Ratio error for {ticker}: {e}")

        return docs

    def _df_to_text(self, df, ticker: str, title: str) -> str:
        """Convert a DataFrame to readable text."""
        lines = [f"# {title} - {ticker}", ""]
        for col in df.columns:
            lines.append(f"## Kỳ: {col}")
            for idx in df.index:
                val = df.loc[idx, col]
                if val is not None and str(val) != "nan":
                    lines.append(f"- {idx}: {val}")
            lines.append("")
        return "\n".join(lines)

    def _profile_to_text(self, profile, ticker: str) -> str:
        """Convert company profile DataFrame to text."""
        lines = [f"# Hồ sơ công ty - {ticker}", ""]
        if hasattr(profile, "columns") and len(profile.columns) > 0:
            for col in profile.columns:
                val = profile[col].iloc[0] if len(profile) > 0 else ""
                if val is not None and str(val) != "nan":
                    lines.append(f"- {col}: {val}")
        return "\n".join(lines)
