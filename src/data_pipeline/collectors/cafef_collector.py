"""Collector for CafeF financial news articles."""
import hashlib
import json
import logging
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from src.config import settings
from src.data_pipeline.collectors.base import BaseCollector, Document

logger = logging.getLogger(__name__)

CAFEF_RSS_URLS = [
    "https://cafef.vn/rss/du-lieu-macro.chn",
    "https://cafef.vn/rss/thi-truong-chung-khoan.chn",
    "https://cafef.vn/rss/doanh-nghiep.chn",
]

CAFEF_SEARCH_URL = "https://cafef.vn/du-lieu.chn"


class CafeFCollector(BaseCollector):
    """Collects financial news from CafeF via RSS feeds and web scraping."""

    def __init__(
        self,
        output_dir: Path | None = None,
        max_items: int = 500,
    ):
        output_dir = output_dir or settings.RAW_DIR / "cafef"
        super().__init__(output_dir, max_items)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; VNStockAnalyst/1.0)"
        })

    def collect(self) -> list[Document]:
        """Collect articles from CafeF RSS feeds."""
        all_docs = []
        seen_ids = set()

        for rss_url in CAFEF_RSS_URLS:
            try:
                docs = self._parse_rss(rss_url)
                for doc in docs:
                    if doc.doc_id not in seen_ids:
                        seen_ids.add(doc.doc_id)
                        all_docs.append(doc)
                logger.info(f"CafeF: Got {len(docs)} from {rss_url}")
                time.sleep(1)
            except Exception as e:
                logger.warning(f"Error fetching {rss_url}: {e}")
                continue

        self.save(all_docs, "cafef_news.json")
        return all_docs[: self.max_items]

    def _parse_rss(self, rss_url: str) -> list[Document]:
        """Parse RSS feed and extract articles."""
        docs = []
        resp = self.session.get(rss_url, timeout=30)
        resp.encoding = "utf-8"

        soup = BeautifulSoup(resp.content, "lxml-xml")
        items = soup.find_all("item")

        for item in items[: self.max_items]:
            try:
                title = item.find("title").text.strip() if item.find("title") else ""
                link = item.find("link").text.strip() if item.find("link") else ""
                description = (
                    item.find("description").text.strip()
                    if item.find("description")
                    else ""
                )
                pub_date = (
                    item.find("pubDate").text.strip()
                    if item.find("pubDate")
                    else ""
                )

                # Try to extract ticker from title
                ticker = self._extract_ticker(title)

                doc_id = hashlib.md5(link.encode()).hexdigest()[:12]

                # Clean HTML from description
                clean_desc = BeautifulSoup(description, "html.parser").get_text()

                docs.append(Document(
                    content=f"{title}\n\n{clean_desc}",
                    metadata={
                        "title": title,
                        "url": link,
                        "date": pub_date,
                        "ticker": ticker,
                        "category": rss_url.split("/")[-1].replace(".chn", ""),
                    },
                    source="cafef",
                    doc_id=doc_id,
                ))
            except Exception as e:
                logger.debug(f"Error parsing item: {e}")
                continue

        return docs

    def _extract_ticker(self, text: str) -> str:
        """Try to extract stock ticker from article title."""
        # Common patterns: (VIC), (VHM), mã VIC
        patterns = [
            r"\(([A-Z]{2,5})\)",
            r"mã\s+([A-Z]{2,5})",
            r"cổ phiếu\s+([A-Z]{2,5})",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return ""
