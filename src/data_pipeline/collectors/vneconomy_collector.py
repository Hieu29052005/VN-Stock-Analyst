"""Collector for VnEconomy financial news articles."""
import hashlib
import logging
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from src.config import settings
from src.data_pipeline.collectors.base import BaseCollector, Document

logger = logging.getLogger(__name__)

VNECONOMY_RSS_URLS = [
    "https://vneconomy.vn/rss.htm",
]

VNECONOMY_SECTIONS = [
    "https://vneconomy.vn/tai-chinh-dau-tu.htm",
    "https://vneconomy.vn/kinh-doanh.htm",
]


class VnEconomyCollector(BaseCollector):
    """Collects financial news from VnEconomy."""

    def __init__(
        self,
        output_dir: Path | None = None,
        max_items: int = 500,
    ):
        output_dir = output_dir or settings.RAW_DIR / "vneconomy"
        super().__init__(output_dir, max_items)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; VNStockAnalyst/1.0)"
        })

    def collect(self) -> list[Document]:
        """Collect articles from VnEconomy."""
        all_docs = []
        seen_ids = set()

        # Try RSS first
        for rss_url in VNECONOMY_RSS_URLS:
            try:
                docs = self._parse_rss(rss_url)
                for doc in docs:
                    if doc.doc_id not in seen_ids:
                        seen_ids.add(doc.doc_id)
                        all_docs.append(doc)
                logger.info(f"VnEconomy RSS: Got {len(docs)} articles")
                time.sleep(1)
            except Exception as e:
                logger.warning(f"RSS error {rss_url}: {e}")

        # Scrape sections if needed
        if len(all_docs) < self.max_items:
            for section_url in VNECONOMY_SECTIONS:
                try:
                    docs = self._scrape_section(section_url)
                    for doc in docs:
                        if doc.doc_id not in seen_ids:
                            seen_ids.add(doc.doc_id)
                            all_docs.append(doc)
                    logger.info(f"VnEconomy section: Got {len(docs)} articles")
                    time.sleep(1)
                except Exception as e:
                    logger.warning(f"Section error {section_url}: {e}")

        self.save(all_docs, "vneconomy_news.json")
        return all_docs[: self.max_items]

    def _parse_rss(self, rss_url: str) -> list[Document]:
        """Parse VnEconomy RSS feed."""
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

                ticker = self._extract_ticker(title)
                doc_id = hashlib.md5(link.encode()).hexdigest()[:12]

                clean_desc = BeautifulSoup(description, "html.parser").get_text()

                docs.append(Document(
                    content=f"{title}\n\n{clean_desc}",
                    metadata={
                        "title": title,
                        "url": link,
                        "date": pub_date,
                        "ticker": ticker,
                    },
                    source="vneconomy",
                    doc_id=doc_id,
                ))
            except Exception as e:
                logger.debug(f"Error parsing RSS item: {e}")
                continue

        return docs

    def _scrape_section(self, section_url: str) -> list[Document]:
        """Scrape a section page for article links."""
        docs = []
        resp = self.session.get(section_url, timeout=30)
        resp.encoding = "utf-8"

        soup = BeautifulSoup(resp.text, "html.parser")
        articles = soup.find_all("article") or soup.find_all("div", class_="item-news")

        for article in articles[:50]:
            try:
                link_tag = article.find("a", href=True)
                if not link_tag:
                    continue

                title = link_tag.text.strip()
                url = link_tag["href"]
                if not url.startswith("http"):
                    url = f"https://vneconomy.vn{url}"

                ticker = self._extract_ticker(title)
                doc_id = hashlib.md5(url.encode()).hexdigest()[:12]

                # Get summary if available
                summary_tag = article.find("p") or article.find("div", class_="sapo")
                summary = summary_tag.text.strip() if summary_tag else ""

                docs.append(Document(
                    content=f"{title}\n\n{summary}" if summary else title,
                    metadata={
                        "title": title,
                        "url": url,
                        "ticker": ticker,
                    },
                    source="vneconomy",
                    doc_id=doc_id,
                ))
            except Exception as e:
                logger.debug(f"Error scraping article: {e}")
                continue

        return docs

    def _extract_ticker(self, text: str) -> str:
        """Try to extract stock ticker from article title."""
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
