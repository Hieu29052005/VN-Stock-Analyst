"""HTML to clean text conversion."""
import re
from typing import Optional

from bs4 import BeautifulSoup


class HTMLCleaner:
    """Clean HTML content and extract readable text."""

    def __init__(self, remove_scripts: bool = True, remove_styles: bool = True):
        self.remove_scripts = remove_scripts
        self.remove_styles = remove_styles

    def clean(self, html: str) -> str:
        """
        Clean HTML and return readable text.

        Args:
            html: Raw HTML string

        Returns:
            Cleaned text
        """
        if not html:
            return ""

        soup = BeautifulSoup(html, "html.parser")

        # Remove unwanted elements
        if self.remove_scripts:
            for tag in soup.find_all("script"):
                tag.decompose()
        if self.remove_styles:
            for tag in soup.find_all("style"):
                tag.decompose()

        # Remove common noise elements
        for tag in soup.find_all(["nav", "footer", "header", "aside"]):
            tag.decompose()

        # Remove ads and social widgets
        for tag in soup.find_all(class_=re.compile(r"(ad|social|share|comment|widget)")):
            tag.decompose()

        # Get text
        text = soup.get_text(separator="\n", strip=True)

        # Clean up whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)

        return text.strip()

    def extract_article(self, html: str) -> dict:
        """
        Extract article content from HTML page.

        Returns:
            dict with title, content, date, author
        """
        soup = BeautifulSoup(html, "html.parser")

        # Extract title
        title = ""
        title_tag = soup.find("h1") or soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)

        # Extract article body
        content = ""
        article = (
            soup.find("article")
            or soup.find("div", class_=re.compile(r"(article|content|post-body)"))
            or soup.find("div", id=re.compile(r"(article|content)"))
        )
        if article:
            content = article.get_text(separator="\n", strip=True)
        else:
            # Fallback to main content area
            main = soup.find("main") or soup.find("div", class_="main")
            if main:
                content = main.get_text(separator="\n", strip=True)

        # Extract date
        date = ""
        date_tag = soup.find("time") or soup.find(
            "span", class_=re.compile(r"(date|time|publish)")
        )
        if date_tag:
            date = date_tag.get("datetime", "") or date_tag.get_text(strip=True)

        # Extract author
        author = ""
        author_tag = soup.find(
            "span", class_=re.compile(r"(author|writer|byline)")
        )
        if author_tag:
            author = author_tag.get_text(strip=True)

        return {
            "title": title,
            "content": content,
            "date": date,
            "author": author,
        }
