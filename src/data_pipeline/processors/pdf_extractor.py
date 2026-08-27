"""PDF text and table extraction using pdfplumber."""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PDFExtractor:
    """Extract text and tables from PDF files using pdfplumber."""

    def __init__(self, extract_tables: bool = True):
        self.extract_tables = extract_tables

    def extract(self, pdf_path: Path) -> dict:
        """
        Extract content from a PDF file.

        Returns:
            dict with keys: text, tables, page_contents
        """
        import pdfplumber

        result = {
            "text": "",
            "tables": [],
            "page_contents": [],
            "source": str(pdf_path),
        }

        with pdfplumber.open(pdf_path) as pdf:
            all_text = []
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                all_text.append(page_text)

                page_content = {
                    "page": page_num + 1,
                    "text": page_text,
                    "tables": [],
                }

                if self.extract_tables:
                    tables = page.extract_tables()
                    for table in tables:
                        if table and len(table) > 1:
                            table_text = self._format_table(table)
                            page_content["tables"].append(table_text)
                            result["tables"].append(table_text)

                result["page_contents"].append(page_content)

            result["text"] = "\n\n".join(all_text)

        return result

    def extract_from_bytes(self, pdf_bytes: bytes, filename: str = "") -> dict:
        """Extract content from PDF bytes."""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            tmp.write(pdf_bytes)
            tmp.flush()
            return self.extract(Path(tmp.name))

    def _format_table(self, table: list[list]) -> str:
        """Format a table (list of lists) into readable text."""
        if not table or not table[0]:
            return ""

        # Filter out empty rows
        table = [row for row in table if any(cell for cell in row if cell)]

        if len(table) < 2:
            return ""

        # Use first row as headers
        headers = [str(cell).strip() if cell else "" for cell in table[0]]

        lines = []
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        for row in table[1:]:
            cells = [str(cell).strip() if cell else "" for cell in row]
            # Pad if needed
            while len(cells) < len(headers):
                cells.append("")
            lines.append("| " + " | ".join(cells[: len(headers)]) + " |")

        return "\n".join(lines)
