"""Table extraction from various document formats."""
import re


class TableExtractor:
    """Extract and format tables from text and structured data."""

    def extract_from_text(self, text: str) -> list[list[list[str]]]:
        """
        Detect and extract markdown-style tables from text.

        Returns:
            List of tables, each table is a list of rows,
            each row is a list of cells.
        """
        tables = []
        lines = text.split("\n")
        current_table = []
        in_table = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("|") and stripped.endswith("|"):
                # Check if this is a separator row
                cells = [c.strip() for c in stripped.split("|")[1:-1]]
                if all(re.match(r"^[-:]+$", c) for c in cells if c):
                    # This is a separator row, skip it
                    continue
                if cells:
                    current_table.append(cells)
                    in_table = True
            else:
                if in_table and current_table:
                    if len(current_table) > 1:
                        tables.append(current_table)
                    current_table = []
                    in_table = False

        if current_table and len(current_table) > 1:
            tables.append(current_table)

        return tables

    def format_as_text(
        self, table: list[list[str]], title: str = ""
    ) -> str:
        """
        Format a table as readable text with labels.

        Args:
            table: List of rows, each row is list of cells
            title: Optional table title

        Returns:
            Formatted text representation
        """
        if not table or len(table) < 2:
            return ""

        lines = []
        if title:
            lines.append(f"## {title}")
            lines.append("")

        headers = table[0]
        for row in table[1:]:
            row_text_parts = []
            for i, cell in enumerate(row):
                if i < len(headers) and headers[i]:
                    row_text_parts.append(f"{headers[i]}: {cell}")
                elif cell:
                    row_text_parts.append(cell)
            if row_text_parts:
                lines.append("- " + ", ".join(row_text_parts))

        return "\n".join(lines)

    def merge_consecutive_tables(
        self, tables: list[list[list[str]]], max_size: int = 3600
    ) -> list[list[list[str]]]:
        """
        Merge consecutive tables that might be parts of the same table
        (e.g., when a table spans multiple pages).
        """
        if not tables:
            return []

        merged = [tables[0]]
        for table in tables[1:]:
            last = merged[-1]
            # Check if headers are similar
            if self._headers_similar(last[0], table[0]):
                # Merge rows (skip header of second table)
                merged[-1] = last + table[1:]
            else:
                merged.append(table)

        # Split oversized tables
        result = []
        for table in merged:
            text = self.format_as_text(table)
            if len(text) > max_size:
                # Split by rows
                chunk_size = max(2, len(table) // ((len(text) // max_size) + 1))
                for i in range(0, len(table), chunk_size):
                    chunk = table[i : i + chunk_size]
                    if i > 0:
                        chunk = [table[0]] + chunk  # Add header
                    result.append(chunk)
            else:
                result.append(table)

        return result

    def _headers_similar(
        self, h1: list[str], h2: list[str], threshold: float = 0.5
    ) -> bool:
        """Check if two header rows are similar enough to merge."""
        if not h1 or not h2:
            return False

        set1 = set(h.lower().strip() for h in h1 if h.strip())
        set2 = set(h.lower().strip() for h in h2 if h.strip())

        if not set1 or not set2:
            return False

        intersection = set1 & set2
        union = set1 | set2
        return len(intersection) / len(union) >= threshold
