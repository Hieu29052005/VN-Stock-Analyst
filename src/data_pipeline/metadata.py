"""Metadata enrichment for document chunks."""
import hashlib
import re
from datetime import datetime


def generate_chunk_id(content: str, prefix: str = "") -> str:
    """Generate a deterministic chunk ID from content."""
    hash_val = hashlib.md5(content.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{hash_val}" if prefix else hash_val


def extract_ticker(text: str) -> str:
    """Extract stock ticker from text content."""
    patterns = [
        r"\b([A-Z]{2,5})\b",
        r"mã\s+([A-Z]{2,5})",
        r"cổ phiếu\s+([A-Z]{2,5})",
    ]
    known_tickers = {
        "ACB", "BCM", "BID", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
        "MBB", "MSN", "MWG", "PGV", "PHR", "POW", "SAB", "SBT", "SSI",
        "STB", "TCB", "TPB", "VIB", "VIC", "VHM", "VNM", "VPB", "VRE",
        "EIB", "LPB", "SHB", "VND", "PVD", "PLX", "PVS", "NT2", "PPC",
    }

    for pattern in patterns:
        matches = re.findall(pattern, text.upper())
        for match in matches:
            if match in known_tickers:
                return match
    return ""


def infer_doc_type(text: str, source: str = "") -> str:
    """Infer document type from content and source."""
    if source in ("vnstock",):
        return "financial_data"
    if any(kw in text.lower() for kw in ["báo cáo tài chính", "income statement", "balance sheet"]):
        return "financial_report"
    if any(kw in text.lower() for kw in ["phân tích", "analysis", "đánh giá"]):
        return "analysis"
    return "news"


def infer_sector(text: str) -> str:
    """Infer industry sector from content."""
    sector_keywords = {
        "Banking": ["ngân hàng", "banking", "tín dụng", "cho vay", "tiền gửi"],
        "Real Estate": ["bất động sản", "real estate", "dự án", "căn hộ", "đất"],
        "Technology": ["công nghệ", "technology", "phần mềm", "AI", "FPT"],
        "Energy": ["năng lượng", "dầu khí", "gas", "điện", "POWER"],
        "Consumer": ["tiêu dùng", "consumer", "bán lẻ", "MWG", "MSN"],
        "Manufacturing": ["sản xuất", "thép", "HPG", "sản phẩm"],
    }
    text_lower = text.lower()
    for sector, keywords in sector_keywords.items():
        if any(kw.lower() in text_lower for kw in keywords):
            return sector
    return "Unknown"


def enrich_metadata(
    content: str,
    base_metadata: dict | None = None,
    source: str = "",
) -> dict:
    """Enrich metadata with inferred fields."""
    meta = dict(base_metadata or {})
    if not meta.get("ticker"):
        meta["ticker"] = extract_ticker(content)
    if not meta.get("doc_type"):
        meta["doc_type"] = infer_doc_type(content, source)
    if not meta.get("sector"):
        meta["sector"] = infer_sector(content)
    if not meta.get("source"):
        meta["source"] = source
    if not meta.get("ingested_at"):
        meta["ingested_at"] = datetime.utcnow().isoformat()
    return meta
