"""Tests for query classifier."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_pipeline.query_classifier import QueryClassifier, QueryIntent


def test_classifier_comparison():
    clf = QueryClassifier()
    assert clf.classify("So sánh P/E của FPT và VIC") == QueryIntent.COMPARISON
    assert clf.classify("HPG vs VHM哪个更好") == QueryIntent.COMPARISON


def test_classifier_price():
    clf = QueryClassifier()
    assert clf.classify("Giá cổ phiếu VIC hiện tại là bao nhiêu?") == QueryIntent.PRICE_LOOKUP
    assert clf.classify("FPT đang trading ở mức nào?") == QueryIntent.PRICE_LOOKUP


def test_classifier_fundamental():
    clf = QueryClassifier()
    assert clf.classify("P/E của FPT là bao nhiêu?") == QueryIntent.FUNDAMENTAL
    assert clf.classify("Tỷ suất lợi nhuận gộp của HPG") == QueryIntent.FUNDAMENTAL
    assert clf.classify("Doanh thu FPT quý gần nhất") == QueryIntent.FUNDAMENTAL


def test_classifier_analysis():
    clf = QueryClassifier()
    assert clf.classify("Phân tích triển vọng ngành ngân hàng 2025") == QueryIntent.ANALYSIS
    assert clf.classify("Đánh giá chiến lược đầu tư vào VIC") == QueryIntent.ANALYSIS


def test_classifier_general():
    clf = QueryClassifier()
    assert clf.classify("FPT là công ty gì?") == QueryIntent.GENERAL
