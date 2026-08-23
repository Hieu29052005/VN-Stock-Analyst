"""Tests for agent tools."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.tools.financial_ratio_tool import FinancialRatioTool
from src.agents.tools.stock_price_tool import StockPriceTool
from src.agents.memory import ConversationMemory


def test_financial_ratio_cached():
    tool = FinancialRatioTool()
    result = tool.calculate("FPT")
    assert "pe" in result
    assert result["ticker"] == "FPT"


def test_financial_ratio_unknown():
    tool = FinancialRatioTool()
    result = tool.calculate("XYZ")
    assert "error" in result


def test_tool_descriptions():
    price_tool = StockPriceTool()
    ratio_tool = FinancialRatioTool()
    assert "name" in price_tool.get_tool_description()
    assert "name" in ratio_tool.get_tool_description()


def test_memory_add_and_get():
    memory = ConversationMemory()
    memory.add("user", "Hello")
    memory.add("assistant", "Hi there")
    ctx = memory.get_context(2)
    assert "Hello" in ctx
    assert "Hi there" in ctx


def test_memory_max_limit():
    memory = ConversationMemory(max_messages=3)
    for i in range(5):
        memory.add("user", f"msg_{i}")
    assert len(memory.messages) == 3
    assert memory.messages[-1].content == "msg_4"
