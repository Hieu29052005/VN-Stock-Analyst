"""Demo script for VN Stock Analyst - Full RAG Pipeline Test."""
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.knowledge_hub.bm25_store import BM25Store
from src.knowledge_hub.retriever import HybridRetriever
from src.knowledge_hub.vector_store import VectorStore
from src.rag_pipeline.context_builder import FinancialContextBuilder
from src.rag_pipeline.generator import LLMGenerator
from src.rag_pipeline.guardrails import CitationGuardrails
from src.rag_pipeline.query_classifier import QueryClassifier, QueryIntent
from src.rag_pipeline.query_transform.hyde import HyDETransform
from src.rag_pipeline.query_transform.multi_query import MultiQueryTransform
from src.rag_pipeline.reranker import CrossEncoderReranker


# ─────────────────────────────────────────────────────────────
# Colors for terminal output
# ─────────────────────────────────────────────────────────────
class C:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    END = "\033[0m"


def header(text):
    print(f"\n{C.HEADER}{C.BOLD}{'=' * 60}{C.END}")
    print(f"{C.HEADER}{C.BOLD}  {text}{C.END}")
    print(f"{C.HEADER}{C.BOLD}{'=' * 60}{C.END}")


def step(text):
    print(f"\n{C.CYAN}{C.BOLD}▶ {text}{C.END}")


def info(text):
    print(f"  {C.DIM}{text}{C.END}")


def success(text):
    print(f"  {C.GREEN}✓ {text}{C.END}")


def warning(text):
    print(f"  {C.YELLOW}⚠ {text}{C.END}")


def error(text):
    print(f"  {C.RED}✗ {text}{C.END}")


# ─────────────────────────────────────────────────────────────
# 1. Demo: System Components Check
# ─────────────────────────────────────────────────────────────
def demo_system_check():
    header("1. SYSTEM CHECK")

    # Vector store
    step("Checking Vector Store...")
    try:
        vs = VectorStore()
        success(f"ChromaDB: {vs.count} documents indexed")
    except Exception as e:
        error(f"VectorStore failed: {e}")
        return False

    # BM25
    step("Checking BM25 Index...")
    try:
        bm25 = BM25Store()
        success(f"BM25: {bm25.count} documents indexed")
    except Exception as e:
        error(f"BM25 failed: {e}")
        return False

    # Embedding model
    step("Checking Embedding Model...")
    try:
        from src.knowledge_hub.embeddings import get_embedding_model
        # Just check import, don't load model (slow)
        success("Embedding module loaded")
    except Exception as e:
        error(f"Embedding failed: {e}")

    # Reranker
    step("Checking Reranker Model...")
    try:
        from src.rag_pipeline.reranker import CrossEncoderReranker
        success("Reranker module loaded")
    except Exception as e:
        error(f"Reranker failed: {e}")

    return True


# ─────────────────────────────────────────────────────────────
# 2. Demo: Query Classification
# ─────────────────────────────────────────────────────────────
def demo_query_classification():
    header("2. QUERY CLASSIFICATION")

    classifier = QueryClassifier()

    test_queries = [
        "P/E của FPT là bao nhiêu?",
        "So sánh VHM và NVL",
        "Giá cổ phiếu VIC hiện tại",
        "Phân tích triển vọng ngành ngân hàng 2025",
        "FPT hoạt động trong lĩnh vực nào?",
    ]

    for q in test_queries:
        intent = classifier.classify(q)
        color = {
            QueryIntent.FUNDAMENTAL: C.GREEN,
            QueryIntent.COMPARISON: C.BLUE,
            QueryIntent.PRICE_LOOKUP: C.YELLOW,
            QueryIntent.ANALYSIS: C.RED,
            QueryIntent.GENERAL: C.DIM,
        }.get(intent, C.END)

        print(f"  {C.BOLD}Q:{C.END} {q}")
        print(f"  {C.BOLD}→{C.END} {color}{intent.value}{C.END}\n")


# ─────────────────────────────────────────────────────────────
# 3. Demo: Hybrid Retrieval
# ─────────────────────────────────────────────────────────────
def demo_retrieval():
    header("3. HYBRID RETRIEVAL")

    vs = VectorStore()
    bm25 = BM25Store()
    retriever = HybridRetriever(vs, bm25)

    queries = [
        "P/E của FPT là bao nhiêu?",
        "So sánh VHM và NVL",
        "Triển vọng ngành ngân hàng 2025",
    ]

    for q in queries:
        step(f"Query: {q}")
        start = time.time()
        results = retriever.retrieve(q, n_results=5)
        latency = (time.time() - start) * 1000

        success(f"Found {len(results)} results in {latency:.0f}ms")
        for i, r in enumerate(results[:3], 1):
            ticker = r.metadata.get("ticker", "")
            doc_type = r.metadata.get("doc_type", "")
            preview = r.text[:80].replace("\n", " ")
            print(
                f"    {C.DIM}[{i}] {C.END}"
                f"{C.CYAN}{r.score:.4f}{C.END} "
                f"{C.YELLOW}{ticker}{C.END} "
                f"({doc_type}) "
                f"{C.DIM}{preview}...{C.END}"
            )
        print()


# ─────────────────────────────────────────────────────────────
# 4. Demo: Cross-Encoder Reranking
# ─────────────────────────────────────────────────────────────
def demo_reranking():
    header("4. CROSS-ENCODER RERANKING")

    vs = VectorStore()
    bm25 = BM25Store()
    retriever = HybridRetriever(vs, bm25)

    step("Loading reranker model (first time may download ~1GB)...")
    try:
        reranker = CrossEncoderReranker()
        success("Reranker loaded")
    except Exception as e:
        warning(f"Reranker not available: {e}")
        info("Install with: pip install sentence-transformers")
        return

    q = "P/E và ROE của FPT hiện tại"
    step(f"Query: {q}")

    # Retrieve
    results = retriever.retrieve(q, n_results=10)
    docs_for_rerank = [
        {"doc_id": r.doc_id, "text": r.text, "score": r.score, "metadata": r.metadata}
        for r in results
    ]

    # Rerank
    start = time.time()
    reranked = reranker.rerank(q, docs_for_rerank, top_k=5)
    latency = (time.time() - start) * 1000

    success(f"Reranked 10 → 5 in {latency:.0f}ms")

    for i, r in enumerate(reranked, 1):
        preview = r.text[:60].replace("\n", " ")
        print(
            f"    {C.DIM}[{i}] {C.END}"
            f"{C.GREEN}{r.score:.4f}{C.END} (was {r.original_score:.4f}) "
            f"{C.DIM}{preview}...{C.END}"
        )
    print()


# ─────────────────────────────────────────────────────────────
# 5. Demo: Full Pipeline (without LLM if no API key)
# ─────────────────────────────────────────────────────────────
def demo_full_pipeline():
    header("5. FULL RAG PIPELINE")

    has_api_key = bool(settings.OPENAI_API_KEY and settings.OPENAI_API_KEY != "")

    vs = VectorStore()
    bm25 = BM25Store()
    retriever = HybridRetriever(vs, bm25)

    try:
        reranker = CrossEncoderReranker()
    except Exception:
        reranker = None

    context_builder = FinancialContextBuilder()
    classifier = QueryClassifier()
    guardrails = CitationGuardrails()
    hyde = HyDETransform(use_llm=has_api_key)
    multi_query = MultiQueryTransform(use_llm=has_api_key)

    if has_api_key:
        generator = LLMGenerator()
        success("LLM available — full pipeline")
    else:
        generator = None
        warning("No OPENAI_API_KEY — showing retrieval + context only")

    test_questions = [
        "P/E của FPT là bao nhiêu?",
        "So sánh VHM và NVL về tỷ suất lợi nhuận",
        "Triển vọng ngành ngân hàng 2025",
    ]

    for q in test_questions:
        step(f"Q: {q}")

        # Classify
        intent = classifier.classify(q)
        info(f"Intent: {intent.value}")

        # Transform queries
        if intent == QueryIntent.COMPARISON:
            queries = multi_query.transform(q, n=3)
        elif intent in (QueryIntent.FUNDAMENTAL, QueryIntent.ANALYSIS):
            hyde_q = hyde.transform(q)
            queries = [q, hyde_q]
        else:
            queries = [q]
        info(f"Query variations: {len(queries)}")

        # Retrieve
        all_results = []
        for tq in queries:
            results = retriever.retrieve(tq, n_results=10)
            all_results.extend(results)

        # Deduplicate
        seen = set()
        unique = []
        for r in all_results:
            if r.doc_id not in seen:
                seen.add(r.doc_id)
                unique.append(r)
        info(f"Unique results: {len(unique)}")

        # Rerank
        if reranker:
            docs = [
                {"doc_id": r.doc_id, "text": r.text, "score": r.score, "metadata": r.metadata}
                for r in unique
            ]
            reranked = reranker.rerank(q, docs, top_k=5)
            info(f"Reranked to top 5")
        else:
            reranked = [
                type("R", (), {
                    "doc_id": r.doc_id, "text": r.text, "score": r.score,
                    "metadata": r.metadata, "original_score": r.score
                })()
                for r in unique[:5]
            ]

        # Build context
        context = context_builder.build(q, reranked, intent=intent.value)
        info(f"Context length: {len(context)} chars")

        # Generate (if LLM available)
        if generator:
            answer = asyncio.run(generator.generate(context))
            answer = guardrails.verify(answer, context, q)
            print(f"\n  {C.GREEN}{C.BOLD}ANSWER:{C.END}")
            for line in answer.split("\n"):
                if line.strip():
                    print(f"    {line}")
        else:
            print(f"\n  {C.YELLOW}{C.BOLD}RETRIEVED CONTEXT (no LLM):{C.END}")
            for i, r in enumerate(reranked[:3], 1):
                preview = r.text[:120].replace("\n", " ")
                ticker = r.metadata.get("ticker", "")
                print(f"    {C.DIM}[{i}]{C.END} {C.CYAN}{ticker}{C.END} {C.DIM}{preview}...{C.END}")

        # Show sources
        print(f"\n  {C.BOLD}Sources:{C.END}")
        for i, r in enumerate(reranked[:3], 1):
            ticker = r.metadata.get("ticker", "?")
            src = r.metadata.get("source", "?")
            dtype = r.metadata.get("doc_type", "?")
            print(f"    [{i}] {ticker} | {src} | {dtype}")

        print()


# ─────────────────────────────────────────────────────────────
# 6. Demo: Agent Tools
# ─────────────────────────────────────────────────────────────
def demo_agent_tools():
    header("6. AGENT TOOLS")

    from src.agents.tools.financial_ratio_tool import FinancialRatioTool
    from src.agents.tools.stock_price_tool import StockPriceTool
    from src.agents.memory import ConversationMemory

    # Price tool
    step("Stock Price Tool")
    price_tool = StockPriceTool()
    for ticker in ["FPT", "VIC", "HPG"]:
        result = price_tool.get_price(ticker)
        if "error" not in result:
            price = result.get("price", "N/A")
            change = result.get("change_pct", "N/A")
            print(f"    {C.CYAN}{ticker}{C.END}: {price} ({change}%)")
        else:
            info(f"{ticker}: {result['error']}")

    # Ratio tool
    step("Financial Ratio Tool")
    ratio_tool = FinancialRatioTool()
    for ticker in ["FPT", "VIC", "HPG", "TCB"]:
        result = ratio_tool.calculate(ticker)
        pe = result.get("pe", "N/A")
        roe = result.get("roe", "N/A")
        print(f"    {C.CYAN}{ticker}{C.END}: P/E={pe}, ROE={roe}%")

    # Memory
    step("Conversation Memory")
    memory = ConversationMemory(max_messages=5)
    memory.add("user", "P/E của FPT là bao nhiêu?")
    memory.add("assistant", "P/E của FPT hiện tại khoảng 25x")
    memory.add("user", "Vậy so với VIC thì sao?")
    ctx = memory.get_context(3)
    for line in ctx.split("\n"):
        print(f"    {C.DIM}{line}{C.END}")
    print()


# ─────────────────────────────────────────────────────────────
# 7. Demo: Interactive Chat
# ─────────────────────────────────────────────────────────────
def demo_interactive_chat():
    header("7. INTERACTIVE CHAT")
    info("Type your question about Vietnamese stocks.")
    info("Type 'quit' or 'exit' to stop.\n")

    vs = VectorStore()
    bm25 = BM25Store()
    retriever = HybridRetriever(vs, bm25)

    try:
        reranker = CrossEncoderReranker()
    except Exception:
        reranker = None

    context_builder = FinancialContextBuilder()
    classifier = QueryClassifier()
    has_api_key = bool(settings.OPENAI_API_KEY and settings.OPENAI_API_KEY != "")
    generator = LLMGenerator() if has_api_key else None
    guardrails = CitationGuardrails()

    while True:
        try:
            query = input(f"\n{C.BOLD}You ▸ {C.END}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{C.DIM}Goodbye!{C.END}")
            break

        if not query or query.lower() in ("quit", "exit", "q"):
            print(f"{C.DIM}Goodbye!{C.END}")
            break

        start = time.time()

        # Classify
        intent = classifier.classify(query)
        print(f"  {C.DIM}Intent: {intent.value}{C.END}")

        # Retrieve
        results = retriever.retrieve(query, n_results=10)

        # Rerank
        if reranker:
            docs = [
                {"doc_id": r.doc_id, "text": r.text, "score": r.score, "metadata": r.metadata}
                for r in results
            ]
            reranked = reranker.rerank(query, docs, top_k=5)
        else:
            reranked = [
                type("R", (), {
                    "doc_id": r.doc_id, "text": r.text, "score": r.score,
                    "metadata": r.metadata, "original_score": r.score
                })()
                for r in results[:5]
            ]

        # Context + Generate
        context = context_builder.build(query, reranked, intent=intent.value)

        if generator:
            answer = asyncio.run(generator.generate(context))
            answer = guardrails.verify(answer, context, query)
            print(f"\n{C.GREEN}Ensa ▸{C.END} {answer}")
        else:
            print(f"\n{C.YELLOW}Retrieved context (no LLM configured):{C.END}")
            for i, r in enumerate(reranked[:3], 1):
                ticker = r.metadata.get("ticker", "")
                preview = r.text[:100].replace("\n", " ")
                print(f"  [{i}] {C.CYAN}{ticker}{C.END} (score={r.score:.4f}) {C.DIM}{preview}...{C.END}")

        latency = (time.time() - start) * 1000
        print(f"  {C.DIM}({latency:.0f}ms){C.END}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    print(f"\n{C.HEADER}{C.BOLD}")
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║   VN Stock Analyst - Full Demo               ║")
    print("  ║   RAG-powered Investment Research Assistant  ║")
    print("  ╚══════════════════════════════════════════════╝")
    print(f"{C.END}")

    # Run all demos
    if not demo_system_check():
        error("System check failed. Run 'python scripts/ingest_data.py' first.")
        return

    demo_query_classification()
    demo_retrieval()
    demo_reranking()
    demo_agent_tools()
    demo_full_pipeline()

    # Interactive mode
    print(f"\n{C.BOLD}All demos complete!{C.END}")
    response = input(f"\n{C.BOLD}Start interactive chat? (y/n): {C.END}").strip().lower()
    if response in ("y", "yes", ""):
        demo_interactive_chat()


if __name__ == "__main__":
    main()
