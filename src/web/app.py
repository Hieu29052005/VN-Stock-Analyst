"""Streamlit frontend for VN Stock Analyst."""
import asyncio
import sys
from pathlib import Path

import streamlit as st

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.agents.agent import StockAgent
from src.config import settings
from src.knowledge_hub.bm25_store import BM25Store
from src.knowledge_hub.retriever import HybridRetriever
from src.knowledge_hub.vector_store import VectorStore
from src.rag_pipeline.pipeline import StockRAGPipeline
from src.rag_pipeline.reranker import CrossEncoderReranker

st.set_page_config(
    page_title="VN Stock Analyst",
    page_icon="📈",
    layout="wide",
)


@st.cache_resource
def init_pipeline():
    """Initialize RAG pipeline (cached)."""
    try:
        vector_store = VectorStore()
        bm25_store = BM25Store()
        retriever = HybridRetriever(vector_store, bm25_store)

        try:
            reranker = CrossEncoderReranker()
        except Exception:
            reranker = None

        pipeline = StockRAGPipeline(
            retriever=retriever,
            reranker=reranker,
            use_hyde=False,
            use_multi_query=False,
        )
        agent = StockAgent(rag_pipeline=pipeline)
        return agent
    except Exception as e:
        st.error(f"Lỗi khởi tạo: {e}")
        return None


def main():
    st.title("📈 VN Stock Analyst")
    st.caption("Trợ lý đầu tư AI cho thị trường chứng khoán Việt Nam")

    agent = init_pipeline()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Nguồn tham khảo"):
                    for i, src in enumerate(message["sources"][:5], 1):
                        st.text(f"[{i}] {src.get('text', '')[:200]}")

    if query := st.chat_input("Đặt câu hỏi về chứng khoán VN..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Đang phân tích..."):
                if agent:
                    response = asyncio.run(agent.query(query))
                    st.markdown(response.answer)
                    if response.sources:
                        with st.expander("Nguồn tham khảo"):
                            for i, src in enumerate(response.sources[:5], 1):
                                st.text(f"[{i}] {src.get('text', '')[:200]}")
                else:
                    response_text = (
                        "Hệ thống chưa sẵn sàng. "
                        "Vui lòng chạy `make ingest` để nạp dữ liệu trước."
                    )
                    st.markdown(response_text)
                    response = type("R", (), {"answer": response_text, "sources": []})()

            st.session_state.messages.append({
                "role": "assistant",
                "content": response.answer,
                "sources": getattr(response, "sources", []),
            })

    with st.sidebar:
        st.header("Cấu hình")
        st.write(f"Model: {settings.LLM_MODEL}")
        st.write(f"Embedding: {settings.EMBEDDING_MODEL}")
        st.write(f"Reranker: {settings.RERANKER_MODEL}")

        if st.button("Xóa lịch sử"):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.markdown("""
        **Ví dụ câu hỏi:**
        - P/E của FPT là bao nhiêu?
        - So sánh VHM và NVL
        - Giá VIC hiện tại
        - Triển vọng ngành ngân hàng 2025
        """)


if __name__ == "__main__":
    main()
