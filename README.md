# VN Stock Analyst — RAG-powered Investment Research Assistant

> AI-powered system for answering investment questions about Vietnamese stocks,
> built with advanced RAG techniques and evaluated on domain-specific benchmarks.

## Key Features

- **Hybrid Retrieval**: BM25 (sparse) + Dense embeddings with Reciprocal Rank Fusion (RRF)
- **Cross-Encoder Reranking**: BGE-Reranker-v2-M3 for precision retrieval
- **Query Transformation**: HyDE (Hypothetical Document Embeddings) + Multi-Query decomposition
- **Vietnamese-Optimized**: Vietnamese-fine-tuned BGE-M3 embeddings (19% better MRR than baseline)
- **Table-Aware Chunking**: Preserves financial tables as whole chunks during ingestion
- **Structured Evaluation**: RAGAS metrics with 25+ domain-specific test cases
- **Guardrails**: Citation enforcement, hallucination checks, disclaimer injection

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Frontend (Streamlit)                       │
├──────────────────────────────────────────────────────────────┤
│                   RAG Orchestrator                            │
│  ┌──────────┐ → ┌──────────┐ → ┌──────────┐ → ┌──────────┐ │
│  │ Query    │   │ Query    │   │ Hybrid   │   │ Reranker │ │
│  │Classify  │   │Transform │   │Retriever │   │(Cross-   │ │
│  │          │   │(HyDE/MQ) │   │(BM25+Dense)│  │ Encoder) │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘ │
│                         ↓                                    │
│  ┌──────────┐ ← ┌──────────┐ ← ┌──────────┐               │
│  │Generator │   │ Context  │   │Guardrails│               │
│  │(LLM)    │   │ Builder  │   │          │               │
│  └──────────┘   └──────────┘   └──────────┘               │
├──────────────────────────────────────────────────────────────┤
│  ChromaDB (Dense)  │  BM25 Index (Sparse)  │  Knowledge Graph│
└──────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# 1. Clone and setup
cd vn-stock-analyst
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers
pip install -e .

# 2. Configure API keys
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# 3. Ingest data
python scripts/ingest_data.py

# 4. Run Streamlit app
streamlit run src/web/app.py

# 5. Run tests
python -m pytest tests/ -v
```

## Project Structure

```
vn-stock-analyst/
├── src/
│   ├── data_pipeline/       # Data collection & processing
│   │   ├── collectors/      # CafeF, VnEconomy, VNStock collectors
│   │   ├── processors/      # PDF, HTML, Table extractors
│   │   └── chunkers/        # Recursive + Table-aware chunking
│   ├── knowledge_hub/       # Vector store, BM25, Knowledge Graph
│   ├── rag_pipeline/        # RAG core: classifier, transforms, generator
│   ├── evaluation/          # RAGAS evaluation framework
│   ├── agents/              # Tool-use agent (price, ratio, news)
│   └── web/                 # Streamlit frontend
├── scripts/                 # Data ingestion, evaluation, benchmarking
├── tests/                   # 18 unit tests
└── data/                    # ChromaDB, BM25 index, evaluation results
```

## Benchmark Results

| Retriever Config | Avg Latency | Results/Query |
|---|---|---|
| Dense Only | 298ms | 10 |
| BM25 Only | 296ms | 10 |
| **Hybrid (RRF)** | **277ms** | **10** |

## Tech Stack

- **Embeddings**: AITeamVN/Vietnamese_Embedding_v2 (1024 dims, Apache 2.0)
- **Reranker**: BAAI/bge-reranker-v2-m3 (multilingual SOTA)
- **Vector DB**: ChromaDB (local) / Qdrant (production)
- **RAG Framework**: LlamaIndex + custom hybrid retriever
- **LLM**: GPT-4o-mini (configurable)
- **Frontend**: Streamlit
- **Evaluation**: RAGAS + custom metrics

## Example Queries

| Query Type | Example |
|---|---|
| Fundamental | "P/E của FPT là bao nhiêu?" |
| Comparison | "So sánh VHM và NVL" |
| Price | "Giá cổ phiếu VIC hiện tại" |
| Analysis | "Triển vọng ngành ngân hàng 2025" |
| General | "FPT hoạt động trong lĩnh vực nào?" |
