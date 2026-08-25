from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    CHROMA_DIR: Path = DATA_DIR / "chroma_db"
    SAMPLE_REPORTS_DIR: Path = DATA_DIR / "sample_reports"

    # LLM
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 2048

    # Embedding & Reranking
    EMBEDDING_MODEL: str = "AITeamVN/Vietnamese_Embedding_v2"
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    EMBEDDING_DIM: int = 1024

    # Vector Store
    CHROMA_COLLECTION: str = "vn_stock_knowledge"

    # Chunking
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 78  # ~15%

    # Retrieval
    RETRIEVAL_K: int = 20
    RERANK_TOP_K: int = 10
    VECTOR_WEIGHT: float = 0.7
    BM25_WEIGHT: float = 0.3
    RRF_K: int = 60

    # Data Pipeline
    MAX_ARTICLES: int = 500
    MAX_STOCKS: int = 30

    # VN30 tickers
    VN30_TICKERS: list[str] = [
        "ACB", "BCM", "BID", "CTG", "FPT",
        "GAS", "GVR", "HDB", "HPG", "MBB",
        "MSN", "MWG", "PGV", "PHR", "POW",
        "SAB", "SBT", "SSI", "STB", "TCB",
        "TPB", "VIB", "VIC", "VHM", "VNM",
        "VPB", "VRE", "EIB", "LPB", "SHB",
    ]

    # Evaluation
    JUDGE_MODEL: str = "gpt-4o-mini"
    FAITHFULNESS_THRESHOLD: float = 0.85
    RELEVANCY_THRESHOLD: float = 0.80
    PRECISION_THRESHOLD: float = 0.75
    RECALL_THRESHOLD: float = 0.80

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
