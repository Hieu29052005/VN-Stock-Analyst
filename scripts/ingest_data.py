"""Script to run the data ingestion pipeline."""
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import settings
from src.data_pipeline.collectors.cafef_collector import CafeFCollector
from src.data_pipeline.collectors.vneconomy_collector import VnEconomyCollector
from src.data_pipeline.collectors.vnstock_collector import VNStockCollector
from src.data_pipeline.pipeline import DataPipeline
from src.knowledge_hub.bm25_store import BM25Store
from src.knowledge_hub.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("VN Stock Analyst - Data Ingestion Pipeline")
    logger.info("=" * 60)

    # Step 1: Collect data
    logger.info("\n[Step 1/4] Collecting data...")
    pipeline = DataPipeline()

    collectors = [
        CafeFCollector(max_items=settings.MAX_ARTICLES),
        VnEconomyCollector(max_items=settings.MAX_ARTICLES),
    ]

    # Only collect vnstock if OPENAI_API_KEY is not needed for collection
    try:
        vnstock_collector = VNStockCollector(max_items=settings.MAX_STOCKS)
        collectors.append(vnstock_collector)
    except Exception as e:
        logger.warning(f"VNStock collector unavailable: {e}")

    docs = pipeline.run_collectors(collectors)
    logger.info(f"Total documents collected: {len(docs)}")

    # Always include sample data to ensure pipeline has content to work with
    from src.data_pipeline.collectors.base import Document
    sample_docs = [
        Document(
            content="FPT Corporation là công ty công nghệ hàng đầu Việt Nam, "
                    "hoạt động trong lĩnh vực công nghệ thông tin, viễn thông, "
                    "phần mềm và AI. Doanh thu năm 2024 khoảng 60,000 tỷ đồng, "
                    "lợi nhuận sau thuế khoảng 8,500 tỷ đồng. Tỷ suất lợi nhuận "
                    "gộp khoảng 35%. P/E hiện tại khoảng 25x, ROE 22%.",
            metadata={"ticker": "FPT", "doc_type": "company_profile"},
            source="sample",
            doc_id="sample_fpt_1",
        ),
        Document(
            content="Vingroup (VIC) là tập đoàn kinh tế tư nhân đa ngành lớn nhất "
                    "Việt Nam, hoạt động trong bất động sản, ô tô điện (VinFast), "
                    "công nghệ và thương mại điện tử. Tỷ lệ P/E hiện tại khoảng 28.5x, "
                    "P/B 2.1x. Tổng tài sản hơn 300,000 tỷ đồng.",
            metadata={"ticker": "VIC", "doc_type": "company_profile"},
            source="sample",
            doc_id="sample_vic_1",
        ),
        Document(
            content="HPG - Tập đoàn Hòa Phát là nhà sản xuất thép lớn nhất Việt Nam. "
                    "Sản lượng thép thô năm 2024 khoảng 8.5 triệu tấn. "
                    "Doanh thu khoảng 150,000 tỷ đồng. Tỷ suất lợi nhuận gộp 22%. "
                    "P/E 10.5x, P/B 1.8x. ROE 17.2%. Nợ/vốn chủ sở hữu 0.6x.",
            metadata={"ticker": "HPG", "doc_type": "financial_data"},
            source="sample",
            doc_id="sample_hpg_1",
        ),
        Document(
            content="Techcombank (TCB) là ngân hàng thương mại cổ phần tư nhân lớn nhất "
                    "Việt Nam. Tỷ lệ ROE khoảng 17.8%, P/E 8.5x, P/B 1.5x. "
                    "Hệ thống hơn 400 chi nhánh trên toàn quốc. Tổng tài sản hơn "
                    "700,000 tỷ đồng. Tỷ lệ nợ xấu NPL dưới 1%.",
            metadata={"ticker": "TCB", "doc_type": "financial_data"},
            source="sample",
            doc_id="sample_tcb_1",
        ),
        Document(
            content="VHM - Vinhomes là công ty con của Vingroup, hoạt động chính trong "
                    "lĩnh vực bất động sản nhà ở. Doanh thu năm 2024 khoảng 65,000 tỷ đồng. "
                    "Tỷ suất lợi nhuận gộp 32.5%, tăng từ 28.1% năm 2023. P/E 12.3x, "
                    "P/B 2.8x. ROE 22.1%.",
            metadata={"ticker": "VHM", "doc_type": "financial_data"},
            source="sample",
            doc_id="sample_vhm_1",
        ),
        Document(
            content="NVL - Tập đoàn Novaland là doanh nghiệp bất động sản lớn thứ hai "
                    "sau Vingroup. Tỷ suất lợi nhuận gộp 21.3% năm 2024. "
                    "P/E 15x, P/B 1.2x. Tổng nợ vay approximately 50,000 tỷ đồng.",
            metadata={"ticker": "NVL", "doc_type": "financial_data"},
            source="sample",
            doc_id="sample_nvl_1",
        ),
        Document(
            content="Vietcombank (VCB) là ngân hàng TMCP lớn nhất Việt Nam theo vốn hóa. "
                    "P/E khoảng 12x, P/B 2.5x. ROE 20%. Cho vay mua nhà với lãi suất "
                    "ưu đãi từ 6.5%/năm đầu. Hệ thống hơn 600 điểm giao dịch.",
            metadata={"ticker": "VCB", "doc_type": "financial_data"},
            source="sample",
            doc_id="sample_vcb_1",
        ),
        Document(
            content="MSN - Tập đoàn Masan hoạt động trong nhiều lĩnh vực: bán lẻ (WinMart), "
                    "tiêu dùng, F&B, khai khoáng. Doanh thuconsolidated khoảng 70,000 tỷ đồng. "
                    "P/E 22x, ROE 15%. Hơn 3,000 cửa hàng WinMart trên toàn quốc.",
            metadata={"ticker": "MSN", "doc_type": "company_profile"},
            source="sample",
            doc_id="sample_msn_1",
        ),
        Document(
            content="SSI - Công ty Cổ phần Chứng khoán SSI là công ty chứng khoán lớn nhất "
                    "Việt Nam. Doanh thu môi giới chiếm thị phần khoảng 15%. "
                    "P/E 15x, P/B 1.8x. ROE 12%. cung cấp dịch vụ môi giới, tự doanh, "
                    "đầu tư ngân hàng.",
            metadata={"ticker": "SSI", "doc_type": "company_profile"},
            source="sample",
            doc_id="sample_ssi_1",
        ),
        Document(
            content="MWG - Thế Giới Di Động là hệ thống bán lẻ điện tử lớn nhất Việt Nam. "
                    "Doanh thu khoảng 130,000 tỷ đồng. P/E 22x, P/B 4.5x. ROE 20.5%. "
                    "Hơn 5,000 cửa hàng trên toàn quốc. Mở rộng sang TopZone và Điện Máy Xanh.",
            metadata={"ticker": "MWG", "doc_type": "company_profile"},
            source="sample",
            doc_id="sample_mwg_1",
        ),
        Document(
            content="Tỷ giá VND/USD ảnh hưởng lớn đến xuất khẩu Việt Nam. Đồng VND yếu hơn "
                    "giúp hàng xuất khẩu Việt Nam cạnh tranh hơn trên thị trường quốc tế. "
                    "Ngân hàng Nhà nước duy trì tỷ giá linh hoạt trong biên độ +/- 5%.",
            metadata={"ticker": "", "doc_type": "analysis", "sector": "Macro"},
            source="sample",
            doc_id="sample_macro_1",
        ),
        Document(
            content="Ngân hàng Nhà nước Việt Nam tăng lãi suất điều hành 0.5% lên 6%/năm "
                    "để kiểm soát lạm phát. Tăng lãi suất tác động tiêu cực đến ngành "
                    "bất động sản qua kênh chi phí vốn tăng và cầu nhà ở giảm.",
            metadata={"ticker": "", "doc_type": "analysis", "sector": "Banking"},
            source="sample",
            doc_id="sample_macro_2",
        ),
        Document(
            content="VN-Index được dự báo đạt 1,300-1,500 điểm cuối năm 2025. "
                    "Tăng trưởng tín dụng toàn ngành ngân hàng dự kiến 15-18%. "
                    "Lạm phát CPI dự kiến维持 ở mức 3-4%.",
            metadata={"ticker": "", "doc_type": "analysis", "sector": "Macro"},
            source="sample",
            doc_id="sample_macro_3",
        ),
    ]
    docs.extend(sample_docs)
    logger.info(f"Total documents after adding samples: {len(docs)}")

    # Step 2: Process and chunk
    logger.info("\n[Step 2/4] Processing and chunking...")
    docs = pipeline.process_documents(docs)
    chunks = pipeline.chunk_documents(docs)
    pipeline.save_chunks(chunks)
    logger.info(f"Chunks created: {len(chunks)}")

    # Step 3: Index into vector store
    logger.info("\n[Step 3/4] Indexing into vector store...")
    vector_store = VectorStore()
    vector_store.delete_all()

    ids = [c["id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    metadatas = [{k: v for k, v in c["metadata"].items()} for c in chunks]

    added = vector_store.add_documents(ids, texts, metadatas)
    logger.info(f"Vector store: {added} documents indexed")

    # Step 4: Build BM25 index
    logger.info("\n[Step 4/4] Building BM25 index...")
    bm25_store = BM25Store()
    bm25_store.corpus = []
    bm25_store.doc_ids = []
    bm25_store.metadata = []
    bm25_store.add_documents(texts, ids, metadatas)
    bm25_store.save()
    logger.info(f"BM25 index: {bm25_store.count} documents")

    logger.info("\n" + "=" * 60)
    logger.info("Pipeline complete!")
    logger.info(f"  Documents: {len(docs)}")
    logger.info(f"  Chunks: {len(chunks)}")
    logger.info(f"  Vector store: {vector_store.count}")
    logger.info(f"  BM25 index: {bm25_store.count}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
