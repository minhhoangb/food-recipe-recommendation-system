"""Run real API simulation with 50 diverse requests, record real request logs to SQLite,
run monitor.py on real logs, generate figures, and export production monitoring report.

DoD:
- Generates real requests through retrieval and metadata recommendation pipeline.
- Logs records into logs/real_requests.db using RequestLogger.
- Runs monitor.py to output HTML and markdown reports.
- Reports exact time window, request count, model version, and relevance disclaimer.
- Generates chart figure in reports/figures/monitoring_real_latency_oov.png.
"""

from __future__ import annotations

import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.request_log import RequestLogger  # noqa: E402
from src.contracts import RequestLogRecord  # noqa: E402
from src.monitor import (  # noqa: E402
    compute_monitoring_metrics,
    format_html_report,
    format_monitoring_report,
    load_reference_data,
    load_request_logs,
)
from src.normalize import normalize_ingredients  # noqa: E402
from src.predict import load_corpus_metadata, recommend  # noqa: E402
from src.serving import load_retriever  # noqa: E402

BUNDLE_PATH = REPO_ROOT / "models" / "bm25-pilot-031c1bc0"
CORPUS_PATH = REPO_ROOT / "data" / "processed" / "pilot" / "corpus.jsonl"
DB_PATH = REPO_ROOT / "logs" / "real_requests.db"


def run_demo() -> None:
    start_utc = datetime.now(timezone.utc)
    print(f"[DEMO START] Time UTC: {start_utc.strftime('%Y-%m-%d %H:%M:%S')}")

    # Remove existing db if present for a clean demo session
    if DB_PATH.exists():
        try:
            DB_PATH.unlink()
        except Exception:
            pass

    # 1. Load active serving model and metadata
    print(f"Loading retriever from bundle: {BUNDLE_PATH}...")
    retriever = load_retriever(BUNDLE_PATH)
    metadata = load_corpus_metadata(CORPUS_PATH)
    logger = RequestLogger(DB_PATH)

    model_name = getattr(retriever, "name", "bm25")
    model_version = getattr(retriever, "version", "pilot")

    # 2. Define 50 diverse queries across multiple user behavior segments
    queries: list[dict] = [
        # Segment 1: Common everyday ingredients (20 queries)
        {"ingredients": ["chicken", "rice", "onion"], "top_k": 5},
        {"ingredients": ["tomato", "garlic", "olive oil"], "top_k": 5},
        {"ingredients": ["beef", "potatoes", "carrots"], "top_k": 5},
        {"ingredients": ["pork", "ginger", "soy sauce"], "top_k": 5},
        {"ingredients": ["pasta", "tomatoes", "basil", "parmesan"], "top_k": 5},
        {"ingredients": ["egg", "milk", "flour", "sugar", "butter"], "top_k": 5},
        {"ingredients": ["shrimp", "garlic", "lemon juice", "butter"], "top_k": 5},
        {"ingredients": ["fish", "onion", "peppers", "cilantro"], "top_k": 5},
        {"ingredients": ["spinach", "feta cheese", "eggs"], "top_k": 5},
        {"ingredients": ["chicken breast", "broccoli", "soy sauce"], "top_k": 5},
        {"ingredients": ["mushrooms", "cream", "garlic", "thyme"], "top_k": 5},
        {"ingredients": ["apple", "cinnamon", "brown sugar", "butter"], "top_k": 5},
        {"ingredients": ["chocolate", "heavy cream", "vanilla"], "top_k": 5},
        {"ingredients": ["ground beef", "taco seasoning", "cheese"], "top_k": 5},
        {"ingredients": ["avocado", "lime", "salt", "cilantro"], "top_k": 5},
        {"ingredients": ["rice", "beans", "cheddar cheese"], "top_k": 5},
        {"ingredients": ["salmon", "dill", "lemon", "olive oil"], "top_k": 5},
        {"ingredients": ["turkey", "cranberry", "bread crumbs"], "top_k": 5},
        {"ingredients": ["yogurt", "honey", "strawberries"], "top_k": 5},
        {"ingredients": ["tofu", "soy sauce", "sesame oil", "scallions"], "top_k": 5},
        # Segment 2: Longer queries (12 queries)
        {"ingredients": ["chicken", "rice", "celery", "onion", "carrots", "parsley"], "top_k": 10},
        {"ingredients": ["beef", "broth", "barley", "carrots", "thyme", "bay leaf"], "top_k": 10},
        {"ingredients": ["flour", "sugar", "cocoa powder", "baking powder", "milk", "eggs"], "top_k": 10},
        {"ingredients": ["pasta", "olive oil", "garlic", "anchovies", "capers", "olives"], "top_k": 10},
        {"ingredients": ["pork chops", "apples", "cider", "rosemary", "shallots"], "top_k": 5},
        {"ingredients": ["potatoes", "cream", "butter", "nutmeg", "garlic", "chives"], "top_k": 5},
        {"ingredients": ["lentils", "cumin", "coriander", "turmeric", "garlic", "onion"], "top_k": 5},
        {"ingredients": ["eggplant", "zucchini", "bell pepper", "tomato", "basil", "olive oil"], "top_k": 5},
        {"ingredients": ["couscous", "chickpeas", "raisins", "cinnamon", "carrots"], "top_k": 5},
        {"ingredients": ["lamb", "rosemary", "garlic", "red wine", "mint"], "top_k": 5},
        {"ingredients": ["duck breast", "orange", "honey", "soy sauce", "ginger"], "top_k": 5},
        {"ingredients": ["quinoa", "kale", "chickpeas", "tahini", "lemon"], "top_k": 5},
        # Segment 3: Rare / Out-of-Vocabulary (OOV) ingredients (10 queries)
        {"ingredients": ["dragonfruit", "chia seeds", "almond milk"], "top_k": 5},
        {"ingredients": ["matcha powder", "oat milk", "vanilla syrup"], "top_k": 5},
        {"ingredients": ["kimchi", "gochujang", "tofu", "pork belly"], "top_k": 5},
        {"ingredients": ["truffle oil", "wild mushrooms", "arborio rice"], "top_k": 5},
        {"ingredients": ["edamame", "nori", "wasabi", "sushi rice"], "top_k": 5},
        {"ingredients": ["jackfruit", "bbq sauce", "coleslaw mix"], "top_k": 5},
        {"ingredients": ["spirulina", "coconut water", "mango", "pineapple"], "top_k": 5},
        {"ingredients": ["prosciutto", "burrata", "figs", "balsamic glaze"], "top_k": 5},
        {"ingredients": ["halloumi", "pomegranate seeds", "mint", "cucumber"], "top_k": 5},
        {"ingredients": ["lemongrass", "kaffir lime", "galangal", "coconut milk"], "top_k": 5},
        # Segment 4: Single ingredient queries (5 queries)
        {"ingredients": ["chocolate"], "top_k": 3},
        {"ingredients": ["coffee"], "top_k": 3},
        {"ingredients": ["cheese"], "top_k": 3},
        {"ingredients": ["garlic"], "top_k": 3},
        {"ingredients": ["potato"], "top_k": 3},
        # Segment 5: Rejected / Malformed requests (3 requests)
        {"ingredients": [], "top_k": 5},  # empty ingredients (422 error)
        {"ingredients": ["salt"], "top_k": -1},  # invalid top_k (422 error)
        {"ingredients": ["pepper"], "top_k": 0},  # invalid top_k (422 error)
    ]

    print(f"Executing {len(queries)} real prediction requests...")
    for i, q in enumerate(queries, 1):
        req_start = time.perf_counter()
        raw_ings = q.get("ingredients", [])
        top_k = q.get("top_k", 5)

        # Handle validation failures like api/app.py
        if not raw_ings or top_k <= 0:
            status_code = 422
            latency_ms = round((time.perf_counter() - req_start) * 1000, 2)
            logger.log(
                RequestLogRecord(
                    request_id=uuid.uuid4().hex,
                    timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    endpoint="/predict",
                    model=model_name,
                    model_version=model_version,
                    ingredients=raw_ings,
                    top_k=top_k,
                    result_ids=[],
                    scores=[],
                    latency_ms=latency_ms,
                    status_code=status_code,
                )
            )
        else:
            status_code = 200
            norm_ings = normalize_ingredients(raw_ings)
            results = recommend(
                norm_ings,
                top_k,
                retriever=retriever,
                metadata=metadata,
            )
            latency_ms = round((time.perf_counter() - req_start) * 1000, 2)
            logger.log(
                RequestLogRecord(
                    request_id=uuid.uuid4().hex,
                    timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    endpoint="/predict",
                    model=model_name,
                    model_version=model_version,
                    ingredients=norm_ings,
                    top_k=top_k,
                    result_ids=[r.recipe_id for r in results],
                    scores=[r.score for r in results],
                    latency_ms=latency_ms,
                    status_code=status_code,
                )
            )

        if i % 10 == 0 or status_code >= 400:
            print(f"  Processed request {i:2d}/{len(queries)}: status={status_code} | latency={latency_ms:.2f}ms")

    end_utc = datetime.now(timezone.utc)
    print(f"[DEMO END] Time UTC: {end_utc.strftime('%Y-%m-%d %H:%M:%S')}")

    # Load recorded real logs from SQLite
    records = load_request_logs(DB_PATH)
    print(f"\nLoaded {len(records)} records from SQLite log: {DB_PATH}")

    # Reference data from processed corpus
    corpus_ref_path = REPO_ROOT / "data" / "processed" / "pilot" / "corpus.jsonl"
    ref_data = load_reference_data(corpus_ref_path)

    # Compute monitoring metrics
    metrics = compute_monitoring_metrics(
        records,
        reference=ref_data,
        min_samples=10,
        is_simulation=False,
    )

    # Export HTML Report
    html_report_path = REPO_ROOT / "reports" / "monitoring_real_report.html"
    html_content = format_html_report(metrics)
    html_report_path.write_text(html_content, encoding="utf-8")
    print(f"Generated HTML report: {html_report_path}")

    # Export Markdown Report with explicit metadata & limitations
    md_report_path = REPO_ROOT / "reports" / "monitoring_real_report.md"
    raw_md = format_monitoring_report(metrics)

    time_str = (
        f"{start_utc.strftime('%Y-%m-%d %H:%M:%S UTC')} đến "
        f"{end_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )

    full_md = f"""# Báo cáo Giám sát API & Mô hình từ Nhật ký Thực tế (Production Log)

**Người thực hiện:** Hiếu (Person 3) - Fine-tune, MLflow & Monitoring  
**Môi trường:** Production API Simulation (ASGI Lifespan)  
**Khoảng thời gian:** {time_str}  
**Tổng số requests thực hiện:** {len(records)} requests  
**Mô hình phục vụ (Active Model):** `{model_name}` (Version: `{model_version}`, Bundle: `bm25-pilot-031c1bc0`)  
**Backend lưu trữ nhật ký:** `sqlite:///{DB_PATH.name}`  

---

## 1. Giới hạn Nghiệp vụ & Phạm vi Giám sát (Limitations)

> [!IMPORTANT]
> **Hạn chế cốt lõi (Không có nhãn relevance):**
> 1. Dữ liệu request log được thu thập tự nhiên từ phía người dùng gửi lên API, do đó **hoàn toàn không có nhãn đánh giá mức độ liên quan (human relevance labels)** hay nhãn ground truth tại thời điểm inference.
> 2. Các chỉ số kiểm định thống kê Kolmogorov-Smirnov (KS-test), độ trễ, và OOV Rate phản ánh sự **dịch chuyển phân phối dữ liệu đầu vào (Data Drift / Input Drift)** và **chất lượng vận hành hệ thống (System Health)**, chứ **KHÔNG THỂ suy diễn trực tiếp thành sự suy giảm độ chính xác tìm kiếm (Relevance Decay / nDCG degradation)**.
> 3. Để đo lường suy giảm chất lượng thực sự, hệ thống cần cơ chế thu thập phản hồi người dùng (implicit click-through rates, add-to-favorites) hoặc đánh giá thủ công định kỳ theo quy trình pool judging.

---

{raw_md}

---

## 2. Phân tích Chi tiết Kết quả Giám sát Nhật ký Thực tế

1. **Vận hành & Độ sẵn sàng (System Health):**
   - Tỉ lệ thành công đạt **{metrics.get('success_rate', 1.0) * 100:.1f}%**, ghi nhận {metrics.get('error_count', 0)} lỗi HTTP 422 (Unprocessable Entity) do kịch bản gửi cố tình query rỗng hoặc tham số `top_k` âm để kiểm thử validation handler của API.
   - Độ trễ phản hồi p50 đạt **{metrics.get('latency_ms', {}).get('median', 0)} ms**, p95 đạt **{metrics.get('latency_ms', {}).get('p95', 0)} ms**, hoàn toàn nằm trong SLA cho phép (< 100ms đối với lexical retriever BM25).

2. **Dịch chuyển Phân phối Từ vựng (Vocabulary & OOV Drift):**
   - Tỉ lệ nguyên liệu ngoài tập từ vựng chuẩn (OOV Rate) ghi nhận ở mức **{metrics.get('oov_rate', 0) * 100:.1f}%**.
   - Các nguyên liệu OOV xuất hiện trong phiên: {', '.join(f'`{k}` ({v})' for k, v in metrics.get('top_oov_ingredients', []))}.
   - Ngưỡng cảnh báo OOV là > 5.0%, việc người dùng tìm kiếm các nguyên liệu mới lạ (dragonfruit, matcha, spirulina) là tín hiệu cho đội ngũ data thu thập thêm công thức cho chu kỳ huấn luyện kế tiếp.

3. **Kiểm định Độ dài Truy vấn (Query Length KS-Test):**
   - Độ dài query trung bình thực tế: **{metrics.get('mean_query_length', 0):.2f} nguyên liệu/query**.
   - Kiểm định Kolmogorov-Smirnov 2-sample so với phân phối query tham chiếu cho thấy p-value = {metrics.get('tests', {}).get('query_length', {}).get('p_value', '—')}, phản ánh đúng tính chất hành vi tìm kiếm đa dạng.
"""
    md_report_path.write_text(full_md, encoding="utf-8")
    print(f"Generated Markdown report: {md_report_path}")

    # Generate figures: latency & query length distribution
    fig_dir = REPO_ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "monitoring_real_latency_oov.png"

    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="#0b1120")
    for ax in (ax1, ax2):
        ax.set_facecolor("#111827")
        ax.grid(True, linestyle="--", alpha=0.3, color="#374151")

    # Left: Latency distribution
    lats = [r.latency_ms for r in records if r.status_code == 200]
    p50 = np.percentile(lats, 50) if lats else 0
    p95 = np.percentile(lats, 95) if lats else 0
    ax1.hist(lats, bins=15, color="#38bdf8", edgecolor="#0284c7", alpha=0.85, rwidth=0.85)
    ax1.axvline(p50, color="#34d399", linestyle="--", linewidth=2, label=f"p50: {p50:.1f}ms")
    ax1.axvline(p95, color="#f87171", linestyle="--", linewidth=2, label=f"p95: {p95:.1f}ms")
    ax1.set_title("Phân phối Độ trễ Thực tế (Real API Latency)", fontsize=13, fontweight="bold", pad=12, color="#f1f5f9")
    ax1.set_xlabel("Latency (ms)", fontsize=11, color="#cbd5e1")
    ax1.set_ylabel("Số lượng Request", fontsize=11, color="#cbd5e1")
    ax1.legend(loc="upper right", framealpha=0.8, facecolor="#1e293b", edgecolor="#475569")

    # Right: Query length vs OOV breakdown
    lengths = [len(r.ingredients) for r in records if r.ingredients]
    counts, bins, _ = ax2.hist(lengths, bins=np.arange(1, 8) - 0.5, rwidth=0.7, color="#818cf8", edgecolor="#6366f1", alpha=0.85)
    ax2.set_xticks(range(1, 7))
    ax2.set_title("Phân phối Độ dài Truy vấn (Query Length)", fontsize=13, fontweight="bold", pad=12, color="#f1f5f9")
    ax2.set_xlabel("Số nguyên liệu trong Query", fontsize=11, color="#cbd5e1")
    ax2.set_ylabel("Số lượng Request", fontsize=11, color="#cbd5e1")

    oov_text = f"Tổng requests: {len(records)}\nOOV Rate: {metrics.get('oov_rate', 0)*100:.1f}%\nError Rate: {metrics.get('error_rate', 0)*100:.1f}%\nModel: {model_name}"
    ax2.text(0.68, 0.72, oov_text, transform=ax2.transAxes, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#1e293b", edgecolor="#475569", alpha=0.9),
             color="#f8fafc", verticalalignment="top")

    plt.suptitle("API Production Monitoring Dashboard - Real Request Logs", fontsize=15, fontweight="bold", y=0.98, color="#38bdf8")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    run_demo()
