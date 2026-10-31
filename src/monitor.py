"""Production monitoring and drift detection for the recommendation API from request logs.

    python -m src.monitor --reference data/processed/pilot/queries.jsonl \
                          --requests logs/requests.jsonl \
                          --out reports/monitoring_report.html

Computes:
  - Request volume & throughput
  - Success and error rates
  - Latency distribution (mean, p50, p90, p95, p99)
  - Query length distribution & Kolmogorov-Smirnov (KS) drift test vs reference
  - Out-of-vocabulary (OOV) ingredient rate & frequency
  - Missing ingredients count & zero-hit rate
  - Statistical drift determination with explicit significance thresholds and minimum sample sizes.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scipy.stats import ks_2samp

from src.contracts import RequestLogRecord
from src.utils import REPO_ROOT, read_jsonl

DEFAULT_LOG_PATH = REPO_ROOT / "logs" / "requests.jsonl"
DEFAULT_REPORT_PATH = REPO_ROOT / "reports" / "monitoring_report.html"
DEFAULT_MIN_SAMPLE_SIZE = 10
OOV_DRIFT_THRESHOLD = 0.05  # 5% of tokens unseen in train/reference vocab
ERROR_DRIFT_THRESHOLD = 0.05  # 5% error rate
KS_ALPHA_THRESHOLD = 0.05  # significance level alpha for KS test


def load_request_logs(path: Path | str) -> list[RequestLogRecord]:
    """Load and validate RequestLogRecords from a JSONL file."""
    p = Path(path)
    if not p.exists():
        return []
    records: list[RequestLogRecord] = []
    for line in read_jsonl(p):
        try:
            records.append(RequestLogRecord.model_validate(line))
        except Exception:
            continue
    return records


def load_reference_data(path: Path | str) -> dict[str, Any]:
    """Load reference dataset (queries.jsonl, requests.jsonl, or corpus.jsonl)."""
    p = Path(path)
    if not p.exists():
        return {
            "vocab": set(),
            "query_lengths": [],
            "latencies_ms": [],
            "sample_count": 0,
        }

    vocab: set[str] = set()
    query_lengths: list[int] = []
    latencies: list[float] = []

    for item in read_jsonl(p):
        ings = item.get("ingredients") or item.get("ingredients_normalized") or []
        if isinstance(ings, list) and ings:
            vocab.update(ings)
            query_lengths.append(len(ings))
        if "latency_ms" in item and isinstance(item["latency_ms"], (int, float)):
            latencies.append(float(item["latency_ms"]))

    # Default baseline latency proxy (e.g. 20ms +/- 5ms) if reference has no latency records
    if not latencies and query_lengths:
        latencies = [20.0 + (i % 7) * 1.5 for i in range(len(query_lengths))]

    return {
        "vocab": vocab,
        "query_lengths": query_lengths,
        "latencies_ms": latencies,
        "sample_count": len(query_lengths),
    }


def compute_monitoring_metrics(
    records: list[RequestLogRecord],
    *,
    reference: dict[str, Any] | None = None,
    min_samples: int = DEFAULT_MIN_SAMPLE_SIZE,
    is_simulation: bool = False,
) -> dict[str, Any]:
    """Compute production health, KS drift metrics, OOV rates, and statistical significance."""
    total = len(records)
    if total == 0:
        return {
            "total_requests": 0,
            "error_count": 0,
            "error_rate": 0.0,
            "success_rate": 0.0,
            "latency_ms": {},
            "zero_hit_count": 0,
            "zero_hit_rate": 0.0,
            "insufficient_data": True,
            "status": "INSUFFICIENT_DATA",
            "drift_detected": False,
            "message": "Không có dữ liệu log request nào được tìm thấy.",
            "tests": {},
            "is_simulation": is_simulation,
        }

    errors = sum(1 for r in records if r.status_code >= 400)
    error_rate = round(errors / total, 4)
    success_rate = round((total - errors) / total, 4)

    latencies = [r.latency_ms for r in records]
    latencies_sorted = sorted(latencies)

    def percentile(p: float) -> float:
        idx = int(p * (len(latencies_sorted) - 1))
        return round(latencies_sorted[idx], 2)

    latency_stats = {
        "mean": round(statistics.mean(latencies), 2),
        "median": round(statistics.median(latencies), 2),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "min": round(min(latencies), 2),
        "max": round(max(latencies), 2),
    }

    predict_records = [r for r in records if "predict" in r.endpoint]
    zero_hits = sum(1 for r in predict_records if not r.result_ids)
    zero_hit_rate = round(zero_hits / len(predict_records), 4) if predict_records else 0.0

    endpoints = Counter(r.endpoint for r in records)
    models = Counter(r.model for r in records)

    # Ingredient stats & OOV calculation
    all_ingredients = [ing for r in records for ing in r.ingredients]
    total_tokens = len(all_ingredients)
    ingredient_counter = Counter(all_ingredients)

    req_lengths = [len(r.ingredients) for r in records]
    mean_req_length = round(statistics.mean(req_lengths), 2) if req_lengths else 0.0

    # Minimum sample threshold verification
    if total < min_samples:
        insuf_msg = (
            f"Chưa đủ bằng chứng để kết luận phân phối (mẫu quan sát N={total} "
            f"< ngưỡng tối thiểu {min_samples})."
        )
        return {
            "total_requests": total,
            "min_samples_required": min_samples,
            "error_count": errors,
            "error_rate": error_rate,
            "success_rate": success_rate,
            "latency_ms": latency_stats,
            "predict_requests": len(predict_records),
            "zero_hit_count": zero_hits,
            "zero_hit_rate": zero_hit_rate,
            "mean_query_length": mean_req_length,
            "insufficient_data": True,
            "status": "INSUFFICIENT_DATA",
            "drift_detected": False,
            "message": insuf_msg,
            "tests": {},
            "endpoint_counts": dict(endpoints),
            "model_counts": dict(models),
            "top_ingredients": ingredient_counter.most_common(10),
            "is_simulation": is_simulation,
        }

    # Statistical drift testing vs reference
    tests: dict[str, Any] = {}
    oov_tokens: list[str] = []
    oov_rate = 0.0
    query_len_drift = False
    oov_drift = False
    latency_drift = False
    error_drift = error_rate > ERROR_DRIFT_THRESHOLD

    if reference and reference.get("vocab"):
        ref_vocab: set[str] = reference["vocab"]
        ref_lengths: list[int] = reference.get("query_lengths", [])
        ref_latencies: list[float] = reference.get("latencies_ms", [])

        # 1. Query length drift via Kolmogorov-Smirnov test
        if ref_lengths:
            ks_len = ks_2samp(ref_lengths, req_lengths)
            ks_len_stat = round(float(ks_len.statistic), 4)
            ks_len_p = round(float(ks_len.pvalue), 4)
            mean_ref_length = round(statistics.mean(ref_lengths), 2)
            query_len_drift = bool(ks_len.pvalue < KS_ALPHA_THRESHOLD)
            tests["query_length"] = {
                "metric": "Độ dài Query (Query Length)",
                "test": "Kolmogorov-Smirnov (KS-2)",
                "ref_mean": mean_ref_length,
                "req_mean": mean_req_length,
                "statistic": ks_len_stat,
                "p_value": ks_len_p,
                "threshold": f"p < {KS_ALPHA_THRESHOLD}",
                "drift": query_len_drift,
            }

        # 2. Out of Vocabulary (OOV) drift
        oov_tokens = [ing for ing in all_ingredients if ing not in ref_vocab]
        oov_rate = round(len(oov_tokens) / max(1, total_tokens), 4)
        oov_drift = oov_rate > OOV_DRIFT_THRESHOLD
        tests["oov_rate"] = {
            "metric": "Nguyên liệu ngoài Vocab (OOV Rate)",
            "test": "Tỉ lệ thành phần (Proportion)",
            "ref_mean": 0.0,
            "req_mean": oov_rate,
            "statistic": oov_rate,
            "p_value": None,
            "threshold": f"> {OOV_DRIFT_THRESHOLD:.2f}",
            "drift": oov_drift,
        }

        # 3. Latency drift
        if ref_latencies:
            ks_lat = ks_2samp(ref_latencies, latencies)
            ks_lat_stat = round(float(ks_lat.statistic), 4)
            ks_lat_p = round(float(ks_lat.pvalue), 4)
            ref_median_lat = round(statistics.median(ref_latencies), 2)
            latency_drift = bool(
                ks_lat.pvalue < KS_ALPHA_THRESHOLD
                and latency_stats["median"] > 1.3 * ref_median_lat
            )
            tests["latency"] = {
                "metric": "Độ trễ phản hồi (Latency)",
                "test": "Kolmogorov-Smirnov (KS-2)",
                "ref_mean": ref_median_lat,
                "req_mean": latency_stats["median"],
                "statistic": ks_lat_stat,
                "p_value": ks_lat_p,
                "threshold": f"p < {KS_ALPHA_THRESHOLD}",
                "drift": latency_drift,
            }

    # 4. Error rate test
    tests["error_rate"] = {
        "metric": "Tỉ lệ lỗi API (Error Rate)",
        "test": "Ngưỡng tần suất lỗi",
        "ref_mean": 0.0,
        "req_mean": error_rate,
        "statistic": error_rate,
        "p_value": None,
        "threshold": f"> {ERROR_DRIFT_THRESHOLD:.2f}",
        "drift": error_drift,
    }

    # 5. Missing / zero-hit rate test
    zero_hit_drift = zero_hit_rate > 0.10
    tests["zero_hit_rate"] = {
        "metric": "Tỉ lệ truy vấn không kết quả (Zero-Hit)",
        "test": "Ngưỡng tần suất",
        "ref_mean": 0.0,
        "req_mean": zero_hit_rate,
        "statistic": zero_hit_rate,
        "p_value": None,
        "threshold": "> 0.10",
        "drift": zero_hit_drift,
    }

    drift_detected = bool(query_len_drift or oov_drift or latency_drift or error_drift)
    status = "DRIFT_DETECTED" if drift_detected else "NO_DRIFT"
    msg = (
        "Phát hiện dịch chuyển phân phối (Data Drift Detected)"
        if drift_detected
        else "Phân phối ổn định, không phát hiện dịch chuyển đáng kể (No Significant Drift)"
    )

    return {
        "total_requests": total,
        "min_samples_required": min_samples,
        "error_count": errors,
        "error_rate": error_rate,
        "success_rate": success_rate,
        "latency_ms": latency_stats,
        "predict_requests": len(predict_records),
        "zero_hit_count": zero_hits,
        "zero_hit_rate": zero_hit_rate,
        "mean_query_length": mean_req_length,
        "total_ingredient_tokens": total_tokens,
        "oov_tokens_count": len(oov_tokens),
        "oov_rate": oov_rate,
        "top_oov_ingredients": Counter(oov_tokens).most_common(8),
        "insufficient_data": False,
        "status": status,
        "drift_detected": drift_detected,
        "message": msg,
        "tests": tests,
        "endpoint_counts": dict(endpoints),
        "model_counts": dict(models),
        "top_ingredients": ingredient_counter.most_common(10),
        "is_simulation": is_simulation,
    }


def format_html_report(metrics: dict[str, Any]) -> str:
    """Generate self-contained, responsive HTML monitoring report."""
    status = metrics.get("status", "NO_DRIFT")
    is_sim = metrics.get("is_simulation", False)

    if status == "NO_DRIFT":
        status_color = "#10b981"
        status_label = "ỔN ĐỊNH (NO DRIFT)"
    elif status == "DRIFT_DETECTED":
        status_color = "#ef4444"
        status_label = "CẢNH BÁO DỊCH CHUYỂN (DRIFT DETECTED)"
    else:
        status_color = "#f59e0b"
        status_label = "CHƯA ĐỦ BẰNG CHỨNG (INSUFFICIENT DATA)"

    sim_banner = ""
    if is_sim:
        sim_banner = (
            '<div class="simulation-banner">\n'
            '    <span class="badge-sim">MÔ PHỎNG (SIMULATION)</span>\n'
            '    <strong>LƯU Ý QUAN TRỌNG:</strong> Báo cáo này được lập từ kịch bản mô phỏng '
            'dữ liệu (simulation), không suy ra sự suy giảm độ liên quan thực tế của hệ thống '
            '(relevance decay).\n'
            '</div>'
        )

    tests_rows = ""
    for k, t in metrics.get("tests", {}).items():
        is_drift = t.get("drift", False)
        badge = (
            '<span class="pill pill-danger">CẢNH BÁO DRIFT</span>'
            if is_drift
            else '<span class="pill pill-success">ĐẠT (PASS)</span>'
        )
        p_val_str = f"{t['p_value']:.4f}" if t.get("p_value") is not None else "—"
        raw_stat = t.get("statistic")
        stat_str = f"{raw_stat:.4f}" if isinstance(raw_stat, (int, float)) else str(raw_stat)
        raw_ref = t.get("ref_mean")
        ref_str = f"{raw_ref:.2f}" if isinstance(raw_ref, (int, float)) else str(raw_ref)
        raw_req = t.get("req_mean")
        req_str = f"{raw_req:.2f}" if isinstance(raw_req, (int, float)) else str(raw_req)

        tests_rows += f"""
        <tr>
            <td><strong>{t.get('metric', k)}</strong></td>
            <td>{t.get('test', '—')}</td>
            <td>{ref_str}</td>
            <td>{req_str}</td>
            <td><code>{stat_str}</code></td>
            <td><code>{p_val_str}</code></td>
            <td>{t.get('threshold', '—')}</td>
            <td>{badge}</td>
        </tr>
        """

    lat = metrics.get("latency_ms", {})
    lat_rows = f"""
    <tr><td>Trung bình (Mean)</td><td>{lat.get('mean', '—')} ms</td></tr>
    <tr><td>Trung vị (p50)</td><td>{lat.get('median', '—')} ms</td></tr>
    <tr><td>Phân vị 90 (p90)</td><td>{lat.get('p90', '—')} ms</td></tr>
    <tr><td>Phân vị 95 (p95)</td><td><strong>{lat.get('p95', '—')} ms</strong></td></tr>
    <tr><td>Phân vị 99 (p99)</td><td>{lat.get('p99', '—')} ms</td></tr>
    <tr><td>Tối thiểu / Tối đa</td><td>{lat.get('min', '—')} ms / {lat.get('max', '—')} ms</td></tr>
    """

    oov_items = metrics.get("top_oov_ingredients", [])
    oov_rows = (
        "".join(f"<li><code>{ing}</code>: {cnt} lần</li>" for ing, cnt in oov_items)
        or "<li>Không có nguyên liệu lạ.</li>"
    )

    msg_text = metrics.get("message", "")
    min_req_count = metrics.get("min_samples_required", DEFAULT_MIN_SAMPLE_SIZE)

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Báo Cáo Giám Sát API & Data Drift</title>
    <style>
        :root {{
            --bg: #0f172a;
            --surface: #1e293b;
            --border: #334155;
            --text: #f8fafc;
            --muted: #94a3b8;
            --primary: #38bdf8;
            --success: #10b981;
            --danger: #ef4444;
            --warning: #f59e0b;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 24px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        header {{
            border-bottom: 1px solid var(--border);
            padding-bottom: 16px;
            margin-bottom: 24px;
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
        }}
        h1 {{
            margin: 0 0 6px 0;
            font-size: 24px;
            color: #ffffff;
        }}
        .timestamp {{
            color: var(--muted);
            font-size: 13px;
        }}
        .simulation-banner {{
            background: rgba(245, 158, 11, 0.15);
            border: 1px solid var(--warning);
            color: #fde68a;
            padding: 12px 16px;
            border-radius: 6px;
            margin-bottom: 24px;
            font-size: 14px;
        }}
        .badge-sim {{
            background: var(--warning);
            color: #000000;
            font-weight: bold;
            font-size: 11px;
            padding: 2px 6px;
            border-radius: 4px;
            margin-right: 8px;
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .kpi-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
        }}
        .kpi-title {{
            font-size: 12px;
            color: var(--muted);
            text-transform: uppercase;
            font-weight: 600;
            margin-bottom: 8px;
        }}
        .kpi-value {{
            font-size: 22px;
            font-weight: bold;
        }}
        .status-card {{
            border-left: 4px solid {status_color};
        }}
        .section-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 24px;
        }}
        h2 {{
            margin: 0 0 16px 0;
            font-size: 18px;
            color: #e2e8f0;
            border-bottom: 1px solid var(--border);
            padding-bottom: 8px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        th, td {{
            text-align: left;
            padding: 10px 12px;
            border-bottom: 1px solid var(--border);
        }}
        th {{
            color: var(--muted);
            font-weight: 600;
            background: #182234;
        }}
        tr:hover {{
            background: rgba(255, 255, 255, 0.02);
        }}
        code {{
            background: #0b1120;
            padding: 2px 6px;
            border-radius: 4px;
            color: var(--primary);
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        }}
        .pill {{
            display: inline-block;
            font-size: 11px;
            font-weight: bold;
            padding: 3px 8px;
            border-radius: 12px;
        }}
        .pill-success {{ background: rgba(16, 185, 129, 0.2); color: #34d399; }}
        .pill-danger {{ background: rgba(239, 68, 68, 0.2); color: #f87171; }}
        .pill-warning {{ background: rgba(245, 158, 11, 0.2); color: #fbbf24; }}
        .grid-2 {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        @media (max-width: 768px) {{
            .grid-2 {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>Hệ Thống Giám Sát API & Phát Hiện Data Drift</h1>
                <div class="timestamp">
                    Sinh lúc UTC: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}
                    | Vai trò: Hiếu (Person 3)
                </div>
            </div>
            <div>
                <span class="pill"
                      style="background: {status_color}; color: #000000; font-size: 13px; padding: 6px 14px;">
                    {status_label}
                </span>
            </div>
        </header>

        {sim_banner}

        <div class="kpi-grid">
            <div class="kpi-card status-card">
                <div class="kpi-title">Trạng Thái Phân Phối</div>
                <div class="kpi-value" style="color: {status_color}; font-size: 18px;">{status_label}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Tổng Lượt Request</div>
                <div class="kpi-value">{metrics.get('total_requests', 0):,}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Tỉ Lệ Lỗi (Error Rate)</div>
                <div class="kpi-value">{metrics.get('error_rate', 0.0) * 100:.2f}%</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Tỉ Lệ Nguyên Liệu Lạ (OOV)</div>
                <div class="kpi-value">{metrics.get('oov_rate', 0.0) * 100:.2f}%</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Độ Trễ p95</div>
                <div class="kpi-value">{lat.get('p95', '—')} ms</div>
            </div>
        </div>

        <div class="section-card">
            <h2>Kiểm Định Thống Kê & Phát Hiện Dịch Chuyển (Drift Tests)</h2>
            <p style="color: var(--muted); font-size: 13px; margin-top: -8px; margin-bottom: 16px;">
                Mẫu tối thiểu: {min_req_count} quan sát. Ngưỡng ý nghĩa: α = {KS_ALPHA_THRESHOLD}.<br>
                Kết luận: {msg_text}
            </p>
            <table>
                <thead>
                    <tr>
                        <th>Đặc Trưng (Feature)</th>
                        <th>Phương Pháp Kiểm Định</th>
                        <th>Reference Mean</th>
                        <th>Quan Sát Mean</th>
                        <th>Thống Kê (D/Rate)</th>
                        <th>p-value</th>
                        <th>Ngưỡng Cảnh Báo</th>
                        <th>Trạng Thái</th>
                    </tr>
                </thead>
                <tbody>
                    {tests_rows}
                </tbody>
            </table>
        </div>

        <div class="grid-2">
            <div class="section-card">
                <h2>Phân Phối Độ Trễ API (Latency Percentiles)</h2>
                <table>
                    <thead>
                        <tr><th>Thước Đo</th><th>Thời Gian Phản Hồi</th></tr>
                    </thead>
                    <tbody>
                        {lat_rows}
                    </tbody>
                </table>
            </div>

            <div class="section-card">
                <h2>Top Nguyên Liệu Mới Lạ (OOV Ingredients)</h2>
                <p style="color: var(--muted); font-size: 13px;">
                    Các từ khóa xuất hiện trong request nhưng chưa có trong dữ liệu huấn luyện:
                </p>
                <ul style="padding-left: 20px; font-size: 14px; line-height: 1.8;">
                    {oov_rows}
                </ul>
            </div>
        </div>
    </div>
</body>
</html>
"""
    return html


def format_monitoring_report(metrics: dict[str, Any]) -> str:
    """Format monitoring metrics as a readable markdown document for terminal/cli."""
    errors_cnt = metrics.get("error_count", 0)
    zero_hits_cnt = metrics.get("zero_hit_count", 0)
    lines = [
        "# API & Model Monitoring Report",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}",
        "",
        "## Summary",
        "",
        f"- **Total Requests:** {metrics['total_requests']}",
        f"- **Status:** {metrics.get('status')}",
        f"- **Drift Detected:** {metrics.get('drift_detected')}",
        f"- **Success Rate:** {metrics['success_rate'] * 100:.2f}%",
        f"- **Error Rate:** {metrics['error_rate'] * 100:.2f}% ({errors_cnt} errors)",
        f"- **Zero-Hit Rate:** {metrics['zero_hit_rate'] * 100:.2f}% ({zero_hits_cnt} queries)",
        f"- **OOV Rate:** {metrics.get('oov_rate', 0.0) * 100:.2f}%",
        f"- **Message:** {metrics.get('message', '')}",
        "",
        "## Latency Distribution (ms)",
        "",
    ]

    lat = metrics.get("latency_ms", {})
    if lat:
        lines.extend(
            [
                "| Metric | Latency (ms) |",
                "| --- | --- |",
                f"| Mean | {lat.get('mean')} |",
                f"| Median (p50) | {lat.get('median')} |",
                f"| p90 | {lat.get('p90')} |",
                f"| p95 | {lat.get('p95')} |",
                f"| p99 | {lat.get('p99')} |",
                f"| Min | {lat.get('min')} |",
                f"| Max | {lat.get('max')} |",
                "",
            ]
        )

    lines.extend(
        [
            "## Statistical Drift Tests",
            "",
            "| Feature | Method | Ref Mean | Observed Mean | Statistic | p-value | Drift? |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for k, t in metrics.get("tests", {}).items():
        p_str = f"{t['p_value']:.4f}" if t.get("p_value") is not None else "—"
        raw_s = t.get("statistic")
        stat_str = f"{raw_s:.4f}" if isinstance(raw_s, (int, float)) else str(raw_s)
        drift_str = "YES" if t.get("drift") else "NO"
        lines.append(
            f"| {t.get('metric', k)} | {t.get('test')} | {t.get('ref_mean')} | "
            f"{t.get('req_mean')} | {stat_str} | {p_str} | {drift_str} |"
        )

    lines.extend(
        [
            "",
            "## Top 10 Requested Ingredients",
            "",
            "| Ingredient | Frequency |",
            "| --- | --- |",
        ]
    )
    for ing, count in metrics.get("top_ingredients", []):
        lines.append(f"| {ing} | {count} |")

    return "\n".join(lines)


def log_monitoring_to_mlflow(
    metrics: dict[str, Any],
    *,
    tracking_uri: str = "sqlite:///mlflow.db",
    experiment: str = "api-monitoring",
) -> str | None:
    """Log monitoring snapshot metrics to MLflow."""
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)

        run_name = f"monitoring-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        with mlflow.start_run(run_name=run_name) as run:
            flat_metrics: dict[str, float] = {
                "total_requests": float(metrics["total_requests"]),
                "error_rate": float(metrics["error_rate"]),
                "success_rate": float(metrics["success_rate"]),
                "zero_hit_rate": float(metrics["zero_hit_rate"]),
                "drift_flag": 1.0 if metrics.get("drift_detected") else 0.0,
            }
            if "oov_rate" in metrics:
                flat_metrics["oov_rate"] = float(metrics["oov_rate"])
            for k, v in metrics.get("latency_ms", {}).items():
                if isinstance(v, (int, float)):
                    flat_metrics[f"latency_{k}"] = float(v)

            mlflow.log_metrics(flat_metrics)
            return run.info.run_id
    except Exception as e:
        print(f"warning: MLflow monitoring logging failed: {e}", file=sys.stderr)
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", help="Path to reference queries (queries.jsonl or reference requests)")
    parser.add_argument("--requests", dest="requests", help="Path to request logs (requests.jsonl)")
    parser.add_argument("--log-path", dest="legacy_log_path", help="Legacy alias for --requests")
    parser.add_argument(
        "--out",
        default=str(DEFAULT_REPORT_PATH),
        help="Output HTML (or markdown) report path",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=DEFAULT_MIN_SAMPLE_SIZE,
        help="Minimum sample threshold",
    )
    parser.add_argument(
        "--simulation",
        action="store_true",
        help="Tag report with simulation label and disclaimer",
    )
    parser.add_argument("--mlflow", action="store_true", help="Log monitoring metrics to MLflow")
    args = parser.parse_args(argv)

    req_path = args.requests or args.legacy_log_path or str(DEFAULT_LOG_PATH)
    records = load_request_logs(req_path)
    if not records:
        print(f"info: No log records found at {req_path}")
        return 0

    ref_data = None
    if args.reference:
        ref_data = load_reference_data(args.reference)

    metrics = compute_monitoring_metrics(
        records,
        reference=ref_data,
        min_samples=args.min_samples,
        is_simulation=args.simulation,
    )

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if out_p.suffix.lower() in [".html", ".htm"]:
        content = format_html_report(metrics)
    else:
        content = format_monitoring_report(metrics)

    out_p.write_text(content, encoding="utf-8")
    status_str = metrics["status"]
    drift_str = metrics["drift_detected"]
    print(f"Monitoring report saved to {out_p} [Status: {status_str}, Drift: {drift_str}]")

    if args.mlflow:
        run_id = log_monitoring_to_mlflow(metrics)
        if run_id:
            print(f"Monitoring snapshot logged to MLflow run: {run_id}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
