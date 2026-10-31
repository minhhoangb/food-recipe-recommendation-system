from __future__ import annotations

from pathlib import Path

import pytest

from src.contracts import RequestLogRecord
from src.monitor import (
    compute_monitoring_metrics,
    format_html_report,
    format_monitoring_report,
    load_request_logs,
    main,
)
from src.utils import write_jsonl


@pytest.fixture
def sample_records() -> list[RequestLogRecord]:
    return [
        RequestLogRecord(
            request_id="req-1",
            timestamp_utc="2026-10-26T12:00:00Z",
            endpoint="/predict",
            model="minilm_pretrained",
            model_version="1",
            ingredients=["chicken", "rice"],
            top_k=5,
            result_ids=["rec-1", "rec-2"],
            scores=[0.85, 0.72],
            latency_ms=25.0,
            status_code=200,
        ),
        RequestLogRecord(
            request_id="req-2",
            timestamp_utc="2026-10-26T12:01:00Z",
            endpoint="/predict",
            model="minilm_pretrained",
            model_version="1",
            ingredients=["garlic"],
            top_k=5,
            result_ids=[],
            scores=[],
            latency_ms=10.0,
            status_code=200,
        ),
        RequestLogRecord(
            request_id="req-3",
            timestamp_utc="2026-10-26T12:02:00Z",
            endpoint="/predict",
            model="minilm_pretrained",
            model_version="1",
            ingredients=["bad_input"],
            top_k=5,
            result_ids=[],
            scores=[],
            latency_ms=5.0,
            status_code=422,
        ),
    ]


@pytest.fixture
def reference_data() -> dict:
    vocab = {"chicken", "rice", "garlic", "onion", "butter", "beef", "salt", "pepper", "olive oil"}
    # Standard 2-4 length queries
    lengths = [2, 3, 2, 4, 3, 2, 3, 4, 3, 2] * 5
    latencies = [18.0 + (i % 6) * 2.0 for i in range(len(lengths))]
    return {
        "vocab": vocab,
        "query_lengths": lengths,
        "latencies_ms": latencies,
        "sample_count": len(lengths),
    }


def test_compute_monitoring_metrics(sample_records):
    metrics = compute_monitoring_metrics(sample_records)
    assert metrics["total_requests"] == 3
    assert metrics["error_count"] == 1
    assert metrics["error_rate"] == pytest.approx(1 / 3, abs=1e-3)
    assert metrics["success_rate"] == pytest.approx(2 / 3, abs=1e-3)
    assert metrics["predict_requests"] == 3
    assert metrics["zero_hit_count"] == 2
    assert metrics["zero_hit_rate"] == pytest.approx(2 / 3, abs=1e-3)

    lat = metrics["latency_ms"]
    assert lat["min"] == 5.0
    assert lat["max"] == 25.0
    assert lat["median"] == 10.0

    top_ings = dict(metrics["top_ingredients"])
    assert "chicken" in top_ings
    assert "rice" in top_ings


def test_load_request_logs(tmp_path: Path, sample_records):
    log_file = tmp_path / "requests.jsonl"
    write_jsonl(log_file, [r.model_dump() for r in sample_records])

    loaded = load_request_logs(log_file)
    assert len(loaded) == 3
    assert loaded[0].request_id == "req-1"
    assert loaded[1].ingredients == ["garlic"]


def test_format_monitoring_report(sample_records):
    metrics = compute_monitoring_metrics(sample_records)
    report = format_monitoring_report(metrics)
    assert "# API & Model Monitoring Report" in report
    assert "**Total Requests:** 3" in report
    assert "Latency Distribution (ms)" in report
    assert "Top 10 Requested Ingredients" in report


def test_monitor_no_drift_scenario(reference_data):
    """Scenario 1: Synthetic requests without drift (sampled from reference distribution)."""
    clean_vocab = list(reference_data["vocab"])
    ref_lengths = reference_data["query_lengths"]
    records = []
    for i in range(30):
        length = ref_lengths[i % len(ref_lengths)]
        ings = [clean_vocab[(i + j) % len(clean_vocab)] for j in range(length)]
        records.append(
            RequestLogRecord(
                request_id=f"req-nodrift-{i}",
                timestamp_utc="2026-10-27T10:00:00Z",
                endpoint="/predict",
                model="minilm_finetuned",
                model_version="1",
                ingredients=ings,
                top_k=5,
                result_ids=["r1", "r2"],
                scores=[0.9, 0.8],
                latency_ms=reference_data["latencies_ms"][i % len(reference_data["latencies_ms"])],
                status_code=200,
            )
        )

    metrics = compute_monitoring_metrics(records, reference=reference_data, min_samples=10, is_simulation=True)
    assert metrics["drift_detected"] is False
    assert metrics["status"] == "NO_DRIFT"
    assert metrics["oov_rate"] == 0.0
    assert metrics["error_rate"] == 0.0
    assert "query_length" in metrics["tests"]
    assert metrics["tests"]["query_length"]["drift"] is False


def test_monitor_drift_scenario(reference_data):
    """Scenario 2: Synthetic requests with significant drift (longer queries, exotic OOV ingredients)."""
    novel_ingredients = ["dragonfruit_powder", "spirulina_extract", "matcha_collagen", "truffle_essence", "chia_seeds"]
    records = []
    for i in range(30):
        # 6 to 9 ingredients (much longer than reference lengths of 2-4)
        ings = ["chicken", "garlic"] + novel_ingredients[: 3 + (i % 3)]
        records.append(
            RequestLogRecord(
                request_id=f"req-drift-{i}",
                timestamp_utc="2026-10-27T11:00:00Z",
                endpoint="/predict",
                model="minilm_finetuned",
                model_version="1",
                ingredients=ings,
                top_k=5,
                result_ids=["r1"],
                scores=[0.5],
                latency_ms=95.0 + i,
                status_code=200 if i % 10 != 0 else 422,
            )
        )

    metrics = compute_monitoring_metrics(records, reference=reference_data, min_samples=10, is_simulation=True)
    assert metrics["drift_detected"] is True
    assert metrics["status"] == "DRIFT_DETECTED"
    assert metrics["oov_rate"] > 0.05
    assert metrics["tests"]["query_length"]["drift"] is True
    assert metrics["tests"]["oov_rate"]["drift"] is True
    assert len(metrics["top_oov_ingredients"]) > 0


def test_monitor_insufficient_data(reference_data):
    """Scenario 3: Insufficient sample size (e.g. 3 requests < min_samples 10)."""
    records = [
        RequestLogRecord(
            request_id=f"req-small-{i}",
            timestamp_utc="2026-10-27T12:00:00Z",
            endpoint="/predict",
            model="minilm_finetuned",
            model_version="1",
            ingredients=["chicken", "rice"],
            top_k=5,
            result_ids=["r1"],
            scores=[0.8],
            latency_ms=20.0,
            status_code=200,
        )
        for i in range(3)
    ]

    metrics = compute_monitoring_metrics(records, reference=reference_data, min_samples=10)
    assert metrics["insufficient_data"] is True
    assert metrics["status"] == "INSUFFICIENT_DATA"
    assert metrics["drift_detected"] is False
    assert "Chưa đủ bằng chứng" in metrics["message"]


def test_format_html_report_contains_simulation_notice(reference_data, tmp_path: Path):
    """HTML report should contain simulation warning banner and disclaimer."""
    clean_vocab = list(reference_data["vocab"])
    ref_lengths = reference_data["query_lengths"]
    records = [
        RequestLogRecord(
            request_id=f"req-{i}",
            timestamp_utc="2026-10-27T12:00:00Z",
            endpoint="/predict",
            model="minilm_finetuned",
            model_version="1",
            ingredients=[clean_vocab[(i + j) % len(clean_vocab)] for j in range(ref_lengths[i % len(ref_lengths)])],
            top_k=5,
            result_ids=["r1"],
            scores=[0.8],
            latency_ms=20.0,
            status_code=200,
        )
        for i in range(20)
    ]
    metrics = compute_monitoring_metrics(records, reference=reference_data, is_simulation=True)
    html = format_html_report(metrics)
    assert "<!DOCTYPE html>" in html
    assert "MÔ PHỎNG (SIMULATION)" in html
    assert "không suy ra sự suy giảm độ liên quan thực tế" in html
    assert "ỔN ĐỊNH (NO DRIFT)" in html


def test_monitor_cli_generates_html(tmp_path: Path, reference_data):
    """CLI python -m src.monitor --reference ... --requests ... --out ... works cleanly."""
    ref_file = tmp_path / "reference_queries.jsonl"
    write_jsonl(ref_file, [{"query_id": f"q{i}", "ingredients": ["chicken", "rice"]} for i in range(20)])

    req_file = tmp_path / "requests.jsonl"
    write_jsonl(
        req_file,
        [
            RequestLogRecord(
                request_id=f"r-{i}",
                timestamp_utc="2026-10-27T12:00:00Z",
                endpoint="/predict",
                model="minilm_finetuned",
                model_version="1",
                ingredients=["chicken", "garlic"],
                top_k=5,
                result_ids=["rec-1"],
                scores=[0.88],
                latency_ms=18.5,
                status_code=200,
            ).model_dump()
            for i in range(15)
        ],
    )

    out_html = tmp_path / "report.html"
    ret = main(["--reference", str(ref_file), "--requests", str(req_file), "--out", str(out_html), "--simulation"])
    assert ret == 0
    assert out_html.exists()
    content = out_html.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "Hệ Thống Giám Sát API" in content
