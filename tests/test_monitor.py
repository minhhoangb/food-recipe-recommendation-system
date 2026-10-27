from __future__ import annotations

from pathlib import Path

import pytest

from src.contracts import RequestLogRecord
from src.monitor import compute_monitoring_metrics, format_monitoring_report, load_request_logs
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


def test_compute_monitoring_metrics(sample_records):
    metrics = compute_monitoring_metrics(sample_records)
    assert metrics["total_requests"] == 3
    assert metrics["error_count"] == 1
    assert metrics["error_rate"] == pytest.approx(1 / 3, abs=1e-3)
    assert metrics["success_rate"] == pytest.approx(2 / 3, abs=1e-3)
    assert metrics["predict_requests"] == 3
    # req-2 and req-3 had empty result_ids
    assert metrics["zero_hit_count"] == 2
    assert metrics["zero_hit_rate"] == pytest.approx(2 / 3, abs=1e-3)

    # Latency checks
    lat = metrics["latency_ms"]
    assert lat["min"] == 5.0
    assert lat["max"] == 25.0
    assert lat["median"] == 10.0

    # Top ingredients
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
