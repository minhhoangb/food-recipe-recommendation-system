"""Request log for the API: one `RequestLogRecord` per prediction, in SQLite.

SQLite is used because several uvicorn workers (and several threads inside one worker)
may write at the same time: every write opens its own connection in WAL mode with a
busy timeout, so a concurrent write waits for the lock instead of failing or
interleaving half a line.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from src.contracts import RequestLogRecord

TABLE = "request_log"
COLUMNS = (
    "request_id",
    "timestamp_utc",
    "endpoint",
    "model",
    "model_version",
    "ingredients",
    "top_k",
    "result_ids",
    "scores",
    "latency_ms",
    "status_code",
    "missing_ingredient_counts",
)


class RequestLogger:
    """Append-only log of the API requests. `path=None` disables logging."""

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with closing(self._connect()) as connection, connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(f"CREATE TABLE IF NOT EXISTS {TABLE} ({', '.join(COLUMNS)})")
                columns = {row[1] for row in connection.execute(f"PRAGMA table_info({TABLE})")}
                if "missing_ingredient_counts" not in columns:
                    connection.execute(f"ALTER TABLE {TABLE} ADD COLUMN missing_ingredient_counts")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def log(self, record: RequestLogRecord) -> None:
        """Write one record. A failing log must never break a prediction."""
        if self.path is None:
            return
        row = record.model_dump()
        try:
            with closing(self._connect()) as connection, connection:
                connection.execute(
                    f"INSERT INTO {TABLE} ({', '.join(COLUMNS)}) VALUES ({', '.join('?' for _ in COLUMNS)})",
                    (
                        row["request_id"],
                        row["timestamp_utc"],
                        row["endpoint"],
                        row["model"],
                        row["model_version"],
                        ",".join(row["ingredients"]),
                        row["top_k"],
                        ",".join(row["result_ids"]),
                        ",".join(str(score) for score in row["scores"]),
                        row["latency_ms"],
                        row["status_code"],
                        json.dumps(row["missing_ingredient_counts"])
                        if row["missing_ingredient_counts"] is not None
                        else None,
                    ),
                )
        except sqlite3.Error:
            pass  # the log is observability, not a hard dependency of the API


__all__ = ["COLUMNS", "TABLE", "RequestLogger"]
