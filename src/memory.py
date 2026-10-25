"""Peak resident memory of the current process, in MB.

Uses `psutil` when it is installed on Windows (where `memory_full_info` exposes the peak
working set), and `/proc/self/status` (VmHWM) on Linux. Returns the current RSS as the
fallback, and None only when nothing can be read, so RAM is never guessed.
"""

from __future__ import annotations

import os


def _psutil_peak_mb() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    try:
        if os.name == "nt":
            full = psutil.Process().memory_full_info()
            if hasattr(full, "peak_wset") and full.peak_wset:
                return round(full.peak_wset / (1024 * 1024), 1)
        return round(psutil.Process().memory_info().rss / (1024 * 1024), 1)
    except (AttributeError, OSError):
        return None


def _linux_peak_mb() -> float | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        return None
    return None


def peak_rss_mb() -> float | None:
    """Peak resident set size of this process in MB, or None if it cannot be read."""
    return _psutil_peak_mb() or _linux_peak_mb()
