"""Peak resident memory of the current process, in MB.

Portable and dependency-free: `/proc/self/status` on Linux, `GetProcessMemoryInfo` on
Windows, `None` when the platform exposes neither. Used to report the real RAM cost of
the encoding and evaluation steps (the assignment asks for measured RAM, not a guess).
"""

from __future__ import annotations

import sys


def peak_rss_mb() -> float | None:
    """Peak resident set size of this process in MB, or None if it cannot be read."""
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/self/status", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("VmHWM:"):
                        return round(int(line.split()[1]) / 1024, 1)
        except OSError:
            return None
        return None
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class _Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _Counters()
        counters.cb = ctypes.sizeof(_Counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()  # type: ignore[attr-defined]
        psapi = ctypes.windll.psapi  # type: ignore[attr-defined]
        if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return round(counters.PeakWorkingSetSize / (1024 * 1024), 1)
    return None
