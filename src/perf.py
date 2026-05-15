"""Performance utilities: profiling, memory tracking, cache statistics."""

from __future__ import annotations

import cProfile
import io
import pstats
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import psutil


# ---------------------------------------------------------------------------
# Memory + wall time
# ---------------------------------------------------------------------------
@dataclass
class RunMetrics:
    label: str
    wall_seconds: float = 0.0
    cpu_seconds: float = 0.0
    peak_rss_mb: float = 0.0

    def __str__(self) -> str:
        return (
            f"[{self.label}] wall={self.wall_seconds:.2f}s "
            f"cpu={self.cpu_seconds:.2f}s peak_rss={self.peak_rss_mb:.0f}MB"
        )


@contextmanager
def measure(label: str = "block") -> Iterator[RunMetrics]:
    """Wall-clock + CPU + peak-RSS for a code block."""
    proc = psutil.Process()
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    start_rss = proc.memory_info().rss
    metrics = RunMetrics(label=label)
    try:
        yield metrics
    finally:
        metrics.wall_seconds = time.perf_counter() - start_wall
        metrics.cpu_seconds = time.process_time() - start_cpu
        end_rss = proc.memory_info().rss
        metrics.peak_rss_mb = max(start_rss, end_rss) / (1024 * 1024)


# ---------------------------------------------------------------------------
# cProfile wrapper
# ---------------------------------------------------------------------------
def profile_to_text(func, *args, top_n: int = 20, **kwargs) -> tuple[object, str]:
    """Run ``func`` under cProfile and return (result, formatted-top-N-text)."""
    pr = cProfile.Profile()
    pr.enable()
    result = func(*args, **kwargs)
    pr.disable()
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(top_n)
    return result, buf.getvalue()


def profile_to_file(func, *args, out_path: Path, top_n: int = 20, **kwargs):
    """Run ``func`` under cProfile and write stats to a .prof + a .txt file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pr = cProfile.Profile()
    pr.enable()
    result = func(*args, **kwargs)
    pr.disable()
    pr.dump_stats(str(out_path))
    txt = out_path.with_suffix(".txt")
    with txt.open("w", encoding="utf-8") as f:
        pstats.Stats(pr, stream=f).sort_stats("cumulative").print_stats(top_n)
    return result, txt


# ---------------------------------------------------------------------------
# Cache statistics (used by the scorer + price cache)
# ---------------------------------------------------------------------------
@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    saved_calls: int = 0
    estimated_dollars_saved: float = 0.0
    by_module: dict[str, dict[str, int]] = field(default_factory=dict)

    def hit(self, module: str = "default", dollars: float = 0.0) -> None:
        self.hits += 1
        self.saved_calls += 1
        self.estimated_dollars_saved += dollars
        self.by_module.setdefault(module, {"hits": 0, "misses": 0})["hits"] += 1

    def miss(self, module: str = "default") -> None:
        self.misses += 1
        self.by_module.setdefault(module, {"hits": 0, "misses": 0})["misses"] += 1

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def __str__(self) -> str:
        return (
            f"cache: hits={self.hits} misses={self.misses} "
            f"hit-rate={self.hit_rate:.1%} "
            f"saved≈${self.estimated_dollars_saved:.2f} on {self.saved_calls} calls"
        )
