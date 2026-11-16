import ctypes
import ctypes.wintypes
import json
import os
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

from findex.index import ExecutorType, build_index_parallel, collect_corpus_paths


class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.wintypes.DWORD),
        ("PageFaultCount", ctypes.wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def get_peak_rss_mb() -> float:
    if sys.platform == "win32":
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.wintypes.HANDLE
        kernel32.K32GetProcessMemoryInfo.argtypes = [
            ctypes.wintypes.HANDLE,
            ctypes.POINTER(_PROCESS_MEMORY_COUNTERS),
            ctypes.wintypes.DWORD,
        ]
        kernel32.K32GetProcessMemoryInfo.restype = ctypes.wintypes.BOOL

        counters = _PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(_PROCESS_MEMORY_COUNTERS)
        handle = kernel32.GetCurrentProcess()
        if kernel32.K32GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        ):
            return counters.PeakWorkingSetSize / (1024 * 1024)
    return 0.0


def _measure_once(
    paths: list[Path], executor: ExecutorType, workers: int
) -> tuple[float, float, float, float]:
    t_wall_0 = time.perf_counter()
    t_cpu_0 = time.process_time()
    _, merge_sec = build_index_parallel(
        paths, executor=executor, workers=workers, with_positions=True
    )
    wall = time.perf_counter() - t_wall_0
    cpu = time.process_time() - t_cpu_0
    rss_mb = get_peak_rss_mb()
    return wall, cpu, rss_mb, merge_sec


def run_io_bound_demo(paths: list[Path], workers: int = 8) -> tuple[float, float]:
    def read_many(p: Path) -> int:
        total = 0
        for _ in range(20):
            total += len(p.read_bytes())
        return total

    t0 = time.perf_counter()
    for p in paths:
        read_many(p)
    serial_io = time.perf_counter() - t0

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(read_many, paths))
    threaded_io = time.perf_counter() - t0
    return serial_io, threaded_io


def run_race_demo(
    iterations: int = 200_000, num_threads: int = 8
) -> tuple[int, int, int]:
    expected = iterations * num_threads
    unsafe_counter = 0

    def inc_unsafe() -> None:
        nonlocal unsafe_counter
        for _ in range(iterations):
            unsafe_counter += 1

    threads = [threading.Thread(target=inc_unsafe) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    safe_counter = 0
    lock = threading.Lock()

    def inc_safe() -> None:
        nonlocal safe_counter
        for _ in range(iterations):
            with lock:
                safe_counter += 1

    threads_safe = [threading.Thread(target=inc_safe) for _ in range(num_threads)]
    for t in threads_safe:
        t.start()
    for t in threads_safe:
        t.join()

    return expected, unsafe_counter, safe_counter


def generate_plot() -> None:
    p_gil = Path("bench_313.json")
    p_nogil = Path("bench_313t.json")
    if not p_gil.exists():
        return

    import matplotlib.pyplot as plt  # type: ignore[import-untyped]

    data_gil = cast(list[dict[str, Any]], json.loads(p_gil.read_text(encoding="utf-8")))
    data_nogil: list[dict[str, Any]] = (
        cast(list[dict[str, Any]], json.loads(p_nogil.read_text(encoding="utf-8")))
        if p_nogil.exists()
        else []
    )

    threads_pts: list[tuple[int, float]] = [(1, 1.0)] + [
        (int(r["workers"]), float(r["speedup"]))
        for r in data_gil
        if r["executor"] == "threads"
    ]
    proc_pts: list[tuple[int, float]] = [
        (int(r["workers"]), float(r["speedup"]))
        for r in data_gil
        if r["executor"] == "processes"
    ]
    nogil_pts: list[tuple[int, float]] = [(1, 1.0)] + [
        (int(r["workers"]), float(r["speedup"]))
        for r in data_nogil
        if r["executor"] == "threads"
    ]

    all_workers = sorted({w for w, _ in (threads_pts + proc_pts + nogil_pts)})

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)  # type: ignore[misc]
    ax.plot(  # type: ignore[misc]
        all_workers,
        [float(w) for w in all_workers],
        linestyle="--",
        color="gray",
        label="Ideal linear speedup",
    )
    if threads_pts:
        ax.plot(  # type: ignore[misc]
            [w for w, _ in threads_pts],
            [s for _, s in threads_pts],
            marker="o",
            label="threads (Python 3.13 GIL)",
        )
    if proc_pts:
        ax.plot(  # type: ignore[misc]
            [w for w, _ in proc_pts],
            [s for _, s in proc_pts],
            marker="s",
            label="processes (Python 3.13)",
        )
    if len(nogil_pts) > 1:
        ax.plot(  # type: ignore[misc]
            [w for w, _ in nogil_pts],
            [s for _, s in nogil_pts],
            marker="^",
            label="threads (Python 3.13t no-GIL)",
        )

    ax.set_title("findex Parallel Indexing Speedup vs Workers")  # type: ignore[misc]
    ax.set_xlabel("Number of Workers")  # type: ignore[misc]
    ax.set_ylabel("Speedup (x)")  # type: ignore[misc]
    ax.set_xticks(all_workers)  # type: ignore[misc]
    ax.set_ylim(0, max(max(all_workers) * 0.4, 3.0))  # type: ignore[misc]
    ax.grid(True, linestyle=":", alpha=0.6)  # type: ignore[misc]
    ax.legend()  # type: ignore[misc]

    out_dir = Path("docs")
    out_dir.mkdir(exist_ok=True)
    out_png = out_dir / "speedup.png"
    fig.tight_layout()  # type: ignore[misc]
    fig.savefig(out_png)  # type: ignore[misc]
    print(f"Saved speedup plot to {out_png}")


def main() -> None:
    corpus_dir = Path("data")
    paths = collect_corpus_paths(corpus_dir)
    if not paths:
        print("Корпус data/ порожній!")
        return

    gil_fn = getattr(sys, "_is_gil_enabled", None)
    gil_enabled = bool(gil_fn()) if callable(gil_fn) else True
    cpu_count = os.cpu_count() or 8
    worker_counts = sorted({1, 2, 4, 8, cpu_count})

    print(f"Python: {sys.version.split()[0]} | sys._is_gil_enabled() = {gil_enabled}")
    print(f"Files: {len(paths)} | Logical CPUs: {cpu_count}")

    # Warmup
    build_index_parallel(paths[: min(50, len(paths))], executor="serial", workers=1)

    results: list[dict[str, Any]] = []

    serial_runs = [_measure_once(paths, "serial", 1) for _ in range(3)]
    base_wall = statistics.median(r[0] for r in serial_runs)
    base_cpu = statistics.median(r[1] for r in serial_runs)
    base_rss = statistics.median(r[2] for r in serial_runs)
    base_merge = statistics.median(r[3] for r in serial_runs)
    results.append(
        {
            "label": "serial" if gil_enabled else "serial (3.13t)",
            "executor": "serial",
            "workers": 1,
            "wall": base_wall,
            "cpu": base_cpu,
            "rss": base_rss,
            "merge": base_merge,
            "speedup": 1.0,
        }
    )

    executors_to_run: list[ExecutorType] = (
        ["threads", "processes"] if gil_enabled else ["threads"]
    )
    for exc in executors_to_run:
        for w in worker_counts:
            if exc == "threads" and w == 1:
                continue
            runs = [_measure_once(paths, exc, w) for _ in range(3)]
            med_wall = statistics.median(r[0] for r in runs)
            med_cpu = statistics.median(r[1] for r in runs)
            med_rss = statistics.median(r[2] for r in runs)
            med_merge = statistics.median(r[3] for r in runs)
            speedup = base_wall / med_wall if med_wall > 0 else 0.0
            label = exc if gil_enabled else "threads (3.13t no-GIL)"
            results.append(
                {
                    "label": label,
                    "executor": exc,
                    "workers": w,
                    "wall": med_wall,
                    "cpu": med_cpu,
                    "rss": med_rss,
                    "merge": med_merge,
                    "speedup": speedup,
                }
            )

    print(
        "\n| Executor | Workers | Wall (s, median) | CPU (s) | Peak RSS (MB) | Merge (s) | Speedup |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in results:
        print(
            f"| {row['label']} | {row['workers']} | {row['wall']:.4f} | "
            f"{row['cpu']:.4f} | {row['rss']:.1f} | {row['merge']:.4f} | {row['speedup']:.2f}x |"
        )

    io_serial, io_threads = run_io_bound_demo(paths, workers=8)
    print(
        f"\nI/O-bound read ({len(paths)} files x 20): serial={io_serial:.4f}s, "
        f"threads(8)={io_threads:.4f}s (speedup={io_serial / max(io_threads, 1e-6):.2f}x)"
    )

    exp, unsafe_c, safe_c = run_race_demo()
    print(
        f"Race condition demo (expected={exp}): without Lock={unsafe_c} "
        f"(lost={exp - unsafe_c}), with Lock={safe_c}"
    )

    out_json = Path("bench_313.json" if gil_enabled else "bench_313t.json")
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved raw results to {out_json}")

    if gil_enabled:
        generate_plot()


if __name__ == "__main__":
    main()
