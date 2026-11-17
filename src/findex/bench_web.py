import asyncio
import math
import os
import sys
import time
from pathlib import Path
from typing import Annotated, Literal

import httpx
from fastapi import Depends, FastAPI, Query

from findex.index import Index
from findex.search import BM25, Scorer, TfIdf, ranked_search
from findex.web.app import create_app, get_index
from findex.web.schemas import ApiSearchResult, SearchResponse


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, math.ceil((pct / 100.0) * len(sorted_vals)) - 1)
    return sorted_vals[idx]


def build_bench_app() -> FastAPI:
    app = create_app()

    @app.get("/search-async", response_model=SearchResponse)
    async def search_async_blocking(
        q: Annotated[str, Query(min_length=1, max_length=200)],
        k: Annotated[int, Query(ge=1, le=100)] = 10,
        scorer: Annotated[Literal["bm25", "tfidf"], Query()] = "bm25",
        page: Annotated[int, Query(ge=1, le=1000)] = 1,
        idx: Annotated[Index, Depends(get_index)] = None,  # type: ignore[assignment]
    ) -> SearchResponse:
        t0 = time.perf_counter()
        active_scorer: Scorer = BM25() if scorer == "bm25" else TfIdf()
        all_hits = ranked_search(
            idx, q, scorer=active_scorer, k=max(idx.num_docs, k * page)
        )
        total = len(all_hits)
        total_pages = max(1, math.ceil(total / k)) if total > 0 else 1
        slice_hits = all_hits[(page - 1) * k : page * k]

        results: list[ApiSearchResult] = []
        for h in slice_hits:
            meta = idx.doc_meta.get(h.doc_id)
            url = meta.path if meta else f"doc://{h.doc_id}"
            title = url.rstrip("/").split("/")[-1] or url
            snippet = h.snippet.strip() or f"Document #{h.doc_id} ({url})"
            results.append(
                ApiSearchResult(
                    doc_id=h.doc_id,
                    title=title,
                    url=url,
                    score=round(h.score, 4),
                    snippet=snippet,
                )
            )

        return SearchResponse(
            query=q,
            scorer=scorer,
            page=page,
            page_size=k,
            total=total,
            total_pages=total_pages,
            took_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            results=results,
        )

    return app


bench_app = build_bench_app()


async def _measure_concurrent(
    base_url: str, path: str, concurrency: int, total_requests: int
) -> tuple[float, float, float, float, float]:
    sem = asyncio.Semaphore(concurrency)
    latencies_ms: list[float] = []

    async with httpx.AsyncClient(base_url=base_url, timeout=15.0) as client:
        await client.get(path)

        async def _one() -> None:
            async with sem:
                t0 = time.perf_counter()
                r = await client.get(path)
                r.raise_for_status()
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        t_start = time.perf_counter()
        await asyncio.gather(*[_one() for _ in range(total_requests)])
        total_wall = time.perf_counter() - t_start

    latencies_ms.sort()
    rps = total_requests / max(total_wall, 1e-6)
    p50 = _percentile(latencies_ms, 50)
    p95 = _percentile(latencies_ms, 95)
    p99 = _percentile(latencies_ms, 99)
    return total_wall, rps, p50, p95, p99


async def _wait_healthy(base_url: str) -> None:
    async with httpx.AsyncClient(base_url=base_url, timeout=2.0) as client:
        for _ in range(50):
            try:
                r = await client.get("/health")
                if r.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.1)
    raise RuntimeError(f"Server at {base_url} did not become healthy")


async def main_async() -> None:
    if not Path("crawl_index.bin").exists():
        print("Спочатку створіть crawl_index.bin через findex index!")
        return

    env_with_index = {**os.environ, "INDEX_PATH": "crawl_index.bin"}

    proc1 = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "uvicorn",
        "findex.bench_web:bench_app",
        "--host",
        "127.0.0.1",
        "--port",
        "8011",
        "--workers",
        "1",
        "--log-level",
        "warning",
        env=env_with_index,
    )
    try:
        await _wait_healthy("http://127.0.0.1:8011")
        w20_async, _, _, _, _ = await _measure_concurrent(
            "http://127.0.0.1:8011",
            "/search-async?q=mystery",
            concurrency=20,
            total_requests=20,
        )
        w20_def, _, _, _, _ = await _measure_concurrent(
            "http://127.0.0.1:8011",
            "/search?q=mystery",
            concurrency=20,
            total_requests=20,
        )
        print(
            f"20 concurrent requests: async def = {w20_async * 1000:.2f} ms | "
            f"def = {w20_def * 1000:.2f} ms\n"
        )

        _, rps_async, p50_a, p95_a, p99_a = await _measure_concurrent(
            "http://127.0.0.1:8011",
            "/search-async?q=mystery",
            concurrency=50,
            total_requests=400,
        )
        _, rps_def, p50_d, p95_d, p99_d = await _measure_concurrent(
            "http://127.0.0.1:8011",
            "/search?q=mystery",
            concurrency=50,
            total_requests=400,
        )
    finally:
        proc1.terminate()
        await proc1.wait()

    proc4 = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "uvicorn",
        "findex.bench_web:bench_app",
        "--host",
        "127.0.0.1",
        "--port",
        "8014",
        "--workers",
        "4",
        "--log-level",
        "warning",
        env=env_with_index,
    )
    try:
        await _wait_healthy("http://127.0.0.1:8014")
        _, rps_w4, p50_w4, p95_w4, p99_w4 = await _measure_concurrent(
            "http://127.0.0.1:8014",
            "/search?q=mystery",
            concurrency=50,
            total_requests=400,
        )
    finally:
        proc4.terminate()
        await proc4.wait()

    print("| Target | Route | Concurrency | RPS | p50 (ms) | p95 (ms) | p99 (ms) |")
    print("|---|---|---:|---:|---:|---:|---:|")
    print(
        f"| local, 1 worker, async def | /search | 50 | "
        f"{rps_async:.1f} | {p50_a:.2f} | {p95_a:.2f} | {p99_a:.2f} |"
    )
    print(
        f"| local, 1 worker, def | /search | 50 | "
        f"{rps_def:.1f} | {p50_d:.2f} | {p95_d:.2f} | {p99_d:.2f} |"
    )
    print(
        f"| local, 4 workers | /search | 50 | "
        f"{rps_w4:.1f} | {p50_w4:.2f} | {p95_w4:.2f} | {p99_w4:.2f} |"
    )


if __name__ == "__main__":
    asyncio.run(main_async())
