import asyncio
import logging
import time

import httpx

from findex.bench_parallel import get_peak_rss_mb
from findex.crawler.crawl import CrawlStats, crawl
from findex.crawler.fetch import FetchResult, fetch

SEED_URL = "https://books.toscrape.com/"


async def _bench_one(
    concurrency: int, max_pages: int = 200
) -> tuple[int, float, float, int, float]:
    stats = CrawlStats()
    url_log: list[FetchResult] = []
    t0 = time.perf_counter()
    async for _ in crawl(
        [SEED_URL],
        max_pages=max_pages,
        concurrency=concurrency,
        per_host=concurrency,
        per_host_delay=0.02 if concurrency > 1 else 0.05,
        stats=stats,
        url_log=url_log,
        timeout=10.0,
    ):
        pass
    wall = time.perf_counter() - t0
    pps = stats.pages_fetched / max(wall, 1e-6)
    rss = get_peak_rss_mb()
    return stats.pages_fetched, wall, pps, stats.errors, rss


async def _demo_handled_failure() -> None:
    """Моделює реальний збій 429 Too Many Requests із Retry-After та таймаут для логу."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )
    attempt = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempt
        attempt += 1
        if attempt == 1:
            return httpx.Response(
                429, headers={"Retry-After": "0.5"}, text="Rate limit exceeded"
            )
        if attempt == 2:
            raise httpx.ReadTimeout("Simulated socket read timeout")
        return httpx.Response(
            200, text="<html><title>Recovered</title><body>OK</body></html>"
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        res = await fetch(
            client,
            "https://books.toscrape.com/catalogue/page-42.html",
            backoff_base=0.2,
        )
        print(
            f"\nHandled failure summary: status={res.status}, attempts={res.attempts}, "
            f"bytes={res.num_bytes}, elapsed={res.elapsed:.3f}s, url={res.url}"
        )


async def main_async() -> None:
    print(f"Seed: {SEED_URL} | max_pages: 200\n")
    print("| Concurrency | Pages | Wall (s) | Pages/s | Errors | Peak RSS (MB) |")
    print("|---:|---:|---:|---:|---:|---:|")
    for c in (1, 5, 20):
        pages, wall, pps, errors, rss = await _bench_one(c, max_pages=200)
        print(f"| {c} | {pages} | {wall:.2f} | {pps:.2f} | {errors} | {rss:.1f} |")

    await _demo_handled_failure()


if __name__ == "__main__":
    asyncio.run(main_async())
