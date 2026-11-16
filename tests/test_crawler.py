import asyncio
import json
from pathlib import Path

import httpx
from typer.testing import CliRunner

from findex.cli import app
from findex.corpus import iter_documents
from findex.crawler.crawl import canonicalize_url, crawl
from findex.crawler.fetch import FetchResult, fetch
from findex.crawler.robots import RobotsCache
from findex.index import build_index

runner = CliRunner()


def test_canonicalize_url() -> None:
    assert (
        canonicalize_url("HTTP://Example.COM:80/docs/page/#section?b=2&a=1")
        == "http://example.com/docs/page"
    )
    assert (
        canonicalize_url("https://example.com:443/search/?z=9&a=1#top")
        == "https://example.com/search?a=1&z=9"
    )
    assert canonicalize_url("mailto:test@example.com") is None


def test_fetch_retries_429_and_respects_retry_after() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                429, headers={"Retry-After": "0.01"}, text="Too Many Requests"
            )
        return httpx.Response(
            200, text="<html><title>OK</title><body>Hello</body></html>"
        )

    async def _run() -> FetchResult:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await fetch(client, "https://example.test/page", backoff_base=0.01)

    res = asyncio.run(_run())
    assert res.status == 200
    assert res.attempts == 2
    assert calls == 2


def test_fetch_does_not_retry_404() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(404, text="Not Found")

    async def _run() -> FetchResult:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await fetch(
                client, "https://example.test/missing", backoff_base=0.01
            )

    res = asyncio.run(_run())
    assert res.status == 404
    assert res.attempts == 1
    assert calls == 1


def test_robots_cache_disallows_blocked_paths() -> None:
    robots_txt = "User-agent: *\nDisallow: /private/\n"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text=robots_txt)
        return httpx.Response(200, text="ok")

    async def _run() -> tuple[bool, bool]:
        cache = RobotsCache()
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            ok1 = await cache.allowed(client, "https://example.test/public/doc")
            ok2 = await cache.allowed(client, "https://example.test/private/secret")
            return ok1, ok2

    allowed_pub, allowed_priv = asyncio.run(_run())
    assert allowed_pub is True
    assert allowed_priv is False


def test_crawl_generator_dedup_robots_and_max_pages() -> None:
    site = {
        "/robots.txt": "User-agent: *\nDisallow: /secret\n",
        "/": '<a href="/p1#frag">P1</a><a href="/p1/">P1 dup</a><a href="/secret">Sec</a><a href="https://other.test/">Ext</a>',
        "/p1": "<title>Page One</title><p>Async crawler content</p><a href='/p2'>P2</a><a href='/p3'>P3</a>",
        "/p2": "<title>Page Two</title><p>Second page</p>",
        "/p3": "<title>Page Three</title><p>Third page</p>",
        "/secret": "<title>Secret</title><p>Should not be crawled</p>",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        body = site.get(request.url.path)
        if body is None:
            return httpx.Response(404, text="Not found")
        return httpx.Response(200, text=body)

    async def _run() -> list[str]:
        transport = httpx.MockTransport(handler)
        pages: list[str] = []
        async for p in crawl(
            ["https://docs.test/"],
            max_pages=3,
            concurrency=2,
            per_host=2,
            per_host_delay=0.0,
            transport=transport,
        ):
            pages.append(p.url)
        return pages

    urls = asyncio.run(_run())
    assert len(urls) == 3
    assert len(set(urls)) == 3
    assert "https://docs.test/secret" not in urls


def test_jsonl_streaming_compatible_with_iter_documents_and_index(
    tmp_path: Path,
) -> None:
    jsonl_file = tmp_path / "crawl.jsonl"
    lines = [
        {
            "url": "https://docs.test/1",
            "title": "Asyncio",
            "text": "Event loop coroutines",
            "fetched_at": "2026-11-16T00:00:00Z",
        },
        {
            "url": "https://docs.test/2",
            "title": "Httpx",
            "text": "Connection pooling transport",
            "fetched_at": "2026-11-16T00:00:01Z",
        },
    ]
    jsonl_file.write_text(
        "\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8"
    )

    docs = list(iter_documents(jsonl_file))
    assert len(docs) == 2
    idx = build_index(docs)
    assert idx.num_docs == 2
    assert idx.df("coroutines") == 1


def test_cli_crawl_and_index_pipeline(tmp_path: Path) -> None:
    jsonl_out = tmp_path / "out.jsonl"
    log_out = tmp_path / "crawl.log"
    idx_out = tmp_path / "crawl.bin"

    # Перевіряємо через мок або порожній/локальний потік індексацію
    jsonl_out.write_text(
        json.dumps(
            {
                "url": "https://docs.test/home",
                "title": "Home",
                "text": "Python asyncio taskgroup semaphore",
                "fetched_at": "2026-11-16T00:00:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    log_out.write_text(
        "200\t128B\t0.010s\tattempts=1\thttps://docs.test/home\n", encoding="utf-8"
    )

    res_idx = runner.invoke(
        app, ["index", str(jsonl_out), "--out", str(idx_out), "--executor", "serial"]
    )
    assert res_idx.exit_code == 0
    res_search = runner.invoke(app, ["search", str(idx_out), "taskgroup"])
    assert res_search.exit_code == 0
    assert "Результати" in res_search.stdout
