import asyncio
import logging
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html.parser import HTMLParser
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import httpx

from findex.crawler.fetch import USER_AGENT, FetchResult, fetch
from findex.crawler.robots import RobotsCache

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Page:
    url: str
    title: str
    text: str
    fetched_at: str


@dataclass(slots=True)
class CrawlStats:
    pages_fetched: int = 0
    in_flight: int = 0
    errors: int = 0
    queue_depth: int = 0
    started_at: float = field(default_factory=time.perf_counter)

    @property
    def pages_per_sec(self) -> float:
        elapsed = max(time.perf_counter() - self.started_at, 1e-6)
        return self.pages_fetched / elapsed


class _LinkAndTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []
        self._title_parts: list[str] = []
        self._text_parts: list[str] = []
        self._in_title = False
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        if t in {"script", "style", "noscript"}:
            self._skip_depth += 1
        elif t == "title":
            self._in_title = True
        elif t == "a":
            for k, v in attrs:
                if k.lower() == "href" and v:
                    self.links.append(v.strip())

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in {"script", "style", "noscript"} and self._skip_depth > 0:
            self._skip_depth -= 1
        elif t == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        cleaned = " ".join(data.split())
        if not cleaned:
            return
        if self._in_title:
            self._title_parts.append(cleaned)
        self._text_parts.append(cleaned)

    @property
    def title(self) -> str:
        return " ".join(self._title_parts).strip()

    @property
    def text(self) -> str:
        return " ".join(self._text_parts).strip()


def canonicalize_url(raw_url: str, base_url: str | None = None) -> str | None:
    """Канонізує URL: нижній регістр схеми/хоста, без #фрагмента, нормалізований слеш і query."""
    joined = urljoin(base_url, raw_url) if base_url else raw_url
    parts = urlsplit(joined)
    scheme = parts.scheme.lower()
    netloc = parts.netloc.lower()
    if scheme not in {"http", "https"} or not netloc:
        return None

    if netloc.endswith(":80") and scheme == "http":
        netloc = netloc[:-3]
    elif netloc.endswith(":443") and scheme == "https":
        netloc = netloc[:-4]

    path = parts.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    sorted_query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
    return urlunsplit((scheme, netloc, path, sorted_query, ""))


def extract_page_and_links(url: str, html: str) -> tuple[Page, list[str]]:
    parser = _LinkAndTextParser()
    parser.feed(html)
    links: list[str] = []
    for href in parser.links:
        canon = canonicalize_url(href, base_url=url)
        if canon is not None:
            links.append(canon)
    page = Page(
        url=url,
        title=parser.title or url,
        text=parser.text,
        fetched_at=datetime.now(UTC).isoformat(),
    )
    return page, links


async def crawl(
    seeds: Sequence[str],
    *,
    max_pages: int = 200,
    concurrency: int = 10,
    per_host: int = 2,
    per_host_delay: float = 0.25,
    allowed_domains: set[str] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    stats: CrawlStats | None = None,
    url_log: list[FetchResult] | None = None,
    timeout: float = 10.0,
) -> AsyncIterator[Page]:
    """Асинхронний генератор сторінок із чергою, TaskGroup, семафорами та robots.txt."""
    if max_pages <= 0:
        return

    active_stats = stats if stats is not None else CrawlStats()
    queue: asyncio.Queue[str] = asyncio.Queue()
    out_queue: asyncio.Queue[Page | None] = asyncio.Queue()
    seen: set[str] = set()
    domains: set[str] = set()

    if allowed_domains is not None:
        domains = {d.lower() for d in allowed_domains}

    for seed in seeds:
        canon = canonicalize_url(seed)
        if canon is not None and canon not in seen:
            seen.add(canon)
            queue.put_nowait(canon)
            if allowed_domains is None:
                domains.add(urlsplit(canon).netloc.lower())

    active_stats.queue_depth = queue.qsize()
    if queue.empty():
        return

    global_sem = asyncio.Semaphore(max(1, concurrency))
    host_sems: dict[str, asyncio.Semaphore] = {}
    host_next_allowed: dict[str, float] = {}
    host_rate_locks: dict[str, asyncio.Lock] = {}
    robots = RobotsCache(user_agent=USER_AGENT, timeout=timeout)
    stop_event = asyncio.Event()
    claimed_pages = 0

    async def _respect_host_delay(host: str) -> None:
        if per_host_delay <= 0:
            return
        lock = host_rate_locks.setdefault(host, asyncio.Lock())
        async with lock:
            now = time.perf_counter()
            ready_at = host_next_allowed.get(host, 0.0)
            wait_for = ready_at - now
            host_next_allowed[host] = max(now, ready_at) + per_host_delay
        if wait_for > 0:
            await asyncio.sleep(wait_for)

    async def _worker(client: httpx.AsyncClient) -> None:
        nonlocal claimed_pages
        while not stop_event.is_set():
            try:
                async with asyncio.timeout(0.1):
                    url = await queue.get()
            except TimeoutError:
                if queue.empty() and active_stats.in_flight == 0:
                    return
                continue

            active_stats.in_flight += 1
            active_stats.queue_depth = queue.qsize()
            try:
                if stop_event.is_set():
                    continue

                host = urlsplit(url).netloc.lower()
                if domains and host not in domains:
                    continue

                if not await robots.allowed(client, url):
                    log.info("Disallowed by robots.txt: %s", url)
                    if url_log is not None:
                        url_log.append(
                            FetchResult(
                                url=url,
                                final_url=url,
                                status=0,
                                text="",
                                num_bytes=0,
                                elapsed=0.0,
                                attempts=0,
                                error="Disallowed by robots.txt",
                            )
                        )
                    continue

                h_sem = host_sems.setdefault(host, asyncio.Semaphore(max(1, per_host)))
                async with global_sem, h_sem:
                    if stop_event.is_set():
                        continue
                    await _respect_host_delay(host)
                    if stop_event.is_set():
                        continue
                    res = await fetch(client, url, timeout=timeout)

                if url_log is not None:
                    url_log.append(res)

                if res.status >= 400 or res.status == 0 or res.error is not None:
                    active_stats.errors += 1
                    continue

                final_canon = canonicalize_url(res.final_url)
                if final_canon:
                    seen.add(final_canon)

                if claimed_pages >= max_pages:
                    stop_event.set()
                    continue

                claimed_pages += 1
                page, links = extract_page_and_links(res.final_url, res.text)
                active_stats.pages_fetched = claimed_pages
                await out_queue.put(page)

                if claimed_pages >= max_pages:
                    stop_event.set()
                    continue

                for link in links:
                    if link not in seen:
                        l_host = urlsplit(link).netloc.lower()
                        if not domains or l_host in domains:
                            seen.add(link)
                            queue.put_nowait(link)
                active_stats.queue_depth = queue.qsize()
            finally:
                active_stats.in_flight -= 1
                queue.task_done()

    async def _run_pool() -> None:
        headers = {"User-Agent": USER_AGENT}
        async with httpx.AsyncClient(headers=headers, transport=transport) as client:
            try:
                async with asyncio.TaskGroup() as tg:
                    workers = [
                        tg.create_task(_worker(client))
                        for _ in range(max(1, concurrency))
                    ]
                    while not stop_event.is_set():
                        if queue.empty() and active_stats.in_flight == 0:
                            break
                        await asyncio.sleep(0.05)
                    stop_event.set()
                    for w in workers:
                        w.cancel()
            except* asyncio.CancelledError:
                pass
            finally:
                await out_queue.put(None)

    pool_task = asyncio.create_task(_run_pool())
    try:
        while True:
            item = await out_queue.get()
            if item is None:
                break
            yield item
    finally:
        stop_event.set()
        if not pool_task.done():
            pool_task.cancel()
            try:
                await pool_task
            except asyncio.CancelledError:
                pass
