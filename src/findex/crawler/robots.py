import asyncio
import logging
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from findex.crawler.fetch import USER_AGENT

log = logging.getLogger(__name__)


class RobotsCache:
    """Кешує правила robots.txt на рівні кожного хоста (scheme://netloc)."""

    def __init__(self, user_agent: str = USER_AGENT, timeout: float = 5.0) -> None:
        self.user_agent = user_agent
        self.timeout = timeout
        self._parsers: dict[str, RobotFileParser] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _host_key(self, url: str) -> str:
        parts = urlsplit(url)
        return f"{parts.scheme.lower()}://{parts.netloc.lower()}"

    async def allowed(self, client: httpx.AsyncClient, url: str) -> bool:
        host_key = self._host_key(url)
        if host_key not in self._parsers:
            lock = self._locks.setdefault(host_key, asyncio.Lock())
            async with lock:
                if host_key not in self._parsers:
                    self._parsers[host_key] = await self._fetch_robots(client, host_key)
        return self._parsers[host_key].can_fetch(self.user_agent, url)

    async def _fetch_robots(
        self, client: httpx.AsyncClient, host_key: str
    ) -> RobotFileParser:
        robots_url = f"{host_key}/robots.txt"
        rp = RobotFileParser()
        rp.set_url(robots_url)
        try:
            async with asyncio.timeout(self.timeout):
                resp = await client.get(robots_url, follow_redirects=True)
            if resp.status_code == 200:
                rp.parse(resp.text.splitlines())
                log.debug("Loaded robots.txt for %s", host_key)
            elif resp.status_code in (401, 403):
                rp.disallow_all = True  # type: ignore[attr-defined]
                log.warning(
                    "robots.txt returned %d for %s -> disallow all",
                    resp.status_code,
                    host_key,
                )
            else:
                rp.allow_all = True  # type: ignore[attr-defined]
                log.debug(
                    "No robots.txt (%d) for %s -> allow all", resp.status_code, host_key
                )
        except (TimeoutError, httpx.HTTPError) as exc:
            rp.allow_all = True  # type: ignore[attr-defined]
            log.warning("Failed to fetch %s (%s) -> allow all", robots_url, exc)
        return rp
