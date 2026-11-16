import asyncio
import email.utils
import logging
import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx

log = logging.getLogger(__name__)

USER_AGENT = "findex-bot/0.6 (+https://github.com/w1der-leg/findex; contact: stepan4enco2007@gmail.com)"


@dataclass(frozen=True, slots=True)
class FetchResult:
    url: str
    final_url: str
    status: int
    text: str
    num_bytes: int
    elapsed: float
    attempts: int
    error: str | None = None


def _parse_retry_after(header_value: str | None) -> float | None:
    if not header_value:
        return None
    stripped = header_value.strip()
    try:
        seconds = float(stripped)
        return max(0.0, seconds)
    except ValueError:
        pass
    try:
        dt = email.utils.parsedate_to_datetime(stripped)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        delta = (dt - datetime.now(UTC)).total_seconds()
        return max(0.0, delta)
    except (TypeError, ValueError):
        return None


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout: float = 10.0,
    max_retries: int = 3,
    backoff_base: float = 0.5,
    backoff_cap: float = 10.0,
) -> FetchResult:
    """Асинхронно завантажує URL із таймаутом та backoff + jitter для 429 / 5xx / таймаутів."""
    t_start = time.perf_counter()
    last_status = 0
    last_error: str | None = None
    final_url = url

    for attempt in range(max_retries + 1):
        try:
            async with asyncio.timeout(timeout):
                resp = await client.get(url, follow_redirects=True)
            final_url = str(resp.url)
            last_status = resp.status_code
            raw_bytes = resp.content
            elapsed = time.perf_counter() - t_start

            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                last_error = f"HTTP {resp.status_code}"
                if attempt < max_retries:
                    retry_after = (
                        _parse_retry_after(resp.headers.get("Retry-After"))
                        if resp.status_code == 429
                        else None
                    )
                    delay = (
                        retry_after
                        if retry_after is not None
                        else min(backoff_cap, backoff_base * (2**attempt))
                        + random.uniform(0.0, 0.5)
                    )
                    log.warning(
                        "Retry %d/%d for %s after %s (sleep %.2fs)",
                        attempt + 1,
                        max_retries,
                        url,
                        last_error,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue

            return FetchResult(
                url=url,
                final_url=final_url,
                status=resp.status_code,
                text=resp.text if resp.status_code < 400 else "",
                num_bytes=len(raw_bytes),
                elapsed=elapsed,
                attempts=attempt + 1,
                error=None if resp.status_code < 400 else f"HTTP {resp.status_code}",
            )

        except (TimeoutError, httpx.TimeoutException):
            last_error = "TimeoutError"
            if attempt < max_retries:
                delay = min(backoff_cap, backoff_base * (2**attempt)) + random.uniform(
                    0.0, 0.5
                )
                log.warning(
                    "Timeout retry %d/%d for %s (sleep %.2fs)",
                    attempt + 1,
                    max_retries,
                    url,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
        except httpx.TooManyRedirects:
            elapsed = time.perf_counter() - t_start
            log.error("Redirect loop for %s", url)
            return FetchResult(
                url=url,
                final_url=final_url,
                status=310,
                text="",
                num_bytes=0,
                elapsed=elapsed,
                attempts=attempt + 1,
                error="TooManyRedirects",
            )
        except httpx.HTTPError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < max_retries:
                delay = min(backoff_cap, backoff_base * (2**attempt)) + random.uniform(
                    0.0, 0.5
                )
                log.warning(
                    "HTTP error retry %d/%d for %s: %s (sleep %.2fs)",
                    attempt + 1,
                    max_retries,
                    url,
                    last_error,
                    delay,
                )
                await asyncio.sleep(delay)
                continue

    elapsed = time.perf_counter() - t_start
    return FetchResult(
        url=url,
        final_url=final_url,
        status=last_status,
        text="",
        num_bytes=0,
        elapsed=elapsed,
        attempts=max_retries + 1,
        error=last_error or "UnknownError",
    )
