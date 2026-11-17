# findex (`v0.7.0`)

**Live Demo (Public URL):** https://ingenious-encouragement-production-87cc2290.up.railway.app  
**Interactive OpenAPI Docs:** https://ingenious-encouragement-production-87cc2290.up.railway.app/docs

Повнотекстовий локальний пошуковий рушій, асинхронний веб-краулер і веб-сервіс на Python (`FastAPI`, `Pydantic`, `Uvicorn`, `Jinja2`).

---

## Можливості

- **CLI (`findex`)**:
  - `findex crawl <url>` — ввічливий асинхронний краулер (`asyncio` + `httpx`, `robots.txt`, rate limiting на хост, потоковий запис у JSONL).
  - `findex index <corpus>` — паралельна індексація через `ProcessPoolExecutor` / `ThreadPoolExecutor` із позиціями слів (`with_positions=True`).
  - `findex search <index.bin> "<query>"` — булеві запити (`AND`, `OR`, `NOT`), фразовий пошук (`"..."`), ранжування `BM25` та `TF-IDF`.
  - `findex serve --port 8000 --workers 1` — запуск HTTP API та веб-інтерфейсу на `Uvicorn`.
- **Веб-сервіс (`src/findex/web/`)**:
  - `GET /` — пошуковий інтерфейс на `Jinja2` з підсвіткою знайдених термів, вибором алгоритму (`BM25` / `TF-IDF`), пагінацією та переглядом документів (`GET /doc/{doc_id}`).
  - `GET /search?q=&k=&scorer=&page=` — валідований JSON API (моделі `Pydantic` лише на межі входу/виходу; всередині залишаються `dataclass`-моделі).
  - `GET /docs/{doc_id}`, `GET /stats`, `GET /health` (`503`, якщо індекс не завантажений).
  - Завантаження індексу один раз під час старту через `lifespan`, ін'єкція залежностей через `Depends`, конфігурація через змінні оточення (`pydantic-settings`).

---

## Швидкий старт

### Локально через `uv`
```bash
uv sync
uv run findex serve --index crawl_index.bin --port 8000
```

### У контейнері Docker (multi-stage build, non-root `appuser`)
```bash
docker build -t findex:0.7.0 .
docker run -d -p 8000:8000 -e INDEX_PATH=/app/data/crawl_index.bin findex:0.7.0
```

---

## Навантажувальне тестування та вибір `def` проти `async def` (Lab 07)

### Порівняння 20 одночасних запитів на `/search` (1 воркер)
- **`async def /search`**: `124.10 ms` (CPU-bound ранжування виконується прямо в циклі подій `asyncio` і блокує прийом інших з'єднань).
- **`def /search`**: **`108.41 ms`** (FastAPI автоматично виносить синхронний обробник у пул потоків `anyio.to_thread`, залишаючи event loop вільним).

Тому для основного маршруту `GET /search` у коді залишено **`def`**.

### Результати навантажувального тестування (`RPS`, `p50`, `p95`, `p99`)

| Target | Route | Concurrency | RPS | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---|---:|---:|---:|---:|---:|
| local, 1 worker, async def | /search | 50 | 131.8 | 266.40 | 947.04 | 1404.17 |
| local, 1 worker, def | /search | 50 | 121.5 | 266.89 | 1078.22 | 1453.00 |
| local, 4 workers | /search | 50 | 88.4 | 397.29 | 1385.26 | 2394.25 |
| deployed | /search | 20 | 76.8 | 205.64 | 408.04 | 424.99 |

**Зв'язок із теорією конкурентності (Labs 5–6):**
1. **`1 worker, async def` vs `1 worker, def` (пул потоків і GIL):** ранжування BM25/TF-IDF та побудова сніпетів — чистий CPU-bound код на Python, який утримує GIL (Lab 5). При 50 одночасних клієнтах у `def`-режимі потоки пулу змагаються за GIL і перемикання контексту, через що `p95`/`p99` близькі до `async def`, проте при помірній конкурентності (20 запитів) винесення в пул потоків не блокує цикл подій Uvicorn для легких маршрутів (`/health`, `/stats`).
2. **`4 workers` (кілька процесів):** `--workers 4` піднімає 4 незалежні процеси Uvicorn зі власним GIL і власною копією індексу в пам'яті. На локальній машині, де одночасно працює генератор навантаження (`50` клієнтів) і 4 процеси воркерів на спільному CPU, конкуренція за ядра та перемикання процесів піднімають хвіст затримок (`p99 = 2394.25 ms`).
3. **`deployed` (хмарний контейнер на Railway):** затримки `p50 = 205.64 ms` і `p99 = 424.99 ms` при `concurrency = 20` включають мережевий RTT через TLS до дата-центру US West, але завдяки невеликому індексу та відсутності локальної боротьби з клієнтом тестування хвіст `p99` залишається стабільним (`~2×` від `p50`).

---

## Тести та якість коду

```bash
uv run ruff check .
uv run pyright
uv run pytest
```
