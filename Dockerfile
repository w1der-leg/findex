# Stage 1: Builder
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev

COPY src ./src
RUN uv sync --frozen --no-dev

# Stage 2: Runtime (slim, non-root)
FROM python:3.13-slim-bookworm AS runtime

RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --create-home appuser

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appuser /app/src /app/src

RUN mkdir -p /app/data && chown -R appuser:appuser /app/data

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    INDEX_PATH="/app/data/crawl_index.bin" \
    PORT="8000"

USER appuser

# Генеруємо корпус і будуємо індекс прямо під час збірки образу
RUN printf "Python asyncio FastAPI web search engine with BM25 and TF-IDF ranking.\n" > /app/data/doc1.txt && \
    printf "Mystery detective novel classic story about books and investigations.\n" > /app/data/doc2.txt && \
    printf "Docker containers deployment healthcheck and performance benchmarks p50 p95 p99.\n" > /app/data/doc3.txt && \
    findex index /app/data --out /app/data/crawl_index.bin --executor serial

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["sh", "-c", "findex serve --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]