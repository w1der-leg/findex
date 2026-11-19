import logging
import math
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape

from findex.index import Index
from findex.query import QuerySyntaxError
from findex.search import BM25, Scorer, TfIdf, load_index, ranked_search
from findex.semantic import (
    SearchMode,
    SemanticIndex,
    hybrid_search,
    load_semantic_index,
)
from findex.tokenize import tokenize
from findex.web.config import Settings
from findex.web.schemas import (
    ApiSearchResult,
    DocumentResponse,
    HealthResponse,
    SearchResponse,
    StatsResponse,
)

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))


def get_settings(request: Request) -> Settings:
    settings: Settings | None = getattr(request.app.state, "settings", None)
    if settings is None:
        settings = Settings()
        request.app.state.settings = settings
    return settings


def get_index(request: Request) -> Index:
    idx: Index | None = getattr(request.app.state, "index", None)
    if idx is None:
        raise HTTPException(status_code=503, detail="Search index is not loaded")
    return idx


def _extract_title_and_snippet(
    idx: Index, doc_id: int, raw_snippet: str
) -> tuple[str, str]:
    meta = idx.doc_meta.get(doc_id)
    url = meta.path if meta else f"doc://{doc_id}"
    parts = [p for p in url.rstrip("/").split("/") if p and p.lower() != "index.html"]
    slug = parts[-1] if parts else url
    title = slug.replace("-", " ").replace("_", " ").title() or url
    snippet = (
        raw_snippet.strip() if raw_snippet.strip() else f"Document #{doc_id} ({url})"
    )
    return title, snippet


def _highlight_text(text: str, query: str) -> Markup:
    terms = set(tokenize(query))
    escaped = str(escape(text))
    if not terms:
        return Markup(escaped)
    words = escaped.split(" ")
    highlighted: list[str] = []
    for w in words:
        clean = "".join(ch.lower() for ch in w if ch.isalnum())
        if clean in terms:
            highlighted.append(f"<mark>{w}</mark>")
        else:
            highlighted.append(w)
    return Markup(" ".join(highlighted))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    settings = Settings()
    app.state.settings = settings
    app.state.started_at = time.perf_counter()
    app.state.index = None
    app.state.semantic_index = None

    if settings.index_path.exists():
        log.info("Loading search index from %s", settings.index_path)
        idx = load_index(settings.index_path)
        _ = idx.numpy_postings
        _ = idx.doc_len_array
        app.state.index = idx
        app.state.semantic_index = load_semantic_index(settings.index_path)
        log.info(
            "Loaded index: %r (semantic=%s)",
            app.state.index,
            app.state.semantic_index is not None,
        )
    else:
        log.warning(
            "Index path %s does not exist; /health will return 503", settings.index_path
        )

    try:
        yield
    finally:
        idx_close: Index | None = getattr(app.state, "index", None)
        if idx_close is not None:
            idx_close.close()
            log.info("Closed search index on shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="findex Search Engine",
        version="1.0.0",
        description="Повнотекстовий та гібридний пошуковий рушій на FastAPI (BM25 / TF-IDF / Semantic RRF)",
        lifespan=lifespan,
    )
    app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")

    @app.middleware("http")
    async def request_id_and_error_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = req_id
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = req_id
            return response
        except Exception:
            log.exception(
                "Unhandled error on %s %s [request_id=%s]",
                request.method,
                request.url.path,
                req_id,
            )
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal Server Error", "request_id": req_id},
                headers={"X-Request-ID": req_id},
            )

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> JSONResponse:
        idx: Index | None = getattr(request.app.state, "index", None)
        sem_idx: SemanticIndex | None = getattr(
            request.app.state, "semantic_index", None
        )
        if idx is None:
            payload = HealthResponse(
                status="unavailable",
                index_loaded=False,
                semantic_loaded=False,
                num_docs=0,
            )
            return JSONResponse(status_code=503, content=payload.model_dump())
        payload = HealthResponse(
            status="ok",
            index_loaded=True,
            semantic_loaded=sem_idx is not None,
            num_docs=idx.num_docs,
        )
        return JSONResponse(status_code=200, content=payload.model_dump())

    @app.get("/stats", response_model=StatsResponse)
    def stats(
        request: Request,
        idx: Annotated[Index, Depends(get_index)],
        settings: Annotated[Settings, Depends(get_settings)],
    ) -> StatsResponse:
        started_at: float = getattr(
            request.app.state, "started_at", time.perf_counter()
        )
        sem_idx: SemanticIndex | None = getattr(
            request.app.state, "semantic_index", None
        )
        return StatsResponse(
            index_path=str(settings.index_path),
            num_docs=idx.num_docs,
            num_terms=len(idx),
            avg_doc_length=round(idx.avg_doc_length, 2),
            semantic_chunks=sem_idx.num_chunks if sem_idx is not None else 0,
            uptime_sec=round(time.perf_counter() - started_at, 2),
        )

    def _execute_search(
        idx: Index,
        sem_idx: SemanticIndex | None,
        q: str,
        k: int,
        scorer: Literal["bm25", "tfidf"],
        mode: SearchMode,
        page: int,
    ) -> SearchResponse:
        t0 = time.perf_counter()
        active_scorer: Scorer = BM25() if scorer == "bm25" else TfIdf()
        effective_mode: SearchMode = (
            mode if (mode == "keyword" or sem_idx is not None) else "keyword"
        )
        try:
            top_limit = max(idx.num_docs, k * page)
            if effective_mode == "keyword":
                all_hits = ranked_search(idx, q, scorer=active_scorer, k=top_limit)
            else:
                all_hits = hybrid_search(
                    idx,
                    sem_idx,
                    q,
                    mode=effective_mode,
                    scorer=active_scorer,
                    k=top_limit,
                )
        except QuerySyntaxError as exc:
            raise HTTPException(
                status_code=422, detail=f"Invalid query syntax: {exc}"
            ) from None

        total = len(all_hits)
        total_pages = max(1, math.ceil(total / k)) if total > 0 else 1
        start = (page - 1) * k
        slice_hits = all_hits[start : start + k]

        api_results: list[ApiSearchResult] = []
        for hit in slice_hits:
            meta = idx.doc_meta.get(hit.doc_id)
            url = meta.path if meta else f"doc://{hit.doc_id}"
            title, snippet = _extract_title_and_snippet(idx, hit.doc_id, hit.snippet)
            api_results.append(
                ApiSearchResult(
                    doc_id=hit.doc_id,
                    title=title,
                    url=url,
                    score=round(hit.score, 4),
                    snippet=snippet,
                )
            )

        took_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return SearchResponse(
            query=q,
            scorer=scorer,
            mode=effective_mode,
            page=page,
            page_size=k,
            total=total,
            total_pages=total_pages,
            took_ms=took_ms,
            results=api_results,
        )

    @app.get("/search", response_model=SearchResponse)
    def search_api(
        request: Request,
        q: Annotated[
            str, Query(min_length=1, max_length=200, description="Пошуковий запит")
        ],
        k: Annotated[
            int, Query(ge=1, le=100, description="Результатів на сторінці")
        ] = 10,
        scorer: Annotated[
            Literal["bm25", "tfidf"], Query(description="Алгоритм ранжування")
        ] = "bm25",
        mode: Annotated[
            SearchMode, Query(description="Режим пошуку: keyword, semantic або hybrid")
        ] = "keyword",
        page: Annotated[int, Query(ge=1, le=1000, description="Номер сторінки")] = 1,
        idx: Annotated[Index, Depends(get_index)] = None,  # type: ignore[assignment]
    ) -> SearchResponse:
        if not q.strip():
            raise HTTPException(status_code=422, detail="Query must not be blank")
        sem_idx: SemanticIndex | None = getattr(
            request.app.state, "semantic_index", None
        )
        return _execute_search(idx, sem_idx, q, k, scorer, mode, page)

    @app.get("/docs/{doc_id}", response_model=DocumentResponse)
    def get_document_api(
        doc_id: int,
        idx: Annotated[Index, Depends(get_index)],
    ) -> DocumentResponse:
        if doc_id not in idx.doc_meta:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        meta = idx.doc_meta[doc_id]
        title, snippet = _extract_title_and_snippet(idx, doc_id, "")
        return DocumentResponse(
            doc_id=doc_id,
            title=title,
            url=meta.path,
            length=idx.doc_length(doc_id),
            snippet=snippet,
        )

    @app.get("/", response_class=HTMLResponse)
    def home_ui(
        request: Request,
        q: Annotated[str, Query(max_length=200)] = "",
        k: Annotated[int, Query(ge=1, le=50)] = 10,
        scorer: Annotated[Literal["bm25", "tfidf"], Query()] = "bm25",
        mode: Annotated[SearchMode, Query()] = "keyword",
        page: Annotated[int, Query(ge=1, le=1000)] = 1,
    ) -> HTMLResponse:
        override_fn = app.dependency_overrides.get(get_index)
        idx: Index | None = (
            override_fn()
            if callable(override_fn)
            else getattr(request.app.state, "index", None)
        )
        sem_idx: SemanticIndex | None = getattr(
            request.app.state, "semantic_index", None
        )
        search_data: SearchResponse | None = None
        highlighted_items: list[dict[str, object]] = []
        error_msg: str | None = None

        if idx is None:
            error_msg = "Пошуковий індекс ще не завантажений (перевірте INDEX_PATH)."
        elif q.strip():
            try:
                search_data = _execute_search(idx, sem_idx, q, k, scorer, mode, page)
                for item in search_data.results:
                    highlighted_items.append(
                        {
                            "doc_id": item.doc_id,
                            "title": item.title,
                            "url": item.url,
                            "score": item.score,
                            "snippet_html": _highlight_text(item.snippet, q),
                        }
                    )
            except HTTPException as exc:
                error_msg = str(exc.detail)

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "q": q,
                "k": k,
                "scorer": scorer,
                "mode": mode,
                "semantic_available": sem_idx is not None,
                "page": page,
                "search_data": search_data,
                "items": highlighted_items,
                "error_msg": error_msg,
                "num_docs": idx.num_docs if idx is not None else 0,
                "num_terms": len(idx) if idx is not None else 0,
            },
        )

    @app.get("/doc/{doc_id}", response_class=HTMLResponse)
    def document_ui(
        request: Request,
        doc_id: int,
        idx: Annotated[Index, Depends(get_index)],
    ) -> HTMLResponse:
        if doc_id not in idx.doc_meta:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        meta = idx.doc_meta[doc_id]
        title, _ = _extract_title_and_snippet(idx, doc_id, "")
        terms_in_doc = sorted(
            (term, next(p.tf for p in plist if p.doc_id == doc_id))
            for term, plist in idx.postings.items()
            if any(p.doc_id == doc_id for p in plist)
        )
        terms_in_doc.sort(key=lambda x: x[1], reverse=True)

        return templates.TemplateResponse(
            request=request,
            name="doc.html",
            context={
                "doc_id": doc_id,
                "title": title,
                "url": meta.path,
                "length": idx.doc_length(doc_id),
                "top_terms": terms_in_doc[:40],
            },
        )

    return app


app = create_app()
