from typing import Literal

from pydantic import BaseModel, Field


class ApiSearchResult(BaseModel):
    doc_id: int
    title: str
    url: str
    score: float
    snippet: str


class SearchResponse(BaseModel):
    query: str
    scorer: Literal["bm25", "tfidf"]
    page: int
    page_size: int
    total: int
    total_pages: int
    took_ms: float
    results: list[ApiSearchResult]


class DocumentResponse(BaseModel):
    doc_id: int
    title: str
    url: str
    length: int
    snippet: str


class StatsResponse(BaseModel):
    index_path: str
    num_docs: int
    num_terms: int
    avg_doc_length: float
    uptime_sec: float


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    index_loaded: bool
    num_docs: int = Field(ge=0)
