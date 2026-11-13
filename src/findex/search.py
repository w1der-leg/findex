import functools
import heapq
import logging
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from findex.index import Index, InvertedIndex, Posting
from findex.query import QueryNode, parse_query
from findex.store import load_index_json, load_index_pickle
from findex.tokenize import tokenize

log = logging.getLogger(__name__)

EngineType = Literal["merge", "set"]
ScorerType = Literal["bm25", "tfidf"]


def timed[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            log.debug("%s took %.2f ms", fn.__name__, elapsed_ms)

    return wrapper


@dataclass(frozen=True, order=True, slots=True)
class SearchResult:
    score: float
    doc_id: int
    title: str = ""
    snippet: str = ""


class Scorer(Protocol):
    def score(self, index: Index, term: str, posting: Posting) -> float: ...


@dataclass(frozen=True, slots=True)
class TfIdf:
    def score(self, index: Index, term: str, posting: Posting) -> float:
        df = index.df(term)
        if df == 0 or index.num_docs == 0:
            return 0.0
        idf = math.log((index.num_docs + 1) / (df + 1)) + 1.0
        return float(posting.tf) * idf


@dataclass(frozen=True, slots=True)
class BM25:
    k1: float = 1.5
    b: float = 0.75

    def score(self, index: Index, term: str, posting: Posting) -> float:
        df = index.df(term)
        n = index.num_docs
        if df == 0 or n == 0:
            return 0.0
        idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
        doc_len = index.doc_length(posting.doc_id)
        avg_dl = index.avg_doc_length or 1.0
        denom = posting.tf + self.k1 * (1.0 - self.b + self.b * (doc_len / avg_dl))
        return idf * (posting.tf * (self.k1 + 1.0)) / denom


@functools.lru_cache(maxsize=256)
def _cached_parse(query: str) -> QueryNode:
    log.debug("Cache miss for query parse: %r", query)
    return parse_query(query)


def extract_snippet(text: str, terms: set[str], window: int = 80) -> str:
    if not text:
        return ""
    lower_text = text.casefold()
    best_pos = -1
    for term in terms:
        pos = lower_text.find(term.casefold())
        if pos != -1 and (best_pos == -1 or pos < best_pos):
            best_pos = pos

    if best_pos == -1:
        snippet = text[: window * 2].replace("\n", " ").strip()
        return snippet

    start = max(0, best_pos - window)
    end = min(len(text), best_pos + window)
    snippet = text[start:end].replace("\n", " ").strip()

    for term in sorted(terms, key=len, reverse=True):
        if term:
            pattern = re.compile(rf"\b({re.escape(term)})\b", re.IGNORECASE)
            snippet = pattern.sub(r"[bold yellow]\1[/bold yellow]", snippet)

    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{snippet}{suffix}"


def _build_doc_snippet(index: Index, doc_id: int, terms: set[str]) -> str:
    meta = index.doc_meta.get(doc_id)
    if meta is None:
        return ""
    path = Path(meta.path)
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return extract_snippet(text, terms)


def intersect(posting_a: list[Posting], posting_b: list[Posting]) -> list[int]:
    i = 0
    j = 0
    result: list[int] = []

    while i < len(posting_a) and j < len(posting_b):
        doc_a = posting_a[i].doc_id
        doc_b = posting_b[j].doc_id

        if doc_a == doc_b:
            result.append(doc_a)
            i += 1
            j += 1
        elif doc_a < doc_b:
            i += 1
        else:
            j += 1
    return result


def union(posting_a: list[Posting], posting_b: list[Posting]) -> list[int]:
    i = 0
    j = 0
    result: list[int] = []

    while i < len(posting_a) and j < len(posting_b):
        doc_a = posting_a[i].doc_id
        doc_b = posting_b[j].doc_id

        if doc_a == doc_b:
            result.append(doc_a)
            i += 1
            j += 1
        elif doc_a < doc_b:
            result.append(doc_a)
            i += 1
        else:
            result.append(doc_b)
            j += 1

    while i < len(posting_a):
        result.append(posting_a[i].doc_id)
        i += 1

    while j < len(posting_b):
        result.append(posting_b[j].doc_id)
        j += 1

    return result


def difference(posting_a: list[Posting], posting_b: list[Posting]) -> list[int]:
    i = 0
    j = 0
    result: list[int] = []

    while i < len(posting_a) and j < len(posting_b):
        doc_a = posting_a[i].doc_id
        doc_b = posting_b[j].doc_id
        if doc_a == doc_b:
            i += 1
            j += 1
        elif doc_a < doc_b:
            result.append(doc_a)
            i += 1
        else:
            j += 1

    while i < len(posting_a):
        result.append(posting_a[i].doc_id)
        i += 1
    return result


def intersect_sets(posting_a: list[Posting], posting_b: list[Posting]) -> list[int]:
    set_a = {p.doc_id for p in posting_a}
    set_b = {p.doc_id for p in posting_b}
    return sorted(set_a & set_b)


def search_and(
    index: InvertedIndex, query: str, engine: EngineType = "merge"
) -> list[int]:
    tokens = list(tokenize(query))
    if not tokens:
        return []
    lists_to_merge: list[list[Posting]] = []

    for word in tokens:
        postings_list = index.postings.get(word)
        if postings_list is None:
            return []
        lists_to_merge.append(postings_list)

    current_postings = lists_to_merge[0]
    result_ids = [p.doc_id for p in current_postings]

    for next_posting in lists_to_merge[1:]:
        if engine == "merge":
            result_ids = intersect(current_postings, next_posting)
        else:
            result_ids = intersect_sets(current_postings, next_posting)

        if not result_ids:
            return []
        current_postings = [Posting(doc_id=d_id, tf=0) for d_id in result_ids]

    return result_ids


def boolean_search(
    index: InvertedIndex, query: str, engine: EngineType = "merge"
) -> list[int]:
    if not query.strip():
        return []
    node = _cached_parse(query)
    return sorted(node.evaluate(index))


@timed
def ranked_search(
    index: Index,
    query: str,
    scorer: Scorer | None = None,
    k: int = 10,
) -> list[SearchResult]:
    active_scorer: Scorer = scorer if scorer is not None else BM25()
    if not query.strip():
        return []

    node = _cached_parse(query)
    matched_docs = node.evaluate(index)
    if not matched_docs:
        return []

    query_terms = node.terms()
    scores: dict[int, float] = {doc_id: 0.0 for doc_id in matched_docs}

    for term in query_terms:
        for posting in index.postings.get(term, ()):
            if posting.doc_id in scores:
                scores[posting.doc_id] += active_scorer.score(index, term, posting)

    top_pairs = heapq.nlargest(k, scores.items(), key=lambda item: item[1])
    results: list[SearchResult] = []
    for doc_id, s in top_pairs:
        meta = index.doc_meta.get(doc_id)
        title = Path(meta.path).name if meta else f"doc_{doc_id}"
        snippet = _build_doc_snippet(index, doc_id, query_terms)
        results.append(
            SearchResult(score=s, doc_id=doc_id, title=title, snippet=snippet)
        )

    return results


@timed
def load_index(path: Path) -> Index:
    if path.suffix == ".json":
        return load_index_json(path)
    return load_index_pickle(path)
