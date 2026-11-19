from pathlib import Path
from typing import Any

import pytest

from findex.corpus import iter_documents
from findex.index import Index, build_index
from findex.search import BM25, SearchResult, TfIdf, ranked_search, ranked_search_python


@pytest.fixture(scope="module")
def bench_corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("bench_corpus")
    topics = [
        "Python performance optimization profiling cProfile bytecode interpreter loops",
        "NumPy vectorized array operations broadcasting ufuncs int32 argpartition speed",
        "Information retrieval full text search engine BM25 TF-IDF inverted index postings",
        "Asyncio web crawler httpx rate limiting robots parser concurrency FastAPI Uvicorn",
        "Semantic search embeddings cosine similarity dot product sentence transformers RRF",
    ]
    for i in range(120):
        text = " ".join(topics[(i + j) % len(topics)] for j in range(12))
        (root / f"doc_{i:03d}.txt").write_text(f"Title {i}\n{text}\n", encoding="utf-8")
    return root


def test_benchmark_build_index(benchmark: Any, bench_corpus: Path) -> None:
    def _run() -> Index:
        return build_index(iter_documents(bench_corpus))

    idx: Index = benchmark(_run)
    assert idx.num_docs == 120


def test_benchmark_search_bm25_three_terms(benchmark: Any, bench_corpus: Path) -> None:
    idx = build_index(iter_documents(bench_corpus))
    _ = idx.numpy_postings
    _ = idx.doc_len_array

    results: list[SearchResult] = benchmark(
        ranked_search,
        idx,
        "python performance search",
        scorer=BM25(),
        k=10,
    )
    assert len(results) == 10


def test_benchmark_search_tfidf_three_terms(benchmark: Any, bench_corpus: Path) -> None:
    idx = build_index(iter_documents(bench_corpus))
    _ = idx.numpy_postings
    _ = idx.doc_len_array

    results: list[SearchResult] = benchmark(
        ranked_search,
        idx,
        "python performance search",
        scorer=TfIdf(),
        k=10,
    )
    assert len(results) == 10


def test_numpy_scorer_matches_lab3_ranking(bench_corpus: Path) -> None:
    idx = build_index(iter_documents(bench_corpus))
    queries = [
        "python",
        "python performance search",
        "vectorized OR embeddings",
        "crawler NOT mystery",
    ]
    for q in queries:
        for scorer in (BM25(), TfIdf()):
            res_py = ranked_search_python(idx, q, scorer=scorer, k=10)
            res_np = ranked_search(idx, q, scorer=scorer, k=10)
            assert [r.doc_id for r in res_np] == [r.doc_id for r in res_py]
            assert [r.score for r in res_np] == pytest.approx(
                [r.score for r in res_py], rel=1e-12
            )
