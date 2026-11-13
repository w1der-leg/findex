from pathlib import Path

import pytest
from typer.testing import CliRunner

from findex.cli import app
from findex.corpus import Document, iter_documents
from findex.index import InvertedIndex, Posting, build_index
from findex.search import (
    BM25,
    TfIdf,
    boolean_search,
    difference,
    intersect,
    intersect_sets,
    ranked_search,
    search_and,
    union,
)
from findex.store import (
    load_index_json,
    load_index_pickle,
    save_index_json,
    save_index_pickle,
)
from findex.tokenize import tokenize

runner = CliRunner()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello, World!", ["hello", "world"]),
        ("Python 3.13 & FastAPI", ["python", "3", "13", "fastapi"]),
        ("Привіт, світе! Київ 2026", ["привіт", "світе", "київ", "2026"]),
        ("   \n\t  ", []),
        ("UPPER lower MiXeD", ["upper", "lower", "mixed"]),
        ("one-two_three", ["one", "two_three"]),
    ],
)
def test_tokenize_parametrized(text: str, expected: list[str]) -> None:
    assert list(tokenize(text)) == expected


def test_iter_documents(sample_corpus_dir: Path) -> None:
    docs = list(iter_documents(sample_corpus_dir))
    assert len(docs) == 5
    assert [d.doc_id for d in docs] == [0, 1, 2, 3, 4]


def test_build_index_properties(sample_index: InvertedIndex) -> None:
    assert sample_index.num_docs == 5
    assert sample_index.avg_doc_length > 0
    assert sample_index.df("python") == 4
    assert sample_index.df("java") == 1
    assert sample_index.df("nonexistent") == 0


def test_empty_index_properties() -> None:
    empty = build_index([])
    assert empty.num_docs == 0
    assert empty.avg_doc_length == 0.0
    assert empty.df("python") == 0


def test_intersect_merge() -> None:
    a = [Posting(1, 1), Posting(3, 2), Posting(5, 1)]
    b = [Posting(2, 1), Posting(3, 1), Posting(5, 4), Posting(7, 1)]
    assert intersect(a, b) == [3, 5]


def test_union_merge() -> None:
    a = [Posting(1, 1), Posting(3, 2)]
    b = [Posting(3, 1), Posting(4, 1)]
    assert union(a, b) == [1, 3, 4]


def test_difference_merge() -> None:
    a = [Posting(1, 1), Posting(3, 2), Posting(5, 1)]
    b = [Posting(3, 1)]
    assert difference(a, b) == [1, 5]


def test_intersect_sets() -> None:
    a = [Posting(1, 1), Posting(3, 2)]
    b = [Posting(3, 1), Posting(4, 1)]
    assert intersect_sets(a, b) == [3]


@pytest.mark.parametrize("engine", ["merge", "set"])
def test_search_and_engines(sample_index: InvertedIndex, engine: str) -> None:
    res = search_and(sample_index, "python async", engine=engine)  # type: ignore[arg-type]
    assert res == [0]


def test_boolean_search_operators(sample_index: InvertedIndex) -> None:
    assert boolean_search(sample_index, "java OR async") == [0, 1]
    assert boolean_search(sample_index, "python NOT async") == [2, 3, 4]
    assert boolean_search(sample_index, "") == []


@pytest.mark.parametrize("scorer", [BM25(), TfIdf()])
def test_sanity_rare_term_higher_than_common(
    sample_index: InvertedIndex, scorer: BM25 | TfIdf
) -> None:
    rare = ranked_search(sample_index, "bm25", scorer=scorer, k=5)
    common = ranked_search(sample_index, "python", scorer=scorer, k=5)
    assert rare[0].score > common[0].score


def test_sanity_bm25_saturation() -> None:
    docs = [
        Document(0, "d0.txt", "python " * 5),
        Document(1, "d1.txt", "python " * 20),
        Document(2, "d2.txt", "other words here"),
    ]
    idx = build_index(docs)
    bm25 = BM25()
    s5 = bm25.score(idx, "python", idx.postings["python"][0])
    s20 = bm25.score(idx, "python", idx.postings["python"][1])
    assert s20 > s5
    assert (s20 - s5) < s5


def test_sanity_bm25_short_doc_beats_long_doc(sample_index: InvertedIndex) -> None:
    results = ranked_search(sample_index, "python", scorer=BM25(), k=5)
    ranked_ids = [r.doc_id for r in results]
    assert ranked_ids.index(3) < ranked_ids.index(4)


def test_store_pickle_roundtrip(sample_index: InvertedIndex, tmp_path: Path) -> None:
    out = tmp_path / "test.bin"
    save_index_pickle(sample_index, out)
    loaded = load_index_pickle(out)
    assert loaded == sample_index


def test_store_json_roundtrip(sample_index: InvertedIndex, tmp_path: Path) -> None:
    out = tmp_path / "test.json"
    save_index_json(sample_index, out)
    loaded = load_index_json(out)
    assert loaded == sample_index


def test_cli_index_search_stats(sample_corpus_dir: Path, tmp_path: Path) -> None:
    idx_file = tmp_path / "cli_idx.bin"
    res_idx = runner.invoke(
        app, ["index", str(sample_corpus_dir), "--out", str(idx_file)]
    )
    assert res_idx.exit_code == 0
    assert idx_file.exists()

    res_search = runner.invoke(
        app, ["search", str(idx_file), "python", "--scorer", "bm25"]
    )
    assert res_search.exit_code == 0
    assert "Результати" in res_search.stdout

    res_json = runner.invoke(app, ["search", str(idx_file), "python", "--json"])
    assert res_json.exit_code == 0
    assert '"doc_id"' in res_json.stdout

    res_stats = runner.invoke(app, ["stats", str(idx_file)])
    assert res_stats.exit_code == 0
    assert "Документів" in res_stats.stdout


def test_cli_errors_exit_nonzero(tmp_path: Path) -> None:
    missing = tmp_path / "missing.bin"
    res = runner.invoke(app, ["search", str(missing), "python"])
    assert res.exit_code != 0


@pytest.mark.slow
def test_slow_large_synthetic_corpus() -> None:
    docs = [
        Document(i, f"doc_{i}.txt", f"python search engine document number {i}")
        for i in range(1500)
    ]
    idx = build_index(docs)
    assert idx.num_docs == 1500
    assert len(ranked_search(idx, "python engine", k=10)) == 10
