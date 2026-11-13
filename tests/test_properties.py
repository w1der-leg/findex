import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from findex.corpus import Document
from findex.index import Posting, build_index
from findex.search import intersect, intersect_sets
from findex.store import load_index_pickle, save_index_pickle
from findex.tokenize import tokenize


@given(st.text())
def test_prop_tokenize_idempotent(text: str) -> None:
    tokens = list(tokenize(text))
    assert list(tokenize(" ".join(tokens))) == tokens


@given(st.text())
def test_prop_tokenize_no_empty_tokens(text: str) -> None:
    for token in tokenize(text):
        assert len(token) > 0


@given(
    st.lists(st.integers(min_value=0, max_value=500)),
    st.lists(st.integers(min_value=0, max_value=500)),
)
def test_prop_merge_matches_set_oracle(xs: list[int], ys: list[int]) -> None:
    a = [Posting(doc_id=d, tf=1) for d in sorted(set(xs))]
    b = [Posting(doc_id=d, tf=1) for d in sorted(set(ys))]
    assert intersect(a, b) == intersect_sets(a, b)


@settings(max_examples=25)
@given(st.lists(st.text(min_size=1, max_size=40), min_size=1, max_size=10))
def test_prop_index_roundtrip_and_sorted_postings(texts: list[str]) -> None:
    docs = [Document(doc_id=i, path=f"d{i}.txt", text=t) for i, t in enumerate(texts)]
    idx = build_index(docs)

    for postings in idx.postings.values():
        doc_ids = [p.doc_id for p in postings]
        assert doc_ids == sorted(doc_ids)

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "prop.bin"
        save_index_pickle(idx, path)
        assert load_index_pickle(path) == idx
