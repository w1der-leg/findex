from pathlib import Path

import numpy as np
import pytest

from findex.corpus import Document
from findex.index import build_index
from findex.semantic import (
    SemanticIndex,
    chunk_text,
    hybrid_search,
    l2_normalize,
    load_semantic_index,
    rrf_fuse,
    save_semantic_index,
    semantic_search,
)


def test_chunk_text_splits_with_overlap() -> None:
    words = [f"w{i}" for i in range(600)]
    chunks = chunk_text(" ".join(words), chunk_words=250, overlap=40)
    assert len(chunks) == 3
    assert chunks[0].split()[0] == "w0"
    assert chunks[1].split()[0] == "w210"


def test_semantic_best_chunk_per_doc_and_persistence(tmp_path: Path) -> None:
    docs = [
        Document(doc_id=0, path="doc0.txt", text="alpha beta gamma"),
        Document(doc_id=1, path="doc1.txt", text="delta epsilon zeta"),
    ]
    idx = build_index(docs)

    # Doc 0 має 2 чанки: слабкий і сильний; Doc 1 має 1 середній чанк
    raw = np.array(
        [
            [0.0, 1.0, 0.0],  # doc 0, chunk 0 -> dot=0.0
            [0.95, 0.31, 0.0],  # doc 0, chunk 1 -> dot=0.95 (найкращий фрагмент!)
            [0.6, 0.8, 0.0],  # doc 1, chunk 0 -> dot=0.60
        ],
        dtype=np.float32,
    )
    sem = SemanticIndex(
        embeddings=l2_normalize(raw),
        chunk_to_doc=np.array([0, 0, 1], dtype=np.int32),
        model_name="test-dummy",
    )

    idx_path = tmp_path / "test_index.bin"
    save_semantic_index(sem, idx_path)
    loaded = load_semantic_index(idx_path)
    assert loaded is not None
    assert loaded.num_chunks == 3
    assert loaded.dim == 3

    q_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    hits = semantic_search(idx, loaded, "query", k=2, query_vector=q_vec)
    assert [h.doc_id for h in hits] == [0, 1]
    assert hits[0].score > hits[1].score


def test_hybrid_rrf_combines_keyword_and_semantic() -> None:
    docs = [
        Document(doc_id=0, path="kw_only.txt", text="python compiler"),
        Document(doc_id=1, path="both.txt", text="python python speed"),
        Document(doc_id=2, path="sem_only.txt", text="fast runtime"),
    ]
    idx = build_index(docs)

    raw = np.array(
        [
            [0.0, 0.0, 1.0],  # doc 0 -> далеко від q_vec
            [0.8, 0.6, 0.0],  # doc 1 -> близько до q_vec + вищий BM25 за слово python
            [1.0, 0.0, 0.0],  # doc 2 -> найближчий до q_vec, але без слова python
        ],
        dtype=np.float32,
    )
    sem = SemanticIndex(
        embeddings=l2_normalize(raw),
        chunk_to_doc=np.array([0, 1, 2], dtype=np.int32),
        model_name="test-dummy",
    )
    q_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    hits = hybrid_search(idx, sem, "python", mode="hybrid", k=3, query_vector=q_vec)
    # Doc 1 присутній в обох списках (1-й у BM25, 2-й у semantic), тому RRF ставить його вище за doc 0
    assert {h.doc_id for h in hits} == {0, 1, 2}
    assert hits[0].doc_id in {1, 2}

    kw_hits = hybrid_search(idx, sem, "python", mode="keyword", k=3)
    sem_hits = hybrid_search(idx, sem, "python", mode="semantic", k=3, query_vector=q_vec)
    fused = rrf_fuse(kw_hits, sem_hits, k=3)
    assert fused[0].score == pytest.approx(1.0 / 61 + 1.0 / 62)