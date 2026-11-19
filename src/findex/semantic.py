from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from findex.index import Index
from findex.search import BM25, Scorer, SearchResult, build_doc_snippet, ranked_search

log = logging.getLogger(__name__)

SearchMode = Literal["keyword", "semantic", "hybrid"]

# Мапінг короткої назви з CLI на модель fastembed (обидві 384-вимірні, ~80 МБ на ONNX)
MODEL_ALIASES: dict[str, str] = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
    "BAAI/bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
}


@dataclass(frozen=True, slots=True)
class SemanticIndex:
    embeddings: NDArray[np.float32]  # shape: (N_chunks, d), L2-normalized
    chunk_to_doc: NDArray[np.int32]  # shape: (N_chunks,)
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    @property
    def num_chunks(self) -> int:
        return int(self.embeddings.shape[0])

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1]) if self.embeddings.ndim == 2 else 0


def chunk_text(text: str, chunk_words: int = 250, overlap: int = 40) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [" ".join(words)]
    step = max(1, chunk_words - overlap)
    chunks: list[str] = []
    for start in range(0, len(words), step):
        piece = words[start : start + chunk_words]
        if piece:
            chunks.append(" ".join(piece))
        if start + chunk_words >= len(words):
            break
    return chunks


@functools.lru_cache(maxsize=2)
def get_embedder(model_name: str):
    from fastembed import TextEmbedding

    resolved = MODEL_ALIASES.get(model_name, model_name)
    log.info("Loading TextEmbedding model: %s", resolved)
    return TextEmbedding(model_name=resolved)


def l2_normalize(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    if vectors.size == 0:
        return vectors.astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (vectors / norms).astype(np.float32)


def semantic_paths_for(index_path: Path) -> tuple[Path, Path, Path]:
    base = index_path.with_suffix("")
    return (
        Path(f"{base}.embeddings.npy"),
        Path(f"{base}.chunk_docs.npy"),
        Path(f"{base}.semantic.meta.txt"),
    )


def save_semantic_index(sem_idx: SemanticIndex, index_path: Path) -> Path:
    emb_path, map_path, meta_path = semantic_paths_for(index_path)
    np.save(emb_path, sem_idx.embeddings.astype(np.float32))
    np.save(map_path, sem_idx.chunk_to_doc.astype(np.int32))
    meta_path.write_text(sem_idx.model_name, encoding="utf-8")
    return emb_path


def load_semantic_index(index_path: Path) -> SemanticIndex | None:
    emb_path, map_path, meta_path = semantic_paths_for(index_path)
    if not emb_path.exists() or not map_path.exists():
        return None
    embeddings = np.load(emb_path).astype(np.float32)
    chunk_to_doc = np.load(map_path).astype(np.int32)
    model_name = (
        meta_path.read_text(encoding="utf-8").strip()
        if meta_path.exists()
        else "sentence-transformers/all-MiniLM-L6-v2"
    )
    return SemanticIndex(
        embeddings=embeddings,
        chunk_to_doc=chunk_to_doc,
        model_name=model_name,
    )


def embed_query(query: str, model_name: str) -> NDArray[np.float32]:
    embedder = get_embedder(model_name)
    vec = next(iter(embedder.embed([query])))
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm > 1e-12:
        arr = arr / norm
    return arr


def semantic_search(
    index: Index,
    sem_idx: SemanticIndex,
    query: str,
    *,
    k: int = 10,
    query_vector: NDArray[np.float32] | None = None,
) -> list[SearchResult]:
    if not query.strip() or k <= 0 or sem_idx.num_chunks == 0:
        return []

    q_vec = (
        query_vector
        if query_vector is not None
        else embed_query(query, sem_idx.model_name)
    )
    # Матрично-векторний добуток: косинусна подібність усіх фрагментів за один крок BLAS
    chunk_scores = sem_idx.embeddings @ q_vec

    # Документу зараховується найкращий фрагмент (best-chunk-per-doc)
    best_per_doc: dict[int, float] = {}
    for chunk_i in range(sem_idx.num_chunks):
        doc_id = int(sem_idx.chunk_to_doc[chunk_i])
        score = float(chunk_scores[chunk_i])
        prev = best_per_doc.get(doc_id)
        if prev is None or score > prev:
            best_per_doc[doc_id] = score

    ranked = sorted(
        best_per_doc.items(),
        key=lambda item: (-item[1], item[0]),
    )[:k]

    query_terms = {w.lower() for w in query.split() if len(w) > 2}
    results: list[SearchResult] = []
    for doc_id, score in ranked:
        meta = index.doc_meta.get(doc_id)
        title = Path(meta.path).name if meta else f"doc_{doc_id}"
        snippet = build_doc_snippet(index, doc_id, query_terms)
        results.append(
            SearchResult(score=score, doc_id=doc_id, title=title, snippet=snippet)
        )
    return results


def rrf_fuse(
    keyword_results: list[SearchResult],
    semantic_results: list[SearchResult],
    *,
    k: int = 10,
    rrf_k: int = 60,
) -> list[SearchResult]:
    """Злиття двох рейтингів через Reciprocal Rank Fusion: sum(1 / (60 + rank_i))."""
    fused_scores: dict[int, float] = {}
    by_doc: dict[int, SearchResult] = {}

    for rank, res in enumerate(keyword_results, start=1):
        fused_scores[res.doc_id] = fused_scores.get(res.doc_id, 0.0) + 1.0 / (
            rrf_k + rank
        )
        by_doc[res.doc_id] = res

    for rank, res in enumerate(semantic_results, start=1):
        fused_scores[res.doc_id] = fused_scores.get(res.doc_id, 0.0) + 1.0 / (
            rrf_k + rank
        )
        if res.doc_id not in by_doc:
            by_doc[res.doc_id] = res

    ordered = sorted(
        fused_scores.items(),
        key=lambda item: (-item[1], item[0]),
    )[:k]

    return [
        SearchResult(
            score=score,
            doc_id=doc_id,
            title=by_doc[doc_id].title,
            snippet=by_doc[doc_id].snippet,
        )
        for doc_id, score in ordered
    ]


def hybrid_search(
    index: Index,
    sem_idx: SemanticIndex | None,
    query: str,
    *,
    mode: SearchMode = "hybrid",
    scorer: Scorer | None = None,
    k: int = 10,
    query_vector: NDArray[np.float32] | None = None,
) -> list[SearchResult]:
    if mode == "keyword" or sem_idx is None:
        return ranked_search(index, query, scorer=scorer or BM25(), k=k)

    if mode == "semantic":
        return semantic_search(
            index, sem_idx, query, k=k, query_vector=query_vector
        )

    pool_k = max(k * 3, 30)
    kw_res = ranked_search(index, query, scorer=scorer or BM25(), k=pool_k)
    sem_res = semantic_search(
        index, sem_idx, query, k=pool_k, query_vector=query_vector
    )
    return rrf_fuse(kw_res, sem_res, k=k)