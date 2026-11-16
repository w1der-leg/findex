import functools
import logging
import os
import time
from collections import defaultdict
from collections.abc import Generator, Iterable, Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from findex.corpus import Document, iter_documents
from findex.tokenize import tokenize

log = logging.getLogger(__name__)

ExecutorType = Literal["serial", "threads", "processes"]


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class DocMeta:
    doc_id: int
    path: str


@dataclass(frozen=True, slots=True)
class PartialIndex:
    postings: dict[str, list[Posting]]
    doc_lengths: dict[int, int]
    doc_meta: dict[int, DocMeta]


@dataclass
class Index(Mapping[str, list[Posting]]):
    postings: dict[str, list[Posting]]
    doc_lengths: dict[int, int]
    doc_meta: dict[int, DocMeta]
    _closed: bool = field(default=False, repr=False)

    def __len__(self) -> int:
        return len(self.postings)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self.postings

    def __getitem__(self, term: str) -> list[Posting]:
        return self.postings[term]

    def __iter__(self) -> Iterator[str]:
        return iter(self.postings)

    def __repr__(self) -> str:
        return f"Index(terms={len(self):_}, docs={self.num_docs:_})"

    @property
    def num_docs(self) -> int:
        return len(self.doc_meta)

    @functools.cached_property
    def avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def doc_length(self, doc_id: int) -> int:
        return self.doc_lengths.get(doc_id, 0)

    def df(self, term: str) -> int:
        return len(self.postings.get(term, ()))

    def close(self) -> None:
        self._closed = True
        log.debug("Index closed")


InvertedIndex = Index


def collect_corpus_paths(corpus_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in corpus_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".txt", ".md"}
    )


def build_partial(
    doc_paths: list[Path],
    start_doc_id: int = 0,
    *,
    with_positions: bool = True,
) -> PartialIndex:
    """Рівень модуля (picklable): читає чанк шляхів і будує локальний індекс із діапазону doc_id."""
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}
    postings: defaultdict[str, list[Posting]] = defaultdict(list)

    for offset, path in enumerate(doc_paths):
        doc_id = start_doc_id + offset
        text = path.read_text(encoding="utf-8", errors="replace")
        tokens = list(tokenize(text))

        doc_meta[doc_id] = DocMeta(doc_id=doc_id, path=str(path))
        doc_lengths[doc_id] = len(tokens)

        term_positions: defaultdict[str, list[int]] = defaultdict(list)
        for pos, term in enumerate(tokens):
            term_positions[term].append(pos)

        for term, pos_list in term_positions.items():
            positions_tuple = tuple(pos_list) if with_positions else ()
            postings[term].append(
                Posting(doc_id=doc_id, tf=len(pos_list), positions=positions_tuple)
            )

    return PartialIndex(
        postings=dict(postings),
        doc_lengths=doc_lengths,
        doc_meta=doc_meta,
    )


def _build_partial_task(task: tuple[list[Path], int, bool]) -> PartialIndex:
    paths, start_id, with_positions = task
    return build_partial(paths, start_doc_id=start_id, with_positions=with_positions)


def merge(partials: Iterable[PartialIndex]) -> Index:
    """Послідовно об'єднує часткові індекси у фінальний Index зі збереженням порядку doc_id."""
    merged_doc_lengths: dict[int, int] = {}
    merged_doc_meta: dict[int, DocMeta] = {}
    merged_postings: defaultdict[str, list[Posting]] = defaultdict(list)

    for part in partials:
        merged_doc_meta.update(part.doc_meta)
        merged_doc_lengths.update(part.doc_lengths)
        for term, plist in part.postings.items():
            merged_postings[term].extend(plist)

    for plist in merged_postings.values():
        plist.sort(key=lambda p: p.doc_id)

    return Index(
        postings=dict(merged_postings),
        doc_lengths=merged_doc_lengths,
        doc_meta=merged_doc_meta,
    )


def _chunk_paths(
    paths: list[Path], num_chunks: int, with_positions: bool
) -> list[tuple[list[Path], int, bool]]:
    if not paths:
        return []
    n = max(1, min(num_chunks, len(paths)))
    chunk_size = (len(paths) + n - 1) // n
    tasks: list[tuple[list[Path], int, bool]] = []
    for i in range(0, len(paths), chunk_size):
        chunk = paths[i : i + chunk_size]
        tasks.append((chunk, i, with_positions))
    return tasks


def build_index_parallel(
    doc_paths: list[Path],
    *,
    executor: ExecutorType = "processes",
    workers: int | None = None,
    with_positions: bool = True,
) -> tuple[Index, float]:
    """Будує індекс обраним виконавцем і повертає (Index, merge_time_sec)."""
    if not doc_paths:
        return Index(postings={}, doc_lengths={}, doc_meta={}), 0.0

    max_workers = (
        workers if workers is not None and workers > 0 else (os.cpu_count() or 1)
    )
    partials: list[PartialIndex] = []

    if executor == "serial":
        tasks = _chunk_paths(doc_paths, 1, with_positions)
        partials = [_build_partial_task(t) for t in tasks]
    elif executor == "threads":
        tasks = _chunk_paths(doc_paths, max_workers, with_positions)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_build_partial_task, t) for t in tasks]
            for fut in as_completed(futures):
                partials.append(fut.result())
    else:
        tasks = _chunk_paths(doc_paths, max_workers, with_positions)
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_build_partial_task, t) for t in tasks]
            for fut in as_completed(futures):
                partials.append(fut.result())

    t0 = time.perf_counter()
    final_index = merge(partials)
    merge_elapsed = time.perf_counter() - t0

    log.info(
        "Built index (%s, workers=%d): %d docs, %d terms (merge=%.4fs)",
        executor,
        max_workers,
        final_index.num_docs,
        len(final_index),
        merge_elapsed,
    )
    return final_index, merge_elapsed


def build_index(documents: Iterable[Document], *, with_positions: bool = True) -> Index:
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}
    postings: defaultdict[str, list[Posting]] = defaultdict(list)

    for doc in documents:
        tokens = list(tokenize(doc.text))
        doc_meta[doc.doc_id] = DocMeta(doc_id=doc.doc_id, path=str(doc.path))
        doc_lengths[doc.doc_id] = len(tokens)

        term_positions: defaultdict[str, list[int]] = defaultdict(list)
        for pos, term in enumerate(tokens):
            term_positions[term].append(pos)

        for term, pos_list in term_positions.items():
            positions_tuple = tuple(pos_list) if with_positions else ()
            postings[term].append(
                Posting(doc_id=doc.doc_id, tf=len(pos_list), positions=positions_tuple)
            )

    log.info("Built index with %d documents and %d terms", len(doc_meta), len(postings))
    return Index(
        postings=dict(postings),
        doc_lengths=doc_lengths,
        doc_meta=doc_meta,
    )


@contextmanager
def open_index(path: Path) -> Generator[Index]:
    from findex.store import load_index_json, load_index_pickle

    idx = load_index_json(path) if path.suffix == ".json" else load_index_pickle(path)
    try:
        yield idx
    finally:
        idx.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from findex.store import save_index_pickle

    idx = build_index(iter_documents(Path("data")), with_positions=True)
    save_index_pickle(idx, Path("index.bin"))
    log.info("Saved %r", idx)
