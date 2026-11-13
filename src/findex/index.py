import functools
import logging
from collections import defaultdict
from collections.abc import Generator, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from findex.corpus import Document, iter_documents
from findex.tokenize import tokenize

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class DocMeta:
    doc_id: int
    path: str


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
