import json
import logging
import pickle
from pathlib import Path
from typing import Any, cast

from findex.index import DocMeta, InvertedIndex, Posting

log = logging.getLogger(__name__)


def save_index_pickle(index: InvertedIndex, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(index, f)
    log.debug("Saved pickle index to %s", path)


def load_index_pickle(path: Path) -> InvertedIndex:
    # SECURITY: Never load pickle files from untrusted sources (RCE risk).
    with open(path, "rb") as f:
        return cast(InvertedIndex, pickle.load(f))


def save_index_json(index: InvertedIndex, path: Path) -> None:
    data: dict[str, Any] = {
        "doc_meta": {
            str(doc_id): {"doc_id": meta.doc_id, "path": meta.path}
            for doc_id, meta in index.doc_meta.items()
        },
        "doc_lengths": {
            str(doc_id): length for doc_id, length in index.doc_lengths.items()
        },
        "postings": {
            word: [
                {"doc_id": p.doc_id, "tf": p.tf, "positions": list(p.positions)}
                for p in postings_list
            ]
            for word, postings_list in index.postings.items()
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    log.debug("Saved JSON index to %s", path)


def load_index_json(path: Path) -> InvertedIndex:
    with open(path, "r", encoding="utf-8") as f:
        raw_data = cast(dict[str, Any], json.load(f))

    raw_doc_meta = cast(dict[str, dict[str, Any]], raw_data["doc_meta"])
    restored_doc_meta: dict[int, DocMeta] = {
        int(doc_id_str): DocMeta(doc_id=int(doc_id_str), path=str(meta["path"]))
        for doc_id_str, meta in raw_doc_meta.items()
    }

    raw_doc_lengths = cast(dict[str, int], raw_data["doc_lengths"])
    restored_doc_lengths: dict[int, int] = {
        int(doc_id_str): int(length) for doc_id_str, length in raw_doc_lengths.items()
    }

    raw_postings = cast(dict[str, list[dict[str, Any]]], raw_data["postings"])
    restored_postings: dict[str, list[Posting]] = {
        word: [
            Posting(
                doc_id=int(p["doc_id"]),
                tf=int(p["tf"]),
                positions=tuple(int(x) for x in p.get("positions", ())),
            )
            for p in postings_list
        ]
        for word, postings_list in raw_postings.items()
    }

    return InvertedIndex(
        postings=restored_postings,
        doc_lengths=restored_doc_lengths,
        doc_meta=restored_doc_meta,
    )
