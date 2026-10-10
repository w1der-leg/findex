import pickle
from pathlib import Path
from findex.index import Posting, InvertedIndex, build_index, DocMeta
from findex.corpus import iter_documents, Document
import json
from findex.index import DocMeta, InvertedIndex, Posting
def save_index_pickle(index: InvertedIndex, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(index, f)

def load_index_pickle(path: Path) -> InvertedIndex:
    with open(path, "rb") as f:
        return pickle.load(f)
        

def save_index_json(index: InvertedIndex, path: Path) -> None:
    
    data = {
        "doc_meta": {
            doc_id: {"doc_id": meta.doc_id, "path": str(meta.path)}
            for doc_id, meta in index.doc_meta.items()
        },
        "doc_lengths": index.doc_lengths,  # Зберігаємо довжини документів
        "postings": {
            word: [{"doc_id": p.doc_id, "tf": p.tf} for p in postings_list]
            for word, postings_list in index.postings.items()
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_index_json(path: Path) -> InvertedIndex:
    
    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    restored_doc_meta = {
        int(doc_id_str): DocMeta(
            doc_id=int(doc_id_str), path=Path(meta["path"])
        )
        for doc_id_str, meta in raw_data["doc_meta"].items()
    }

    
    restored_doc_lengths = {
        int(doc_id_str): length
        for doc_id_str, length in raw_data["doc_lengths"].items()
    }

    restored_postings = {
        word: [Posting(doc_id=p["doc_id"], tf=p["tf"]) for p in postings_list]
        for word, postings_list in raw_data["postings"].items()
    }

    return InvertedIndex(
        postings=restored_postings,
        doc_lengths=restored_doc_lengths,
        doc_meta=restored_doc_meta,
    )