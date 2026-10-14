from dataclasses import dataclass
from collections import Counter, defaultdict
from pathlib import Path
from collections.abc import Iterator
from findex.corpus import iter_documents, Document
from findex.tokenize import tokenize

root = Path("data")

@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    tf: int

@dataclass(frozen=True, slots=True)
class DocMeta:
    doc_id: int
    path: str

@dataclass(frozen=True, slots=True)
class InvertedIndex:
    postings: dict[str, list[Posting]]
    doc_lengths: dict[int, int]
    doc_meta: dict[int, DocMeta]

def build_index(documents: Iterator[Document]) -> InvertedIndex:
    doc_lengths = {}
    doc_meta = {}
    postings = defaultdict(list)

    for doc in documents:
        tokens = list(tokenize(doc.text))
        counts = Counter(tokens)
        doc_meta[doc.doc_id] = DocMeta(doc_id=doc.doc_id, path=str(doc.path))
        doc_lengths[doc.doc_id] = len(tokens)

        for term, tf in counts.items():
            postings[term].append(Posting(doc_id=doc.doc_id, tf=tf))

    return InvertedIndex(
        postings=dict(postings),
        doc_lengths=doc_lengths,
        doc_meta=doc_meta,
    )

if __name__ == "__main__":
    import time
    import tracemalloc
    from findex.store import save_index_pickle
    
    data_dir = Path("data")
    
    tracemalloc.start()
    start_time = time.perf_counter()
    
    print(f"Побудова індексу з папки {data_dir}...")
    index = build_index(iter_documents(data_dir))
    
    output_path = Path("index.bin")
    save_index_pickle(index, output_path)
    
    elapsed_time = time.perf_counter() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"Готово! Індекс збережено у {output_path} (документів: {len(index.doc_meta)})")
    print(f" Час виконання: {elapsed_time:.4f} сек.")
    print(f" Пік пам'яті:   {peak / 1024 / 1024:.2f} MB")