from dataclasses import dataclass
from collections import defaultdict
from collections import Counter
from pathlib import Path
from findex.corpus import iter_documents, Document
import argparse
from findex.store import load_index_json, load_index_pickle
from findex.index import Posting, InvertedIndex, build_index, DocMeta
from findex.tokenize import tokenize

def intersect(posting_a, posting_b):
    i = 0
    j = 0
    result = []

    while i < len(posting_a) and j < len(posting_b):
        doc_a = posting_a[i].doc_id
        doc_b = posting_b[j].doc_id

        if  doc_a == doc_b:
             result.append(doc_a)
             i += 1
             j += 1
        elif doc_a < doc_b:
            i += 1
        else:
            j += 1
    return result

def union(posting_a, posting_b):
    i = 0
    j = 0
    result = []

    while i < len(posting_a) and j < len(posting_b):
        doc_a = posting_a[i].doc_id
        doc_b = posting_b[j].doc_id

        if doc_a == doc_b:
            result.append(doc_a) 
            i += 1
            j += 1
        elif doc_a < doc_b:
            result.append(doc_a) 
            i += 1 
            
        else:
            result.append(doc_b) 
            j += 1
            
    while i < len(posting_a):
      result.append(posting_a[i].doc_id)
      i += 1

    while j < len(posting_b):
      result.append(posting_b[j].doc_id)
      j += 1
    
    return result

def difference(posting_a, posting_b):
    i = 0
    j = 0
    result = []

    while i < len(posting_a) and j < len(posting_b):
        doc_a = posting_a[i].doc_id
        doc_b = posting_b[j].doc_id
        if doc_a == doc_b:
         i += 1
         j += 1
        elif doc_a < doc_b:
         result.append(doc_a) 
         i += 1        
        else:
         j += 1
    while i < len(posting_a):
         result.append(posting_a[i].doc_id)
         i += 1
    return result

def search_and(index: InvertedIndex, query: str, engine: str = "merge") -> list[int]:

  if engine not in ("merge", "set"):
        raise ValueError(f"Unknown engine: {engine}. Allowed: 'merge', 'set'")

  tokens = list(tokenize(query))
  if not tokens:
     return []
  lists_to_merge = []

  for word in tokens:
     postings_list = index.postings.get(word)
     if postings_list is None:
        return []
     lists_to_merge.append(postings_list)

  current_postings = lists_to_merge[0]
  result_ids = [p.doc_id for p in current_postings]

  for next_posting in lists_to_merge[1:]:
      if engine == "merge":
          result_ids = intersect(current_postings, next_posting)
      elif engine == "set":
          result_ids = intersect_sets(current_postings, next_posting)

      if not result_ids:
          return []
      current_postings = [Posting(doc_id=d_id, tf=0) for d_id in result_ids]

  return  result_ids



def intersect_sets(posting_a, posting_b):
    set_a = {p.doc_id for p in posting_a}
    set_b = {p.doc_id for p in posting_b}
    inter = set_a & set_b
    return sorted(inter)


def boolean_search(index: InvertedIndex, query: str, engine: str = "merge") -> list[int]:
    words = query.split()
    result_postings = None
    op = "AND"

    for item in words:
        if item in ["AND", "OR", "NOT"]:
            op = item
            continue

        tokens = list(tokenize(item))
        if not tokens:
            continue
        term = tokens[0]
        postings = index.postings.get(term, [])

        if result_postings is None:
            result_postings = postings
            continue

        if op == "AND":
            if engine == "set":
                set_a = [p.doc_id for p in result_postings]
                set_b = [p.doc_id for p in postings]
                doc_ids = intersect_sets(
                    [Posting(doc_id=d, tf=0) for d in set_a], 
                    [Posting(doc_id=d, tf=0) for d in set_b]
                )
            else:
                doc_ids = intersect(result_postings, postings)
        elif op == "OR":
            doc_ids = union(result_postings, postings)
        elif op == "NOT":
            doc_ids = difference(result_postings, postings)

        result_postings = [Posting(doc_id=d_id, tf=0) for d_id in doc_ids]
        op = "AND"

    if result_postings is None:
        return []
       
    return [p.doc_id for p in result_postings]

def load_index(path: Path):
    if path.suffix == ".json":
        return load_index_json(path)
    else:
        return load_index_pickle(path)

if __name__ == "__main__":
    import time
    import tracemalloc

    parser = argparse.ArgumentParser(description="findex Boolean Search Engine CLI")
    parser.add_argument("index_path", type=Path, help="Path to the index file (.bin or .json)")
    parser.add_argument("query", type=str, help="Search query string (e.g. 'physical OR settings')")
    parser.add_argument(
        "--engine", 
        choices=["merge", "set"], 
        default="merge", 
        help="Search execution strategy (default: merge)"
    )

    args = parser.parse_args()

    tracemalloc.start()
    start_time = time.perf_counter()

    print(f"Завантаження індексу з {args.index_path}...")
    index = load_index(args.index_path)
    print(f"Індекс завантажено! Документів у базі: {len(index.doc_meta)}")

    print(f"Шукаємо: '{args.query}' (рушій: {args.engine})")
    matched_doc_ids = boolean_search(index, args.query, engine=args.engine)

    elapsed_time = time.perf_counter() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\nЗнайдено збігів: {len(matched_doc_ids)}")
    for doc_id in matched_doc_ids:
        meta = index.doc_meta.get(doc_id)
        if meta:
            print(f"  [doc_id={doc_id}] -> {meta.path}")

    print(f"\n Час пошуку/завантаження: {elapsed_time:.4f} сек.")
    print(f" Пік пам'яті:           {peak / 1024 / 1024:.2f} MB")




    
   
                   