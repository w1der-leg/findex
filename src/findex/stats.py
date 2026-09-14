import argparse
import itertools
import time
import tracemalloc
from collections import Counter
from pathlib import Path

from findex.corpus import iter_documents
from findex.tokenize import tokenize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute corpus and token statistics in constant memory."
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Шлях до папки з текстовими файлами",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Максимальна кількість документів для читання",
    )
    return parser.parse_args()


def run_eager(
    root: Path, limit: int | None = None
) -> tuple[int, int, int, float, float]:
    tracemalloc.start()
    t_start = time.perf_counter()

    all_files = list(root.rglob("*.txt"))
    if limit is not None:
        all_files = all_files[:limit]

    documents = []
    for doc_id, file_path in enumerate(all_files):
        with open(file_path, encoding="utf-8", errors="replace") as f:
            documents.append((doc_id, str(file_path), f.read()))

    all_tokens = []
    for _, _, text in documents:
        all_tokens.append(list(tokenize(text)))

    doc_count = len(documents)
    total_tokens = sum(len(toks) for toks in all_tokens)
    term_counts = Counter(token for toks in all_tokens for token in toks)

    t_elapsed = time.perf_counter() - t_start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return (
        doc_count,
        total_tokens,
        len(term_counts),
        t_elapsed,
        peak_bytes / (1024 * 1024),
    )


def run_lazy(
    root: Path, limit: int | None = None
) -> tuple[int, int, int, float, float]:
    tracemalloc.start()
    t_start = time.perf_counter()

    docs_stream = iter_documents(root)
    if limit is not None:
        docs_stream = itertools.islice(docs_stream, limit)

    doc_count = 0
    total_tokens = 0
    term_counts = Counter()

    for doc in docs_stream:
        doc_count += 1
        for token in tokenize(doc.text):
            total_tokens += 1
            term_counts[token] += 1

    t_elapsed = time.perf_counter() - t_start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return (
        doc_count,
        total_tokens,
        len(term_counts),
        t_elapsed,
        peak_bytes / (1024 * 1024),
    )


def main() -> None:
    args = parse_args()

    print("Запуск eager версії...")
    e_docs, _, _, e_time, e_mem = run_eager(args.root, args.limit)

    print("Запуск lazy версії...")
    l_docs, _, _, l_time, l_mem = run_lazy(args.root, args.limit)

    print("\n| Version          | Documents | Peak memory | Elapsed   |")
    print("|------------------|-----------|-------------|-----------|")
    print(f"| eager (lists)    | {e_docs:<9} | {e_mem:.2f} MB     | {e_time:.4f} s  |")
    print(f"| lazy (generators)| {l_docs:<9} | {l_mem:.2f} MB     | {l_time:.4f} s  |")


if __name__ == "__main__":
    main()
