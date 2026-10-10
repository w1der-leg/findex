import argparse
from collections import Counter
from itertools import islice
from pathlib import Path
import time
import tracemalloc

from findex.corpus import iter_documents
from findex.tokenize import tokenize


def run_eager(folder_path: Path, limit: int | None = None):
    tracemalloc.start()
    start_time = time.perf_counter()

    all_files = list(folder_path.rglob("*.txt"))
    if limit is not None:
        all_files = all_files[:limit]

    all_texts = [
        f.read_text(encoding="utf-8", errors="replace") for f in all_files
    ]
    all_tokens = [list(tokenize(text)) for text in all_texts]

    doc_count = len(all_texts)
    stats_eager = Counter()
    for tokens in all_tokens:
        stats_eager.update(tokens)

    elapsed_time = time.perf_counter() - start_time
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)

    return doc_count, stats_eager, elapsed_time, peak_mb


def run_lazy(folder_path: Path, limit: int | None = None):
    tracemalloc.start()
    start_time = time.perf_counter()

    docs_stream = iter_documents(folder_path)
    if limit is not None:
        docs_stream = islice(docs_stream, limit)

    doc_count = 0
    stats = Counter()
    for doc in docs_stream:
        doc_count += 1
        stats.update(tokenize(doc.text))

    elapsed_time = time.perf_counter() - start_time
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)

    return doc_count, stats, elapsed_time, peak_mb


def main():
    parser = argparse.ArgumentParser(description="Аналіз статистики слів у файлах")
    parser.add_argument("folder", type=Path, help="Шлях до папки з файлами")
    parser.add_argument("--limit", type=int, default=None, help="Ліміт кількості документів")
    args = parser.parse_args()

    e_docs, e_stats, e_time, e_mem = run_eager(args.folder, args.limit)
    l_docs, l_stats, l_time, l_mem = run_lazy(args.folder, args.limit)

    print("\n| Version            | Documents | Peak memory | Elapsed   |")
    print("|--------------------|-----------|-------------|-----------|")
    print(f"| eager (lists)      | {e_docs:<9} | {e_mem:.2f} MB     | {e_time:.4f} s |")
    print(f"| lazy (generators)  | {l_docs:<9} | {l_mem:.2f} MB     | {l_time:.4f} s |")

    total_tokens = sum(l_stats.values())
    unique_words = len(l_stats)

    print("\nТоп-50 найчастіших слів:")
    for word, count in l_stats.most_common(50):
        print(f"  • {word}: {count}")

    print(f"\nУнікальних слів: {unique_words}")
    print(f"Сума всіх токенів: {total_tokens}")


if __name__ == "__main__":
    main()