import tracemalloc
from pathlib import Path
from findex.corpus import iter_documents
from findex.index import build_index

def profile_index_memory():
    data_dir = Path("data")
    
    # Запускаємо трекінг пам'яті
    tracemalloc.start()
    
    print("Побудова індексу для вимірювання пам'яті...")
    index = build_index(iter_documents(data_dir))
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print("\n" + "="*40)
    print("РЕЗУЛЬТАТИ TRACEMALLOC (Slots=True):")
    print("="*40)
    print(f"Поточне використання пам'яті: {current / 1024 / 1024:.2f} MB")
    print(f"Пікове споживання пам'яті:   {peak / 1024 / 1024:.2f} MB")
    print(f"Всього документів в індексі:    {len(index.doc_meta)}")
    print(f"Всього унікальних термінів:   {len(index.postings)}")
    print("="*40)

if __name__ == "__main__":
    profile_index_memory()