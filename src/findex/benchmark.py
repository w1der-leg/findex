import time
from pathlib import Path
from findex.corpus import iter_documents
from findex.index import build_index
from findex.store import (
    save_index_pickle,
    load_index_pickle,
    save_index_json,
    load_index_json,
)

def run_benchmark():
    data_dir = Path("data")
    bin_path = Path("index.bin")
    json_path = Path("index.json")

    print("1. Будуємо індекс у пам'яті...")
    start_time = time.perf_counter()
    index = build_index(iter_documents(data_dir))
    build_time = time.perf_counter() - start_time
    print(f"   Час побудови: {build_time:.4f} сек. (Документів: {len(index.doc_meta)})")

   
    print("\n2. Тестуємо Pickle (бінарний формат):")
    
    start_time = time.perf_counter()
    save_index_pickle(index, bin_path)
    pickle_save_time = time.perf_counter() - start_time
    
    start_time = time.perf_counter()
    _ = load_index_pickle(bin_path)
    pickle_load_time = time.perf_counter() - start_time
    
    pickle_size_kb = bin_path.stat().st_size / 1024
    print(f"   Розмір файлу: {pickle_size_kb:.2f} KB")
    print(f"   Час збереження: {pickle_save_time:.4f} сек.")
    print(f"   Час завантаження: {pickle_load_time:.4f} сек.")

   
    print("\n3. Тестуємо JSON (текстовий формат):")
    
    start_time = time.perf_counter()
    save_index_json(index, json_path)
    json_save_time = time.perf_counter() - start_time
    
    start_time = time.perf_counter()
    _ = load_index_json(json_path)
    json_load_time = time.perf_counter() - start_time
    
    json_size_kb = json_path.stat().st_size / 1024
    print(f"   Розмір файлу: {json_size_kb:.2f} KB")
    print(f"   Час збереження: {json_save_time:.4f} сек.")
    print(f"   Час завантаження: {json_load_time:.4f} сек.")

    
    print("\n" + "="*40)
    print("ПІДСУМКОВА ТАБЛИЦЯ ДЛЯ ЗВІТУ:")
    print("="*40)
    print(f"Формат | Розмір (KB) | Час збереження (s) | Час завантаження (s)")
    print(f"-------+-------------+--------------------+--------------------")
    print(f"Pickle | {pickle_size_kb:11.2f} | {pickle_save_time:18.4f} | {pickle_load_time:18.4f}")
    print(f"JSON   | {json_size_kb:11.2f} | {json_save_time:18.4f} | {json_load_time:18.4f}")
    print("="*40)

if __name__ == "__main__":
    run_benchmark()