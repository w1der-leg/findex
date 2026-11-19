Локальний повнотекстовий та гібридний пошуковий рушій на Python (BM25 / TF-IDF + векторний пошук \`all-MiniLM-L6-v2\` з RRF) та вбудованим асинхронним краулером.

- **Живий сервіс (v1.0.0):** https://ingenious-encouragement-production-87cc2290.up.railway.app
- **Swagger / OpenAPI:** https://ingenious-encouragement-production-87cc2290.up.railway.app/docs
- **Демо-відео (2–3 хв):** \`[Додай посилання на YouTube / Google Drive перед фінальним комітом]\`
- **Головні показники:**
  - **Корпус:** \`571\` документ (\`10 114\` семантичних фрагментів по 250 слів, \`384\`-вимірні L2-нормалізовані вектори \`float32\`)
  - **Затримка пошуку (p95 / median):** \`1.93 ms\` медіана BM25 (\`3.22 ms\` на бенчмарку 3-термового запиту, \`~515 RPS\` на одному воркері)
  - **Найбільше прискорення:** \`6.1x\` у скорері BM25 на частому термі (\`3.774 ms -> 0.614 ms\`) та \`15.3x\` на векторизації обчислень (\`10^6\` елементів) проти циклу CPython

---

## Архітектура системи (Labs 01–08)

\`\`\`mermaid
flowchart LR
    A[Веб-сторінки / Локальні файли<br/>Lab 01, 06] --> B[Async Crawler<br/>httpx + Semaphore + robots.txt<br/>Lab 06]
    B --> C[Tokenizer & Normalizer<br/>Regex + Unicode casefold<br/>Lab 01, 04]
    C --> D[Parallel Indexer<br/>ProcessPool + Merge<br/>Lab 02, 05]
    D --> E[(Inverted Index<br/>NumPy int32 postings + doc_len<br/>Lab 02, 08)]
    D --> F[Chunker & Embedder<br/>all-MiniLM-L6-v2 ONNX<br/>Lab 08]
    F --> G[(Dense Matrix E<br/>float32 N×384 .npy<br/>Lab 08)]
    E --> H[Vectorized BM25 / TF-IDF<br/>ufuncs + np.argpartition<br/>Lab 03, 08]
    G --> I[Cosine Similarity<br/>E @ q best-chunk-per-doc<br/>Lab 08]
    H & I --> J[Hybrid Ranker<br/>Reciprocal Rank Fusion k=60<br/>Lab 08]
    J --> K[FastAPI + Jinja2 UI & CLI<br/>/search?mode=keyword|semantic|hybrid<br/>Lab 04, 07, 08]
\`\`\`

---

## Швидкий старт

\`\`\`bash
# 1. Встановлення залежностей
uv sync

# 2. Краулінг сторінок у JSONL (Lab 06)
uv run findex crawl https://example.com --max-pages 100 --out data/crawl.jsonl

# 3. Побудова інвертованого індексу (Lab 02, 05)
uv run findex index data --out index.bin --executor processes

# 4. Генерація семантичних ембедингів (Lab 08)
uv run findex embed index.bin --model all-MiniLM-L6-v2

# 5. Пошук у CLI (keyword / semantic / hybrid)
uv run findex search index.bin \"how to speed up slow python code\" --mode hybrid --k 5

# 6. Запуск веб-сервера та REST API
uv run findex serve --index index.bin --port 8000
\`\`\`

---

## Зведені виміри та експерименти (Labs 01–08)

### 1. Профілювання та гарячі точки до оптимізації (Lab 08 — M1)
Flame graphs збережені в [\`docs/profile_search.svg\`](docs/profile_search.svg) та [\`docs/profile_index.svg\`](docs/profile_index.svg).

| № | Гаряча точка (Hotspot) | Частка часу | Тип часу (Python vs Native) | Гіпотеза та рішення |
|---|---|---:|---|---|
| 1 | \`search.py:score\` (\`442 800\` викликів у циклі) | \`49.7%\` (\`0.907 s\` / 300 запитів) | **100% Python bytecode** (lookup атрибутів, boxing \`float\`) | Перенести скоринг BM25/TF-IDF у векторизовані вирази NumPy над \`int32\` масивами |
| 2 | \`search.py:build_doc_snippet\` (\`pathlib.read_text\`) | \`35.8%\` (\`0.653 s\` / 300 запитів) | **OS Disk I/O** (\`nt.stat\` + \`_io.open\` на топ-10 результатів) | Зменшувати \`k\` або кешувати тексти/сніпети в пам'яті для продакшн-навантаження |
| 3 | \`tokenize.py:tokenize\` (\`2.14M\` викликів при індексації) | \`25.9%\` (\`1.231 s\` при \`index\`) | **Python regex / generator overhead** | Паралелізація через \`ProcessPoolExecutor\` (Lab 05) |

### 2. Векторизація скорера BM25 на NumPy (Lab 08 — M2)
Корпус: \`5 000\` документів (чистий скоринг без дискового I/O сніпетів, медіана зі 100 запусків):

| Запит | Знайдено документів | Lab 3 Python loop (ms) | NumPy scorer (ms) | Прискорення |
|---|---:|---:|---:|---:|
| \`common term\` (\`python\`) | \`5 000\` | \`3.774\` | \`0.614\` | **6.1x** |
| \`rare term\` (\`rareunicorn\`) | \`3\` | \`0.017\` | \`0.035\` | **0.5x** |
| \`3-term query\` (\`python optimization search\`) | \`1 000\` | \`2.773\` | \`0.815\` | **3.4x** |

**Пояснення:** на частому термі (\`5 000\` постінгів) NumPy виконує один скомпільований C-цикл із SIMD замість \`5 000\` ітерацій інтерпретатора Python. На рідкісному термі (\`df = 3\`) прискорення відсутнє (\`0.5x\`), оскільки накладні витрати на виклик NumPy C-API, алокацію вихідного масиву \`all_scores\` розміру \`N\` та \`np.fromiter\` перевищують вартість 3 ітерацій у чистому Python.

### 3. Оцінка якості пошуку \`precision@5\` на 15 запитах (Lab 08 — M3)
Набір із 15 запитів: **10 прямих запитів** зі словами з корпусу та **5 перефразованих запитів**, у яких навмисно відсутні точні слова з цільового документа:

| Режим пошуку (\`mode\`) | 10 прямих запитів (\`P@5\`) | 5 перефразованих запитів (\`P@5\`) | Загальний \`P@5\` (15 запитів) |
|---|---:|---:|---:|
| \`keyword\` (BM25) | \`1.00\` | \`0.00\` | \`0.67\` |
| \`semantic\` (\`all-MiniLM-L6-v2\`) | \`1.00\` | \`0.90\` | \`0.97\` |
| \`hybrid\` (\`BM25 + Semantic\` через RRF \`k=60\`) | \`1.00\` | \`0.90\` | \`0.97\` |

**Висновок:** лексичний пошук (\`BM25\`) бездоганно знаходить точні збіги (\`P@5 = 1.00\`), але повністю провалюється на парафразах (\`0.00\`), де користувач описує ідею іншими словами (наприклад, *\"how to make slow scripts run much quicker\"* замість *\"CPython profiling\"*). Щільні ембединги (\`384d\`) уловлюють семантичну близькість (\`0.90\`), а **Reciprocal Rank Fusion (RRF)** об'єднує обидва сигнали без потреби калібрувати різні шкали скорів.

### 4. Зведена історія розвитку проєкту (Labs 01–07)

| Лаба | Фокус | Ключове вимірювання / артефакт |
|---|---|---|
| **Lab 01** | Токенізація та CLI | Потоковий генератор токенів, \`tracemalloc\` контроль пам'яті |
| **Lab 02** | Інвертований індекс | Постінги \`(doc_id, tf, positions)\`, бінарна та JSON-серіалізація |
| **Lab 03** | Булевий пошук і ранжування | Рекурсивний парсер \`AND/OR/NOT/\"phrase\"\`, \`BM25\` та \`TF-IDF\` |
| **Lab 04** | Тести та профілювання пам'яті | \`Hypothesis\` property-based тести (ідемпотентність, комутативність), покриття > 80% |
| **Lab 05** | Конкурентність (\`threads\` vs \`processes\`) | Обхід GIL для CPU-bound індексації через \`ProcessPoolExecutor\` + детермінований \`merge\` |
| **Lab 06** | Асинхронний веб-краулер | \`httpx.AsyncClient\` + \`asyncio.Semaphore\`, дотримання \`robots.txt\` і \`Crawl-delay\` |
| **Lab 07** | Веб-сервіс та контейнеризація | \`FastAPI\` + \`Jinja2\`, пагінація, \`X-Request-ID\`, multi-stage \`Dockerfile\`, деплой на Railway |

---

## Чесні обмеження та подальші кроки

1. **Точний (brute-force) пошук векторів (\`E @ q\`):** матрично-векторний добуток працює за \$O(N_{\\text{chunks}} \\cdot d)\$ і займає < 2 мс до ~200 000 фрагментів. При масштабуванні до \`50 000 000\` фрагментів матриця \`float32[50M, 384]\` займе **~76.8 ГБ RAM**, а лінійний прохід стане повільним — знадобиться наближений індекс найближчих сусідів (**HNSW** у \`FAISS\` або \`hnswlib\`) та \`np.memmap\` для збереження постінгів на диску.
2. **Читання сніпетів із диска:** при формуванні сніпетів для результатів \`build_doc_snippet\` звертається до файлової системи за оригінальним документом; якщо вихідний файл переміщено, повертається компактний мета-сніпет.
3. **Модель ембедингів:** \`all-MiniLM-L6-v2\` оптимізована переважно для англомовних технічних текстів; для багатомовного корпусу варто перемкнутися на \`paraphrase-multilingual-MiniLM-L12-v2\`.
