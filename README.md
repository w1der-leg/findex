# findex

Локальний повнотекстовий пошуковий рушій на Python із підтримкою ранжування (BM25 / TF-IDF), мови булевих і фразових запитів, інтерактивного CLI та суворої статичної типізації.

![CI](https://github.com/<твій-нік>/findex/actions/workflows/ci.yml/badge.svg)

## Встановлення

Встановлення як системної CLI-утиліти через `uv`:

```bash
uv tool install .
```

Встановлення для розробки:

```bash
uv sync
```

## Використання CLI

```bash
# Побудова та збереження інвертованого індексу
findex index data --out index.bin

# Пошук із ранжуванням BM25 (за замовчуванням) та підсвіченими сніпетами
findex search index.bin "physical AND (settings OR network)" --scorer bm25

# Пошук із ранжуванням TF-IDF та виводом у форматі JSON
findex search index.bin '"physical settings"' --scorer tfidf --json

# Перегляд статистики індексу
findex stats index.bin

# Увімкнення детального логування (-v для INFO, -vv для DEBUG)
findex -vv search index.bin "network"
```

## Архітектура та типізація

- **Пакування (`src/`-layout):** конфігурація згідно з PEP 621 у `pyproject.toml`, маркер `py.typed` для розповсюдження типів.
- **Сувора типізація:** проєкт проходить перевірку `pyright` у режимі `typeCheckingMode = "strict"` з `0 errors`.
- **Протоколи:** інтерфейс ранжування `Scorer` реалізований через `typing.Protocol` (structural subtyping) з двома взаємозамінними реалізаціями — `BM25` та `TfIdf`.
- **Логування:** замість прямого виводу через `print` у бібліотечних модулях використовується стандартний `logging` з налаштуванням рівнів через прапорці `-v` / `-vv`.

## Тестування та якість коду

```bash
# Лінтер та форматування
uv run ruff check .
uv run ruff format --check .

# Статичний аналіз типів
uv run pyright

# Запуск тестів із вимірюванням покриття
uv run pytest --cov=findex --cov-report=term-missing
```

### Покриття тестами

- **Набір тестів (`pytest` + `Hypothesis`):** параметризовані юніт-тести токенізатора (включно з Unicode), перевірки збірки індексу та операцій над постінгами, тестування CLI через `typer.testing.CliRunner`, а також property-based тести (ідемпотентність токенізації, інваріант сортування постінгів, round-trip серіалізація `load(save(index)) == index`).
- **Покриття коду (`pytest --cov=findex`):** `64%` загалом по пакету (основні модулі `corpus.py`, `store.py`, `tokenize.py` — `100%`, `cli.py` — `90%`, `search.py` — `84%`; поза покриттям свідомо залишені допоміжні скрипти бенчмарків та профілювання пам'яті `benchmark.py`, `profile_memory.py`, `stats.py`).

## Збірка дистрибутиву

```bash
uv build
```

Зібраний `.whl`-пакет (`dist/findex-0.4.0-py3-none-any.whl`) доступний у розділі [Releases](../../releases/tag/v0.4.0).