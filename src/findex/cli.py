import json
import logging
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from findex.corpus import iter_documents
from findex.index import build_index
from findex.query import QuerySyntaxError
from findex.search import BM25, Scorer, TfIdf, load_index, ranked_search
from findex.store import save_index_pickle

app = typer.Typer(help="findex — локальний пошуковий рушій")
console = Console()
err_console = Console(stderr=True)
log = logging.getLogger(__name__)


def setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )


@app.callback()
def main(
    verbose: Annotated[
        int,
        typer.Option(
            "-v", "--verbose", count=True, help="Рівень деталізації логів (-v, -vv)"
        ),
    ] = 0,
) -> None:
    setup_logging(verbose)


@app.command()
def index(
    corpus: Annotated[Path, typer.Argument(help="Шлях до папки з документами")],
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Куди зберегти індекс")
    ] = None,
) -> None:
    """Побудувати та зберегти інвертований індекс."""
    if not corpus.exists() or not corpus.is_dir():
        err_console.print(f"[red]Помилка:[/red] папка корпусу '{corpus}' не існує.")
        raise typer.Exit(code=1)

    output_path = out if out is not None else Path("index.bin")
    log.info("Починаємо індексацію з %s", corpus)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=err_console,
    ) as progress:
        progress.add_task(description="Індексація документів...", total=None)
        idx = build_index(iter_documents(corpus), with_positions=True)
        save_index_pickle(idx, output_path)

    console.print(
        f"[green]Готово![/green] Проіндексовано документів: {idx.num_docs} -> {output_path}"
    )


@app.command()
def search(
    index_path: Annotated[Path, typer.Argument(help="Шлях до файлу індексу")],
    query: Annotated[str, typer.Argument(help="Пошуковий запит")],
    k: Annotated[int, typer.Option("--k", help="Кількість результатів")] = 10,
    scorer: Annotated[
        Literal["bm25", "tfidf"],
        typer.Option("--scorer", help="Алгоритм ранжування (bm25 або tfidf)"),
    ] = "bm25",
    as_json: Annotated[
        bool, typer.Option("--json", help="Вивести результат у JSON (stdout)")
    ] = False,
) -> None:
    """Виконати ранжований пошук по збереженому індексу."""
    if not index_path.exists():
        err_console.print(
            f"[red]Помилка:[/red] файл індексу '{index_path}' не знайдено."
        )
        raise typer.Exit(code=1)

    if not query.strip():
        err_console.print("[red]Помилка:[/red] пошуковий запит не може бути порожнім.")
        raise typer.Exit(code=1)

    log.debug("Завантаження індексу з %s (scorer=%s)", index_path, scorer)
    idx = load_index(index_path)

    active_scorer: Scorer = BM25() if scorer == "bm25" else TfIdf()
    try:
        results = ranked_search(idx, query, scorer=active_scorer, k=k)
    except QuerySyntaxError as exc:
        err_console.print(f"[red]Помилка синтаксису запиту:[/red] {exc}")
        raise typer.Exit(code=1) from None

    if as_json:
        for res in results:
            meta = idx.doc_meta.get(res.doc_id)
            record = {
                "doc_id": res.doc_id,
                "score": round(res.score, 4),
                "path": meta.path if meta else "",
                "snippet": res.snippet,
            }
            print(json.dumps(record, ensure_ascii=False))
        return

    table = Table(title=f"Результати для '{query}' ({scorer.upper()})")
    table.add_column("Rank", justify="right", style="bold")
    table.add_column("ID", justify="right", style="cyan")
    table.add_column("Score", justify="right", style="magenta")
    table.add_column("Файл", style="green")
    table.add_column("Сніпет")

    for rank, res in enumerate(results, start=1):
        meta = idx.doc_meta.get(res.doc_id)
        table.add_row(
            str(rank),
            str(res.doc_id),
            f"{res.score:.4f}",
            meta.path if meta else "N/A",
            res.snippet,
        )

    console.print(table)


@app.command()
def stats(
    index_path: Annotated[Path, typer.Argument(help="Шлях до файлу індексу")],
) -> None:
    """Показати статистику збереженого індексу."""
    if not index_path.exists():
        err_console.print(
            f"[red]Помилка:[/red] файл індексу '{index_path}' не знайдено."
        )
        raise typer.Exit(code=1)

    idx = load_index(index_path)
    table = Table(title=f"Статистика {index_path}")
    table.add_column("Метрика", style="bold")
    table.add_column("Значення", justify="right")
    table.add_row("Документів", str(idx.num_docs))
    table.add_row("Унікальних термів", str(len(idx)))
    table.add_row("Середня довжина документа", f"{idx.avg_doc_length:.1f}")
    console.print(table)
