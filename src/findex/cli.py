import asyncio
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from findex.crawler.crawl import CrawlStats, crawl
from findex.crawler.fetch import FetchResult
from findex.index import ExecutorType, build_index_parallel, collect_corpus_paths
from findex.query import QuerySyntaxError
from findex.search import BM25, Scorer, TfIdf, load_index, ranked_search
from findex.store import save_index_pickle

app = typer.Typer(help="findex — локальний пошуковий рушій та асинхронний краулер")
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


def _render_crawl_table(stats: CrawlStats, max_pages: int) -> Table:
    table = Table(title="findex crawl — Live Status")
    table.add_column("Метрика", style="bold")
    table.add_column("Значення", justify="right", style="cyan")
    table.add_row("Зібрано сторінок", f"{stats.pages_fetched} / {max_pages}")
    table.add_row("Сторінок / сек", f"{stats.pages_per_sec:.2f}")
    table.add_row("Активні запити (in-flight)", str(stats.in_flight))
    table.add_row("Помилки", f"[red]{stats.errors}[/red]" if stats.errors else "0")
    table.add_row("Глибина черги (queue depth)", str(stats.queue_depth))
    return table


@app.command("crawl")
def crawl_cmd(
    seed_url: Annotated[str, typer.Argument(help="Початковий URL для обходу")],
    max_pages: Annotated[
        int, typer.Option("--max-pages", help="Максимум сторінок")
    ] = 500,
    concurrency: Annotated[
        int, typer.Option("--concurrency", help="Максимум одночасних запитів")
    ] = 10,
    per_host: Annotated[
        int, typer.Option("--per-host", help="Ліміт запитів на один хост")
    ] = 2,
    delay: Annotated[
        float, typer.Option("--delay", help="Мінімальна пауза на хост (сек)")
    ] = 0.2,
    out: Annotated[
        Path, typer.Option("--out", "-o", help="Шлях до вихідного JSONL")
    ] = Path("data/crawl.jsonl"),
    log_path: Annotated[
        Path, typer.Option("--log", help="Файл логу статусів URL")
    ] = Path("crawl.log"),
    debug: Annotated[
        bool, typer.Option("--debug", help="Увімкнути asyncio debug mode")
    ] = False,
) -> None:
    """Асинхронно обійти веб-сторінки та зберегти їх потоково у JSONL."""
    out.parent.mkdir(parents=True, exist_ok=True)
    stats = CrawlStats()
    url_log: list[FetchResult] = []

    async def _run() -> None:
        with (
            out.open("w", encoding="utf-8") as jsonl_file,
            Live(
                _render_crawl_table(stats, max_pages),
                console=err_console,
                refresh_per_second=8,
            ) as live,
        ):
            async for page in crawl(
                [seed_url],
                max_pages=max_pages,
                concurrency=concurrency,
                per_host=per_host,
                per_host_delay=delay,
                stats=stats,
                url_log=url_log,
            ):
                jsonl_file.write(json.dumps(asdict(page), ensure_ascii=False) + "\n")
                jsonl_file.flush()
                live.update(_render_crawl_table(stats, max_pages))
            live.update(_render_crawl_table(stats, max_pages))

    asyncio.run(_run(), debug=debug)

    with log_path.open("w", encoding="utf-8") as lf:
        for entry in url_log:
            lf.write(
                f"{entry.status}\t{entry.num_bytes}B\t{entry.elapsed:.3f}s\t"
                f"attempts={entry.attempts}\t{entry.url}"
                + (f"\terror={entry.error}" if entry.error else "")
                + "\n"
            )

    console.print(
        f"[green]Обхід завершено![/green] Сторінок: {stats.pages_fetched}, "
        f"швидкість: {stats.pages_per_sec:.2f} стор/с, помилок: {stats.errors} -> {out}"
    )


@app.command()
def index(
    corpus: Annotated[
        Path, typer.Argument(help="Шлях до папки або .jsonl файлу корпусу")
    ],
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Куди зберегти індекс")
    ] = None,
    workers: Annotated[
        int | None,
        typer.Option(
            "--workers",
            "-w",
            help="Кількість воркерів (за замовчуванням os.cpu_count())",
        ),
    ] = None,
    executor: Annotated[
        ExecutorType,
        typer.Option(
            "--executor", "-e", help="Виконавець: serial, threads або processes"
        ),
    ] = "processes",
) -> None:
    """Побудувати та зберегти інвертований індекс."""
    if not corpus.exists():
        err_console.print(f"[red]Помилка:[/red] шлях корпусу '{corpus}' не існує.")
        raise typer.Exit(code=1)

    output_path = out if out is not None else Path("index.bin")
    actual_workers = (
        workers if workers is not None and workers > 0 else (os.cpu_count() or 1)
    )
    log.info(
        "Починаємо індексацію з %s (executor=%s, workers=%d)",
        corpus,
        executor,
        actual_workers,
    )

    doc_paths = collect_corpus_paths(corpus)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=err_console,
    ) as progress:
        progress.add_task(
            description=f"Індексація ({executor}, workers={actual_workers})...",
            total=None,
        )
        idx, merge_sec = build_index_parallel(
            doc_paths,
            executor=executor,
            workers=actual_workers,
            with_positions=True,
        )
        save_index_pickle(idx, output_path)

    console.print(
        f"[green]Готово![/green] Проіндексовано документів: {idx.num_docs} "
        f"({executor}, workers={actual_workers}, merge={merge_sec:.3f}s) -> {output_path}"
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


if __name__ == "__main__":
    app()
