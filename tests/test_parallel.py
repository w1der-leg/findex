from pathlib import Path

import pytest
from typer.testing import CliRunner

from findex.cli import app
from findex.corpus import iter_documents
from findex.index import (
    build_index,
    build_index_parallel,
    build_partial,
    collect_corpus_paths,
    merge,
)

runner = CliRunner()


def test_serial_partial_merge_matches_lab4_build_index(sample_corpus_dir: Path) -> None:
    expected = build_index(iter_documents(sample_corpus_dir), with_positions=True)
    paths = collect_corpus_paths(sample_corpus_dir)
    actual = merge([build_partial(paths, start_doc_id=0, with_positions=True)])
    assert actual == expected


@pytest.mark.parametrize("executor", ["serial", "threads", "processes"])
@pytest.mark.parametrize("workers", [1, 2, 4])
def test_parallel_executors_produce_identical_index(
    sample_corpus_dir: Path,
    executor: str,
    workers: int,
) -> None:
    expected = build_index(iter_documents(sample_corpus_dir), with_positions=True)
    paths = collect_corpus_paths(sample_corpus_dir)

    actual, _ = build_index_parallel(
        paths,
        executor=executor,  # type: ignore[arg-type]
        workers=workers,
        with_positions=True,
    )
    assert actual == expected


def test_worker_exception_propagates(tmp_path: Path) -> None:
    missing_file = tmp_path / "does_not_exist.txt"
    with pytest.raises(FileNotFoundError):
        build_index_parallel([missing_file], executor="processes", workers=2)


def test_cli_index_with_executor_and_workers(
    sample_corpus_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "par.bin"
    res = runner.invoke(
        app,
        [
            "index",
            str(sample_corpus_dir),
            "--out",
            str(out),
            "--executor",
            "threads",
            "--workers",
            "2",
        ],
    )
    assert res.exit_code == 0
    assert out.exists()
