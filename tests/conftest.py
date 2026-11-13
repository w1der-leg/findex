from pathlib import Path

import pytest

from findex.corpus import iter_documents
from findex.index import InvertedIndex, build_index


@pytest.fixture(scope="session")
def sample_corpus_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    corpus_dir = tmp_path_factory.mktemp("corpus")
    (corpus_dir / "doc0.txt").write_text(
        "Python async programming with event loop and coroutines.",
        encoding="utf-8",
    )
    (corpus_dir / "doc1.txt").write_text(
        "Java enterprise systems and virtual machines.",
        encoding="utf-8",
    )
    (corpus_dir / "doc2.txt").write_text(
        "Python search engine indexing with BM25 and TF-IDF ranking algorithms.",
        encoding="utf-8",
    )
    (corpus_dir / "doc3.txt").write_text(
        "Short python note.",
        encoding="utf-8",
    )
    (corpus_dir / "doc4.txt").write_text(
        "Python " + "filler " * 100 + "document with lots of extra words.",
        encoding="utf-8",
    )
    return corpus_dir


@pytest.fixture(scope="session")
def sample_index(sample_corpus_dir: Path) -> InvertedIndex:
    return build_index(iter_documents(sample_corpus_dir))
