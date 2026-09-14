from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple


class Document(NamedTuple):
    doc_id: int
    path: str
    text: str


def iter_documents(root: Path) -> Iterator[Document]:
    current_id = 0
    for file_path in root.rglob("*.txt"):
        with open(file_path, encoding="utf-8", errors="replace") as f:
            text = f.read()
            yield Document(doc_id=current_id, path=str(file_path), text=text)
            current_id += 1


if __name__ == "__main__":
    docs_stream = iter_documents(Path("data"))
    print(next(docs_stream))
