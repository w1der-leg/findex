import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: int
    path: str
    text: str


def iter_documents(corpus_dir: Path) -> Iterator[Document]:
    """Потоково читає .txt, .md та .jsonl файли корпусу без завантаження всього корпусу в ОЗП."""
    doc_id = 0
    if corpus_dir.is_file():
        files = [corpus_dir]
    else:
        files = sorted(
            p
            for p in corpus_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in {".txt", ".md", ".jsonl"}
        )

    for file_path in files:
        if file_path.suffix.lower() == ".jsonl":
            with file_path.open("r", encoding="utf-8", errors="replace") as f:
                for line_no, line in enumerate(f, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    rec = json.loads(stripped)
                    title = str(rec.get("title", ""))
                    body = str(rec.get("text", ""))
                    url = str(rec.get("url", f"{file_path}:{line_no}"))
                    yield Document(
                        doc_id=doc_id, path=url, text=f"{title}\n{body}".strip()
                    )
                    doc_id += 1
        else:
            text = file_path.read_text(encoding="utf-8", errors="replace")
            yield Document(doc_id=doc_id, path=str(file_path), text=text)
            doc_id += 1
