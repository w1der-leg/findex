# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

from fastapi.testclient import TestClient

from findex.corpus import Document
from findex.index import Index, build_index
from findex.web.app import create_app, get_index


def _make_fixture_index() -> Index:
    docs = [
        Document(
            doc_id=0,
            path="https://books.toscrape.com/catalogue/python-101/index.html",
            text="Python programming language asyncio fastapi web search",
        ),
        Document(
            doc_id=1,
            path="https://books.toscrape.com/catalogue/mystery-house/index.html",
            text="Mystery detective novel classic story",
        ),
    ]
    return build_index(docs, with_positions=True)


def test_health_returns_503_when_index_unloaded() -> None:
    app = create_app()
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 503
    assert resp.json()["status"] == "unavailable"


def test_search_api_pagination_and_validation() -> None:
    app = create_app()
    fixture_idx = _make_fixture_index()
    app.dependency_overrides[get_index] = lambda: fixture_idx
    client = TestClient(app)

    ok = client.get("/search?q=python&k=5&scorer=bm25&page=1")
    assert ok.status_code == 200
    body = ok.json()
    assert body["total"] == 1
    assert body["results"][0]["doc_id"] == 0

    bad_k = client.get("/search?q=python&k=abc")
    assert bad_k.status_code == 422

    blank_q = client.get("/search?q=%20%20%20")
    assert blank_q.status_code == 422


def test_document_endpoint_and_404() -> None:
    app = create_app()
    fixture_idx = _make_fixture_index()
    app.dependency_overrides[get_index] = lambda: fixture_idx
    client = TestClient(app)

    ok = client.get("/docs/1")
    assert ok.status_code == 200
    assert ok.json()["doc_id"] == 1

    not_found = client.get("/docs/999")
    assert not_found.status_code == 404


def test_web_ui_renders_html_and_highlights() -> None:
    app = create_app()
    fixture_idx = _make_fixture_index()
    app.dependency_overrides[get_index] = lambda: fixture_idx
    client = TestClient(app)

    home = client.get("/?q=mystery")
    assert home.status_code == 200
    assert "text/html" in home.headers["content-type"]
    assert "Mystery" in home.text

    doc_page = client.get("/doc/0")
    assert doc_page.status_code == 200
    assert "Doc ID #0" in doc_page.text
