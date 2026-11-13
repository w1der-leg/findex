import re
from dataclasses import dataclass

from findex.index import Index
from findex.tokenize import tokenize


class QuerySyntaxError(ValueError):
    """Помилка синтаксису пошукового запиту."""


class QueryNode:
    def evaluate(self, index: Index) -> set[int]:
        raise NotImplementedError

    def terms(self) -> set[str]:
        return set()

    def __and__(self, other: "QueryNode") -> "And":
        return And(self, other)

    def __or__(self, other: "QueryNode") -> "Or":
        return Or(self, other)

    def __invert__(self) -> "Not":
        return Not(self)


@dataclass(frozen=True)
class Term(QueryNode):
    value: str

    def evaluate(self, index: Index) -> set[int]:
        return {p.doc_id for p in index.postings.get(self.value, ())}

    def terms(self) -> set[str]:
        return {self.value}


@dataclass(frozen=True)
class Phrase(QueryNode):
    words: tuple[str, ...]

    def evaluate(self, index: Index) -> set[int]:
        if not self.words:
            return set()
        if len(self.words) == 1:
            return Term(self.words[0]).evaluate(index)

        first_postings = {
            p.doc_id: p.positions for p in index.postings.get(self.words[0], ())
        }
        if not first_postings:
            return set()

        matched_docs: set[int] = set()
        for doc_id, start_positions in first_postings.items():
            if not start_positions:
                continue
            current_positions = set(start_positions)
            ok = True
            for offset, word in enumerate(self.words[1:], start=1):
                word_postings = {
                    p.doc_id: p.positions
                    for p in index.postings.get(word, ())
                    if p.doc_id == doc_id
                }
                next_pos = word_postings.get(doc_id)
                if not next_pos:
                    ok = False
                    break
                next_pos_set = set(next_pos)
                current_positions = {
                    p for p in current_positions if (p + offset) in next_pos_set
                }
                if not current_positions:
                    ok = False
                    break
            if ok:
                matched_docs.add(doc_id)

        return matched_docs

    def terms(self) -> set[str]:
        return set(self.words)


@dataclass(frozen=True)
class And(QueryNode):
    left: QueryNode
    right: QueryNode

    def evaluate(self, index: Index) -> set[int]:
        return self.left.evaluate(index) & self.right.evaluate(index)

    def terms(self) -> set[str]:
        return self.left.terms() | self.right.terms()


@dataclass(frozen=True)
class Or(QueryNode):
    left: QueryNode
    right: QueryNode

    def evaluate(self, index: Index) -> set[int]:
        return self.left.evaluate(index) | self.right.evaluate(index)

    def terms(self) -> set[str]:
        return self.left.terms() | self.right.terms()


@dataclass(frozen=True)
class Not(QueryNode):
    operand: QueryNode

    def evaluate(self, index: Index) -> set[int]:
        all_docs = set(index.doc_meta.keys())
        return all_docs - self.operand.evaluate(index)

    def terms(self) -> set[str]:
        return set()


_TOKEN_RE = re.compile(r'\(|\)|"[^"]*"|[^\s()"]+')


class _Parser:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self) -> str:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def parse_or(self) -> QueryNode:
        node = self.parse_and()
        while self.peek() == "OR":
            self.consume()
            right = self.parse_and()
            node = Or(node, right)
        return node

    def parse_and(self) -> QueryNode:
        node = self.parse_not()
        while (tok := self.peek()) is not None and tok not in ("OR", ")"):
            if tok == "AND":
                self.consume()
            right = self.parse_not()
            node = And(node, right)
        return node

    def parse_not(self) -> QueryNode:
        if self.peek() == "NOT":
            self.consume()
            return Not(self.parse_not())
        return self.parse_atom()

    def parse_atom(self) -> QueryNode:
        tok = self.peek()
        if tok is None:
            raise QuerySyntaxError("Неочікуваний кінець запиту")
        if tok == "(":
            self.consume()
            expr = self.parse_or()
            if self.peek() != ")":
                raise QuerySyntaxError("Очікувалась закриваюча дужка ')'")
            self.consume()
            return expr
        if tok == ")":
            raise QuerySyntaxError("Зайва закриваюча дужка ')'")
        if tok in ("AND", "OR"):
            raise QuerySyntaxError(f"Неочікуваний оператор '{tok}'")

        self.consume()
        if tok.startswith('"') and tok.endswith('"') and len(tok) >= 2:
            words = tuple(tokenize(tok[1:-1]))
            if not words:
                raise QuerySyntaxError("Порожня фраза в лапках")
            return Phrase(words)

        norm_tokens = list(tokenize(tok))
        if not norm_tokens:
            raise QuerySyntaxError(f"Некоректний терм: '{tok}'")
        if len(norm_tokens) == 1:
            return Term(norm_tokens[0])
        node: QueryNode = Term(norm_tokens[0])
        for extra in norm_tokens[1:]:
            node = And(node, Term(extra))
        return node


def parse_query(text: str) -> QueryNode:
    if '"' in text and text.count('"') % 2 != 0:
        raise QuerySyntaxError("Незакриті лапки у запиті")
    raw_tokens = _TOKEN_RE.findall(text)
    if not raw_tokens:
        raise QuerySyntaxError("Порожній пошуковий запит")
    parser = _Parser(raw_tokens)
    node = parser.parse_or()
    if parser.peek() is not None:
        raise QuerySyntaxError(f"Неочікуваний токен: '{parser.peek()}'")
    return node
