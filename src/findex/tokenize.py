import re
import unicodedata
from collections.abc import Iterator

WORD_PATTERN = re.compile(r"\w+")


def tokenize(text: str) -> Iterator[str]:
    clean_text = unicodedata.normalize("NFC", text).casefold()
    for match in WORD_PATTERN.finditer(clean_text):
        yield match.group()
