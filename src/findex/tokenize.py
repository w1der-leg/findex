
from collections.abc import Iterator
import re
import unicodedata

WORD_PATTERN = re.compile(r"\w+")


def tokenize(text: str) -> Iterator[str]:
    clear_text = unicodedata.normalize("NFC", text).casefold()
    for match in WORD_PATTERN.finditer(clear_text):
        yield match.group()