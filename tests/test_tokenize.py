from findex.tokenize import tokenize


def test_basic_ascii():
    tokens = list(tokenize("Hello world"))
    assert tokens == ["hello", "world"]


def test_mixed_case():
    tokens = list(tokenize("PyThOn AnD c++"))
    assert tokens == ["python", "and", "c"]


def test_cyrillic():
    tokens = list(tokenize("Привіт, світе! Пошуковий рушій працює."))
    assert tokens == ["привіт", "світе", "пошуковий", "рушій", "працює"]


def test_unicode_nfc_normalization():
    # "cafe\u0301" - це 'e' + окремий комбінований наголос.
    # Після NFC воно нормалізується до цільного символу 'é'.
    decomposed = "cafe\u0301"
    composed = "café"
    assert list(tokenize(decomposed)) == [composed]


def test_punctuation_and_whitespace():
    tokens = list(tokenize("  one... two!? -- three \t\n four  "))
    assert tokens == ["one", "two", "three", "four"]


def test_empty_string():
    assert list(tokenize("")) == []
    assert list(tokenize("   \n\t   ")) == []


def test_numbers_and_alphanumeric():
    tokens = list(tokenize("Shader v1.20 and year 2026"))
    assert tokens == ["shader", "v1", "20", "and", "year", "2026"]
