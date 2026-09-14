# findex (Lab 01)

Streaming document corpus loader, tokenizer, and vocabulary stats collector.

## Corpus

The corpus consists of local game configuration and shader logs located in `data/`.
This folder is excluded from version control via `.gitignore`.

## Tokenization Policy

- **Normalization**: Unicode NFC normalization via `unicodedata.normalize("NFC", text)` to ensure composed and decomposed characters match consistently.
- **Casing**: Case folding via `str.casefold()` for caseless matching.
- **Apostrophes & Hyphens**: Stripped / treated as token delimiters by default (tokens are matched using `\w+`).
- **Digits**: Preserved as part of alphanumeric tokens.

## Eager vs. Lazy Benchmark

| Version          | Documents | Peak memory | Elapsed   |
|------------------|-----------|-------------|-----------|
| eager (lists)    | 7         | 2.23 MB     | 0.1506 s  |
| lazy (generators)| 7         | 0.55 MB     | 0.1282 s  |

### Memory & Performance Analysis

The eager implementation loads all documents from the filesystem simultaneously into a Python list of strings, and then materializes all extracted tokens into an intermediate list of lists before passing them to `collections.Counter`. This duplicates the raw text and keeps all token objects alive in RAM simultaneously, causing a high memory peak (2.23 MB). 

In contrast, the lazy pipeline uses Python generators (`yield`). It opens and holds only one document in memory at any given time, tokenizing it as a stream. Each token is consumed and recorded in `Counter` immediately, allowing previous document strings and transient token instances to be freed by the garbage collector. As a result, the peak memory (0.55 MB) reflects only the vocabulary dictionary and a single active buffer.