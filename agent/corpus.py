"""
corpus.py — Load and index the local support corpus.

Walks data/{hackerrank,claude,visa}/ and indexes all text files.
Provides a retriever that returns the top-k most relevant chunks
for a query using TF-IDF cosine similarity (no network needed).
"""

from __future__ import annotations

import os
import re
import math
import json
from dataclasses import dataclass, field
from collections import defaultdict
from typing import List, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A single text chunk from the corpus."""
    source: str          # relative file path
    domain: str          # hackerrank | claude | visa | unknown
    title: str           # filename stem or first heading
    text: str            # raw text content
    tokens: List[str] = field(default_factory=list, repr=False)


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DOMAIN_MAP = {
    "hackerrank": "HackerRank",
    "claude":     "Claude",
    "visa":       "Visa",
}

_STOP = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would could should may might must shall can i we you they he she it "
    "of in on at to for with by from as into that this these those and or but "
    "not no nor so yet both either neither just also very any all some more most "
    "its their our your my his her its".split()
)


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    return [t for t in tokens if t not in _STOP and len(t) > 1]


def _chunk_text(text: str, max_words: int = 200, overlap: int = 40) -> List[str]:
    """Split text into overlapping windows of ~max_words words."""
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks = []
    step = max_words - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + max_words])
        chunks.append(chunk)
        if i + max_words >= len(words):
            break
    return chunks


def _extract_title(text: str, filename: str) -> str:
    for line in text.splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            return line[:80]
    return os.path.splitext(filename)[0]


# ---------------------------------------------------------------------------
# TF-IDF index
# ---------------------------------------------------------------------------

class CorpusIndex:
    def __init__(self):
        self.chunks: List[Chunk] = []
        self._df: Dict[str, int] = defaultdict(int)   # doc frequency
        self._built = False

    # ---- loading -----------------------------------------------------------

    def load_directory(self, corpus_dir: str):
        """Walk corpus_dir and ingest all readable text files."""
        corpus_dir = os.path.abspath(corpus_dir)
        if not os.path.isdir(corpus_dir):
            raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

        loaded = 0
        for root, _, files in os.walk(corpus_dir):
            for fname in files:
                if not self._is_text_file(fname):
                    continue
                fpath = os.path.join(root, fname)
                rel   = os.path.relpath(fpath, corpus_dir)
                domain = self._infer_domain(rel)
                try:
                    text = self._read_file(fpath)
                except Exception:
                    continue
                if not text.strip():
                    continue

                title = _extract_title(text, fname)
                for chunk_text in _chunk_text(text):
                    chunk = Chunk(
                        source=rel,
                        domain=domain,
                        title=title,
                        text=chunk_text,
                    )
                    chunk.tokens = _tokenize(chunk_text)
                    self.chunks.append(chunk)
                loaded += 1

        self._build_index()
        return loaded

    def _is_text_file(self, fname: str) -> bool:
        ext = os.path.splitext(fname)[1].lower()
        return ext in {".txt", ".md", ".html", ".json", ".csv", ".rst", ""}

    def _read_file(self, path: str) -> str:
        # Try UTF-8 first, fall back to latin-1
        for enc in ("utf-8", "latin-1"):
            try:
                with open(path, encoding=enc, errors="replace") as f:
                    raw = f.read()
                # Strip HTML tags if HTML file
                if path.endswith(".html"):
                    raw = re.sub(r"<[^>]+>", " ", raw)
                    raw = re.sub(r"&[a-z]+;", " ", raw)
                return re.sub(r"\s+", " ", raw).strip()
            except Exception:
                continue
        return ""

    def _infer_domain(self, rel_path: str) -> str:
        parts = rel_path.lower().replace("\\", "/").split("/")
        for part in parts:
            if part in DOMAIN_MAP:
                return DOMAIN_MAP[part]
        return "unknown"

    # ---- index build -------------------------------------------------------

    def _build_index(self):
        """Compute IDF weights."""
        N = len(self.chunks)
        if N == 0:
            return
        for chunk in self.chunks:
            for term in set(chunk.tokens):
                self._df[term] += 1
        self._N = N
        self._built = True

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log((self._N + 1) / (df + 1)) + 1.0

    def _tf_idf_vec(self, tokens: List[str]) -> Dict[str, float]:
        tf: Dict[str, float] = defaultdict(float)
        for t in tokens:
            tf[t] += 1.0
        n = len(tokens) or 1
        return {t: (cnt / n) * self._idf(t) for t, cnt in tf.items()}

    def _cosine(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        dot = sum(a.get(t, 0) * v for t, v in b.items())
        norm_a = math.sqrt(sum(v * v for v in a.values())) or 1e-9
        norm_b = math.sqrt(sum(v * v for v in b.values())) or 1e-9
        return dot / (norm_a * norm_b)

    # ---- retrieval ---------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        domain_filter: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """Return top_k chunks most relevant to query."""
        if not self._built or not self.chunks:
            return []

        q_tokens = _tokenize(query)
        q_vec    = self._tf_idf_vec(q_tokens)

        results = []
        for chunk in self.chunks:
            if domain_filter and chunk.domain != domain_filter:
                continue
            c_vec  = self._tf_idf_vec(chunk.tokens)
            score  = self._cosine(q_vec, c_vec)
            results.append(RetrievalResult(chunk=chunk, score=score))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def domain_chunks(self, domain: str) -> List[Chunk]:
        return [c for c in self.chunks if c.domain == domain]

    @property
    def stats(self) -> Dict:
        by_domain: Dict[str, int] = defaultdict(int)
        for c in self.chunks:
            by_domain[c.domain] += 1
        return {"total_chunks": len(self.chunks), "by_domain": dict(by_domain)}
