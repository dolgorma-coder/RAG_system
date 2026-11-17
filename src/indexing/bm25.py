"""BM25 lexical index over code and documentation chunks."""
import re
from typing import List, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from src.indexing.chunking import Chunk

_WORD = re.compile(r"[A-Za-z0-9_]+")
_SUBWORD = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")
_QUERY_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "of", "to", "in",
    "on", "for", "and", "or", "by", "as", "at", "from", "with", "it", "its",
    "this", "that", "these", "those", "how", "what", "which", "who", "when",
    "where", "why", "does", "do", "did", "can", "could", "should", "would",
    "there", "i", "me", "my", "we", "you", "your",
})


def tokenize(text: str) -> List[str]:
    """Split text into lowercase search tokens.

    Identifiers are indexed twice: as a whole (``get_num_requests``) so a
    question quoting it verbatim matches, and split into parts (``get``,
    ``num``, ``requests``, also at camelCase boundaries) so a paraphrase
    using plain words matches too. One-character tokens are dropped.

    Args:
        text: Text to tokenize.

    Returns:
        List of tokens, possibly with repetitions.
    """
    tokens: List[str] = []
    for word in _WORD.findall(text):
        if word.isalpha() and word.islower():
            if len(word) > 1:
                tokens.append(word)
            continue
        lowered = word.lower()
        parts = [part.lower() for part in _SUBWORD.findall(word)]
        if parts != [lowered]:
            tokens.append(lowered)
        tokens.extend(part for part in parts if len(part) > 1)
    return tokens


def tokenize_query(text: str) -> List[str]:
    """Tokenize a question and drop common English stop words.

    Args:
        text: Question text.

    Returns:
        Query tokens without stop words.
    """
    return [t for t in tokenize(text) if t not in _QUERY_STOPWORDS]


class BM25:
    """BM25 (Okapi) index built from chunks."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        """Create an empty index.

        Args:
            k1: Term frequency saturation parameter.
            b: Document length normalization parameter.
        """
        self.k1 = k1
        self.b = b
        self.model: Optional[BM25Okapi] = None
        self.chunks: List[Chunk] = []

    def build(self, chunks: List[Chunk]) -> None:
        """Build the index.

        The file path of each chunk is indexed together with its content,
        so a question mentioning a feature finds the matching file even if
        the chunk itself never repeats the name.

        Args:
            chunks: All chunks of the corpus.

        Raises:
            ValueError: If there are no chunks.
        """
        if not chunks:
            raise ValueError("Cannot build a BM25 index without chunks.")
        self.chunks = chunks
        corpus = [
            tokenize(chunk.source.file_path) + tokenize(chunk.content)
            for chunk in chunks
        ]
        self.model = BM25Okapi(corpus, k1=self.k1, b=self.b)

    def search(self, query: str, k: int = 5) -> List[Chunk]:
        """Return the k most relevant chunks, best first.

        Args:
            query: Question text.
            k: Number of chunks to return.

        Returns:
            Chunks with a positive score; empty if the query has no usable
            words or matches nothing.

        Raises:
            RuntimeError: If the index has not been built.
        """
        if self.model is None:
            raise RuntimeError("The index is not built. Call build() first.")
        tokens = tokenize_query(query)
        if k <= 0 or not tokens:
            return []

        scores = np.asarray(self.model.get_scores(tokens))
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        ranked = sorted(top.tolist(), key=lambda i: (-scores[i], i))
        return [self.chunks[i] for i in ranked if scores[i] > 0]
