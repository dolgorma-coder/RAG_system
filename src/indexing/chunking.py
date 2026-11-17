"""Text chunking strategies for the RAG indexer.

Every strategy works with *spans*: ``(start, end)`` character offsets in the
original text. A chunk's content is always ``text[start:end]``, so the
reported indices can never drift away from the real file content.
"""
import ast
import os
import re
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel

from src.models.models import MinimalSource

Span = Tuple[int, int]

MAX_ALLOWED_CHUNK_SIZE = 2000

_NEWLINE = re.compile(r"\r\n|\r|\n")
_BLANK_LINES = re.compile(r"\r?\n(?:[ \t]*\r?\n)+")
_CODE_START = re.compile(r"(?:async\s+def|def|class)\s|@")


class Chunk(BaseModel):
    """A piece of text together with its location in the source file."""

    content: str
    source: MinimalSource


def _split_long(text: str, start: int, end: int, limit: int) -> List[Span]:
    """Cut one span into pieces of at most ``limit`` characters.

    Cuts are made after a newline when one is available in the second half
    of the window, otherwise the span is cut hard at the limit.
    """
    spans: List[Span] = []
    pos = start
    while end - pos > limit:
        idx = text.rfind("\n", pos, pos + limit)
        cut = idx + 1 if idx + 1 > pos + limit // 2 else pos + limit
        spans.append((pos, cut))
        pos = cut
    spans.append((pos, end))
    return spans


def _fit_spans(text: str, spans: List[Span], limit: int) -> List[Span]:
    """Split every span that is longer than ``limit``."""
    fitted: List[Span] = []
    for start, end in spans:
        if end - start > limit:
            fitted.extend(_split_long(text, start, end, limit))
        else:
            fitted.append((start, end))
    return fitted


def _merge_spans(
    text: str,
    spans: List[Span],
    limit: int,
    break_before: Optional[Callable[[str], bool]] = None,
) -> List[Span]:
    """Greedily merge neighbouring spans while the result fits ``limit``.

    If ``break_before`` returns True for the next span and the current chunk
    is already reasonably big (a third of the limit), a new chunk is started
    instead, so logical units are not glued together needlessly.
    """
    merged: List[Span] = []
    for start, end in spans:
        if merged:
            m_start, m_end = merged[-1]
            fits = end - m_start <= limit
            brk = (
                break_before is not None
                and m_end - m_start >= limit // 3
                and break_before(text[start:end])
            )
            if fits and not brk:
                merged[-1] = (m_start, end)
                continue
        merged.append((start, end))
    return merged


def _spans_from_cuts(cuts: List[int], length: int) -> List[Span]:
    """Turn sorted cut offsets into contiguous spans covering the text."""
    points = sorted(set(cuts) | {0, length})
    return [(a, b) for a, b in zip(points, points[1:]) if b > a]


def _to_chunks(file_path: str, text: str, spans: List[Span]) -> List[Chunk]:
    """Build Chunk objects from spans, skipping whitespace-only pieces."""
    chunks: List[Chunk] = []
    for start, end in spans:
        content = text[start:end]
        if not content.strip():
            continue
        source = MinimalSource(
            file_path=file_path,
            first_character_index=start,
            last_character_index=end,
        )
        chunks.append(Chunk(content=content, source=source))
    return chunks


class BaseChunk(ABC):
    """Abstract chunking strategy."""

    def __init__(self, max_chunk_size: int = MAX_ALLOWED_CHUNK_SIZE) -> None:
        """Store the size limit.

        Args:
            max_chunk_size: Maximum chunk length in characters (1..2000).

        Raises:
            ValueError: If the limit is not in the allowed range.
        """
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be positive.")
        if max_chunk_size > MAX_ALLOWED_CHUNK_SIZE:
            raise ValueError(
                f"max_chunk_size cannot exceed {MAX_ALLOWED_CHUNK_SIZE}."
            )
        self.max_chunk_size = max_chunk_size

    @abstractmethod
    def chunk_text(self, file_path: str, text: str) -> List[Chunk]:
        """Split a file into chunks.

        Args:
            file_path: Path of the file, copied into each chunk's source.
            text: Full text of the file.

        Returns:
            Chunks, each at most ``max_chunk_size`` characters long.
        """


class CodeChunk(BaseChunk):
    """Chunking strategy for Python files.

    Cut points are placed at the start of top-level statements and of class
    members (found with ``ast``), so functions and methods stay whole when
    they fit. Oversized units are split by lines. If the file cannot be
    parsed, every line start is a cut point instead.
    """

    @staticmethod
    def _cut_points(text: str) -> List[int]:
        """Return character offsets where a new code unit starts."""
        starts = [0] + [m.end() for m in _NEWLINE.finditer(text)]
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            return starts

        cuts: List[int] = []

        def visit(body: List[ast.stmt]) -> None:
            for node in body:
                first = node.lineno
                if isinstance(
                    node,
                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                ) and node.decorator_list:
                    first = min(first, node.decorator_list[0].lineno)
                idx = first - 1
                # Keep comments right above a definition with it.
                while idx > 0 and text[starts[idx - 1]:starts[idx]].lstrip(
                ).startswith("#"):
                    idx -= 1
                cuts.append(starts[idx])
                if isinstance(node, ast.ClassDef):
                    visit(node.body)

        visit(tree.body)
        return cuts

    def chunk_text(self, file_path: str, text: str) -> List[Chunk]:
        """Split Python source into chunks along definition boundaries."""
        if not text:
            return []
        limit = self.max_chunk_size
        spans = _spans_from_cuts(self._cut_points(text), len(text))
        spans = _fit_spans(text, spans, limit)
        spans = _merge_spans(
            text,
            spans,
            limit,
            lambda piece: bool(_CODE_START.match(piece.lstrip())),
        )
        return _to_chunks(file_path, text, spans)


class MarkdownChunk(BaseChunk):
    """Chunking strategy for Markdown and plain text.

    Text is divided into paragraphs (separated by blank lines). A heading
    line is always kept together with the paragraph that follows it, and a
    new chunk is preferably started at a heading. Oversized paragraphs are
    split by lines.
    """

    @staticmethod
    def _is_heading(piece: str) -> bool:
        """Tell whether a piece of text starts with a Markdown heading."""
        return piece.lstrip().startswith("#")

    def _paragraph_spans(self, text: str) -> List[Span]:
        """Split text into paragraph spans, gluing headings to their body."""
        cuts = [m.end() for m in _BLANK_LINES.finditer(text)]
        spans = _spans_from_cuts(cuts, len(text))
        glued: List[Span] = []
        pending: Optional[int] = None
        for start, end in spans:
            if pending is not None:
                start = pending
                pending = None
            piece = text[start:end]
            single_line = len(piece.strip().splitlines()) == 1
            if self._is_heading(piece) and single_line and end < len(text):
                pending = start
                continue
            glued.append((start, end))
        if pending is not None:
            glued.append((pending, len(text)))
        return glued

    def chunk_text(self, file_path: str, text: str) -> List[Chunk]:
        """Split Markdown or plain text into chunks."""
        if not text:
            return []
        limit = self.max_chunk_size
        spans = _fit_spans(text, self._paragraph_spans(text), limit)
        spans = _merge_spans(text, spans, limit, self._is_heading)
        return _to_chunks(file_path, text, spans)


class ChunkFactory:
    """Choose a chunking strategy from the file extension."""

    _EXTENSIONS: Dict[str, Type[BaseChunk]] = {
        ".py": CodeChunk,
        ".md": MarkdownChunk,
        ".markdown": MarkdownChunk,
        ".txt": MarkdownChunk,
        ".rst": MarkdownChunk,
    }

    @classmethod
    def from_file(
        cls,
        file_path: str,
        max_chunk_size: int = MAX_ALLOWED_CHUNK_SIZE,
    ) -> BaseChunk:
        """Return a chunker suited to the given file.

        Args:
            file_path: Path whose extension selects the strategy.
            max_chunk_size: Maximum chunk length in characters.

        Returns:
            A chunker instance; text chunking is the default.
        """
        _, ext = os.path.splitext(file_path.lower())
        chunker_class = cls._EXTENSIONS.get(ext, MarkdownChunk)
        return chunker_class(max_chunk_size=max_chunk_size)
