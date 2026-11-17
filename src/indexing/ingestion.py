"""File discovery and exact text reading for the indexing stage."""
from pathlib import Path
from typing import List

VALID_EXTENSIONS = frozenset({".py", ".md", ".markdown", ".txt", ".rst"})
SKIP_DIRS = frozenset({".git", "__pycache__", "node_modules", ".venv"})


def read_text_exact(path: Path) -> str:
    """Read a UTF-8 text file without any newline translation.

    Character offsets used by the pipeline refer to this exact text, so
    CRLF line endings must be kept as they are.

    Args:
        path: File to read.

    Returns:
        The full text of the file as a single string.

    Raises:
        OSError: If the file cannot be read.
        UnicodeDecodeError: If the file is not valid UTF-8.
    """
    with open(path, "r", encoding="utf-8", newline="") as file:
        return file.read()


def to_corpus_path(path: Path) -> str:
    """Return the path as written in the corpus (POSIX, relative to cwd).

    The grader compares ``file_path`` verbatim, e.g.
    ``data/raw/vllm-0.10.1/docs/features/lora.md``.

    Args:
        path: Path of a file found in the raw data directory.

    Returns:
        The path with forward slashes, relative to the current directory
        whenever that is possible.
    """
    if path.is_absolute():
        try:
            path = path.relative_to(Path.cwd())
        except ValueError:
            pass
    return path.as_posix()


class CodebaseIngestor:
    """Finds the files of the corpus that must be indexed."""

    def __init__(self, raw_data_dir: str) -> None:
        """Remember the directory with the raw repository files.

        Args:
            raw_data_dir: Directory that contains the corpus.
        """
        self.raw_data_dir = Path(raw_data_dir)

    def get_files(self) -> List[Path]:
        """List all files suitable for indexing, in a stable order.

        Returns:
            Sorted paths of the files with a supported extension.

        Raises:
            FileNotFoundError: If the raw data directory does not exist.
        """
        if not self.raw_data_dir.is_dir():
            raise FileNotFoundError(
                f"Raw data directory not found: {self.raw_data_dir}"
            )

        files: List[Path] = []
        for path in self.raw_data_dir.rglob("*"):
            parent_dirs = path.relative_to(self.raw_data_dir).parts[:-1]
            if SKIP_DIRS.intersection(parent_dirs):
                continue
            if path.suffix.lower() in VALID_EXTENSIONS and path.is_file():
                files.append(path)
        return sorted(files)
