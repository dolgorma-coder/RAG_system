"""Indexing orchestrator: chunk the corpus, build BM25, save to disk."""
import pickle
import time
from pathlib import Path
from typing import List

from tqdm import tqdm

from src.indexing.bm25 import BM25
from src.indexing.chunking import MAX_ALLOWED_CHUNK_SIZE, Chunk, ChunkFactory
from src.indexing.ingestion import (
    CodebaseIngestor,
    read_text_exact,
    to_corpus_path,
)

INDEX_FILE_NAME = "bm25_index.pkl"


class CodebaseIndexer:
    """Traverses the repository, chunks files and builds a BM25 index."""

    def __init__(self, raw_data_dir: str, processed_data_dir: str) -> None:
        """Store the input and output locations.

        Args:
            raw_data_dir: Directory with the raw corpus.
            processed_data_dir: Directory where the index is saved.
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.index = BM25()
        self.ingestor = CodebaseIngestor(raw_data_dir)

    def build_index(self, max_chunk_size: int = 2000) -> Path:
        """Build the index for the whole corpus and save it.

        Args:
            max_chunk_size: Maximum chunk length in characters.

        Returns:
            Path of the saved index file.

        Raises:
            ValueError: If the chunk size is invalid or nothing was indexed.
            FileNotFoundError: If the raw data directory does not exist.
        """
        if not 0 < max_chunk_size <= MAX_ALLOWED_CHUNK_SIZE:
            raise ValueError(
                "max_chunk_size must be between 1 and "
                f"{MAX_ALLOWED_CHUNK_SIZE}, got {max_chunk_size}."
            )

        started = time.perf_counter()
        files = self.ingestor.get_files()
        if not files:
            raise ValueError(f"No indexable files in {self.raw_data_dir}.")

        all_chunks: List[Chunk] = []
        skipped = 0
        for path in tqdm(files, desc="Chunking files"):
            try:
                text = read_text_exact(path)
            except (OSError, UnicodeDecodeError) as error:
                tqdm.write(f"Skipping {path}: {error}")
                skipped += 1
                continue
            corpus_path = to_corpus_path(path)
            chunker = ChunkFactory.from_file(corpus_path, max_chunk_size)
            all_chunks.extend(chunker.chunk_text(corpus_path, text))

        if not all_chunks:
            raise ValueError("No chunks were created.")

        print(f"Indexing {len(all_chunks)} chunks...")
        self.index.build(all_chunks)

        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        output_file = self.processed_data_dir / INDEX_FILE_NAME
        with open(output_file, "wb") as file:
            pickle.dump(self.index, file)

        elapsed = time.perf_counter() - started
        print(
            f"Ingestion complete! {len(files) - skipped} files, "
            f"{len(all_chunks)} chunks in {elapsed:.1f}s. "
            f"Indices saved under {self.processed_data_dir}/"
        )
        return output_file
