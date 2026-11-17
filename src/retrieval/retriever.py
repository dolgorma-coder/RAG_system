"""Retrieval: load the saved index and search it, for one query or a dataset.

Degenerate input (empty query, ``k <= 0``) yields an empty result instead
of an error, so the CLI never crashes on it.
"""
import json
import pickle
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from src.indexing.bm25 import BM25
from src.models.models import (
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)


class CodebaseRetriever:
    """Loads a BM25 index and returns the most relevant source locations."""

    def __init__(
        self, index_path: str = "data/processed/bm25_index.pkl"
    ) -> None:
        """Load the saved index from disk.

        Args:
            index_path: Path of the pickled index.

        Raises:
            FileNotFoundError: If the index file does not exist.
            RuntimeError: If the index cannot be deserialized.
        """
        self.index_path = Path(index_path)
        self.index: Optional[BM25] = None
        self._load_index()

    def _load_index(self) -> None:
        """Safely load the pickled index."""
        if not self.index_path.is_file():
            raise FileNotFoundError(
                f"Index file not found at {self.index_path}. "
                "Please run the index command first."
            )
        try:
            with open(self.index_path, "rb") as file:
                self.index = pickle.load(file)
        except Exception as error:
            raise RuntimeError(
                f"Failed to load the index {self.index_path}: {error}"
            ) from error

    def retrieve(self, query: str, k: int = 5) -> List[MinimalSource]:
        """Return the top-k source locations for a query.

        Args:
            query: Question text.
            k: Number of sources to return.

        Returns:
            Ranked sources; empty for an empty query or ``k <= 0``.
        """
        if not query.strip() or k <= 0:
            return []
        if self.index is None:
            raise RuntimeError("Index is not loaded.")
        chunks = self.index.search(query, k=k)
        return [chunk.source for chunk in chunks]

    def search_dataset(
        self, dataset_path: str, save_directory: str, k: int = 5
    ) -> Path:
        """Search every question of a JSON dataset and save the results.

        Args:
            dataset_path: JSON file with a ``RagDataset``.
            save_directory: Directory for the output file, which keeps the
                name of the dataset file.
            k: Number of sources per question.

        Returns:
            Path of the written results file.

        Raises:
            FileNotFoundError: If the dataset file does not exist.
            ValueError: If the dataset is not valid JSON or not a dataset.
        """
        dataset_file = Path(dataset_path)
        if not dataset_file.is_file():
            raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

        try:
            with open(dataset_file, "r", encoding="utf-8") as file:
                dataset = RagDataset.model_validate(json.load(file))
        except (ValueError, UnicodeDecodeError) as error:
            raise ValueError(
                f"Malformed dataset {dataset_file}: {error}"
            ) from error

        results: List[MinimalSearchResults] = []
        for item in tqdm(dataset.rag_questions, desc="Searching dataset"):
            results.append(
                MinimalSearchResults(
                    question_id=item.question_id,
                    question=item.question,
                    retrieved_sources=self.retrieve(item.question, k),
                )
            )

        output = StudentSearchResults(search_results=results, k=k)
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        output_file = save_dir / dataset_file.name
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(output.model_dump_json(indent=4))

        print(f"Saved student_search_results to {output_file}")
        return output_file
