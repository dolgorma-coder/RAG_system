"""Command-line interface of the RAG pipeline (built with Python Fire).

Usage: ``uv run python -m src <command> [options]``
"""
import sys
from typing import Any, NoReturn

import fire

from src.indexing.indexer import CodebaseIndexer
from src.retrieval.retriever import CodebaseRetriever

DEFAULT_INDEX_PATH = "data/processed/bm25_index.pkl"
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_SEARCH_DIR = "data/output/search_results"
DEFAULT_ANSWER_DIR = "data/output/search_results_and_answer"


def _fail(message: str) -> NoReturn:
    """Print an error message and exit with a non-zero status."""
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(1)


def _to_int(name: str, value: Any) -> int:
    """Convert a CLI value to int or exit with a clear message."""
    try:
        return int(value)
    except (TypeError, ValueError):
        _fail(f"--{name} must be an integer, got {value!r}.")


class RAGPipelineCLI:
    """Commands of the RAG pipeline over the vLLM repository."""

    def index(
        self,
        max_chunk_size: int = 2000,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
    ) -> None:
        """Ingest the corpus and build the index.

        Args:
            max_chunk_size: Maximum chunk length in characters (1..2000).
            raw_dir: Directory with the corpus.
            processed_dir: Directory where the index is saved.
        """
        size = _to_int("max_chunk_size", max_chunk_size)
        try:
            CodebaseIndexer(raw_dir, processed_dir).build_index(size)
        except Exception as error:
            _fail(f"indexing failed: {error}")

    def search(
        self,
        query: str = "",
        k: int = 5,
        index_path: str = DEFAULT_INDEX_PATH,
    ) -> None:
        """Print the top-k sources for a single query.

        Args:
            query: Question text.
            k: Number of sources to return.
            index_path: Path of the saved index.
        """
        query = str(query)
        k_value = _to_int("k", k)
        if not query.strip():
            print("Warning: empty query, no sources returned.")
            return
        if k_value <= 0:
            print("Warning: k must be greater than 0, no sources returned.")
            return
        try:
            sources = CodebaseRetriever(index_path).retrieve(query, k_value)
        except Exception as error:
            _fail(f"search failed: {error}")

        print(f"Top-{k_value} sources for query '{query}':")
        for rank, source in enumerate(sources, 1):
            print(
                f"{rank}. {source.file_path} "
                f"[{source.first_character_index}:"
                f"{source.last_character_index}]"
            )

    def search_dataset(
        self,
        dataset_path: str,
        save_directory: str = DEFAULT_SEARCH_DIR,
        k: int = 5,
        index_path: str = DEFAULT_INDEX_PATH,
    ) -> None:
        """Search a whole dataset and save a StudentSearchResults file.

        Args:
            dataset_path: JSON file with the questions.
            save_directory: Directory for the output file.
            k: Number of sources per question.
            index_path: Path of the saved index.
        """
        k_value = _to_int("k", k)
        if k_value <= 0:
            print("Warning: k must be greater than 0, nothing to do.")
            return
        try:
            CodebaseRetriever(index_path).search_dataset(
                dataset_path, save_directory, k_value
            )
        except Exception as error:
            _fail(f"dataset search failed: {error}")

    def answer(
        self,
        query: str = "",
        k: int = 5,
        index_path: str = DEFAULT_INDEX_PATH,
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        """Answer a single question using the retrieved context.

        Args:
            query: Question text.
            k: Number of sources to retrieve.
            index_path: Path of the saved index.
            model_name: Hugging Face model used for generation.
        """
        query = str(query)
        k_value = _to_int("k", k)
        if not query.strip():
            print("Warning: empty query, nothing to answer.")
            return
        if k_value <= 0:
            print("Warning: k must be greater than 0, nothing to answer.")
            return
        try:
            # Imported here: loading torch is slow and not needed elsewhere.
            from src.generation.generator import CodebaseAnswerGenerator

            sources = CodebaseRetriever(index_path).retrieve(query, k_value)
            generator = CodebaseAnswerGenerator(model_name=model_name)
            text = generator.generate_answer(query, sources)
        except Exception as error:
            _fail(f"answer generation failed: {error}")

        print(f"[Question]: {query}")
        print(f"[Answer]: {text}")

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = DEFAULT_ANSWER_DIR,
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        """Generate answers for saved search results.

        Args:
            student_search_results_path: JSON file from search_dataset.
            save_directory: Directory for the output file.
            model_name: Hugging Face model used for generation.
        """
        try:
            from src.generation.generator import CodebaseAnswerGenerator

            CodebaseAnswerGenerator(model_name=model_name).answer_dataset(
                student_search_results_path, save_directory
            )
        except Exception as error:
            _fail(f"dataset answer generation failed: {error}")

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
    ) -> None:
        """Report recall@k against a ground-truth dataset.

        Args:
            student_search_results_path: JSON file from search_dataset.
            dataset_path: Ground-truth dataset with sources.
        """
        from src.evaluation.evaluator import evaluate_recall

        try:
            recalls = evaluate_recall(
                student_search_results_path, dataset_path
            )
        except Exception as error:
            _fail(f"evaluation failed: {error}")

        print("Evaluation Results")
        print("=" * 40)
        print("  ".join(f"Recall@{k}: {v:.3f}" for k, v in recalls.items()))


def main() -> None:
    """Expose the pipeline commands through Python Fire."""
    fire.Fire(RAGPipelineCLI)


if __name__ == "__main__":
    main()
