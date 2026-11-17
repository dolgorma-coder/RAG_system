"""Recall@k of saved search results against a ground-truth dataset.

This is only a local approximation of the official moulinette: a correct
source is found when one of the top-k results lies in the same file and has
an intersection-over-union of at least 0.05 with it.
"""
import json
from pathlib import Path
from typing import Dict, List, Sequence

from src.models.models import (
    AnsweredQuestion,
    MinimalSource,
    StudentSearchResults,
)

IOU_THRESHOLD = 0.05
DEFAULT_KS = (1, 3, 5, 10)


def sources_overlap(
    found: MinimalSource,
    expected: MinimalSource,
    threshold: float = IOU_THRESHOLD,
) -> bool:
    """Tell whether a retrieved source matches an expected one.

    Args:
        found: Retrieved source.
        expected: Ground-truth source.
        threshold: Minimum intersection-over-union.

    Returns:
        True if both are in the same file and overlap enough.
    """
    if found.file_path != expected.file_path:
        return False
    inter = min(found.last_character_index, expected.last_character_index) \
        - max(found.first_character_index, expected.first_character_index)
    if inter <= 0:
        return False
    union = (
        (found.last_character_index - found.first_character_index)
        + (expected.last_character_index - expected.first_character_index)
        - inter
    )
    return union > 0 and inter / union >= threshold


def _load_ground_truth(dataset_path: str) -> Dict[str, List[MinimalSource]]:
    """Read the expected sources of every answered question."""
    with open(dataset_path, "r", encoding="utf-8") as file:
        raw = json.load(file)
    items = raw.get("rag_questions") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        raise ValueError("Dataset must contain a 'rag_questions' list.")
    truth: Dict[str, List[MinimalSource]] = {}
    for item in items:
        if isinstance(item, dict) and "sources" in item:
            question = AnsweredQuestion.model_validate(item)
            truth[question.question_id] = question.sources
    return truth


def evaluate_recall(
    student_search_results_path: str,
    dataset_path: str,
    ks: Sequence[int] = DEFAULT_KS,
) -> Dict[int, float]:
    """Compute the mean recall@k over all questions of the dataset.

    Args:
        student_search_results_path: JSON with ``StudentSearchResults``.
        dataset_path: JSON ground-truth dataset with sources.
        ks: Values of k to report.

    Returns:
        Mapping from k to recall.

    Raises:
        FileNotFoundError: If one of the files does not exist.
        ValueError: If a file is malformed or no question ids match.
    """
    if not Path(student_search_results_path).is_file():
        raise FileNotFoundError(
            f"Search results file not found: {student_search_results_path}"
        )
    with open(student_search_results_path, "r", encoding="utf-8") as file:
        results = StudentSearchResults.model_validate_json(file.read())
    truth = _load_ground_truth(dataset_path)

    totals = {k: 0.0 for k in ks}
    counted = 0
    for entry in results.search_results:
        expected = truth.get(entry.question_id)
        if not expected:
            continue
        counted += 1
        for k in ks:
            top = entry.retrieved_sources[:k]
            found = sum(
                1 for target in expected
                if any(sources_overlap(source, target) for source in top)
            )
            totals[k] += found / len(expected)

    if counted == 0:
        raise ValueError("No question ids of the results match the dataset.")
    return {k: totals[k] / counted for k in ks}
