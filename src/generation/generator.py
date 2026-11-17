"""Answer generation with a small local model (Qwen3-0.6B) on CPU."""
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.indexing.ingestion import read_text_exact
from src.models.models import (
    MinimalAnswer,
    MinimalSource,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)

SYSTEM_PROMPT = (
    "You answer questions about the vLLM codebase and documentation. "
    "Use ONLY the numbered context snippets provided by the user. "
    "If the snippets do not contain the answer, say that the provided "
    "context does not contain it. Do not invent facts, options or "
    "function names. Answer concisely and mention the relevant file, "
    "class, function or option names."
)
NO_CONTEXT_ANSWER = "No relevant source context could be read."
ERROR_ANSWER = "The answer could not be generated."
MAX_SNIPPET_CHARS = 2000
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)

Snippet = Tuple[MinimalSource, str]


class CodebaseAnswerGenerator:
    """Generates grounded answers from retrieved source locations."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        max_new_tokens: int = 256,
        max_sources: int = 3,
    ) -> None:
        """Load the model and tokenizer on the CPU.

        Args:
            model_name: Hugging Face model identifier.
            max_new_tokens: Maximum length of a generated answer.
            max_sources: How many of the retrieved sources go into the
                prompt (keeps the prompt short enough for CPU inference).
        """
        self.max_new_tokens = max_new_tokens
        self.max_sources = max_sources
        print(f"[Generation] Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Typed as Any: the transformers type stubs reject valid calls to
        # .to() and .generate() on models returned by from_pretrained().
        self.model: Any = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.float()  # float32 weights; converts in place
        self.model.eval()
        print("[Generation] Model loaded.")

    @staticmethod
    def _read_snippets(sources: List[MinimalSource]) -> List[Snippet]:
        """Read the text of each source location from disk.

        Args:
            sources: Retrieved source locations.

        Returns:
            Pairs of source and snippet text, skipping unreadable or empty
            sources.
        """
        cache: Dict[str, str] = {}
        snippets: List[Snippet] = []
        for source in sources:
            try:
                if source.file_path not in cache:
                    cache[source.file_path] = read_text_exact(
                        Path(source.file_path)
                    )
            except (OSError, UnicodeDecodeError) as error:
                tqdm.write(f"Warning: cannot read {source.file_path}: {error}")
                continue
            text = cache[source.file_path]
            snippet = text[
                source.first_character_index:source.last_character_index
            ]
            if snippet.strip():
                snippets.append((source, snippet[:MAX_SNIPPET_CHARS]))
        return snippets

    @staticmethod
    def _build_messages(
        question: str, snippets: List[Snippet]
    ) -> List[Dict[str, str]]:
        """Build the chat messages: instructions, context and question."""
        blocks = [
            f"[{number}] {source.file_path}\n{text}"
            for number, (source, text) in enumerate(snippets, 1)
        ]
        user_content = (
            "Context:\n\n" + "\n\n".join(blocks)
            + f"\n\nQuestion: {question}\nAnswer:"
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def generate_answer(
        self, question: str, retrieved_sources: List[MinimalSource]
    ) -> str:
        """Answer a question from its retrieved sources.

        Args:
            question: Question text.
            retrieved_sources: Ranked source locations.

        Returns:
            The generated answer text.
        """
        snippets = self._read_snippets(retrieved_sources[:self.max_sources])
        if not snippets:
            return NO_CONTEXT_ANSWER

        prompt = str(
            self.tokenizer.apply_chat_template(
                self._build_messages(question, snippets),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        text = str(self.tokenizer.decode(new_tokens, skip_special_tokens=True))
        return _THINK_BLOCK.sub("", text).strip()

    def answer_dataset(
        self, student_search_results_path: str, save_directory: str
    ) -> Path:
        """Generate answers for a saved search results file.

        Args:
            student_search_results_path: JSON file with search results.
            save_directory: Directory for the output file, which keeps the
                name of the input file.

        Returns:
            Path of the written file.

        Raises:
            FileNotFoundError: If the results file does not exist.
            ValueError: If the file is malformed.
        """
        results_file = Path(student_search_results_path)
        if not results_file.is_file():
            raise FileNotFoundError(
                f"Search results file not found: {results_file}"
            )
        try:
            search_results = StudentSearchResults.model_validate_json(
                results_file.read_text(encoding="utf-8")
            )
        except (ValueError, UnicodeDecodeError) as error:
            raise ValueError(
                f"Malformed search results {results_file}: {error}"
            ) from error

        total = len(search_results.search_results)
        print(f"Loaded {total} questions from search results.")

        answered: List[MinimalAnswer] = []
        for entry in tqdm(
            search_results.search_results, desc="Generating answers"
        ):
            try:
                answer = self.generate_answer(
                    entry.question, entry.retrieved_sources
                )
            except Exception as error:
                tqdm.write(f"Generation failed for {entry.question_id}: "
                           f"{error}")
                answer = ERROR_ANSWER
            answered.append(
                MinimalAnswer(
                    question_id=entry.question_id,
                    question=entry.question,
                    retrieved_sources=entry.retrieved_sources,
                    answer=answer,
                )
            )

        output = StudentSearchResultsAndAnswer(
            search_results=answered, k=search_results.k
        )
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        output_file = save_dir / results_file.name
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(output.model_dump_json(indent=4))

        print(f"Processed {len(answered)} of {total} questions")
        print(f"Saved student_search_results_and_answer to {output_file}")
        return output_file
