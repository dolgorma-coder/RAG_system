.PHONY: install run debug clean fclean lint lint-strict

DATASET ?= data/datasets/UnansweredQuestions/dataset_docs_public.json
K ?= 10
SAVE_DIR ?= data/output/search_results/UnansweredQuestions
MAX_CHUNK_SIZE ?= 2000

install:
	uv sync

run:
	uv run python -m src index --max_chunk_size $(MAX_CHUNK_SIZE)
	uv run python -m src search_dataset --dataset_path $(DATASET) \
		--k $(K) --save_directory $(SAVE_DIR)

debug:
	uv run python -m pdb -m src index --max_chunk_size $(MAX_CHUNK_SIZE)

clean:
	find . -type d -name __pycache__ -not -path "./.venv/*" -prune -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache

fclean: clean
	rm -rf data/processed/* data/output/*

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any \
		--warn-unused-ignores --ignore-missing-imports \
		--disallow-untyped-defs --check-untyped-defs
