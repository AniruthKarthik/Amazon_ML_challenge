PYTHON ?= python3
TRAIN_DIR ?= dataset/train
TEST_DIR ?= dataset/test
OUTPUT_DIR ?= chimera_submission/output
FOLDS ?= 5
SEED ?= 42

.PHONY: help install test run generate validate package clean

help:
	@echo "Amazon ML Challenge 2026 — Business Entity Resolution"
	@echo ""
	@echo "Available targets:"
	@echo "  make run        Run end-to-end pipeline and generate TSVs"
	@echo "  make validate   Validate generated output TSVs using official validator"
	@echo "  make test       Run the automated pytest test suite"
	@echo "  make install    Install required python dependencies"
	@echo "  make package    Package submission zip archive"
	@echo "  make clean      Clean python caches and temporary build artifacts"
	@echo ""
	@echo "Configurable variables:"
	@echo "  PYTHON=$(PYTHON)"
	@echo "  TRAIN_DIR=$(TRAIN_DIR)"
	@echo "  TEST_DIR=$(TEST_DIR)"
	@echo "  OUTPUT_DIR=$(OUTPUT_DIR)"
	@echo "  FOLDS=$(FOLDS)"
	@echo "  SEED=$(SEED)"

install:
	$(PYTHON) -m pip install -r chimera_submission/code/business_entity_resolution/requirements.txt

test:
	PYTHONPATH=. $(PYTHON) -m pytest tests/ -v

generate: run

run:
	PYTHONPATH=. $(PYTHON) chimera_submission/code/business_entity_resolution/src/main.py \
		--train-dir $(TRAIN_DIR) \
		--test-dir $(TEST_DIR) \
		--output-dir $(OUTPUT_DIR) \
		--k-folds $(FOLDS) \
		--seed $(SEED)

validate:
	$(PYTHON) utils/validate_submission.py \
		--matching $(OUTPUT_DIR)/matching_results.tsv \
		--candidate $(OUTPUT_DIR)/candidate_pairs.tsv \
		--test-dir $(TEST_DIR)

package:
	@echo "Packaging chimera_submission.zip..."
	rm -f chimera_submission.zip
	cd chimera_submission && zip -r ../chimera_submission.zip output code Documentation_template.md -x "*__pycache__*" "*.pyc" "*.DS_Store*"
	@echo "chimera_submission.zip ready for upload."

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
