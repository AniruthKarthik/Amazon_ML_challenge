SYSTEM_PYTHON ?= python3
VENV_DIR ?= .venv
VENV_PYTHON = $(VENV_DIR)/bin/python
VENV_PIP = $(VENV_DIR)/bin/pip
VENV_STAMP = $(VENV_DIR)/.installed_stamp

REQUIREMENTS = chimera_submission/code/business_entity_resolution/requirements.txt

# Auto-detect dataset layout (supports standard dataset/train as well as nested dataset/student_resource/dataset/train)
ifeq ($(wildcard dataset/train/train_source1.tsv),)
  ifneq ($(wildcard dataset/student_resource/dataset/train/train_source1.tsv),)
    DEFAULT_TRAIN_DIR = dataset/student_resource/dataset/train
    DEFAULT_TEST_DIR = dataset/student_resource/dataset/test
  else
    DEFAULT_TRAIN_DIR = dataset/train
    DEFAULT_TEST_DIR = dataset/test
  endif
else
  DEFAULT_TRAIN_DIR = dataset/train
  DEFAULT_TEST_DIR = dataset/test
endif

TRAIN_DIR ?= $(DEFAULT_TRAIN_DIR)
TEST_DIR ?= $(DEFAULT_TEST_DIR)
OUTPUT_DIR ?= chimera_submission/output
FOLDS ?= 5
SEED ?= 42
JOBS ?= -1
MAX_TRAIN_QUERIES ?= 40000
DEVICE ?= auto

.PHONY: help venv install test run generate validate package collab-prep clean clean-all

help:
	@echo "Amazon ML Challenge 2026 — Business Entity Resolution"
	@echo ""
	@echo "Virtual environment targets:"
	@echo "  make venv       Initialize .venv and install dependencies"
	@echo "  make install    Sync dependencies into .venv"
	@echo ""
	@echo "Pipeline targets (automatically initializes .venv if needed):"
	@echo "  make run        Run end-to-end pipeline and generate TSVs"
	@echo "  make generate   Alias for 'make run'"
	@echo "  make validate   Validate generated output TSVs using official validator"
	@echo "  make test       Run the automated pytest test suite in .venv"
	@echo "  make package    Package submission zip archive"
	@echo "  make collab-prep Package repository and dataset for Google Colab"
	@echo ""
	@echo "Cleanup targets:"
	@echo "  make clean      Clean python caches and build artifacts"
	@echo "  make clean-all  Clean caches and remove .venv"
	@echo ""
	@echo "Configurable variables:"
	@echo "  SYSTEM_PYTHON=$(SYSTEM_PYTHON)"
	@echo "  VENV_DIR=$(VENV_DIR)"
	@echo "  TRAIN_DIR=$(TRAIN_DIR)"
	@echo "  TEST_DIR=$(TEST_DIR)"
	@echo "  OUTPUT_DIR=$(OUTPUT_DIR)"
	@echo "  FOLDS=$(FOLDS)"
	@echo "  SEED=$(SEED)"
	@echo "  JOBS=$(JOBS)"
	@echo "  MAX_TRAIN_QUERIES=$(MAX_TRAIN_QUERIES)"

# Create virtual environment and install requirements
$(VENV_DIR)/bin/activate:
	@echo "Creating virtual environment in $(VENV_DIR)..."
	$(SYSTEM_PYTHON) -m venv $(VENV_DIR)
	$(VENV_PIP) install --upgrade pip

$(VENV_STAMP): $(VENV_DIR)/bin/activate $(REQUIREMENTS)
	@echo "Installing dependencies into $(VENV_DIR)..."
	$(VENV_PIP) install -r $(REQUIREMENTS)
	@touch $(VENV_STAMP)

venv: $(VENV_STAMP)

install: venv

test: $(VENV_STAMP)
	PYTHONPATH=. $(VENV_PYTHON) -m pytest tests/ -v

generate: run

run: $(VENV_STAMP)
	PYTHONPATH=. $(VENV_PYTHON) chimera_submission/code/business_entity_resolution/src/main.py \
		--train-dir $(TRAIN_DIR) \
		--test-dir $(TEST_DIR) \
		--output-dir $(OUTPUT_DIR) \
		--k-folds $(FOLDS) \
		--seed $(SEED) \
		--n-jobs $(JOBS) \
		--max-train-queries $(MAX_TRAIN_QUERIES) \
		--device $(DEVICE)

validate: $(VENV_STAMP)
	$(VENV_PYTHON) utils/validate_submission.py \
		--matching $(OUTPUT_DIR)/matching_results.tsv \
		--candidate $(OUTPUT_DIR)/candidate_pairs.tsv \
		--test-dir $(TEST_DIR)

package:
	@echo "Packaging chimera_submission.zip..."
	rm -f chimera_submission.zip
	cd chimera_submission && zip -r ../chimera_submission.zip output code Documentation_template.md -x "*__pycache__*" "*.pyc" "*.DS_Store*"
	@echo "chimera_submission.zip ready for upload."

collab-prep:
	@echo "Packaging codebase, configs, utilities, and full dataset for Google Colab..."
	rm -f aml_collab.zip
	zip -r -q aml_collab.zip \
		chimera_submission \
		dataset \
		utils \
		Makefile \
		colab_run.ipynb \
		COLAB_GUIDE.md \
		-x "*__pycache__*" "*.pyc" "*.DS_Store*" ".venv*" "chimera_submission/output/*"
	@echo "======================================================================"
	@echo "Google Colab archive created: aml_collab.zip ($$(du -h aml_collab.zip | cut -f1))"
	@echo "Upload 'aml_collab.zip' to Google Colab and run the Colab cell."
	@echo "======================================================================"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache

clean-all: clean
	@echo "Removing virtual environment $(VENV_DIR)..."
	rm -rf $(VENV_DIR)
