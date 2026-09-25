SYSTEM_PYTHON ?= python
VENV_DIR ?= .venv
VENV_PYTHON = $(VENV_DIR)\Scripts\python
VENV_PIP = $(VENV_DIR)\Scripts\pip
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
	@echo Amazon ML Challenge 2026 - Business Entity Resolution
	@echo.
	@echo Virtual environment targets:
	@echo   make venv       Initialize .venv and install dependencies
	@echo   make install    Sync dependencies into .venv
	@echo.
	@echo Pipeline targets (automatically initializes .venv if needed):
	@echo   make run        Run end-to-end pipeline and generate TSVs
	@echo   make generate   Alias for 'make run'
	@echo   make validate   Validate generated output TSVs using official validator
	@echo   make test       Run the automated pytest test suite in .venv
	@echo   make package    Package submission zip archive
	@echo   make collab-prep Package repository and dataset for Google Colab
	@echo.
	@echo Cleanup targets:
	@echo   make clean      Clean python caches and build artifacts
	@echo   make clean-all  Clean caches and remove .venv
	@echo.
	@echo Configurable variables:
	@echo   SYSTEM_PYTHON=$(SYSTEM_PYTHON)
	@echo   VENV_DIR=$(VENV_DIR)
	@echo   TRAIN_DIR=$(TRAIN_DIR)
	@echo   TEST_DIR=$(TEST_DIR)
	@echo   OUTPUT_DIR=$(OUTPUT_DIR)
	@echo   FOLDS=$(FOLDS)
	@echo   SEED=$(SEED)
	@echo   JOBS=$(JOBS)
	@echo   MAX_TRAIN_QUERIES=$(MAX_TRAIN_QUERIES)

# Create virtual environment and install requirements
$(VENV_DIR)/Scripts/activate:
	@echo Creating virtual environment in $(VENV_DIR)...
	$(SYSTEM_PYTHON) -m venv $(VENV_DIR)
	$(VENV_PIP) install --upgrade pip

$(VENV_STAMP): $(VENV_DIR)/Scripts/activate $(REQUIREMENTS)
	@echo Installing dependencies into $(VENV_DIR)...
	$(VENV_PIP) install -r $(REQUIREMENTS)
	@type nul > $(VENV_STAMP)

venv: $(VENV_STAMP)

install: venv

test: $(VENV_STAMP)
	cmd /C "set PYTHONPATH=. && $(VENV_PYTHON) -m pytest tests/ -v"

generate: run

run: $(VENV_STAMP)
	cmd /C "set PYTHONPATH=. && $(VENV_PYTHON) chimera_submission/code/business_entity_resolution/src/main.py --train-dir $(TRAIN_DIR) --test-dir $(TEST_DIR) --output-dir $(OUTPUT_DIR) --k-folds $(FOLDS) --seed $(SEED) --n-jobs $(JOBS) --max-train-queries $(MAX_TRAIN_QUERIES) --device $(DEVICE)"

validate: $(VENV_STAMP)
	$(VENV_PYTHON) utils/validate_submission.py --matching $(OUTPUT_DIR)/matching_results.tsv --candidate $(OUTPUT_DIR)/candidate_pairs.tsv --test-dir $(TEST_DIR)

package:
	@echo Packaging chimera_submission.zip...
	@if exist chimera_submission.zip del /f chimera_submission.zip
	@powershell -Command "Compress-Archive -Path chimera_submission/output, chimera_submission/code, chimera_submission/Documentation_template.md -DestinationPath chimera_submission.zip -Force"
	@echo chimera_submission.zip ready for upload.

collab-prep:
	@echo Packaging codebase, configs, utilities, and full dataset for Google Colab...
	@if exist aml_collab.zip del /f aml_collab.zip
	@powershell -Command "Compress-Archive -Path chimera_submission, dataset, utils, Makefile, colab_run.ipynb, COLAB_GUIDE.md -DestinationPath aml_collab.zip -Force"
	@echo ======================================================================
	@echo Google Colab archive created: aml_collab.zip
	@echo Upload 'aml_collab.zip' to Google Colab and run the Colab cell.
	@echo ======================================================================

clean:
	@powershell -Command "Get-ChildItem -Recurse -Filter '__pycache__' -Directory | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue"
	@powershell -Command "Get-ChildItem -Recurse -Filter '*.pyc' -File | Remove-Item -Force -ErrorAction SilentlyContinue"
	@if exist .pytest_cache rd /s /q .pytest_cache

clean-all: clean
	@echo Removing virtual environment $(VENV_DIR)...
	@if exist $(VENV_DIR) rd /s /q $(VENV_DIR)
