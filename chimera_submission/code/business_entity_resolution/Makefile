# Business Entity Resolution — CPU/GPU pipeline Makefile
#
# Run everything from this directory:
#   chimera_submission/code/business_entity_resolution/
#
# Quick start:
#   make venv          # create .venv and install pinned requirements
#   make test          # synthetic unit/integration tests
#   make run           # provisional end-to-end pipeline (sampled training)
#   make gpu-run       # A1000/32 GB profile; requires a working OpenCL GPU
#
# Every target honours PYTHON=, so an existing interpreter works too:
#   make test PYTHON=/tmp/er_venv/bin/python
#
# Staged workflow (large jobs — run each stage separately):
#   make phase4-train  (via `run`, resumable)  ->  make train  ->  make prepare-test  ->  make predict  ->  make validate
#
# All thresholds/models are tuned on training OOF data only; test data is
# never used for model selection. See README.md for pipeline details.

.DEFAULT_GOAL := help

# ---------------------------------------------------------------- environment ---

VENV ?= .venv
PYTHON ?= $(VENV)/bin/python

# ---------------------------------------------------------------- data / work dirs ---
# Paths are relative to this directory (override with absolute paths if needed).

TRAIN_DIR ?= ../../../dataset/train
TEST_DIR ?= ../../../dataset/test
WORK_ROOT ?= ../../../cpu-run
OUTPUT_DIR ?= ../../../provisional-output

# Derived stage directories (override independently when resuming guests).
MODEL_DIR ?= $(WORK_ROOT)/model
TEST_WORK ?= $(WORK_ROOT)/test-work
PHASE4_WORK ?=
PHASE4_REPORT ?=

# ---------------------------------------------------------------- pipeline settings ---

TRAIN_ENTITIES ?= 3000
TOP_K ?= 50
MAX_CANDIDATES ?= 250
MAX_TRAIN_PAIRS ?= 1000000
FOLDS ?= 3
GRID_POINTS ?= 3
FEATURE_MAX_FEATURES ?= 30000
BOOST_ROUNDS ?= 100
MIN_DATA_IN_LEAF ?= 20
BATCH_ENTITIES ?= 128
SEED ?= 42
THREADS ?= auto
DEVICE ?= auto
MAX_BIN ?= 63
GPU_PLATFORM_ID ?= -1
GPU_DEVICE_ID ?= -1
GPU_USE_DP ?=

# Optional dense retrieval (Phase 13, off by default on 16 GB machines).
DENSE_WORK ?=
DENSE_TOP_K ?= 20
DENSE_TEST_WORK ?=

# Optional meta-model comparison (Phase 12, off by default).
EVALUATE_META ?=
MAX_WORST_FOLD_DROP ?=
MAX_FOLD_STD_INCREASE ?=

# Extra pass-through flags.
RUN_ARGS ?=
TRAIN_ARGS ?=
PREDICT_ARGS ?= --allow-exploratory

# --- internal flag assembly (do not override directly) ---

PHASE4_ARGS = $(if $(PHASE4_WORK),--phase4-work "$(PHASE4_WORK)") $(if $(PHASE4_REPORT),--phase4-report "$(PHASE4_REPORT)")
DENSE_TRAIN_ARGS = $(if $(DENSE_WORK),--dense-work "$(DENSE_WORK)" --dense-top-k "$(DENSE_TOP_K)")
DENSE_PREDICT_ARGS = $(if $(DENSE_TEST_WORK),--dense-test-work "$(DENSE_TEST_WORK)")
META_ARGS = $(if $(EVALUATE_META),--evaluate-meta --max-worst-fold-drop "$(MAX_WORST_FOLD_DROP)" --max-fold-std-increase "$(MAX_FOLD_STD_INCREASE)")
GPU_ARGS = --device "$(DEVICE)" --max-bin "$(MAX_BIN)" \
	--gpu-platform-id "$(GPU_PLATFORM_ID)" --gpu-device-id "$(GPU_DEVICE_ID)" \
	$(if $(GPU_USE_DP),--gpu-use-dp)
TRAIN_OPTS = --train-entities "$(TRAIN_ENTITIES)" --top-k "$(TOP_K)" \
	--max-candidates "$(MAX_CANDIDATES)" --max-train-pairs "$(MAX_TRAIN_PAIRS)" \
	--folds "$(FOLDS)" --grid-points "$(GRID_POINTS)" \
	--feature-max-features "$(FEATURE_MAX_FEATURES)" --boost-rounds "$(BOOST_ROUNDS)" \
	--min-data-in-leaf "$(MIN_DATA_IN_LEAF)" --threads "$(THREADS)" --seed "$(SEED)" \
	$(GPU_ARGS)

.PHONY: help venv install install-dense cpu-info gpu-check test test-verbose compile \
	run gpu-run train prepare-test predict validate validate-official \
	freeze-check clean clean-work

help:
	@echo 'Usage: make <target> [VAR=value ...]'
	@echo ''
	@echo 'Environment:'
	@echo '  venv             Create .venv and install pinned requirements'
	@echo '  install          Install pinned requirements into $$(PYTHON)'
	@echo '  install-dense    Install optional dense-retrieval requirements'
	@echo '  cpu-info         Show logical CPU cores visible to the process'
	@echo '  gpu-check        Verify that LightGBM can train on the selected OpenCL GPU'
	@echo '  freeze-check     Verify installed packages match requirements.txt'
	@echo ''
	@echo 'Tests:'
	@echo '  test             Run synthetic unit/integration tests (quiet)'
	@echo '  test-verbose     Run tests with verbose output'
	@echo '  compile          Byte-compile all sources (fast syntax check)'
	@echo ''
	@echo 'Pipeline (provisional lexical baseline, sampled training):'
	@echo '  run              End-to-end: retrieval + OOF training + inference + validation'
	@echo '  gpu-run          End-to-end NVIDIA GPU profile for a 32 GB RAM laptop'
	@echo '  train            Staged: OOF training only (needs phase4-work)'
	@echo '  prepare-test     Staged: build test store + retrieval channels'
	@echo '  predict          Staged: frozen inference to matching_results.tsv'
	@echo '  validate         Revalidate existing output (internal checks)'
	@echo '  validate-official  Also run official --check-ids validation (high memory)'
	@echo ''
	@echo 'Cleanup:'
	@echo '  clean            Remove __pycache__ trees'
	@echo '  clean-work       Delete WORK_ROOT + OUTPUT_DIR (needs CONFIRM=1)'
	@echo ''
	@echo 'Common overrides: TRAIN_ENTITIES=5000 THREADS=8 DEVICE=gpu TOP_K=20 MAX_CANDIDATES=100'
	@echo '  FOLDS=5 GRID_POINTS=15 BOOST_ROUNDS=500 PYTHON=/tmp/er_venv/bin/python'

# ---------------------------------------------------------------- environment ---

venv:
	python3 -m venv "$(VENV)"
	"$(VENV)/bin/pip" install --upgrade pip wheel setuptools
	"$(VENV)/bin/pip" install -r requirements.txt

install:
	$(PYTHON) -m pip install -r requirements.txt

install-dense:
	$(PYTHON) -m pip install -r requirements-dense.txt

freeze-check:
	$(PYTHON) -m pip freeze --exclude-editable | sort > .freeze-actual.txt
	@echo '--- packages in requirements.txt missing from the environment: ---'
	@comm -23 <(sed 's/==.*//' requirements.txt | sort) \
		<(sed 's/==.*//' .freeze-actual.txt | sort) || true
	@rm -f .freeze-actual.txt

cpu-info:
	$(PYTHON) -B -c 'from src.cpu_resources import available_cpu_count; print(available_cpu_count())'

gpu-check:
	$(PYTHON) -B -c 'from src.pair_model import BaselineConfig, resolve_training_device; print(resolve_training_device(BaselineConfig(device_type="gpu", max_bin=$(MAX_BIN), gpu_platform_id=$(GPU_PLATFORM_ID), gpu_device_id=$(GPU_DEVICE_ID)))[1])'

# ---------------------------------------------------------------- tests ---

test:
	$(PYTHON) -B -m unittest discover -s tests -p 'test_*.py' -q

test-verbose:
	$(PYTHON) -B -m unittest discover -s tests -p 'test_*.py' -v

compile:
	$(PYTHON) -B -m compileall -q src tests && echo 'compile ok'

# ---------------------------------------------------------------- pipeline ---

run:
	$(PYTHON) -B -m src.main run \
		--train-dir "$(TRAIN_DIR)" --test-dir "$(TEST_DIR)" \
		--work-root "$(WORK_ROOT)" --output-dir "$(OUTPUT_DIR)" $(PHASE4_ARGS) \
		$(TRAIN_OPTS) $(META_ARGS) \
		--batch-entities "$(BATCH_ENTITIES)" \
		--allow-provisional $(RUN_ARGS)

# The numeric matrix stays comfortably below 32 GB while the 63-bin histogram
# is small enough for laptop-class VRAM. Explicit GPU mode fails instead of
# silently running a multi-hour training stage on CPU.
gpu-run: DEVICE = gpu
gpu-run: TRAIN_ENTITIES = 10000
gpu-run: MAX_TRAIN_PAIRS = 2500000
gpu-run: FEATURE_MAX_FEATURES = 50000
gpu-run: BATCH_ENTITIES = 512
gpu-run: run

train:
	$(PYTHON) -B -m src.main train \
		--phase4-work "$(if $(PHASE4_WORK),$(PHASE4_WORK),$(WORK_ROOT)/phase4-work)" \
		--model-dir "$(MODEL_DIR)" \
		$(TRAIN_OPTS) $(META_ARGS) $(DENSE_TRAIN_ARGS) $(TRAIN_ARGS)

prepare-test:
	$(PYTHON) -B -m src.main prepare-test \
		--test-dir "$(TEST_DIR)" --model-dir "$(MODEL_DIR)" \
		--test-work "$(TEST_WORK)" --threads "$(THREADS)"

predict:
	$(PYTHON) -B -m src.main predict \
		--model-dir "$(MODEL_DIR)" --test-work "$(TEST_WORK)" \
		--output-dir "$(OUTPUT_DIR)" --batch-entities "$(BATCH_ENTITIES)" \
		$(DENSE_PREDICT_ARGS) $(PREDICT_ARGS)

validate:
	$(PYTHON) -B -m src.main validate \
		--test-work "$(TEST_WORK)" --output-dir "$(OUTPUT_DIR)"

validate-official:
	$(PYTHON) -B -m src.main validate \
		--test-work "$(TEST_WORK)" --output-dir "$(OUTPUT_DIR)" \
		--test-dir "$(TEST_DIR)" --official-check-ids

# ---------------------------------------------------------------- cleanup ---

clean:
	find src tests -type d -name __pycache__ -prune -exec rm -rf {} +

clean-work:
ifndef CONFIRM
	$(error Refusing to delete work dirs. Re-run with CONFIRM=1)
endif
	rm -rf "$(WORK_ROOT)" "$(OUTPUT_DIR)"
