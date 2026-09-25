# Business Entity Resolution — CPU pipeline Makefile
#
# Run everything from this directory:
#   chimera_submission/code/business_entity_resolution/
#
# Quick start:
#   make venv          # create .venv and install pinned requirements
#   make test          # synthetic unit/integration tests
#   make run           # provisional end-to-end pipeline (sampled training)
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
TRAIN_OPTS = --train-entities "$(TRAIN_ENTITIES)" --top-k "$(TOP_K)" \
	--max-candidates "$(MAX_CANDIDATES)" --max-train-pairs "$(MAX_TRAIN_PAIRS)" \
	--folds "$(FOLDS)" --grid-points "$(GRID_POINTS)" \
	--feature-max-features "$(FEATURE_MAX_FEATURES)" --boost-rounds "$(BOOST_ROUNDS)" \
	--min-data-in-leaf "$(MIN_DATA_IN_LEAF)" --threads "$(THREADS)" --seed "$(SEED)"

.PHONY: help venv install install-dense cpu-info test test-verbose compile \
	run train prepare-test predict validate validate-official \
	freeze-check clean clean-work

help:
	@echo 'Usage: make <target> [VAR=value ...]'
	@echo ''
	@echo 'Environment:'
	@echo '  venv             Create .venv and install pinned requirements'
	@echo '  install          Install pinned requirements into $$(PYTHON)'
	@echo '  install-dense    Install optional dense-retrieval requirements'
	@echo '  cpu-info         Show logical CPU cores visible to the process'
	@echo '  freeze-check     Verify installed packages match requirements.txt'
	@echo ''
	@echo 'Tests:'
	@echo '  test             Run synthetic unit/integration tests (quiet)'
	@echo '  test-verbose     Run tests with verbose output'
	@echo '  compile          Byte-compile all sources (fast syntax check)'
	@echo ''
	@echo 'Pipeline (provisional lexical CPU baseline, sampled training):'
	@echo '  run              End-to-end: retrieval + OOF training + inference + validation'
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
	@echo 'Common overrides: TRAIN_ENTITIES=5000 THREADS=8 TOP_K=20 MAX_CANDIDATES=100'
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
