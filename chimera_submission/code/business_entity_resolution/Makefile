.DEFAULT_GOAL := help

PYTHON ?= .venv/bin/python
TRAIN_DIR ?= ../../../dataset/train
TEST_DIR ?= ../../../dataset/test
WORK_ROOT ?= ../../../cpu-run
OUTPUT_DIR ?= ../../../provisional-output
PHASE4_WORK ?=
PHASE4_REPORT ?=
TRAIN_ENTITIES ?= 3000
THREADS ?= auto
RUN_ARGS ?=
PHASE4_ARGS = $(if $(PHASE4_WORK),--phase4-work "$(PHASE4_WORK)") $(if $(PHASE4_REPORT),--phase4-report "$(PHASE4_REPORT)")

.PHONY: help cpu-info test run validate validate-official

help:
	@echo 'make run                 Provisional lexical CPU pipeline'
	@echo 'make test                Run synthetic unit/integration tests'
	@echo 'make cpu-info            Show available logical CPU cores'
	@echo 'make validate            Revalidate existing output internally'
	@echo 'make validate-official   Also run official ID validation (higher memory)'
	@echo 'Override TRAIN_ENTITIES, THREADS, paths, PYTHON, or RUN_ARGS as needed.'

cpu-info:
	$(PYTHON) -B -c 'from src.cpu_resources import available_cpu_count; print(available_cpu_count())'

test:
	$(PYTHON) -B -m unittest discover -s tests -p 'test_*.py' -q

run:
	$(PYTHON) -B -m src.main run \
		--train-dir "$(TRAIN_DIR)" --test-dir "$(TEST_DIR)" \
		--work-root "$(WORK_ROOT)" --output-dir "$(OUTPUT_DIR)" $(PHASE4_ARGS) \
		--train-entities "$(TRAIN_ENTITIES)" --threads "$(THREADS)" \
		--allow-provisional $(RUN_ARGS)

validate:
	$(PYTHON) -B -m src.main validate \
		--test-work "$(WORK_ROOT)/test-work" --output-dir "$(OUTPUT_DIR)"

validate-official:
	$(PYTHON) -B -m src.main validate \
		--test-work "$(WORK_ROOT)/test-work" --output-dir "$(OUTPUT_DIR)" \
		--test-dir "$(TEST_DIR)" --official-check-ids
