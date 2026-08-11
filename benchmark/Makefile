# strainphase benchmark suite
#
#   make install   install the harness (into an activated spbench conda env)
#   make check     can the configured pipeline commands and comparators run?
#   make run       simulate, run every tool, score, report
#   make plan      list the (dataset, tool) work units
#   make test      unit tests for the metrics, abundance model, truth derivation
#   make clean     remove generated results
#
# Simulation runs Badread, your aligner and your variant caller for real, so it
# needs the conda environment: see envs/README.md. On a cluster use
# scripts/slurm/submit.sh rather than `make run`.

PYTHON ?= python
CONFIG ?= configs/example.yaml
RESULTS ?= results

.PHONY: install check run plan test clean help

help:
	@sed -n '3,13p' $(MAKEFILE_LIST) | sed 's/^# \{0,1\}//'

install:
	$(PYTHON) -m pip install -e .. -e .

check:
	$(PYTHON) -m spbench.cli check-env -c $(CONFIG)
	$(PYTHON) -m spbench.cli check-tools -c $(CONFIG)

run:
	$(PYTHON) -m spbench.cli run -c $(CONFIG) -w $(RESULTS)
	@echo
	@echo "Report: $(RESULTS)/results/report.md"

plan:
	$(PYTHON) -m spbench.cli plan -c $(CONFIG)

test:
	$(PYTHON) -m pytest tests -v

clean:
	rm -rf $(RESULTS)
