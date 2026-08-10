# strainphase benchmark suite
#
#   make install    install the harness and its dependencies
#   make smoke      ~5 min, no downloads, no external tools  (start here)
#   make check      confirm which comparator tools are installed
#   make standard   the sweep behind the reported figures
#   make genomes    fetch reference genomes for the real-genome tier
#   make real       the real-genome tier (needs `make genomes` first)
#   make test       unit tests for the simulator and the metrics
#   make plan       list the (dataset, tool) work units
#   make clean      remove generated results
#
# On a cluster, use scripts/slurm/submit.sh instead of `make standard`; see
# scripts/slurm/README.md.

PYTHON ?= python
RESULTS ?= results
THREADS ?= 4
CONFIG ?= configs/standard.yaml

.PHONY: install smoke check standard genomes real test verify plan clean help

help:
	@sed -n '3,17p' $(MAKEFILE_LIST) | sed 's/^# \{0,1\}//'

install:
	$(PYTHON) -m pip install -e .. -e .
	@echo
	@echo "Installed. Next: make smoke"

# The functional check. Runs only the tools that need no external binaries, so
# it works on a bare clone. Small enough that the numbers are noisy - it proves
# the harness works, it is not evidence about any tool.
smoke:
	$(PYTHON) -m spbench.cli run -c configs/smoke.yaml -w $(RESULTS)/smoke
	$(PYTHON) -m spbench.cli verify -r $(RESULTS)/smoke/results -e expected/smoke.json
	@echo
	@echo "Report: $(RESULTS)/smoke/results/report.md"

check:
	$(PYTHON) -m spbench.cli check-tools

# The tier behind the reported figures: strain count x divergence x coverage,
# three replicate seeds. Comparators that are not installed become `skipped`
# rows rather than failures, so this runs to completion either way.
standard:
	$(PYTHON) -m spbench.cli run -c configs/standard.yaml -w $(RESULTS)/standard
	@echo
	@echo "Report: $(RESULTS)/standard/results/report.md"

genomes:
	$(PYTHON) scripts/fetch_genomes.py --outdir genomes

real: genomes
	$(PYTHON) -m spbench.cli run -c configs/real-genomes.yaml -w $(RESULTS)/real
	@echo
	@echo "Report: $(RESULTS)/real/results/report.md"

test:
	$(PYTHON) -m pytest tests -v

plan:
	$(PYTHON) -m spbench.cli plan -c $(CONFIG)

verify:
	$(PYTHON) -m spbench.cli verify -r $(RESULTS)/smoke/results -e expected/smoke.json

clean:
	rm -rf $(RESULTS)
