# Environments

Each comparator gets its own conda environment. They are kept separate on
purpose: Floria, Strainy, devider and whatshap have conflicting pins (samtools,
minimap2 and Python versions in particular), and resolving them into one
environment either fails or silently downgrades something. Separate environments
also mean the version of each tool that produced a number is recorded in one
place.

```bash
# The harness itself. Everything except the third-party tools runs from here.
conda env create -f envs/spbench.yml && conda activate spbench
pip install -e . -e ..            # spbench + strainphase

# Comparators, one at a time.
conda env create -f envs/floria.yml
conda env create -f envs/strainy.yml
conda env create -f envs/devider.yml
conda env create -f envs/whatshap.yml
```

Because the tools live in separate environments, the harness finds them on
`PATH`. The simplest way to make that work is to prepend each environment's
`bin` directory before running:

```bash
export PATH="$CONDA_PREFIX/../floria/bin:$CONDA_PREFIX/../strainy/bin:$PATH"
spbench check-tools -c configs/standard.yaml
```

`spbench check-tools` prints exactly which tools it can see. Anything it cannot
see becomes a `skipped` row in the results with the reason attached - the
benchmark still completes.

## Pinning

The version pins below are the versions these adapters were written against.
Loosen them if you need to, but record what you actually ran: `spbench` writes
every tool's resolved status into `results/runs.tsv` and the platform and
package versions into `results/provenance.json`.

If a newer release changes an output format, `spbench check-tools` will surface
it as a parse failure on the smoke dataset rather than as a wrong number in a
table.
