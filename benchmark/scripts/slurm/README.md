# SLURM submission

Three dependent stages. `submit.sh` is the only script you invoke.

```bash
cp env.sh.example env.sh
$EDITOR env.sh                    # conda activation + comparator PATHs

../../scripts/slurm/submit.sh \
    -c ../../configs/standard.yaml \
    -w /scratch/$USER/spbench-standard \
    --env-setup env.sh \
    --partition compute --account mylab
```

| File | Role |
|---|---|
| `submit.sh` | Counts work units, submits the three jobs with dependencies. `--dry-run` prints the `sbatch` lines without submitting. |
| `simulate.sbatch` | Stage 1. Generates every dataset. One job. |
| `run_unit.sbatch` | Stage 2. Array, one task per `(dataset, tool)` pair. |
| `evaluate.sbatch` | Stage 3. Scores everything and writes the report. |
| `_common.sh` | Sourced by all three: sources your env file, checks `spbench` imports, prints a provenance header. |
| `env.sh.example` | Template for the site-specific parts. Copy it; do not commit your copy. |

## Why three stages instead of one job per dataset

Simulation has to finish before the array starts. Array tasks share dataset
directories, so letting each task simulate on demand would put several processes
in the same directory at once. A dependency is the fix; a lock file would be a
workaround.

The evaluate stage depends on the array with `afterany`, not `afterok`. One tool
crashing on one dataset should not cost you the other results — that unit is
recorded with `status=failed` and its error, and everything else is still
scored.

## Reruns

The index-to-unit mapping is deterministic for a given config, so a failed unit
can be rerun by index alone:

```bash
spbench plan -c configs/standard.yaml | less   # index -> dataset, tool
sbatch --array=17,31 ... run_unit.sbatch       # rerun just those
spbench evaluate -c configs/standard.yaml -w /scratch/$USER/spbench-standard
```

Scoring reads only the prediction files, so `spbench evaluate` is also how you
rescore at a different `match_threshold` without re-running any tool.

## Resources

Defaults in `submit.sh` are deliberately generous. Tighten them from measured
numbers after a first run — `results/runs.tsv` carries wall time and peak RSS
per unit.

- **The simulate stage is now the heavy one** — it runs Badread, alignment and
  variant calling for every timepoint of every dataset. Give it hours, not
  minutes, and enough memory for your variant caller.
- **Memory** in the array is driven by `strainphase-longitudinal`, which holds every
  timepoint for one MAG at once. 32 GB is comfortable for the standard tier.
- **Wall time** is driven by Strainy, which runs an assembler internally.
- **`--max-concurrent`** throttles the array. Raise it if your partition allows;
  the units are independent and there is no shared state between them.

Set `TMPDIR` to scratch in your `env.sh`. Per-tool working directories are large
and a `$HOME` quota will kill a job hours in.
