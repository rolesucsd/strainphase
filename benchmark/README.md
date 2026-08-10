# strainphase benchmark suite

A self-contained, reproducible comparison of strainphase against other long-read
strain phasing tools. Clone the repository, run one command, get the tables.

```bash
cd benchmark
pip install -e .. -e .        # strainphase + the harness
make smoke                    # ~5 minutes, no downloads, no external tools
```

`make smoke` simulates a three-strain longitudinal mixture, runs strainphase in
two configurations plus a baseline, scores everything, and writes
`results/smoke/results/report.md`. Nothing else needs to be installed for that
to work.

To include the third-party comparators, install them (see `envs/`) and run the
tier the reported figures come from:

```bash
make check                    # which comparators can I see?
make standard                 # the full sweep
```

---

## The problem this suite has to solve

strainphase does something the tools it would naturally be compared against do
not attempt: it reconstructs strain haplotypes from **several timepoints at
once** and carries strain identity across them. Floria, Strainy and devider each
phase one sample in isolation. Running them side by side and reporting that
strainphase found more strains is not a result - it is a description of the
input each tool was given.

Three things keep this comparison honest, and they are the design of the suite:

**1. Every tool is scored on its read partition, not on its own output format.**
Floria emits vartigs, Strainy emits assembly graph paths, devider emits an MSA,
strainphase emits window-linked consensus tracks. Comparing those directly means
comparing four consensus callers as much as four phasing algorithms. The one
thing they all genuinely produce is a grouping of reads, so the harness takes
each tool's read partition and derives the consensus haplotype from it with the
same code for everyone (`spbench/reads.py`). Read-partition agreement - ARI,
V-measure - is the primary comparison surface, and it needs no charitable
interpretation of anyone's output.

**2. strainphase competes against itself.** The like-for-like table contains
`strainphase-single`: strainphase with cross-timepoint rescue disabled, one
timepoint at a time. That is the row that belongs next to the single-sample
tools. The longitudinal claim is then measured as `strainphase-longitudinal`
versus `strainphase-single` - identical code, identical data, one flag apart -
rather than against tools that never attempted the task.

**3. Columns a tool does not claim are `n/a`, not zero.** Every adapter declares
what its tool was designed for, whether it sees all timepoints, and whether it
assigns stable identifiers across samples. Cross-timepoint identity is reported
only for tools that claim it. A tool is never scored on a promise it did not
make, and the report prints each tool's `designed_for` line next to its results.

There is also a **`naive-greedy` floor**: greedy Hamming clustering of read
allele profiles, no model at all. If a published tool cannot beat it on a given
configuration, that configuration is too easy to be evidence for anything. It
also demonstrates the metric's main failure mode - it over-splits, which buys
recall cheaply - which is why the report prints haplotype counts next to recall.

---

## What you have to supply

**Nothing.** That is the point, and it is the main thing the previous benchmark
setup got wrong: it needed a `$GENOME_SOURCE` directory that was never
specified, so even with the code in hand nobody could reproduce a number.

`make smoke` and `make standard` generate their own reference sequence, their
own reads, their own variant calls and their own ground truth from a seed. No
downloads, no BAM you have to find, no reference you have to stage.

The only optional input is real assemblies for the `real-genomes` tier, and
`make genomes` fetches those from the accessions in `data/genomes.tsv`.

### What a generated dataset looks like

Each `(dataset, seed)` pair becomes one self-contained directory:

```
results/standard/datasets/k4-div5-cov30.seed0/
├── reference.fasta            (+ .fai)
├── bam/
│   ├── T1.bam                 (+ .bai)  coordinate-sorted, one per timepoint
│   ├── T2.bam
│   └── ...
├── variants/
│   ├── T1.vcf.gz              (+ .tbi)  per-timepoint calls, DP/AF populated
│   ├── ...
│   └── union.vcf.gz           every site called in any timepoint
├── truth/
│   ├── sites.tsv              contig, pos, ref, alt
│   ├── strains.tsv            each strain's true haplotype
│   ├── abundance.tsv          strain x timepoint, sums to 1 per timepoint
│   └── read_origins.tsv       which strain every read really came from
└── manifest.json              the full config, plus a fingerprint
```

`bam/`, `variants/` and `reference.fasta` are exactly what you would hand any of
these tools on real data. `truth/` is never shown to an adapter - only the
evaluator reads it.

### Pointing it at your own data

Two supported ways to make the simulation resemble your study:

**Use your own reference.** Any dataset entry accepts `reference_fasta`, so you
can build mixtures on one of your own MAG bins rather than on random sequence or
a RefSeq isolate:

```yaml
datasets:
  - name: my-mag-k6
    reference_fasta: /path/to/BF_MAG_01.fasta
    contig_length: 500000      # 0 keeps the full contig
    n_contigs: 1
    n_strains: 6
    n_mutations: 2500
    n_timepoints: 8            # match your study design
    coverage: 30
```

**Match your sequencing.** `coverage`, `mean_read_length`, `read_length_sd`,
`error_rate` and `n_timepoints` should be set from your real runs. Defaults are
HiFi-shaped (15 kb mean, 0.1% error) but the honest thing is to use your own
numbers - a benchmark run at 30x says nothing about a study sequenced at 8x.

Every key in `SimConfig` (`spbench/simulate.py`) is settable from a dataset
entry, and an unrecognised key is a hard error rather than a silent default.

### What is *not* supported, and why

Running the suite on your real longitudinal samples to get accuracy numbers.
There is no ground truth for those, so precision and recall are undefined. You
could compare tools for *concordance* on real data, and that is worth doing, but
it answers a different question and is not implemented here. The nearest honest
substitute is a sequenced mock community - see the limitations section.

---

## What gets measured

| Family | Metrics | Applies to |
|---|---|---|
| Read partition | ARI, AMI, homogeneity, completeness, V-measure, fraction of reads placed | every tool |
| Haplotype reconstruction | precision / recall / F1 at a match threshold, allele (Hamming) error, switch error, span N50, strain-count error | every tool |
| Abundance | MAE, Pearson r, MAE charging missed strains | tools that report abundance |
| Detection sensitivity | recall stratified by true abundance **and** by absolute strain depth | every tool |
| Cross-timepoint identity | ARI of the tool's own identifiers against true strain identity | tools that claim stable IDs |
| Cost | wall time, peak RSS | every tool |

Two details worth knowing before reading a number:

- **Matching is a global optimum, not greedy.** Predicted haplotypes are
  assigned to true strains by the Hungarian algorithm maximising total agreeing
  sites, so results do not depend on file ordering. A pair must share at least
  `min_shared_sites` (default 10) positions to be comparable.
- **Recall must be read against the haplotype count.** Matching is one-to-one,
  so a method emitting fifty fragments per sample will match a rare strain with
  one of them by luck. The report prints `Haps/sample` in the same table.

**Switch error** is generalised from the diploid definition: walking a predicted
haplotype's sites in order, the harness tracks the set of true strains still
consistent with everything seen so far, and counts a switch when that set is
forced empty. It measures chimerism - a reconstruction that looks like a real
strain but never existed.

**Detection is reported on two axes.** Abundance fraction is the intuitive one,
but it is not comparable across coverage levels: a 1% strain at 300x has 3x of
reads and is recoverable, while a 1% strain at 20x has 0.2x and is simply not in
the data. The report also stratifies by `abundance x coverage`, which separates
"the method missed it" from "the reads were not there". The 1-3x band is where
cross-timepoint information can change the outcome, and it is the band to look
at when judging the longitudinal claim.

---

## The simulated data

`spbench/simulate.py` writes a complete dataset: reference FASTA, one sorted and
indexed BAM per timepoint, one VCF per timepoint, and truth tables. Four
decisions in it are load-bearing.

**Strains are related by a tree.** Independently drawn haplotypes are pairwise
equidistant, which is the easy case for every clustering method. Mutations are
placed along the branches of a random tree, so each mixture contains closely
related pairs nested inside more distant clades - the structure that separates
methods.

**Base qualities are calibrated.** Each base's error probability is drawn, its Q
is `-10 log10` of that probability, and the base is corrupted with exactly that
probability. A simulator emitting a flat Q40 would hand an unearned advantage to
quality-weighted methods, strainphase among them.

**Reads are aligned, not placed.** There are two read models, and the choice
matters more than any other setting in the suite:

| `read_model` | Error | Placement | Use |
|---|---|---|---|
| `exact` | substitution-only, uniform | true coordinates, exact CIGAR | smoke tier and CI: fast, no extra deps, deterministic |
| `hifi` | homopolymer-concentrated indels + substitutions | **minimap2**, both strands | `standard.yaml` and anything going in a paper |

`exact` is the model most published phasing benchmarks use, including this
project's previous one, and it is misleading in a specific way: HiFi error is
dominated by indel slippage in homopolymers, and those errors matter chiefly
because an *aligner* has to place them. Emitting the true CIGAR removes exactly
the difficulty the indel axis is meant to test.

So `hifi` gives reads context-aware error, reverse-complements half of them, and
aligns them back to the reference with minimap2 through the `mappy` bindings.
Placement, CIGAR, MAPQ and soft clipping all come from the aligner; nothing tells
it where a read came from. `standard.yaml` also includes a
`k4-div5-cov30-exactreads` dataset identical to `k4-div5-cov30` except for the
read model, so the cost of the shortcut is itself measurable — slice
`per_sample.tsv` on `read_model`.

What `hifi` still does **not** model: a true CCS error process. The parameters
(`hifi_substitution_fraction`, `hifi_homopolymer_exponent`, and the rest) are
literature-shaped approximations, not fitted to an instrument, and every one is
configurable. If you need more fidelity than that, simulate per-strain reads
with PBSIM3 (multi-pass CLR then `ccs`) — the read-to-strain mapping survives
because each strain is simulated separately. Note also that random reference
sequence contains far fewer long homopolymers than a real genome, which is
another reason to run the real-genome tier or point `reference_fasta` at your
own MAG.

**Variant calls are simulated, not handed over.** By default the tools receive a
*called* VCF derived from the simulated reads - a site is called only if enough
reads actually carry the alt allele. Handing every tool the exact truth site
list would remove the low-abundance detection problem, which is the problem this
benchmark most wants to measure. `vcf_mode: truth` is available as an ablation
and `configs/standard.yaml` includes one dataset that uses it, which isolates
how much of each tool's low-abundance loss comes from calling rather than
phasing.

**The rare strain's trough is pinned in coverage, not in abundance.** A fixed 1%
trough means 0.1x at 10x sequencing - no method can recover a strain whose reads
do not exist, so the test would degenerate. `rescue_trough_coverage` (default
1.5x) keeps the scenario meaningful at every coverage: enough reads to be
rescued from a neighbouring timepoint, too few to be discovered de novo within
one.

Every dataset is written with a `manifest.json` recording its full config and a
fingerprint. Re-running reuses datasets whose fingerprint matches and
regenerates those whose config changed.

---

## Reproducibility

- **Seeded end to end.** The simulator is seeded; so is strainphase. The
  adapter pins `random_seed` by default because strainphase's Louvain
  initialisation and read subsampling are both random - unseeded, two runs on
  identical input give different numbers, and a benchmark that cannot be re-run
  to the same answer is not evidence. Two `make smoke` runs produce identical
  result tables.
- **Provenance is written with the results.** `results/provenance.json` records
  the git commit (and whether the tree was dirty), platform, CPU count, package
  versions, seeds, thresholds and each tool's options.
- **Every number carries its condition.** Rows in `per_sample.tsv` carry strain
  count, coverage, timepoint count, error rate, indel fraction, VCF mode, seed
  and tool options, so the table can be sliced without a config lookup.
- **CI guards the harness.** `spbench verify` checks the smoke results against
  `expected/smoke.json`. The bounds are wide and guard the *harness*: a change
  to the simulator or the metrics that silently moves every number fails the
  build. A legitimate change updates the bounds in the same commit.
- **No network for the synthetic tiers.** Smoke and standard generate their own
  references.

---

## Running on a cluster

The standard tier is 189 `(dataset, tool)` pairs. Those are independent, so they
map onto a SLURM array:

```bash
cp scripts/slurm/env.sh.example scripts/slurm/env.sh
$EDITOR scripts/slurm/env.sh          # conda activation, comparator PATHs

scripts/slurm/submit.sh \
    -c configs/standard.yaml \
    -w /scratch/$USER/spbench-standard \
    --env-setup scripts/slurm/env.sh \
    --partition compute --account mylab \
    --threads 8 --run-mem 32G --max-concurrent 20
```

That submits three dependent jobs:

| Stage | Shape | Why |
|---|---|---|
| `spb-sim` | one job | Datasets must exist before the array starts. Array tasks share dataset directories, so simulating on demand would have several processes writing the same files. |
| `spb-run` | array, one task per pair | The only stage that costs real time. Index-to-unit mapping is stable, so a failed task is rerun by index. |
| `spb-eval` | one job, `afterany` | Scores whatever is on disk. `afterany`, not `afterok`: one tool crashing on one dataset should not cost you the other 188 results. |

Add `--dry-run` to print the `sbatch` commands without submitting.

```bash
spbench plan -c configs/standard.yaml       # the index -> (dataset, tool) mapping
spbench plan -c configs/standard.yaml --count
```

Failed units appear in `results/runs.tsv` with `status=failed` and the error
message. Rerun just those, then rescore:

```bash
sbatch --array=17,31,44 ... scripts/slurm/run_unit.sbatch
spbench evaluate -c configs/standard.yaml -w /scratch/$USER/spbench-standard
```

Because scoring reads only the common-format prediction files, `spbench
evaluate` can also be rerun with a different `match_threshold` without
re-running a single tool.

Sizing: strainphase's longitudinal mode holds all timepoints for one MAG in
memory at once and is the peak-RSS driver; 32 GB is comfortable for the standard
tier's 200 kb contigs at 60x. Strainy is the wall-time driver because it runs an
assembler. Both defaults in `submit.sh` are deliberately generous - check
`results/runs.tsv` after a first run and tighten from measured numbers.

---

## Layout

```
benchmark/
├── Makefile                 smoke / standard / real / test / clean
├── configs/
│   ├── smoke.yaml           functional check, ~5 min, no external tools
│   ├── standard.yaml        the reported sweep: k x divergence x coverage x 3 seeds
│   └── real-genomes.yaml    the same on real RefSeq assemblies
├── envs/                    one conda environment per comparator, and why
├── data/genomes.tsv         accessions + checksums for the real-genome tier
├── expected/smoke.json      regression bounds checked by CI
├── scripts/
│   ├── fetch_genomes.py     downloader with checksum verification
│   └── slurm/               three-stage cluster submission
├── spbench/
│   ├── formats.py           the common intermediate format both sides agree on
│   ├── simulate.py          ground-truth simulator
│   ├── reads.py             allele extraction + shared consensus derivation
│   ├── adapters/            one file per tool; each implements partition()
│   ├── metrics/             partition / haplotype / longitudinal
│   ├── evaluate.py          scoring; imports no tool
│   ├── runner.py            orchestration + provenance
│   └── report.py            markdown report and figure
└── tests/                   unit tests for the metrics and the simulator
```

## Outputs

| File | Contents |
|---|---|
| `report.md` | The narrative report: tools, like-for-like table, detection sensitivity, resources, caveats |
| `per_sample.tsv` | One row per dataset x tool x sample x contig, every metric |
| `longitudinal.tsv` | Per-contig detection and cross-timepoint summaries |
| `detection.tsv` | One row per (sample, true strain): abundance and whether it was recovered |
| `runs.tsv` | Status, wall time, peak RSS, and each tool's declared scope |
| `provenance.json` | Commit, platform, versions, seeds, thresholds |
| `detection_vs_abundance.png` | Recall against abundance, per tool (needs matplotlib) |

---

## Adding a tool

One adapter, no metric changes:

```python
class MyToolAdapter(Adapter):
    info = ToolInfo(
        name="mytool",
        designed_for="...",           # printed in the report next to its scores
        citation="...",
        supports_cross_sample_ids=False,
        multi_sample=False,
        requires=["mytool"],          # missing -> `skipped` row, not a crash
    )

    def partition(self, dataset, workdir, threads) -> dict[tuple[str, str], str]:
        """Return {(sample, read_id): cluster_id}."""
```

Register it in `spbench/adapters/__init__.py` and name it in a config. The
harness derives consensus haplotypes, scores every metric, and reports it
alongside everything else. That an adapter *cannot* reach into the metrics is
the property that makes "all tools were scored identically" checkable rather
than asserted.

Adapters never receive the truth directory, so a tool cannot accidentally be run
with information it would not have on real data.

---

## Verification status of the third-party adapters

The Floria, Strainy, devider and whatshap adapters shell out to binaries that
cannot be installed in this repository's CI. They were written against each
tool's documented output format; the Floria parser additionally matches a format
this project previously parsed successfully in production.

**Run `spbench check-tools` after installing them.** A format change in a newer
release shows up there as a parse failure on the smoke dataset, with the command
log attached - not as a wrong number in a table. Any tool whose output does not
parse produces a `failed` row carrying the error.

---

## Known limitations

Stated here and reprinted at the bottom of every generated report:

- **No true CCS error model.** `read_model: hifi` aligns with real minimap2 and
  concentrates indel error in homopolymers, but its error parameters are
  literature-shaped approximations rather than a fit to an instrument. PBSIM3
  per strain is the higher-fidelity option.
- **`read_model: exact` datasets are optimistic.** They emit reads at their true
  coordinates, so no tool pays for alignment error. The smoke tier uses this;
  the reported tier does not, and the two are not directly comparable.
- **One reference per dataset.** Cross-species mismapping, a major source of
  false haplotypes in real metagenomes, is absent by construction.
- **Consensus is harness-derived.** Uniform across tools by design, but these
  numbers are therefore not each tool's native output quality. Rows labelled
  `native` in `per_sample.tsv` report native output where a tool supplies it.
- **No per-tool tuning.** Everything runs at published defaults, strainphase
  included. Tuning strainphase alone would invalidate the comparison; tuning
  every tool fairly is a larger exercise than this suite performs.
- **No real-data track with independent ground truth.** The strongest remaining
  gap. A sequenced mock community or an isolate mixture with known membership
  would test what simulation cannot.

## Tools compared

| Tool | Task it was built for | Reference |
|---|---|---|
| `strainphase-longitudinal` | Multi-timepoint strain reconstruction with cross-timepoint rescue and stable lineage identity | this repository |
| `strainphase-single` | The same method, rescue disabled — the like-for-like row | this repository |
| `floria` | Single-sample strain haplotyping via MEC read clustering and strain-preserving network flow | Shaw, Boucher, Yu, Noyes & Li, *Bioinformatics* 40(Suppl 1), 2024 |
| `strainy` | Single-sample phasing and assembly of strain haplotypes from long-read metagenomes | Kazantseva, Donmez, Frolova, Pop & Kolmogorov, *Nature Methods*, 2024 |
| `devider` | Haplotyping short spans (viruses, plasmids, genes) at high coverage | Shaw & Yu, 2025 |
| `whatshap-diploid` | Diploid read-backed phasing — the "why not a standard phaser" control | Patterson et al., *J Comput Biol* 22, 2015 |
| `naive-greedy` | Nothing. A floor. | — |
