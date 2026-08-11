# Exact benchmark pipeline

What actually runs, as of commit `ba4f6fc`. Every value below is the value in the
committed code, not a description of intent. Where the smoke tier and the
reported tier differ, both are given.

This document is the source for a methods section. If you change a default,
change it here in the same commit.

---

## 1. Setup

### Harness

```bash
git clone https://github.com/rolesucsd/strainphase.git
cd strainphase/benchmark
pip install -e .. -e .
```

That installs `strainphase` and `spbench`. Runtime dependencies of the harness,
from `benchmark/pyproject.toml`:

| Package | Constraint | Used for |
|---|---|---|
| `numpy` | >=1.20 | simulation RNG, metric arithmetic |
| `scipy` | >=1.7 | `linear_sum_assignment` (haplotype matching), `gammaln` (AMI) |
| `pandas` | >=1.3 | results tables and the report |
| `pysam` | >=0.19 | BAM/VCF read and write, bgzip, tabix, faidx |
| `PyYAML` | >=6.0 | config parsing |
| `mappy` | >=2.24 | minimap2 bindings, used by `read_model: hifi` |
| `matplotlib` | optional (`[plots]`) | the detection-vs-abundance figure only |

No `samtools`, `minimap2`, `bcftools` or `conda` binary is required — pysam
carries htslib and mappy carries minimap2.

Versions used in the reference run:

```
Python 3.11.15   numpy 2.4.6    scipy 1.17.1   pandas 3.0.5
pysam 0.24.0     mappy 2.31     networkx 3.6.1 strainphase 0.1.0
```

The exact set is recorded per run in `results/provenance.json`, together with
the git commit, whether the tree was dirty, platform and CPU count.

### Comparators

One conda environment each (`benchmark/envs/`), because their pins conflict:

```bash
conda env create -f envs/floria.yml     # floria=0.0.4
conda env create -f envs/strainy.yml    # strainy>=1.1, python=3.9, flye>=2.9
conda env create -f envs/devider.yml    # devider>=0.1
conda env create -f envs/whatshap.yml   # whatshap>=2.2
```

Put their `bin/` directories on `PATH`, then:

```bash
spbench check-tools
```

Anything not on `PATH` becomes a `skipped` row with the reason attached. The run
still completes.

---

## 2. Commands

```bash
make smoke                       # functional check, ~40 s, no comparators
make standard                    # the reported sweep
make test                        # 34 unit tests over the simulator and metrics
```

Cluster:

```bash
cp scripts/slurm/env.sh.example scripts/slurm/env.sh   # edit for your site
scripts/slurm/submit.sh -c configs/standard.yaml \
    -w /scratch/$USER/spbench --env-setup scripts/slurm/env.sh \
    --partition compute --account yourlab --threads 8 --run-mem 32G
```

Three dependent jobs: `simulate` (one) → `run` (array, one task per
`(dataset, tool)` pair, 210 for `standard.yaml`) → `evaluate` (one, `afterany`).

---

## 3. What the two tiers run

### `configs/smoke.yaml`

One dataset, one seed, three tools, `read_model: exact`.

```
n_strains 3   n_mutations 300    contig_length 60_000   n_timepoints 3
coverage 30   mean_read_length 8_000 (sd 2_000, clipped to [3_000, 15_000])
error_rate 0.001   vcf_mode called   include_sweep true   include_rescue_strain true
tools: naive-greedy, strainphase-single, strainphase-longitudinal
```

Deliberately too small for its numbers to mean anything. It exists to prove the
harness works on a clean clone and to guard against regressions in it.

### `configs/standard.yaml`

Ten datasets × three seeds (0, 1, 2) × seven tools = **210 work units**.
`read_model: hifi` and `aligner: minimap2` throughout, except the last row.

| Dataset | Varies | `n_strains` | `n_mutations` | `coverage` | Other |
|---|---|---|---|---|---|
| `k2-div5-cov30` | strain count | 2 | 1000 | 30 | |
| `k4-div5-cov30` | *(reference point)* | 4 | 1000 | 30 | |
| `k8-div5-cov30` | strain count | 8 | 1000 | 30 | |
| `k4-div1-cov30` | divergence | 4 | 200 | 30 | ~0.1% divergence |
| `k4-div10-cov30` | divergence | 4 | 2000 | 30 | |
| `k4-div5-cov10` | coverage | 4 | 1000 | 10 | |
| `k4-div5-cov60` | coverage | 4 | 1000 | 60 | |
| `k4-div5-cov30-indels` | indels | 4 | 1000 | 30 | `indel_fraction: 0.1` |
| `k4-div5-cov30-truthvcf` | variant calling | 4 | 1000 | 30 | `vcf_mode: truth` |
| `k4-div5-cov30-exactreads` | read realism | 4 | 1000 | 30 | `read_model: exact` |

All at `contig_length: 200_000`, `n_timepoints: 4` (the `SimConfig` default).

The last two rows are ablations, not conditions. `-truthvcf` isolates how much
of each tool's low-abundance loss comes from variant calling rather than
phasing; `-exactreads` is identical to `k4-div5-cov30` except that reads are
placed at their true coordinates, so the gap between them measures how much of a
tool's score came from the simulation being clean.

Scoring parameters for both tiers: `match_threshold: 0.99`,
`min_shared_sites: 10`.

---

## 4. Stage 1 — simulation

`spbench/simulate.py`. One `numpy.random.default_rng(seed)` stream drives the
whole dataset, so `(config, seed)` determines the output byte for byte.

### 4.1 Reference

`reference_fasta` unset → uniform random ACGT of `contig_length`, one contig
named `sim_contig_1`. Set → that FASTA, first `n_contigs` records, each
truncated to `contig_length` (0 keeps the full contig). Written as
`reference.fasta` and indexed with `pysam.faidx`.

### 4.2 Strain tree

`strain_0` is the reference itself — realistic, since one strain usually matches
the assembly, and it exercises reference-allele haplotypes.

For `i` in `1 … n_strains-1`:

1. Parent chosen uniformly from the strains already created (`0 … i-1`).
   Sequential coalescence, so relatedness is nested rather than star-shaped.
2. Branch lengths drawn `Exponential(scale=1.0)`, one per non-root strain.
3. `n_mutations` are apportioned across branches in proportion to branch length
   (minimum 1 per branch).

Each mutation:

- position uniform in `[guard+1, len(contig)-guard)` where
  `guard = max_indel_length + 2 = 14`;
- rejected if it falls inside a region this lineage has already deleted, or (for
  indels) if another variant lies within `guard` bp — overlapping germline
  indels are a normalisation problem, not a phasing one;
- indel with probability `indel_fraction`, else SNV;
- SNV alt drawn uniformly from the three bases other than **the allele this
  lineage currently carries**, so a back-mutation never silently no-ops;
- deletion or insertion with probability 0.5 each, length `U{1 … max_indel_length}`,
  written left-anchored in VCF convention.

At `n_mutations: 1000` over 200 kb this is ~0.5% divergence at the leaves;
`k4-div1` is ~0.1%, near the resolution limit for read-based phasing.

### 4.3 Abundance trajectories

Baseline for every strain is a log-space AR(1):

```
log_level ← 0.75 · log_level + N(0, 0.35)      init N(0, 0.8)
raw[strain, t] = exp(log_level)
```

Then two structures are imposed:

- **Sweep** (`include_sweep`, default on, needs ≥2 strains): `strain_0` follows
  a log-linear decline 3.0 → 0.3 across timepoints and `strain_1` the mirror
  rise 0.3 → 3.0. Tests whether a method tracks identity through a large
  abundance change.
- **Rescue strain** (`include_rescue_strain`, default on, needs ≥3 strains): the
  last strain. It is set to `bloom_abundance` (0.30) at one uniformly chosen
  timepoint and to the trough at all others, with the remaining strains scaled
  to fill `1 − target`.

  The trough is **pinned in coverage, not in abundance**:
  `min(0.10, max(1e-4, rescue_trough_coverage / coverage))` with
  `rescue_trough_coverage = 1.5`. At 30× that is 5% (1.5× of reads); at 60× it
  is 2.5%, still 1.5×. A fixed abundance fraction would mean 0.15× at 10×
  coverage — a strain no method can recover because its reads do not exist, so
  the test would degenerate.

Columns are normalised to sum to 1.0 within each timepoint.

### 4.4 Reads

Per sample: `n_reads = round(coverage × contig_length / mean_read_length)`.
Strain drawn from the timepoint's abundance vector. Length
`Normal(mean_read_length, read_length_sd)` clipped to
`[min_read_length, max_read_length]`. Start uniform.

**`read_model: exact`** — walk the reference applying that strain's variants,
emitting sequence and CIGAR directly (M for a matched or substituted base, D for
a deletion, I for an insertion). Deletions that would run past the read's end
stop cleanly on the anchor base rather than emitting a truncated `D`.

Error and quality, per base:

```
p    ~ Uniform(0, 2 × error_rate)     clipped to [1e-6, 0.5]
Q    = clip(round(-10 log10 p), 2, 60)
base = corrupted with probability p   (substitution only)
```

Q therefore means what it says. A simulator emitting flat Q40 would hand an
unearned advantage to quality-weighted methods, strainphase among them.

**`read_model: hifi`** (`spbench/hifi.py`) — each strain's genome is
materialised once; reads are sampled from *strain* coordinates, then:

1. **Substitution error**, per base, at
   `error_rate × hifi_substitution_fraction` (0.35), multiplied by a homopolymer
   context factor that also feeds the emitted Q — so Q drops inside long runs,
   as real CCS quality does.
2. **Indel error**, allocated as a budget of
   `error_rate × (1 − 0.35) × read_length` expected events, distributed across
   homopolymer runs of length ≥ `hifi_homopolymer_min_length` (4) in proportion
   to `(L − 4 + 1) ^ hifi_homopolymer_exponent` (exponent 2, so an 8 bp run is
   ~25× more error-prone than a 4 bp one). Each event contracts (p=0.6) or
   expands (p=0.4) the run by 1–2 bp. Allocating a budget rather than rolling a
   per-run coin keeps the total error rate at the configured value regardless of
   how homopolymer-rich the reference is, so references of different composition
   stay comparable.
3. **Reverse-complemented with probability 0.5.**
4. **Aligned back to the reference** with minimap2 through `mappy`, preset
   `map-hifi`, `n_threads=1`. The best primary hit by matching length wins.
   `ref_start`, CIGAR, MAPQ, strand and soft clips all come from the aligner.
   Reads that fail to align are dropped and the count is logged.

`read_model: hifi` with `aligner: exact` is rejected at config load: simulating
homopolymer slippage and then handing over the true CIGAR removes the placement
ambiguity that is the only reason to simulate it.

**Not modelled in either path:** a true CCS error process (the parameters above
are literature-shaped approximations, not fitted to an instrument), chimeric
reads, PCR duplicates, coverage bias, or cross-species mismapping. For higher
fidelity, simulate per-strain reads with PBSIM3 (multi-pass CLR → `ccs`); the
read-to-strain mapping survives because each strain is simulated separately.

### 4.5 BAM

Reads sorted by `(contig, ref_start)` and written coordinate-sorted with pysam.
`NM` and `RG` tags set; flag 16 for reverse-strand reads; MAPQ from the aligner
(`hifi`) or fixed at 60 (`exact`). Indexed with `pysam.index`.

### 4.6 VCF

For every variant site, depth and alt-supporting count are computed from the
reads overlapping it, with support attributed by each read's **strain of
origin** rather than by re-genotyping the emitted sequence. Sequencing error
moves individual bases but not a read's true haplotype, and the quantity being
modelled is whether a caller would have the statistical power to call the site —
which is driven by how many molecules carry the allele.

- `vcf_mode: called` (default) — a site is emitted only if
  `alt_count ≥ call_min_alt_reads` (3) **and** `AF ≥ call_min_af` (0.02).
  False-positive sites are added at `call_fp_rate` (2e-5 per bp per sample,
  Poisson) with plausible low AF.
- `vcf_mode: truth` — every site with non-zero depth, no filtering.

Written per timepoint as `variants/{sample}.vcf.gz` with `DP` and `AF` in INFO
and `GT:DP:AD` per sample, bgzipped and tabix-indexed. `variants/union.vcf.gz`
holds every site called in any timepoint, for tools that want one variant file
per run.

Handing every tool the exact truth site list would remove the low-abundance
detection problem, which is the problem this benchmark most wants to measure.
Hence `called` as the default and `truth` as an explicit ablation.

### 4.7 Truth and manifest

```
truth/sites.tsv          contig, pos, ref, alt
truth/strains.tsv        strain_id, contig, n_sites, alleles ("pos:allele,…")
truth/abundance.tsv      sample, strain_id, abundance   (sums to 1 per sample)
truth/read_origins.tsv   sample, read_id, contig, strain_id, start, end
manifest.json            full config, sha256 fingerprint, strain tree, samples
```

Datasets whose manifest fingerprint matches the requested config are reused;
changing any config value regenerates them.

**No adapter is ever given `truth/`.** Adapters receive a `Dataset` object
carrying only the reference, BAMs, VCFs, sample list and contig lengths.

---

## 5. Stage 2 — running the tools

`spbench/runner.py` and `spbench/adapters/`.

Each adapter implements one method:

```python
def partition(self, dataset, workdir, threads) -> dict[tuple[str, str], str]:
    """{(sample, read_id): cluster_id}"""
```

Wall time is measured with `time.monotonic()` around that call; peak RSS from
`resource.getrusage` over `RUSAGE_CHILDREN` and `RUSAGE_SELF`. A missing binary
yields `status="skipped"`; an exception yields `status="failed"` with the
message. Neither stops the run.

### 5.1 What each adapter runs

| Tool | Invocation / entry point |
|---|---|
| `naive-greedy` | in-process; `max_mismatch_frac=0.02`, `min_shared_sites=3`, `min_cluster_reads=3` |
| `strainphase-single` | `process_contig(bam, vcf, contig, length, config, sample_id)` per (sample, contig) |
| `strainphase-longitudinal` | `process_mag_longitudinal(...)` over all samples, then `build_lineage_table(...)` |
| `floria` | `floria -b BAM -v VCF -r REF -o OUT -t N --overwrite` → parse `*.haplosets` |
| `strainy` | `strainy --fasta_ref REF --fastq FQ --bam BAM --snp VCF --mode hifi --stage phase --threads N --output OUT` → parse `alignment_phased.bam`, `HP` then `YC` tag |
| `devider` | `devider -b BAM -v VCF -r REF -o OUT -t N` → parse `ids.txt` |
| `whatshap-diploid` | `whatshap phase` then `whatshap haplotag` → parse `HP` tag |

All third-party tools run at published defaults for HiFi. No per-tool tuning was
done for any tool, strainphase included.

### 5.2 strainphase configuration

Both strainphase adapters build a `HaplotyperConfig` with **stock defaults**,
plus two overrides:

- `random_seed = 0` unless the config sets one. strainphase's Louvain
  initialisation and read subsampling are both random; unseeded, two runs on
  identical input give different numbers. (Louvain seeding was added in this
  work — `strainphase/core.py` now passes `random_state=config.random_seed`.)
- `n_workers = threads` from the benchmark config.

The stock defaults that matter most:

```
window_size 20000          max_mismatch_frac 0.01     min_shared_snvs_for_edge 1
min_reads_per_cluster 3    em_max_iter 30             em_tolerance 1e-5
junk_divergence_rate 0.10  merge_distance_threshold 0.01
assign_confidence_threshold 0.80
min_weight_for_anchor 0.20 rescue_match_distance 0.01 rescued_min_weight 0.02
max_link_distance 0.01     min_shared_snvs_for_link 3
min_mapq 20                min_depth_site 3           include_indels True
```

**Read assignment.** For each window, a read is assigned to
`argmax_k γ[i, k]` over the non-junk components, kept only if that posterior is
≥ **0.9** (the adapter's own threshold, separate from strainphase's
`assign_confidence_threshold`). Windows overlap 50%, so most reads are assigned
twice; the assignment from the window with the highest posterior wins. Reads the
model saw but would not commit to are recorded as `UNASSIGNED`, which lowers
`assigned_fraction` rather than being silently dropped.

**Cluster identity.** `strainphase-single` uses `f"{sample}:{contig}:{track_id}"`
— no cross-sample claim. `strainphase-longitudinal` uses the `lineage_id` from
`build_lineage_table`, falling back to the per-sample track when a track never
entered a lineage.

### 5.3 The shared consensus step

This is the part that makes the comparison a comparison. After *any* adapter
returns its partition, the harness derives haplotypes with the same code
(`spbench/reads.py`):

1. `load_sites(vcf, contig)` — PASS records only, classified SNV/del/ins; MNPs
   skipped.
2. `read_alleles(bam, contig, sites, min_mapq=20)` — secondary and supplementary
   alignments skipped. Per read and site, the literal observed allele:
   - SNV: the base at that reference position;
   - deletion: ALT if every deleted position is a `D` in this read, REF if none
     are, no call on partial overlap or if the read does not span the deletion;
   - insertion: ALT if the inserted length at that anchor matches exactly, REF if
     there is no insertion, no call otherwise.
3. Per cluster, majority vote per site weighted by `1 − 10^(−Q/10)`, requiring
   `min_depth = 3` reads and `min_fraction = 0.5` of the weight. Sites below
   that are left uncalled rather than guessed.
4. Abundance = cluster read count ÷ total assigned reads for that sample and
   contig.

So a tool's score reflects how it grouped reads, not whose consensus caller is
better. strainphase additionally emits its native `lineages.tsv` haplotypes,
scored as separate rows labelled `representation=native` and kept out of the
headline table.

Outputs per `(dataset, tool)`:

```
predictions/{dataset}/{tool}/haplotypes.tsv        sample, contig, hap_id, start, end,
                                                   abundance, n_sites, alleles
predictions/{dataset}/{tool}/read_assignments.tsv  sample, contig, read_id, hap_id, confidence
predictions/{dataset}/{tool}/status.json           status, message, wall_seconds, peak_rss_mb,
                                                   options, declared scope
```

---

## 6. Stage 3 — scoring

`spbench/evaluate.py` and `spbench/metrics/`. Reads only the files above and the
truth tables. Imports no tool.

### 6.1 Read partition (`metrics/partition.py`)

From a single contingency table of true strain × predicted cluster, over reads
the tool actually placed: **ARI**, **AMI** (Vinh–Epps–Bailey expected-MI
correction), **homogeneity**, **completeness**, **V-measure**, plus
`assigned_fraction`, cluster count and largest-cluster fraction. Implemented
here rather than pulled from scikit-learn so the arithmetic is checkable against
the formulas in the docstrings.

### 6.2 Haplotype reconstruction (`metrics/haplotype.py`)

- Agreement between a predicted haplotype and a true strain is computed **only
  over positions where both have a call**.
- A pair must share ≥ `min_shared_sites` (10) positions to be comparable.
- Assignment is the **global optimum** — `scipy.optimize.linear_sum_assignment`
  maximising total agreeing *sites* (not fraction, so a 5000-site haplotype
  outranks a lucky 12-site fragment), one predicted haplotype to at most one
  true strain.
- A true strain counts as recovered when its matched pair agrees at
  ≥ `match_threshold` (0.99).

Reported: `hap_precision`, `hap_recall`, `hap_f1`, `k_error`
(`n_predicted − n_true`), `allele_accuracy` and `hamming_error_rate` (weighted
by shared-site count), `switch_error_rate`, `span_n50`, `mean_sites_per_hap`,
and — only for tools that emit an abundance — `abundance_mae`,
`abundance_pearson_r`, `abundance_mae_with_missed`.

**Switch error** generalises the diploid definition to *n* strains: walking a
predicted haplotype's sites in genomic order, the harness tracks the set of true
strains still consistent with everything seen so far and counts a switch when
that set is forced empty and has to be reseeded. It measures chimerism — a
reconstruction that looks like a real strain but never existed.

### 6.3 Detection and longitudinal (`metrics/longitudinal.py`)

One row per `(sample, true strain)` in `detection.tsv`, recording true
abundance, bin, whether it was recovered, and best agreement. Summarised as
recall in abundance bins `<1%`, `1–5%`, `5–20%`, `>20%`; overall recall; and
mean persistence recall (per-strain mean over the timepoints where it is
present, so a strain seen everywhere and one seen nowhere weigh equally).

The report additionally re-bins by **absolute strain depth**
(`abundance × coverage`) into `<1×`, `1–3×`, `3–10×`, `>10×`. Abundance fraction
is not comparable across coverage levels; depth is. The 1–3× band is where
cross-timepoint information can change the outcome.

**Cross-timepoint identity** (ARI between a tool's own haplotype IDs and the
true strains they matched) is computed **only** for adapters declaring
`supports_cross_sample_ids` — currently `strainphase-longitudinal` alone. For
everything else it is reported `n/a`, never 0: Floria, Strainy and devider never
claim that `HAP3` in T1 is the same organism as `HAP3` in T2, and scoring them
on it would manufacture a win.

### 6.4 Aggregation

Every results row is stamped with `seed`, `n_strains_config`, `coverage`,
`n_timepoints`, `error_rate`, `indel_fraction`, `vcf_mode`, `read_model`,
`aligner` and `tool_options`, so the table can be sliced without a config
lookup.

The report's confidence intervals collapse each seed to one value first, then
take a 95% normal interval over the seeds. Rows within a seed are timepoints and
contigs of the same simulated mixture and are not independent observations;
treating them as such would report an interval several times too narrow.

The headline table contains **only single-sample tools**, so every row saw the
same information. `strainphase-longitudinal` appears in the detection section,
where its comparator is `strainphase-single`.

---

## 7. Outputs

```
results/
├── report.md                     tools, like-for-like table, detection by abundance
│                                 and by depth, resources, limitations
├── per_sample.tsv                dataset × tool × representation × sample × contig
├── longitudinal.tsv              per-contig detection and cross-timepoint summaries
├── detection.tsv                 per (sample, true strain): abundance and recovery
├── runs.tsv                      status, wall time, peak RSS, declared scope
├── provenance.json               commit, dirty flag, platform, versions, seeds, thresholds
└── detection_vs_abundance.png    (requires matplotlib)
```

---

## 8. Reproducibility guarantees

- **Seeded end to end.** Two `make smoke` runs produce byte-identical
  `per_sample.tsv`, `longitudinal.tsv` and `detection.tsv`. CI runs the smoke
  tier twice and fails if they differ.
- **`spbench verify`** checks the smoke results against `expected/smoke.json`.
  Those bounds guard the *harness*: a change to the simulator, the allele
  extractor or the metrics that silently moves every number fails the build. A
  legitimate change updates the bounds in the same commit.
- **Stages are separable.** Simulation is cached by config fingerprint; scoring
  reads only the prediction files, so `spbench evaluate` can rescore at a
  different `match_threshold` without re-running a tool.
- **34 unit tests** cover the metrics against independently known answers
  (perfect partition, random partition, deliberate chimera) and the simulator
  end to end — including that alleles observed in the final BAM match each
  read's true strain at >99% (`exact`) and >97% (`hifi`), the check that would
  collapse if any coordinate, strand or CIGAR translation were wrong.

## 9. Verification status

| Component | Status |
|---|---|
| Simulator, both read models | Tested end to end here |
| Metrics | Unit-tested against known answers |
| `naive-greedy`, both strainphase adapters | Run end to end here |
| SLURM three-stage submission | Run end to end locally; produces results identical to `spbench run` |
| Floria / Strainy / devider / whatshap adapters | **Written from documented output formats; not executed.** Run `spbench check-tools` after installing them — it exercises each on the smoke dataset and reports a parse failure rather than a wrong number. |
| `data/genomes.tsv` checksums | **Recorded as `TBD`.** Run `scripts/fetch_genomes.py --record` once and commit. |
