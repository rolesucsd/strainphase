# strainphase benchmark

A Snakemake workflow comparing strainphase against Floria and Strainy on
longitudinal mixtures built from **real strain assemblies**. The genomes and
their differences are not simulated; the abundances over time are.

```bash
conda env create -f workflow/envs/spbench.yaml && conda activate spbench
pip install -e .. -e .

$EDITOR config/config.yaml            # point `groups` at your assembly directories
snakemake --use-conda --cores 16      # or --profile <your-slurm-profile>
```

Output: `results/report/report.md`.

---

## Layout

```
benchmark/
├── config/config.yaml        the only file you edit
├── workflow/
│   ├── Snakefile             the DAG
│   ├── rules/
│   │   ├── simulate.smk      truth from assemblies, then Badread
│   │   ├── pipeline.smk      alignment + calling — the production commands
│   │   ├── tools.smk         strainphase x2, floria, strainy
│   │   └── evaluate.smk      scoring and the report
│   └── envs/                 one conda env per tool, and why they are separate
├── spbench/                  the scoring library the rules call
└── tests/                    metrics, abundance model, truth derivation
```

Snakemake owns the DAG, the cluster submission and the resume logic. `spbench`
owns the parts Snakemake cannot express — building ground truth, simulating
reads with exact provenance, running strainphase (whose partition comes from EM
posteriors rather than a file), parsing each tool's native output, and scoring.
Everything that is a shell command lives in a rule where you can read it.

### `pipeline.smk` is meant to be replaced

Its alignment and calling rules are copied from the production Snakemake
workflow so the BAMs and VCFs every tool receives are produced the way the real
analysis produces them. **When the production Snakefile is split into rule
modules, replace this file with an `include:` of the real one** and the
duplication disappears. Until then it is the file to diff against production.

Two deliberate departures, both because the benchmark's input differs rather
than its processing:

- no `samtools fastq` step — Badread already emits FASTQ;
- no per-MAG sharding for SNooPy — a dataset is already one strain group on its
  own reference, which is what the sharding produces.

---

## The DAG

```
truth ──> simulate_reads ──> align ──> call_variants ──> union_sites
                               │                              │
                               └──────────────┬───────────────┘
                                              v
                     strainphase-single / strainphase-longitudinal
                                  floria / strainy
                                              v
                                            score
                                              v
                                           report
```

`truth` decides the reference and the abundance trajectories and writes a plan
of per-strain depths; only then can read simulation know how much of each strain
to make. Splitting them that way avoids a Snakemake checkpoint.

Single-sample tools are invoked **once per timepoint** and their partitions
merged; the multi-sample one is invoked **once** over the timecourse. That
asymmetry is the comparison, so it lives in the rule graph rather than inside a
wrapper.

---

## Input

One directory of assemblies per strain group:

```yaml
groups:
  group6: assemblies/group6      # 6 closely related strains, one FASTA each
  group4: assemblies/group4
  group3: assemblies/group3
```

Strain id is the filename stem; `.fasta`, `.fa`, `.fna`, optionally gzipped.
Nothing else is required — reads, BAMs, VCFs and ground truth are all generated.

**One assembly is drawn as the reference** (by seed) and the rest are the strains
in the mixture. Their true haplotypes come from `minimap2 -cx asm5` of each
assembly against that reference. Drawing the reference from the group is
deliberate: every real analysis phases against an assembly that is itself one
strain's genome, so the reference is never equidistant from the members and one
strain always matches it exactly. A neutral consensus reference would be a
friendlier setup than anything that happens in practice.

Replicating over `seeds` averages out both the reference choice and the
trajectories.

---

## Abundance archetypes

The only invented part of a dataset. Trajectories come from named behaviours
rather than noise, so the report can be read per-behaviour and "the method
missed the colonisation events" is a checkable statement.

| Archetype | Shape | Why it is in the set |
|---|---|---|
| `stable` | constant plus small noise | the resident background |
| `bloom` | Gaussian pulse, interior peak | disturbance response; tests tracking through a large abundance change |
| `colonisation` | **exactly 0**, then logistic growth | a strain arriving. Cross-timepoint methods should handle it; single-sample methods cannot see it coming |
| `decline` | exponential loss, sometimes to 0 | clearance |
| `sweep_winner` / `sweep_loser` | crossing exponentials | replacement; the winner ends as far above the loser as it started below |

Assignment guarantees at least one `colonisation` and one `bloom` per group with
≥3 strains. A dataset of six stable residents would test nothing longitudinal.

A strain at zero abundance is **absent from the read plan entirely**, so it
contributes no reads at all — which makes it a false-positive test rather than a
low-abundance one.

---

## Parameter policy

Every tool runs at **its published defaults for PacBio HiFi**. That phrasing is
doing work: "defaults" cannot mean the literal argument-free invocation, because
these tools have separate Nanopore and HiFi configurations and running HiFi data
under Nanopore assumptions would be a strawman, not a fair test.

| | Set it | Example |
|---|---|---|
| **Describes the data** | Yes — interface, not tuning | Strainy `--mode hifi`; Floria `-e 0.001` to match the read error rate |
| **Trades accuracy for accuracy** | No — a sensitivity sweep if anywhere | MEC thresholds, cluster-count priors, coverage cutoffs |

Setting Floria's error-rate parameter to the actual error rate of the reads is
not tuning Floria — it is telling Floria what it asks to be told.

The same restraint applies to strainphase, which is the part that matters: a
**stock `HaplotyperConfig`** with only `window_size` and `max_reads_per_window`
matched to the production workflow, plus a fixed `random_seed` for
reproducibility. No threshold, no merge distance, no rescue parameter touched.

### The site list is a bigger fairness question than parameters

The production pipeline builds a **cohort union** VCF: a site polymorphic in any
one timepoint is genotyped in all of them, so a strain that sweeps to fixation or
drops out keeps a full trajectory. That union is a multi-sample advantage with
nothing to do with the phasing algorithm.

So the `sample_vcf` rule gives **every tool the identical VCF** — including the
union. Floria and Strainy benefit from the better site list too, which is the
point. Set `use_union_vcf: false` for the ablation isolating how much of a result
comes from the union rather than from the phasing; it is the first thing a
careful reviewer will ask for.

---

## What gets measured

| Family | Metrics | Applies to |
|---|---|---|
| Read partition | ARI, AMI, homogeneity, completeness, V-measure, fraction of reads placed | every tool |
| Haplotype reconstruction | precision / recall / F1, allele (Hamming) error, switch error, span N50, strain-count error | every tool |
| Abundance | MAE, Pearson r, MAE charging missed strains | tools reporting abundance |
| Detection sensitivity | recall stratified by true abundance **and** by absolute strain depth | every tool |
| Cross-timepoint identity | ARI of the tool's own IDs against true strain identity | tools claiming stable IDs |

**Every tool is scored on its read partition**, with consensus haplotypes derived
by one function for all of them. Floria emits vartigs, Strainy emits assembly
graph paths, strainphase emits window-linked tracks; scoring those natively would
compare three consensus callers as much as three phasing algorithms.

**strainphase competes against itself.** `strainphase-single` is the row beside
Floria and Strainy; the longitudinal claim is the gap to
`strainphase-longitudinal` — identical code, one flag apart.

**Columns a tool does not claim read `n/a`, not zero.** Cross-timepoint identity
is scored only for tools that claim it.

Two things to know before reading a number:

- **Matching is a global optimum**, by Hungarian assignment on agreeing-site
  counts, so results do not depend on file ordering. A pair must share at least
  `min_shared_sites` (10) positions to be comparable.
- **Recall must be read against the haplotype count.** Matching is one-to-one, so
  a method emitting many fragments will match a rare strain with one by luck.

Read provenance is exact despite the real aligner: Badread runs once per strain
and its reads are renamed `{sample}|{strain}|{n}` before the per-sample FASTQs
are concatenated. `truth/read_origins.tsv` is a fact, not an inference.

---

## Known limitations

Reprinted at the bottom of every generated report:

- **One species per dataset.** Cross-species mismapping, a real source of false
  haplotypes in whole metagenomes, is absent by construction.
- **Truth stops where the asm5 alignment stops.** Accessory genome and large
  rearrangements are outside the scored region. A strain carrying a third allele
  at a shared site is left uncalled there rather than scored as reference.
- **Consensus is harness-derived**, uniformly across tools by design — so these
  are not each tool's native output quality. Rows labelled `native` report native
  output where a tool supplies it.
- **No structural variants.** Production passes `--sv-sidecars` to strainphase;
  here it runs without them, and Badread would not produce the structural
  variation Sniffles looks for anyway.
- **Reads are simulated.** No tool pays for library artefacts, chimeras or
  coverage bias. A sequenced mock community remains the strongest missing
  evidence.

## Tools compared

| Tool | Task it was built for | Reference |
|---|---|---|
| `strainphase-longitudinal` | Multi-timepoint reconstruction with cross-timepoint rescue and stable lineage identity | this repository |
| `strainphase-single` | The same method, rescue disabled — the like-for-like row | this repository |
| `floria` | Single-sample strain haplotyping via MEC read clustering and network flow | Shaw, Boucher, Yu, Noyes & Li, *Bioinformatics* 40(Suppl 1), 2024 |
| `strainy` | Single-sample phasing and assembly of strain haplotypes | Kazantseva, Donmez, Frolova, Pop & Kolmogorov, *Nature Methods*, 2024 |
