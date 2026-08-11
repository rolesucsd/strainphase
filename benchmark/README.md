# strainphase benchmark suite

Compares strainphase against Floria and Strainy on longitudinal mixtures built
from **real strain assemblies**. The genomes and their differences are not
simulated; the abundances over time are.

```bash
conda env create -f envs/spbench.yml && conda activate spbench
pip install -e .. -e .

cp configs/example.yaml configs/mine.yaml
$EDITOR configs/mine.yaml         # point at your assemblies, paste your commands

make check CONFIG=configs/mine.yaml    # can everything run?
make run   CONFIG=configs/mine.yaml
```

---

## The design in one paragraph

Each dataset is one **strain group**: a set of closely related assemblies of the
same species. One of them is drawn as the reference; the rest are the strains in
the mixture. Their true haplotypes come from `minimap2 -cx asm5` of each
assembly against that reference — real strain variation, including whatever
indel and repeat structure the organisms actually have. Abundance trajectories
over the timecourse are simulated from named biological archetypes. Reads come
from **Badread**, run per strain at its abundance × coverage. Alignment and
variant calling run **your commands**, declared in the config. Every tool then
receives the same BAMs and VCFs your real analysis would produce.

### Why each piece is what it is

**Real assemblies, not simulated mutations.** A simulated genome cannot
reproduce a real organism's composition, repeat structure or homopolymer
landscape, and those are exactly what makes phasing hard. Using real close
relatives means the mutations are real mutations, at real spacing, in real
context.

**The reference is one of the group.** Every real analysis phases against an
assembly that is itself one strain's genome, or a MAG close to one — so the
reference is never equidistant from the members, and one strain always matches
it exactly. Drawing a "neutral" consensus reference would be a friendlier setup
than anything that happens in practice.

**Badread for reads.** It is what Strainy's own HiFi benchmark used. Running a
comparator's own read simulator removes the obvious objection, and it is better
calibrated than anything written for this repository would be.

**Your pipeline for alignment and calling.** `align_cmd` and `call_cmd` are
command templates in the config. Whatever they run produces the BAMs and VCFs
every tool sees, so the benchmark measures the pipeline you actually use rather
than an approximation of it.

**strainphase competes against itself.** `strainphase-single` is strainphase
with cross-timepoint rescue disabled, one timepoint at a time. That is the row
that belongs next to Floria and Strainy. The longitudinal claim is the gap
between it and `strainphase-longitudinal` — identical code, one flag apart.

**Every tool is scored on its read partition.** Floria emits vartigs, Strainy
emits assembly graph paths, strainphase emits window-linked tracks. Comparing
those directly compares three consensus callers as much as three phasing
algorithms. The harness takes each tool's read partition and derives the
consensus with the same code for all of them.

---

## Input

One directory of assemblies per strain group:

```
assemblies/
├── group6/     6 closely related strains, one FASTA each
├── group4/     4
└── group3/     3
```

Strain id is the filename stem. `.fasta`, `.fa`, `.fna`, optionally gzipped.
Nothing else is required — reads, BAMs, VCFs and ground truth are all generated.

Point the config at them and set the commands:

```yaml
datasets:
  - name: group6
    assemblies: assemblies/group6
    coverage: 60
    n_timepoints: 6
    align_cmd: >-
      minimap2 -ax map-hifi -t {threads} --secondary=no {reference} {fastq}
      | samtools sort -@ {threads} -o {bam} - && samtools index {bam}
    call_cmd: >-
      run_clair3.sh --bam_fn={bam} --ref_fn={reference} ... --output={vcf_dir}
    call_output: merge_output.vcf.gz
```

Placeholders available to the commands: `{reference} {fastq} {bam} {vcf}
{vcf_dir} {sample} {threads}`. An unknown placeholder is an error, not an empty
string.

`spbench check-env -c <config>` reports which of those binaries are missing and
whether the assembly directories exist — run it before a cluster job.

---

## Abundance archetypes

The only invented part of a dataset. Trajectories are drawn from named
behaviours rather than from noise, so the report can be read per-behaviour and
"the method missed the colonisation events" is a statement that can be checked.

| Archetype | Shape | Why it is in the set |
|---|---|---|
| `stable` | constant plus small noise | the resident background |
| `bloom` | Gaussian pulse, interior peak | disturbance response; tests tracking through a large abundance change |
| `colonisation` | **exactly 0**, then logistic growth | a strain arriving. Cross-timepoint methods should handle it; single-sample methods cannot see it coming |
| `decline` | exponential loss, sometimes to 0 | clearance |
| `sweep_winner` / `sweep_loser` | crossing exponentials | replacement; the winner ends as far above the loser as it started below |

Assignment guarantees at least one `colonisation` and one `bloom` per group with
≥3 strains. A dataset of six stable residents would test nothing longitudinal.

Strains at exactly zero abundance contribute **no reads at all**, which is what
makes them a false-positive test rather than a low-abundance test.

---

## What gets measured

| Family | Metrics | Applies to |
|---|---|---|
| Read partition | ARI, AMI, homogeneity, completeness, V-measure, fraction of reads placed | every tool |
| Haplotype reconstruction | precision / recall / F1, allele (Hamming) error, switch error, span N50, strain-count error | every tool |
| Abundance | MAE, Pearson r, MAE charging missed strains | tools that report abundance |
| Detection sensitivity | recall stratified by true abundance **and** by absolute strain depth | every tool |
| Cross-timepoint identity | ARI of the tool's own IDs against true strain identity | tools that claim stable IDs |
| Cost | wall time, peak RSS | every tool |

Two things to know before reading a number:

- **Matching is a global optimum**, by Hungarian assignment on agreeing-site
  counts, so results do not depend on file ordering. A pair must share at least
  `min_shared_sites` (10) positions to be comparable.
- **Recall must be read against the haplotype count.** Matching is one-to-one,
  so a method emitting many fragments will match a rare strain with one of them
  by luck. The report prints haplotypes-per-sample in the same table.

Read provenance is exact despite the real aligner: Badread runs once per strain
and its reads are renamed with that strain's id before the per-sample FASTQs are
concatenated. `truth/read_origins.tsv` is a fact, not an inference.

---

## Layout

```
benchmark/
├── configs/example.yaml     copy this; it is the only thing you edit
├── envs/                    conda environments, and why they are separate
├── scripts/slurm/           three-stage cluster submission
├── spbench/
│   ├── strains.py           assemblies in, ground truth out (minimap2 asm5 + cs)
│   ├── abundance.py         the biological archetypes
│   ├── simulate.py          orchestration: Badread, your aligner, your caller
│   ├── formats.py           the common intermediate format
│   ├── reads.py             allele extraction + shared consensus derivation
│   ├── adapters/            strainphase x2, floria, strainy
│   ├── metrics/             partition / haplotype / longitudinal
│   ├── evaluate.py          scoring; imports no tool
│   └── report.py            markdown report and figure
└── tests/                   metrics, abundance, truth derivation
```

## Outputs

| File | Contents |
|---|---|
| `report.md` | tools, like-for-like table, detection by abundance and by depth, resources, limitations |
| `per_sample.tsv` | dataset × tool × sample × contig, every metric |
| `longitudinal.tsv` | per-contig detection and cross-timepoint summaries |
| `detection.tsv` | per (sample, true strain): abundance and whether it was recovered |
| `runs.tsv` | status, wall time, peak RSS, declared scope |
| `provenance.json` | commit, platform, versions, seeds, thresholds |

---

## Cluster

```bash
cp scripts/slurm/env.sh.example scripts/slurm/env.sh   # conda activation, PATHs
scripts/slurm/submit.sh -c configs/mine.yaml -w /scratch/$USER/spbench \
    --env-setup scripts/slurm/env.sh --partition compute --account yourlab
```

Three dependent stages: simulate (one job) → run (array, one task per
`(dataset, tool)` pair) → evaluate (one job, `afterany`). The simulate stage is
now the expensive one — it runs Badread, alignment and variant calling for every
timepoint — so give it real time and memory.

---

## Known limitations

Reprinted at the bottom of every generated report:

- **One species per dataset.** Cross-species mismapping, a real source of false
  haplotypes in whole metagenomes, is absent by construction.
- **Truth stops where the asm5 alignment stops.** Accessory genome and large
  rearrangements are outside the scored region. A strain carrying a third allele
  at a site is left uncalled there rather than scored as reference.
- **Consensus is harness-derived**, uniformly across tools by design — so these
  numbers are not each tool's native output quality. Rows labelled `native` in
  `per_sample.tsv` report native output where a tool supplies it.
- **No per-tool tuning.** Everything runs at published defaults, strainphase
  included.
- **Reads are simulated.** No tool pays for library artefacts, chimeras or
  coverage bias. A sequenced mock community remains the strongest missing
  evidence.

## Tools compared

| Tool | Task it was built for | Reference |
|---|---|---|
| `strainphase-longitudinal` | Multi-timepoint reconstruction with cross-timepoint rescue and stable lineage identity | this repository |
| `strainphase-single` | The same method, rescue disabled — the like-for-like row | this repository |
| `floria` | Single-sample strain haplotyping via MEC read clustering and strain-preserving network flow | Shaw, Boucher, Yu, Noyes & Li, *Bioinformatics* 40(Suppl 1), 2024 |
| `strainy` | Single-sample phasing and assembly of strain haplotypes from long-read metagenomes | Kazantseva, Donmez, Frolova, Pop & Kolmogorov, *Nature Methods*, 2024 |
