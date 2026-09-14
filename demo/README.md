# Demo dataset

A small, self-contained dataset for trying StrainPhase end to end. It runs in well
under a minute on a laptop and ships with ground truth, so the output can be checked
rather than just inspected.

## Run it

```bash
bash demo/run_demo.sh
```

That phases all six timepoints and then prints the recovered trajectories beside the
true strain frequencies. Output lands in `demo/output/` (pass a different directory as
the first argument to put it elsewhere).

To run the phasing step on its own:

```bash
strainphase longitudinal \
  --samples T1,T2,T3,T4,T5,T6 \
  --bams "demo/bam/{sample}.bam" \
  --vcfs "demo/variants/{sample}.vcf.gz" \
  --reference demo/reference.fasta \
  --output-dir demo/output
```

## What it contains

Two strains at 50 SNVs/kb divergence over a 100 kb contig, sampled at six timepoints
at ~60x. The strains sweep past each other: `strain1` starts at 0.3% and ends at
fixation while `strain2` does the reverse. That makes the dataset exercise the parts
of the method that matter — cross-window linking, cross-sample merging, and the
cross-timepoint rescue that recovers a strain at a timepoint where it is too rare to
phase from that timepoint's reads alone.

| | T1 | T2 | T3 | T4 | T5 | T6 |
|---|---|---|---|---|---|---|
| strain1 | 0.003 | 0.029 | 0.372 | 0.918 | 0.992 | 1.000 |
| strain2 | 0.997 | 0.971 | 0.628 | 0.082 | 0.008 | 0.000 |

```
demo/
  reference.fasta(.fai)    100 kb reference, contig demo_contig_1
  bam/T{1..6}.bam(.bai)    aligned reads, one BAM per timepoint
  variants/T{1..6}.vcf.gz  per-timepoint variant calls
  truth/                   ground truth (see below)
  expected_output/         lineages.tsv and windows_across_samples.tsv from a reference run
  manifest.json            provenance, including the source dataset fingerprint
  make_demo.py             the script that carved this out of the full benchmark set
```

`truth/` holds `abundance.tsv` (per-sample strain frequencies), `sites.tsv` (the
positions where the two strains differ), `strains.tsv` (each strain's allele at every
such position) and `read_origins.tsv` (the strain each read came from).

## Expected result

Twelve lineages are reported, of which two span all ten windows. Those two are the
real strains; the remaining ten are short single-window fragments that did not merge,
which is expected. The two full-length lineages recover the sweep to a mean absolute
error of about 0.04:

```
demo_contig_1_LIN000000  ->  strain2   MAE = 0.042
demo_contig_1_LIN000001  ->  strain1   MAE = 0.044
```

Exact numbers may shift slightly between releases as the method changes;
`expected_output/` records the run this README describes.

## Reading zeros in the output

`lineages.tsv` has one row per (lineage, sample). An abundance of `0.0` with a
non-zero `total_reads` is a **measured** zero: reads were present at that locus and
none supported the lineage. A (lineage, sample) pair with **no row at all** was not
measured there — the window did not meet the depth floor, or nothing phased. The two
are different, and only the first is evidence of absence. The check script prints the
measured zeros in this dataset; T1 is the interesting case, where `LIN000001` has
`abundance=0.0` against `total_reads=304`.

## Provenance

Carved from the synthetic benchmark set `div0050_k2.cov60.seed0` (fingerprint
`dfbb293e316266d4`), restricted to `contig_1:1,000,000-1,100,000` with coordinates
shifted to the new origin. Reads fully contained in that interval were kept; the
reference, variant calls and truth tables were subset to match, so everything shares
one coordinate system. `make_demo.py` records exactly how.
