# Reference genomes

`genomes.tsv` lists the assemblies used by `configs/real-genomes.yaml`. Four
complete RefSeq genomes spanning the GC and repeat-content range of the human
gut: a Proteobacterium, a Bacteroidetes, a Verrucomicrobium and an
Actinobacterium.

They are not distributed with the repository. Fetch them once:

```bash
make genomes          # or: python scripts/fetch_genomes.py --outdir genomes
```

## Checksums

The `sha256` column ships as `TBD`. Checksums are only meaningful if the person
recording them trusted the download they came from, so the first person to fetch
records them deliberately:

```bash
python scripts/fetch_genomes.py --outdir genomes --record
git add data/genomes.tsv && git commit -m "Pin reference genome checksums"
```

After that every fetch is verified, and an upstream assembly that changes under
the same accession becomes a hard error rather than a quietly different result.
