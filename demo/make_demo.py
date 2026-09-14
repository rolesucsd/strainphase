#!/usr/bin/env python3
"""Carve a small, self-contained demo dataset out of the full synthetic benchmark set.

Keeps all 6 timepoints and the full sweep, restricted to one 100 kb slice of contig_1.
Reads fully contained in the slice are kept and their coordinates shifted to the new
origin, so the BAM, VCF, reference and truth tables all share one coordinate system.
"""
import json, os, sys, collections
import pysam

SRC = "/Users/reneeoles/Desktop/strainphase_tests/data/demo_data/div0050_k2.cov60.seed0"
OUT = sys.argv[1]
SRC_CONTIG = "contig_1"
NEW_CONTIG = "demo_contig_1"          # "<mag>_contig_<n>" so MAG grouping has something to parse
START, END = 1_000_000, 1_100_000
LEN = END - START
SAMPLES = ["T1", "T2", "T3", "T4", "T5", "T6"]

os.makedirs(f"{OUT}/bam", exist_ok=True)
os.makedirs(f"{OUT}/variants", exist_ok=True)
os.makedirs(f"{OUT}/truth", exist_ok=True)

# ---- reference -------------------------------------------------------------
ref = pysam.FastaFile(f"{SRC}/reference.fasta")
seq = ref.fetch(SRC_CONTIG, START, END)
with open(f"{OUT}/reference.fasta", "w") as f:
    f.write(f">{NEW_CONTIG}\n")
    for i in range(0, len(seq), 60):
        f.write(seq[i:i + 60] + "\n")
pysam.faidx(f"{OUT}/reference.fasta")
print(f"reference: {NEW_CONTIG}:1-{LEN}")

# ---- per-sample BAM + VCF --------------------------------------------------
kept_reads = {}                       # sample -> {read_id: strain}
for s in SAMPLES:
    src = pysam.AlignmentFile(f"{SRC}/bam/{s}.bam", "rb")
    header = {"HD": {"VN": "1.6", "SO": "coordinate"},
              "SQ": [{"SN": NEW_CONTIG, "LN": LEN}],
              "RG": [{"ID": s, "SM": s, "PL": "PACBIO"}]}
    out = pysam.AlignmentFile(f"{OUT}/bam/{s}.bam", "wb", header=header)
    keep = {}
    for r in src.fetch(SRC_CONTIG, START, END):
        if r.reference_start < START or r.reference_end is None or r.reference_end > END:
            continue
        if r.is_unmapped or r.is_secondary or r.is_supplementary:
            continue
        a = pysam.AlignedSegment(out.header)
        a.query_name = r.query_name
        a.query_sequence = r.query_sequence
        a.query_qualities = r.query_qualities
        a.flag = r.flag & ~0x1            # drop paired bit; these are single long reads
        a.reference_id = 0
        a.reference_start = r.reference_start - START
        a.mapping_quality = r.mapping_quality
        a.cigartuples = r.cigartuples
        a.next_reference_id = -1
        a.next_reference_start = -1
        a.template_length = 0
        a.set_tag("RG", s)
        out.write(a)
        # read name encodes ground truth: "T1|strain1|0000001"
        parts = r.query_name.split("|")
        keep[r.query_name] = parts[1] if len(parts) > 2 else "unknown"
    out.close()
    src.close()
    pysam.index(f"{OUT}/bam/{s}.bam")
    kept_reads[s] = keep

    # VCF: same slice, coordinates shifted, sample renamed to the timepoint
    vin = pysam.VariantFile(f"{SRC}/variants/{s}.vcf.gz")
    hdr = pysam.VariantHeader()
    for line in str(vin.header).splitlines():
        if line.startswith("##") and not line.startswith("##contig") and not line.startswith("##fileformat"):
            hdr.add_line(line)
    hdr.add_line(f"##contig=<ID={NEW_CONTIG},length={LEN}>")
    hdr.add_sample(s)
    vout = pysam.VariantFile(f"{OUT}/variants/{s}.vcf.gz", "wz", header=hdr)
    nv = 0
    for rec in vin.fetch(SRC_CONTIG, START, END):
        new = vout.new_record(contig=NEW_CONTIG, start=rec.start - START,
                              alleles=rec.alleles, id=rec.id,
                              qual=rec.qual, filter=rec.filter.keys() or ["PASS"])
        for k, v in rec.info.items():
            try: new.info[k] = v
            except Exception: pass
        for k, v in rec.samples[0].items():
            try: new.samples[s][k] = v
            except Exception: pass
        vout.write(new)
        nv += 1
    vout.close(); vin.close()
    pysam.tabix_index(f"{OUT}/variants/{s}.vcf.gz", preset="vcf", force=True)

    n1 = sum(1 for v in keep.values() if v == "strain1")
    print(f"  {s}: {len(keep):5d} reads ({n1:5d} strain1 / {len(keep)-n1:5d} strain2), {nv:5d} variants")

# ---- truth tables, restricted to the slice ---------------------------------
with open(f"{SRC}/truth/sites.tsv") as f, open(f"{OUT}/truth/sites.tsv", "w") as g:
    g.write(f.readline().rstrip("\n") + "\n")
    n = 0
    for line in f:
        c, pos, r_, a_ = line.rstrip("\n").split("\t")
        p = int(pos)
        if c == SRC_CONTIG and START < p <= END:
            g.write(f"{NEW_CONTIG}\t{p - START}\t{r_}\t{a_}\n"); n += 1
print(f"truth/sites.tsv: {n} variant sites in slice")

with open(f"{SRC}/truth/strains.tsv") as f, open(f"{OUT}/truth/strains.tsv", "w") as g:
    g.write(f.readline().rstrip("\n") + "\n")
    for line in f:
        sid, c, _n, alleles = line.rstrip("\n").split("\t")
        sub = []
        for tok in alleles.split(","):
            if not tok: continue
            p, b = tok.split(":")
            p = int(p)
            if START < p <= END:
                sub.append(f"{p - START}:{b}")
        g.write(f"{sid}\t{NEW_CONTIG}\t{len(sub)}\t{','.join(sub)}\n")

# region-specific abundance, recomputed from the reads actually retained
with open(f"{OUT}/truth/abundance.tsv", "w") as g:
    g.write("sample\tstrain_id\tabundance\n")
    for s in SAMPLES:
        c = collections.Counter(kept_reads[s].values())
        tot = sum(c.values())
        for strain in ("strain1", "strain2"):
            g.write(f"{s}\t{strain}\t{c.get(strain,0)/tot:.6f}\n")

with open(f"{OUT}/truth/read_origins.tsv", "w") as g:
    g.write("sample\tread_id\tcontig\tstrain_id\n")
    for s in SAMPLES:
        for rid, strain in sorted(kept_reads[s].items()):
            g.write(f"{s}\t{rid}\t{NEW_CONTIG}\t{strain}\n")

src_manifest = json.load(open(f"{SRC}/manifest.json"))
json.dump({
    "name": "strainphase-demo",
    "derived_from": {"dataset": src_manifest["name"],
                     "fingerprint": src_manifest["fingerprint"],
                     "spbench_version": src_manifest["spbench_version"]},
    "region": {"source_contig": SRC_CONTIG, "start": START, "end": END,
               "contig": NEW_CONTIG, "length": LEN},
    "samples": SAMPLES,
    "strains": src_manifest["strains"],
    "archetypes": src_manifest["archetypes"],
    "coverage": src_manifest["settings"]["coverage"],
}, open(f"{OUT}/manifest.json", "w"), indent=2)
print("done")
