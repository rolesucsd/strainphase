"""Scoring and the report.

Nothing here is tool-aware: `spbench score` reads a read partition, the BAMs and
VCFs the tool was given, and the truth tables. That is what lets the report say
every tool was scored by the same code.
"""


def _native(wildcards):
    """strainphase-longitudinal also emits its own haplotypes; nobody else does."""
    if wildcards.tool == "strainphase-longitudinal":
        return [RESULTS / "partitions" / wildcards.dataset / f"{wildcards.tool}.native.tsv"]
    return []


rule score:
    input:
        partition=RESULTS / "partitions" / "{dataset}" / "{tool}.tsv",
        origins=RESULTS / "datasets" / "{dataset}" / "truth" / "read_origins.tsv",
        sites=RESULTS / "datasets" / "{dataset}" / "truth" / "sites.tsv",
        strains=RESULTS / "datasets" / "{dataset}" / "truth" / "strains.tsv",
        abundance=RESULTS / "datasets" / "{dataset}" / "truth" / "abundance.tsv",
        bams=expand(
            RESULTS / "datasets" / "{{dataset}}" / "bam" / "{sample}.bam", sample=SAMPLES
        ),
        vcfs=expand(
            RESULTS / "datasets" / "{{dataset}}" / "variants" / "{sample}.vcf.gz",
            sample=SAMPLES,
        ),
        native=_native,
    output:
        per_sample=RESULTS / "scored" / "{dataset}" / "{tool}" / "per_sample.tsv",
        longitudinal=RESULTS / "scored" / "{dataset}" / "{tool}" / "longitudinal.tsv",
        detection=RESULTS / "scored" / "{dataset}" / "{tool}" / "detection.tsv",
        runs=RESULTS / "scored" / "{dataset}" / "{tool}" / "runs.tsv",
    params:
        dataset=lambda w: dataset_dir(w.dataset),
        outdir=lambda w: RESULTS / "scored" / w.dataset / w.tool,
        native=lambda w, input: f"--native {input.native[0]}" if input.native else "",
        match_threshold=config["scoring"]["match_threshold"],
        min_shared=config["scoring"]["min_shared_sites"],
    resources:
        mem_mb=32000,
    log:
        RESULTS / "logs" / "score" / "{dataset}.{tool}.log",
    conda:
        "../envs/spbench.yaml"
    shell:
        r"""
        spbench score --tool {wildcards.tool} \
            --partition {input.partition} --dataset {params.dataset} \
            --outdir {params.outdir} {params.native} \
            --match-threshold {params.match_threshold} \
            --min-shared-sites {params.min_shared} > {log} 2>&1
        """


rule report:
    input:
        scored=expand(
            RESULTS / "scored" / "{dataset}" / "{tool}" / "per_sample.tsv",
            dataset=DATASETS,
            tool=TOOLS,
        ),
    output:
        report=RESULTS / "report" / "report.md",
    params:
        dirs=lambda w: " ".join(
            str(RESULTS / "scored" / d / t) for d in DATASETS for t in TOOLS
        ),
        outdir=RESULTS / "report",
    log:
        RESULTS / "logs" / "report.log",
    conda:
        "../envs/spbench.yaml"
    shell:
        "spbench report --inputs {params.dirs} --outdir {params.outdir} > {log} 2>&1"
