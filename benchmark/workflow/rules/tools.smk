"""Running the tools.

Single-sample tools (floria, strainy) are invoked once per timepoint and their
partitions merged; the multi-sample one is invoked once over the whole
timecourse. That asymmetry IS the comparison, so it is expressed in the rule
graph rather than hidden inside a wrapper.

PARAMETER POLICY: every tool at its published defaults FOR HIFI. The only
arguments set are ones describing the data (Strainy --mode, Floria -e matched to
the read error rate) rather than ones trading accuracy for accuracy. strainphase
gets the same restraint - a stock HaplotyperConfig with only window size and max
reads matched to production. See benchmark/README.md.
"""

import json


rule strainphase_single:
    """strainphase with cross-timepoint rescue OFF - the like-for-like row.

    One invocation over the dataset, but each timepoint processed independently,
    so it sees exactly what a single-sample tool sees.
    """
    input:
        manifest=RESULTS / "datasets" / "{dataset}" / "manifest.json",
        bams=expand(
            RESULTS / "datasets" / "{{dataset}}" / "bam" / "{sample}.bam", sample=SAMPLES
        ),
        vcfs=expand(
            RESULTS / "datasets" / "{{dataset}}" / "variants" / "{sample}.vcf.gz",
            sample=SAMPLES,
        ),
    output:
        partition=RESULTS / "partitions" / "{dataset}" / "strainphase-single.tsv",
    params:
        dataset=lambda w: dataset_dir(w.dataset),
        # JSON braces would be read as wildcards; lambda defers expansion.
        config=lambda w: json.dumps(config["tool_params"]["strainphase"]),
    threads: config["resources"]["tool_threads"]
    resources:
        mem_mb=64000,
    log:
        RESULTS / "logs" / "tools" / "{dataset}.strainphase-single.log",
    conda:
        "../envs/spbench.yaml"
    shell:
        r"""
        spbench run-strainphase --mode single \
            --dataset {params.dataset} --out {output.partition} \
            --config '{params.config}' --threads {threads} > {log} 2>&1
        """


rule strainphase_longitudinal:
    """The claim. Its comparator is strainphase-single, not the tools below.

    Also emits strainphase's own lineages, scored separately as
    `representation=native` and kept out of the headline table.
    """
    input:
        manifest=RESULTS / "datasets" / "{dataset}" / "manifest.json",
        bams=expand(
            RESULTS / "datasets" / "{{dataset}}" / "bam" / "{sample}.bam", sample=SAMPLES
        ),
        vcfs=expand(
            RESULTS / "datasets" / "{{dataset}}" / "variants" / "{sample}.vcf.gz",
            sample=SAMPLES,
        ),
    output:
        partition=RESULTS / "partitions" / "{dataset}" / "strainphase-longitudinal.tsv",
        native=RESULTS / "partitions" / "{dataset}" / "strainphase-longitudinal.native.tsv",
    params:
        dataset=lambda w: dataset_dir(w.dataset),
        # JSON braces would be read as wildcards; lambda defers expansion.
        config=lambda w: json.dumps(config["tool_params"]["strainphase"]),
    threads: config["resources"]["tool_threads"]
    resources:
        mem_mb=150000,
    log:
        RESULTS / "logs" / "tools" / "{dataset}.strainphase-longitudinal.log",
    conda:
        "../envs/spbench.yaml"
    shell:
        r"""
        spbench run-strainphase --mode longitudinal \
            --dataset {params.dataset} --out {output.partition} \
            --native {output.native} \
            --config '{params.config}' --threads {threads} > {log} 2>&1
        """


rule floria:
    input:
        reference=RESULTS / "datasets" / "{dataset}" / "reference.fasta",
        bam=RESULTS / "datasets" / "{dataset}" / "bam" / "{sample}.bam",
        bai=RESULTS / "datasets" / "{dataset}" / "bam" / "{sample}.bam.bai",
        vcf=RESULTS / "datasets" / "{dataset}" / "variants" / "{sample}.vcf.gz",
    output:
        partition=RESULTS / "partitions" / "{dataset}" / "floria" / "{sample}.tsv",
    params:
        outdir=lambda w: RESULTS / "work" / w.dataset / "floria" / w.sample,
        extra=config["tool_params"]["floria_extra"],
    threads: config["resources"]["tool_threads"]
    resources:
        mem_mb=32000,
    log:
        RESULTS / "logs" / "tools" / "{dataset}.floria.{sample}.log",
    conda:
        "../envs/floria.yaml"
    shell:
        r"""
        (
        floria -b {input.bam} -v {input.vcf} -r {input.reference} \
            -o {params.outdir} -t {threads} --overwrite {params.extra}
        spbench parse --tool floria --indir {params.outdir} \
            --sample {wildcards.sample} --out {output.partition}
        ) > {log} 2>&1
        """


rule strainy:
    input:
        reference=RESULTS / "datasets" / "{dataset}" / "reference.fasta",
        bam=RESULTS / "datasets" / "{dataset}" / "bam" / "{sample}.bam",
        bai=RESULTS / "datasets" / "{dataset}" / "bam" / "{sample}.bam.bai",
        vcf=RESULTS / "datasets" / "{dataset}" / "variants" / "{sample}.vcf.gz",
        fastq=RESULTS / "datasets" / "{dataset}" / "fastq" / "{sample}.fastq.gz",
    output:
        partition=RESULTS / "partitions" / "{dataset}" / "strainy" / "{sample}.tsv",
    params:
        outdir=lambda w: RESULTS / "work" / w.dataset / "strainy" / w.sample,
        mode=config["tool_params"]["strainy_mode"],
    threads: config["resources"]["tool_threads"]
    resources:
        mem_mb=64000,
    log:
        RESULTS / "logs" / "tools" / "{dataset}.strainy.{sample}.log",
    conda:
        "../envs/strainy.yaml"
    shell:
        r"""
        (
        strainy --fasta_ref {input.reference} --fastq {input.fastq} \
            --bam {input.bam} --snp {input.vcf} --mode {params.mode} \
            --stage phase --threads {threads} --output {params.outdir}
        spbench parse --tool strainy --indir {params.outdir} \
            --sample {wildcards.sample} --out {output.partition}
        ) > {log} 2>&1
        """


rule merge_single_sample_partitions:
    """Per-timepoint partitions from a single-sample tool into one timecourse."""
    input:
        parts=expand(
            RESULTS / "partitions" / "{{dataset}}" / "{{tool}}" / "{sample}.tsv",
            sample=SAMPLES,
        ),
    output:
        partition=RESULTS / "partitions" / "{dataset}" / "{tool}.tsv",
    wildcard_constraints:
        tool="floria|strainy",
    conda:
        "../envs/spbench.yaml"
    shell:
        "spbench merge-partitions --inputs {input.parts} --out {output.partition}"
