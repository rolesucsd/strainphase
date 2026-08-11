"""Alignment and variant calling — the production commands.

These rules are copied from the production Snakemake workflow
(`minimap2_align_sample`, `sort_bam_sample`, the SNooPy chain,
`snoopy_union_sites`) so the BAMs and VCFs every tool receives are produced the
way the real analysis produces them.

They live in their own module on purpose. When the production Snakefile is split
into rule modules, this file should be replaced by an `include:` of the real one
and the duplication disappears. Until then, this is the file to diff against it.

Two deliberate departures from production, both because the benchmark's input
differs, not because the processing does:
  - no `samtools fastq` step: Badread already emits FASTQ;
  - no per-MAG sharding for SNooPy: a dataset is already one strain group on its
    own reference, which is what the sharding produces.
"""


rule index_reference:
    input:
        reference=RESULTS / "datasets" / "{dataset}" / "reference.fasta",
    output:
        fai=RESULTS / "datasets" / "{dataset}" / "reference.fasta.fai",
    conda:
        "../envs/spbench.yaml"
    shell:
        "samtools faidx {input.reference}"


rule align:
    """minimap2_align_sample + sort_bam_sample.

    -N 10 keeps secondaries for multicopy MGE visibility; --MD --eqx are kept
    because they change the CIGAR the downstream tools read (=/X rather than M).
    """
    input:
        reference=RESULTS / "datasets" / "{dataset}" / "reference.fasta",
        fai=RESULTS / "datasets" / "{dataset}" / "reference.fasta.fai",
        fastq=RESULTS / "datasets" / "{dataset}" / "fastq" / "{sample}.fastq.gz",
    output:
        bam=RESULTS / "datasets" / "{dataset}" / "bam" / "{sample}.bam",
        bai=RESULTS / "datasets" / "{dataset}" / "bam" / "{sample}.bam.bai",
    params:
        max_secondary=10,
    threads: config["resources"]["align_threads"]
    resources:
        mem_mb=64000,
    log:
        RESULTS / "logs" / "align" / "{dataset}.{sample}.log",
    conda:
        "../envs/spbench.yaml"
    shell:
        r"""
        (
        minimap2 -ax map-hifi -t {threads} -N {params.max_secondary} --MD --eqx \
            {input.reference} {input.fastq} \
        | samtools sort -@ {threads} -o {output.bam} -
        samtools index {output.bam}
        ) > {log} 2>&1
        """


rule call_variants:
    """SNooPy, the production workflow's SNV source (run_snoopy: true).

    Blocks are atomized (MNP -> SNVs), indels kept, then `bcftools norm -a -m -`
    canonicalizes: left-align, split complex and multiallelic records into atomic
    biallelic ones.
    """
    input:
        reference=RESULTS / "datasets" / "{dataset}" / "reference.fasta",
        fai=RESULTS / "datasets" / "{dataset}" / "reference.fasta.fai",
        bam=RESULTS / "datasets" / "{dataset}" / "bam" / "{sample}.bam",
        bai=RESULTS / "datasets" / "{dataset}" / "bam" / "{sample}.bam.bai",
    output:
        vcf=RESULTS / "datasets" / "{dataset}" / "variable" / "{sample}.vcf.gz",
        tbi=RESULTS / "datasets" / "{dataset}" / "variable" / "{sample}.vcf.gz.tbi",
    params:
        workdir=lambda w: RESULTS / "datasets" / w.dataset / "snoopy" / w.sample,
    threads: config["resources"]["call_threads"]
    resources:
        mem_mb=24000,
    log:
        RESULTS / "logs" / "call" / "{dataset}.{sample}.log",
    conda:
        "../envs/snoopy.yaml"
    shell:
        r"""
        (
        mkdir -p {params.workdir}
        snoopy -b {input.bam} -r {input.reference} -o {params.workdir} -t {threads}
        python workflow/scripts/snoopy_to_vcf.py \
            --snoopy {params.workdir} --ref {input.reference} \
            --sample {wildcards.sample} --output {params.workdir}/raw.vcf.gz
        bcftools norm -a -m - -f {input.reference} -Oz \
            -o {output.vcf} {params.workdir}/raw.vcf.gz
        tabix -f -p vcf {output.vcf}
        ) > {log} 2>&1
        """


rule union_sites:
    """snoopy_union_sites: a site polymorphic in >=1 timepoint is queried in ALL.

    This is what keeps a full trajectory for a strain that sweeps to fixation or
    drops out. It is also a multi-sample advantage that has nothing to do with
    phasing, so the union is handed to EVERY tool identically — see
    `use_union_vcf` below. Giving it to strainphase alone would make the
    comparison meaningless.
    """
    input:
        vcfs=expand(
            RESULTS / "datasets" / "{{dataset}}" / "variable" / "{sample}.vcf.gz",
            sample=SAMPLES,
        ),
    output:
        vcf=RESULTS / "datasets" / "{dataset}" / "union_sites.vcf.gz",
        tbi=RESULTS / "datasets" / "{dataset}" / "union_sites.vcf.gz.tbi",
    resources:
        mem_mb=16000,
    log:
        RESULTS / "logs" / "union" / "{dataset}.log",
    conda:
        "../envs/spbench.yaml"
    shell:
        r"""
        (
        bcftools concat -a -Ou {input.vcfs} \
        | bcftools sort -Ou \
        | bcftools norm -d exact -Ou \
        | bcftools view -G -Oz -o {output.vcf}
        tabix -f -p vcf {output.vcf}
        ) > {log} 2>&1
        """


rule sample_vcf:
    """The VCF every tool gets for a sample.

    With `use_union_vcf` (the default, matching production) this is the cohort
    union, identical for every sample and every tool. Set it false for the
    ablation that isolates how much of a result comes from the union rather than
    from the phasing.
    """
    input:
        vcf=lambda w: (
            RESULTS / "datasets" / w.dataset / "union_sites.vcf.gz"
            if config.get("use_union_vcf", True)
            else RESULTS / "datasets" / w.dataset / "variable" / f"{w.sample}.vcf.gz"
        ),
    output:
        vcf=RESULTS / "datasets" / "{dataset}" / "variants" / "{sample}.vcf.gz",
        tbi=RESULTS / "datasets" / "{dataset}" / "variants" / "{sample}.vcf.gz.tbi",
    conda:
        "../envs/spbench.yaml"
    shell:
        r"""
        cp {input.vcf} {output.vcf}
        tabix -f -p vcf {output.vcf}
        """
