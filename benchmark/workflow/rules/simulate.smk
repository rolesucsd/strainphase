"""Ground truth from real assemblies, then Badread.

Two rules, in this order for a reason: `truth` decides the reference and the
abundance trajectories and writes a plan of per-strain depths, and only then can
read simulation know how much of each strain to make. Splitting them that way
avoids a Snakemake checkpoint - the plan is one file, and the read rule consumes
it per sample.
"""


rule truth:
    """Reference, ground truth and the read plan for one (group, seed).

    Deterministic given the seed: it picks which assembly is the reference and
    which behaviour each strain follows, so replicating over seeds averages out
    both choices rather than baking one in.
    """
    input:
        assemblies=lambda w: GROUPS[group_of(w.dataset)],
    output:
        reference=RESULTS / "datasets" / "{dataset}" / "reference.fasta",
        plan=RESULTS / "datasets" / "{dataset}" / "plan.tsv",
        manifest=RESULTS / "datasets" / "{dataset}" / "manifest.json",
        sites=RESULTS / "datasets" / "{dataset}" / "truth" / "sites.tsv",
        strains=RESULTS / "datasets" / "{dataset}" / "truth" / "strains.tsv",
        abundance=RESULTS / "datasets" / "{dataset}" / "truth" / "abundance.tsv",
    params:
        outdir=lambda w: dataset_dir(w.dataset),
        seed=lambda w: seed_of(w.dataset),
        timepoints=SIM["n_timepoints"],
        coverage=SIM["coverage"],
        asm_preset=SIM["asm_preset"],
    log:
        RESULTS / "logs" / "truth" / "{dataset}.log",
    conda:
        "../envs/spbench.yaml"
    shell:
        r"""
        spbench truth \
            --assemblies {input.assemblies} \
            --outdir {params.outdir} \
            --seed {params.seed} \
            --timepoints {params.timepoints} \
            --coverage {params.coverage} \
            --asm-preset {params.asm_preset} > {log} 2>&1
        """


rule simulate_reads:
    """One timepoint's reads: Badread once per strain, at its own depth.

    Reads are renamed `{sample}|{strain}|{n}` before being concatenated, so read
    provenance survives a real aligner and truth/read_origins.tsv is a fact
    rather than an inference. A strain at zero abundance is absent from the plan
    and so contributes no reads at all.
    """
    input:
        plan=RESULTS / "datasets" / "{dataset}" / "plan.tsv",
    output:
        fastq=RESULTS / "datasets" / "{dataset}" / "fastq" / "{sample}.fastq.gz",
        origins=RESULTS / "datasets" / "{dataset}" / "origins" / "{sample}.tsv",
    params:
        # Carries literal {assembly} / {coverage} placeholders that spbench fills,
        # so it must be a lambda or Snakemake tries to expand them as wildcards.
        reads_cmd=lambda w: SIM["reads_cmd"],
        mean_length=SIM["mean_read_length"],
        length_sd=SIM["read_length_sd"],
        seed=lambda w: seed_of(w.dataset),
    log:
        RESULTS / "logs" / "reads" / "{dataset}.{sample}.log",
    conda:
        "../envs/spbench.yaml"
    shell:
        r"""
        spbench simulate-reads \
            --plan {input.plan} --sample {wildcards.sample} \
            --fastq {output.fastq} --origins {output.origins} \
            --reads-cmd "{params.reads_cmd}" \
            --mean-length {params.mean_length} --length-sd {params.length_sd} \
            --seed {params.seed} > {log} 2>&1
        """


rule collect_read_origins:
    """Per-timepoint origins into the one truth table the scorer reads."""
    input:
        origins=expand(
            RESULTS / "datasets" / "{{dataset}}" / "origins" / "{sample}.tsv",
            sample=SAMPLES,
        ),
    output:
        truth=RESULTS / "datasets" / "{dataset}" / "truth" / "read_origins.tsv",
    shell:
        r"""
        head -1 {input.origins[0]} > {output.truth}
        for f in {input.origins}; do tail -n +2 "$f" >> {output.truth}; done
        """
