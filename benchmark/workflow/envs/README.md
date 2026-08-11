# Environments

`spbench.yml` holds the harness **and the pipeline** — Badread, minimap2,
samtools and a variant caller — because simulation now runs those tools for
real rather than approximating them.

Floria and Strainy get their own environments; their pins conflict with each
other and with the harness.

```bash
conda env create -f envs/spbench.yml && conda activate spbench
pip install -e .. -e .

conda env create -f envs/floria.yml
conda env create -f envs/strainy.yml
```

Put the comparator environments' `bin/` directories on `PATH`, then:

```bash
spbench check-env -c configs/example.yaml   # are the pipeline commands runnable?
spbench check-tools                          # are Floria and Strainy visible?
```

`check-env` inspects the `align_cmd` / `call_cmd` / `reads_cmd` you configured
and reports which of those binaries are missing — so a wrong command surfaces
before a cluster run, not during one.

## Swapping the variant caller

`clair3` here is a default, not a requirement. The benchmark runs whatever
`call_cmd` says. To use your own caller, change `call_cmd` and `call_output` in
the config and add the package here. The only contract is that the command
writes a VCF at `{vcf_dir}/{call_output}`.
