"""The command line refuses unusable input with a message, and reports failure."""

from __future__ import annotations

import importlib.util
import logging
import os

import pytest

import strainphase.cli as cli
import strainphase.longitudinal as L
from strainphase.core import HaplotyperConfig

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _touch(tmp_path, *names):
    for name in names:
        (tmp_path / name).write_text("")


def _run(tmp_path, vcf="v.vcf.gz"):
    return cli.main(["run", "--bam", str(tmp_path / "a.bam"), "--vcf", str(tmp_path / vcf),
                     "--contig", "c1", "--length", "1000",
                     "--output", str(tmp_path / "out.tsv")])


def test_run_refuses_an_uncompressed_vcf(tmp_path, caplog):
    """A plain .vcf cannot be read by region. Say so, with the command that fixes it,
    instead of failing inside the first window with pysam's 'fetch requires an index'."""
    _touch(tmp_path, "a.bam", "a.bam.bai", "v.vcf")
    with caplog.at_level(logging.ERROR):
        assert _run(tmp_path, vcf="v.vcf") == 1
    assert "bgzip" in caplog.text


def test_run_refuses_an_unindexed_vcf_and_bam(tmp_path, caplog):
    _touch(tmp_path, "a.bam", "v.vcf.gz")
    with caplog.at_level(logging.ERROR):
        assert _run(tmp_path) == 1
    assert "samtools index" in caplog.text and "tabix" in caplog.text


def test_longitudinal_reports_every_bad_input_before_phasing(tmp_path, caplog, monkeypatch):
    """All samples are checked up front, so one run lists everything to fix."""
    _touch(tmp_path, "a.bam", "a.bam.bai", "b.bam", "v.vcf.gz", "ref.fa")
    monkeypatch.setattr(L, "process_mag_longitudinal",
                        lambda **kw: pytest.fail("phasing started on unusable input"))
    with caplog.at_level(logging.ERROR):
        rc = cli.main(["longitudinal", "--samples", "a,b",
                       "--bams", str(tmp_path / "{sample}.bam"),
                       "--vcfs", str(tmp_path / "v.vcf.gz"),
                       "--reference", str(tmp_path / "ref.fa"),
                       "--output-dir", str(tmp_path / "out")])
    assert rc == 1
    for expected in ("b.bam", "VCF is not indexed", "samtools faidx"):
        assert expected in caplog.text


def test_a_failing_contig_stops_the_run(monkeypatch):
    """A contig that raises must not be dropped from the output with only a warning."""
    def boom(**kwargs):
        raise ValueError("fetch requires an index")

    monkeypatch.setattr(L, "process_contig", boom)
    with pytest.raises(ValueError, match="fetch requires an index"):
        L.process_mag_longitudinal(
            mag_name="M", mag_contigs={"c1": 1000}, samples=["a"],
            bam_paths={"a": "a.bam"}, vcf_paths={"a": "a.vcf.gz"},
            config=HaplotyperConfig(),
        )


def test_test_command_finds_the_suite_from_any_directory(tmp_path, monkeypatch):
    """`strainphase test` used to run `pytest tests/` relative to the working directory."""
    import subprocess

    seen = {}
    monkeypatch.setattr(subprocess, "run",
                        lambda cmd, **kw: seen.setdefault("cmd", cmd) and type("R", (), {"returncode": 0})())
    monkeypatch.chdir(tmp_path)
    assert cli.main(["test"]) == 0
    assert os.path.join(REPO, "tests") in seen["cmd"]


def _load_adapter():
    spec = importlib.util.spec_from_file_location(
        "sniffles_to_sidecar", os.path.join(REPO, "scripts", "sniffles_to_sidecar.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_SNIFFLES_VCF = """##fileformat=VCFv4.2
##contig=<ID=c1,length=100000>
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="">
##INFO=<ID=SVLEN,Number=1,Type=Integer,Description="">
##INFO=<ID=RNAMES,Number=.,Type=String,Description="">
##FORMAT=<ID=GT,Number=1,Type=String,Description="">
##FORMAT=<ID=DR,Number=1,Type=Integer,Description="">
##FORMAT=<ID=DV,Number=1,Type=Integer,Description="">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1
c1\t5000\tsv1\tN\t<DEL>\t60\tPASS\tSVTYPE=DEL;SVLEN=-800;RNAMES=r1,r2,r3,r4\tGT:DR:DV\t0/1:6:4
c1\t9000\tsv2\tN\t<INS>\t60\tPASS\tSVTYPE=INS;SVLEN=300;RNAMES=r5\tGT:DR:DV\t0/1:9:1
"""


def test_sniffles_adapter_writes_a_sidecar_strainphase_can_load(tmp_path):
    """The README says structural variants are phased; this is the documented way to
    get them into the sidecar format, so it has to be runnable from the repository."""
    from strainphase.sv_encoding import load_sv_sidecar_for_contig

    vcf = tmp_path / "s.sv.vcf"
    vcf.write_text(_SNIFFLES_VCF)
    out = tmp_path / "s.tsv"
    assert _load_adapter().main(["--sniffles", str(vcf), "--out-sidecar", str(out)]) == 0

    sv_pos, _, sv_type, sv_support = load_sv_sidecar_for_contig(str(out), "c1")
    assert sv_pos == [5000], "sv2 has one supporting read and falls below --min-support 3"
    assert sv_type[5000] == "sv"
    assert sv_support[5000] == {"sv1": {"r1", "r2", "r3", "r4"}}
