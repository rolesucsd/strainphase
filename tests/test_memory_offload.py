"""Spilling reads to disk and batching windows must not change a single output value.

Both changes exist purely to bound memory on many-sample MAGs:
  - reads are parked in <output_dir>/tmp/spill after each sample is phased and reloaded
    one sample at a time for the cross-sample rescue pass;
  - windows are dispatched to the worker pool in batches instead of all at once.

Neither touches the algorithm, so the deliverable tables must come out byte-identical to
a run with both switched off. That equivalence is the whole point, so it is asserted
end-to-end through process_mag_longitudinal -> build_window_tables rather than on the
individual helpers.
"""

from __future__ import annotations

import os

import pytest

from strainphase.core import HaplotyperConfig
from strainphase.longitudinal import build_window_tables, process_mag_longitudinal

from .util_io import write_bam, write_vcf

CONTIG = "c1"
CONTIG_LEN = 6000
SNV_POS = list(range(200, CONTIG_LEN - 200, 200))  # 29 sites
SAMPLES = ["t0", "t1", "t2"]

# Two haplotypes: HAP_B carries the alt allele at every third site.
ALT_SITES = set(SNV_POS[::3])


def _cfg(**over) -> HaplotyperConfig:
    base = dict(
        window_size=2000,
        max_reads_per_window=60,
        min_reads_per_window=4,
        min_reads_for_rescue=2,
        min_read_window_overlap_bp=0,
        min_read_read_overlap_bp=0,
        min_entity_overlap_bp=0,
        min_snvs_per_window=2,
        n_workers=1,
        random_seed=7,
    )
    base.update(over)
    return HaplotyperConfig(**base)


def _seq_for(hap: str, start0: int, length: int) -> str:
    """Reference is all A; HAP_B substitutes T at its alt sites."""
    s = ["A"] * length
    if hap == "B":
        for p in ALT_SITES:
            off = (p - 1) - start0  # p is 1-based, start0 is 0-based
            if 0 <= off < length:
                s[off] = "T"
    return "".join(s)


def _build_sample(tmp_path, sample: str, n_b: int):
    """One BAM per sample. `n_b` sets how many reads carry haplotype B, so the two
    haplotypes shift in abundance across timepoints and the rescue pass has work to do.
    """
    reads = []
    read_len = 1500
    starts = list(range(0, CONTIG_LEN - read_len, 250))
    for i, st in enumerate(starts):
        for j in range(6):
            hap = "B" if j < n_b else "A"
            reads.append(
                {
                    "name": f"{sample}_r{i}_{j}",
                    "start": st,
                    "cigar": f"{read_len}M",
                    "seq": _seq_for(hap, st, read_len),
                }
            )
    return write_bam(tmp_path, CONTIG, CONTIG_LEN, reads, name=f"{sample}.bam")


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("offload")
    vcf = write_vcf(
        tmp,
        CONTIG,
        [
            {"pos": p, "ref": "A", "alt": "T", "info": {"DP": 40, "AF": 0.5}}
            for p in SNV_POS
        ],
        contig_length=CONTIG_LEN,
    )
    # n_b varies per timepoint: 1 -> 3 -> 5 of every 6 reads carry haplotype B.
    bams = {s: _build_sample(tmp, s, n_b) for s, n_b in zip(SAMPLES, (1, 3, 5))}
    return tmp, bams, {s: vcf for s in SAMPLES}


# Equivalence is asserted EXACTLY, down to the last bit of every float. That is only
# possible because both randomness sources are now seeded (HaplotyperConfig.random_seed
# feeds the read subsampling and Louvain's random_state); before that, two runs of an
# identical config disagreed on `abundance` at ~1e-7 and this had to be a tolerance.


def _assert_rows_equivalent(rows_a, rows_b, label):
    assert len(rows_a) == len(rows_b), f"{label}: row count {len(rows_a)} != {len(rows_b)}"
    for i, (ra, rb) in enumerate(zip(rows_a, rows_b)):  # noqa: B905
        assert ra.keys() == rb.keys(), f"{label}[{i}]: differing columns"
        for k in ra:
            assert ra[k] == rb[k], f"{label}[{i}].{k}: {ra[k]!r} != {rb[k]!r}"


TABLE_NAMES = ["haplotypes", "within_sample", "across_samples", "edges", "mismatches"]


def _assert_tables_equivalent(a, b):
    for i, name in enumerate(TABLE_NAMES):
        _assert_rows_equivalent(a[i], b[i], name)


def _run(tmp_path, bams, vcfs, config, tag):
    out = str(tmp_path / tag)
    os.makedirs(out, exist_ok=True)
    results, _integrator = process_mag_longitudinal(
        mag_name="MAG1",
        mag_contigs={CONTIG: CONTIG_LEN},
        samples=SAMPLES,
        bam_paths=bams,
        vcf_paths=vcfs,
        config=config,
        output_dir=out,
    )
    tables = build_window_tables(out, {"MAG1": results}, config, sample_order=SAMPLES)
    return out, results, tables


def test_the_fixture_actually_phases_something(dataset):
    """Guard the guard: if the synthetic data produced no haplotypes, the equivalence
    assertions below would pass on two empty tables and prove nothing."""
    tmp, bams, vcfs = dataset
    _out, results, tables = _run(tmp, bams, vcfs, _cfg(), "sanity")
    hap_rows = tables[0]
    assert hap_rows, "fixture produced no haplotype rows - the test would be vacuous"
    assert len({r["sample"] for r in hap_rows}) > 1, "need >1 sample for a rescue pass"


def test_two_runs_of_the_same_config_are_bit_identical(dataset):
    """The pipeline must be reproducible: same inputs, same config, same bits.

    It was not, until both randomness sources were seeded - read subsampling above
    max_reads_per_window (config.get_rng) and Louvain read clustering (random_state).
    Before that, identical runs disagreed on abundance at ~1e-7 AND drew different reads
    in any window above the cap, so a rerun of a published number would not reproduce it.
    """
    tmp, bams, vcfs = dataset
    _o1, _r1, a = _run(tmp, bams, vcfs, _cfg(), "repro_a")
    _o2, _r2, b = _run(tmp, bams, vcfs, _cfg(), "repro_b")
    _assert_tables_equivalent(a, b)
    assert a[0], "fixture degenerate - equivalence would be vacuous"


def test_a_different_seed_actually_changes_something(dataset):
    """Guards the guard above: if the seed were ignored, every reproducibility assertion
    in this file would pass trivially."""
    tmp, bams, vcfs = dataset
    _o1, _r1, a = _run(tmp, bams, vcfs, _cfg(random_seed=1), "seed1")
    _o2, _r2, b = _run(tmp, bams, vcfs, _cfg(random_seed=999, max_reads_per_window=8), "seed999")
    assert a[0] != b[0], "seed has no effect on output - is it reaching the RNG?"


def test_spilling_reads_does_not_change_the_tables(dataset):
    tmp, bams, vcfs = dataset
    _o1, _r1, spilled = _run(tmp, bams, vcfs, _cfg(spill_results_to_disk=True), "spill_on")
    _o2, _r2, kept = _run(tmp, bams, vcfs, _cfg(spill_results_to_disk=False), "spill_off")
    _assert_tables_equivalent(spilled, kept)


def test_window_batching_does_not_change_the_tables(dataset):
    """Batch size must be a pure scheduling knob."""
    tmp, bams, vcfs = dataset
    _o1, _r1, small = _run(tmp, bams, vcfs, _cfg(window_batch_factor=1, n_workers=2), "batch1")
    _o2, _r2, big = _run(tmp, bams, vcfs, _cfg(window_batch_factor=64, n_workers=2), "batch64")
    _assert_tables_equivalent(small, big)


def test_read_assignments_are_off_by_default_and_opt_in(dataset):
    """The field costs one dict per read per window and nothing reads it, so it must be
    empty unless explicitly requested - and still populated when it is."""
    tmp, bams, vcfs = dataset
    _o1, off, _t1 = _run(tmp, bams, vcfs, _cfg(), "assign_off")
    assert not any(
        wr.assignments for contigs in off.values() for wrs in contigs.values() for wr in wrs
    ), "assignments retained despite keep_read_assignments=False"

    _o2, on, _t2 = _run(tmp, bams, vcfs, _cfg(keep_read_assignments=True), "assign_on")
    assert any(
        wr.assignments for contigs in on.values() for wrs in contigs.values() for wr in wrs
    ), "keep_read_assignments=True produced no assignments"


def test_spill_directory_is_cleaned_up(dataset):
    tmp, bams, vcfs = dataset
    out, _results, _tables = _run(tmp, bams, vcfs, _cfg(spill_results_to_disk=True), "cleanup")
    assert not os.path.exists(os.path.join(out, "tmp", "spill", "MAG1")), (
        "spill files outlived the run"
    )


def test_reads_are_released_after_the_run(dataset):
    """The point of the exercise: once a MAG is done, no WindowResult should still be
    pinning its reads."""
    tmp, bams, vcfs = dataset
    _out, results, _tables = _run(tmp, bams, vcfs, _cfg(spill_results_to_disk=True), "released")
    resident = sum(
        len(wr.window.reads)
        for contigs in results.values()
        for wrs in contigs.values()
        for wr in wrs
    )
    assert resident == 0, f"{resident} reads still resident after the run"
