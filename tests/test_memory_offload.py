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


def test_compacting_reads_does_not_change_the_tables(dataset):
    """compact_reads stores each read's alleles/quals as arrays instead of dicts. It is
    a pure representation swap read through the same Mapping interface, so every table -
    haplotypes, abundances and all - must come out bit-for-bit identical, through the
    rescue pass and the disk spill (compact reads are pickled to both)."""
    tmp, bams, vcfs = dataset
    _o1, _r1, dicts = _run(tmp, bams, vcfs, _cfg(compact_reads=False), "compact_off")
    _o2, _r2, arrays = _run(tmp, bams, vcfs, _cfg(compact_reads=True), "compact_on")
    _assert_tables_equivalent(dicts, arrays)
    assert dicts[0], "fixture degenerate - equivalence would be vacuous"


def test_compacting_reads_survives_the_worker_pool(dataset):
    """Compact reads are pickled to worker processes; a global code table would decode to
    the wrong allele there. The self-contained codec must give identical tables at >1
    worker too."""
    tmp, bams, vcfs = dataset
    _o1, _r1, one = _run(tmp, bams, vcfs, _cfg(compact_reads=True, n_workers=1), "cw1")
    _o2, _r2, many = _run(tmp, bams, vcfs, _cfg(compact_reads=True, n_workers=2), "cw2")
    _assert_tables_equivalent(one, many)


def test_compact_read_round_trips_every_allele_kind():
    """freeze() must be a faithful, picklable, idempotent Mapping - for plain bases and
    for the rare multi-char tokens (indels, SV ids, split-read anchors) that take the
    per-read overflow path."""
    import pickle

    from strainphase.core import Read, _CompactAlleles, _CompactQuals

    r = Read(id="r", contig="c", mapq=60)
    r.alleles = {100: "A", 250: "C", 175: "DEL5", 400: "T", 300: "BRK1234", 50: "N"}
    r.quals = {100: 30, 250: 40, 175: 20, 400: 35, 300: 20, 50: 15}
    ref_a, ref_q = dict(r.alleles), dict(r.quals)

    r.freeze()
    assert isinstance(r.alleles, _CompactAlleles) and isinstance(r.quals, _CompactQuals)
    for view, ref in ((r.alleles, ref_a), (r.quals, ref_q)):
        assert dict(view) == ref
        assert list(view) == sorted(ref)  # iterates in position order
        assert len(view) == len(ref)
        assert view.get(999, "DEFLT") == "DEFLT"
        assert 999 not in view and 100 in view
        for k, v in ref.items():
            assert view[k] == v and view.get(k) == v

    r2 = pickle.loads(pickle.dumps(r))  # the worker/spill path
    assert dict(r2.alleles) == ref_a and dict(r2.quals) == ref_q

    r2.freeze()  # idempotent
    assert dict(r2.alleles) == ref_a


def test_compacting_reads_shrinks_the_payload():
    """The whole point: the array form is materially smaller than the two dicts."""
    import sys

    from strainphase.core import Read

    def payload(read):
        a, q = read.alleles, read.quals
        if hasattr(a, "_pos"):
            return sum(sys.getsizeof(x) for x in (a._pos, a._codes, a._extra, q._pos, q._quals, a, q))
        return (sys.getsizeof(a) + sys.getsizeof(q)
                + sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in a.items())
                + sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in q.items()))

    r = Read(id="r", contig="c", mapq=60)
    r.alleles = {p: "ACGT"[p % 4] for p in range(1000, 1000 + 75 * 137, 137)}  # 75 markers
    r.quals = {p: 40 for p in r.alleles}
    before = payload(r)
    r.freeze()
    after = payload(r)
    assert after * 5 < before, f"compact not much smaller: {before} -> {after}"


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
    pinning a read's PAYLOAD - the two position-keyed dicts that make a read expensive.

    Released is not the same as gone. This asserted `len(window.reads) == 0` for a while,
    which is a stricter thing than the offload exists to do and is wrong: a caller
    scoring reads rather than haplotypes needs to know which read each gamma row is, and
    with the list emptied every returned window read as "phased nothing". What must
    survive is the id and the row order; what must not is `.alleles`.
    """
    from strainphase.core import _ReadRef

    tmp, bams, vcfs = dataset
    _out, results, _tables = _run(tmp, bams, vcfs, _cfg(spill_results_to_disk=True), "released")
    windows = [wr for contigs in results.values() for wrs in contigs.values() for wr in wrs]
    assert windows, "the fixture produced no windows - the assertions below are vacuous"

    resident = sum(
        1
        for wr in windows
        for read in wr.window.reads
        if getattr(read, "alleles", None)
    )
    assert resident == 0, f"{resident} reads still carrying their payload after the run"
    assert all(
        isinstance(read, _ReadRef) for wr in windows for read in wr.window.reads
    ), "a released read must be an id-only stand-in, not a stripped Read"
    # The stand-ins ARE the read partition, so they must still line up with gamma.
    for wr in windows:
        assert len(wr.window.reads) == wr.gamma.shape[0], (
            "gamma row count and read count disagree - the partition cannot be recovered"
        )
        assert all(read.id for read in wr.window.reads)


def test_a_read_partition_can_still_be_built_from_the_returned_windows(dataset):
    """REGRESSION (N1): the returned WindowResults ARE the read partition.

    The post-rescue cleanup used to call offload_heavy() on the very objects it handed
    back, so every returned window held zero reads against a gamma of 50-odd rows. A
    caller scoring reads rather than haplotypes - which is what the benchmark's
    longitudinal partition is - then saw nothing assigned and read as "phased nothing".
    The failure looked like a bad score rather than like missing plumbing, because
    single-sample mode never offloads and so never showed it.

    This walks the return value the way such a caller does: gamma row i belongs to
    window.reads[i], and its cluster is the winning column.
    """
    tmp, bams, vcfs = dataset
    _out, results, _tables = _run(tmp, bams, vcfs, _cfg(spill_results_to_disk=True), "partition")

    assignment = {}
    for sample, per_contig in results.items():
        for contig, wrs in per_contig.items():
            for wr in wrs:
                junk_col = wr.gamma.shape[1] - 1
                for row, read in zip(wr.gamma, wr.window.reads):  # noqa: B905
                    k = int(row.argmax())
                    if k == junk_col:
                        continue
                    assignment[(sample, read.id)] = f"{contig}:{wr.window.start}:{k}"

    assert assignment, "no read could be assigned - the partition is empty"
    assert len({s for s, _ in assignment}) == len(SAMPLES), "every sample must contribute"
    assert len({c for c in assignment.values()}) > 1, "one cluster is not a partition"


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------
# The tests above call process_mag_longitudinal directly, so they all passed while
# `strainphase longitudinal` was in fact running with spilling DISABLED: cli.py's
# cmd_longitudinal never forwarded output_dir, so _SpillStore.create fell back to
# _NullSpill and every sample's reads stayed resident. Nothing caught it because
# nothing exercised the console-script path. These tests do.


def _run_cli(tmp_path, monkeypatch, extra):
    """Drive the real `strainphase longitudinal ...` entry point and capture the kwargs
    that reach process_mag_longitudinal.

    Deliberately goes through cli.main rather than hand-building a Namespace: a
    hand-built one silently drifts from the parser (it grows attributes the test does
    not know about), and it would not catch an 'unrecognized arguments' failure at all -
    which is precisely how --seed reached a cluster run before dying.
    """
    import strainphase.cli as cli
    import strainphase.longitudinal as L

    for name in ("a.bam", "b.bam", "v.vcf.gz", "ref.fa"):
        (tmp_path / name).write_text("")

    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        raise SystemExit(99)  # stop before any real work

    monkeypatch.setattr(L, "process_mag_longitudinal", fake)
    monkeypatch.setattr(L, "parse_reference_contigs", lambda *a, **k: {"MAG1": {"c1": 1000}})

    argv = ["longitudinal",
            "--samples", "a,b",
            "--bams", str(tmp_path / "{sample}.bam"),
            "--vcfs", str(tmp_path / "v.vcf.gz"),
            "--reference", str(tmp_path / "ref.fa"),
            "--output-dir", str(tmp_path / "OUTDIR"),
            "--mags", "MAG1"] + extra
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    # 2 is argparse's "bad command line"; anything else means we got past parsing.
    assert exc.value.code != 2, f"argparse rejected: {' '.join(argv)}"
    return seen


def test_longitudinal_subcommand_accepts_the_memory_and_seed_flags(tmp_path, monkeypatch):
    """`strainphase longitudinal --seed ...` must not die with 'unrecognized arguments'.

    cli.py's long_parser and longitudinal.py's parser are two hand-maintained arg lists
    over one config; a flag added to only the second is accepted by
    `python -m strainphase.longitudinal` and rejected by `strainphase longitudinal`.
    That mismatch shipped --seed to a cluster run that died on it immediately.
    """
    seen = _run_cli(tmp_path, monkeypatch,
                    ["--seed", "7", "--window-batch-factor", "2", "--keep-read-assignments"])
    cfg = seen.get("config")
    assert cfg is not None, "process_mag_longitudinal was never reached"
    assert cfg.random_seed == 7
    assert cfg.window_batch_factor == 2
    assert cfg.keep_read_assignments is True


def test_cmd_longitudinal_forwards_output_dir_so_spilling_is_actually_on(
    tmp_path, monkeypatch
):
    """The regression that mattered: without output_dir the spill store is a no-op, so
    every sample's reads stay resident and the OOM the spill work exists to prevent
    comes straight back - with every unit test still green."""
    seen = _run_cli(tmp_path, monkeypatch, [])
    assert seen.get("output_dir") == str(tmp_path / "OUTDIR"), (
        "cmd_longitudinal did not forward output_dir -> _SpillStore falls back to "
        "_NullSpill and reads are never spilled"
    )


def test_spilling_is_on_by_default_and_no_spill_turns_it_off(tmp_path, monkeypatch):
    on = _run_cli(tmp_path, monkeypatch, [])["config"]
    assert on.spill_results_to_disk is True, "spilling must be the default"
    assert on.random_seed == 42, "the run must be seeded even with no --seed"

    off = _run_cli(tmp_path, monkeypatch, ["--no-spill"])["config"]
    assert off.spill_results_to_disk is False


# ---------------------------------------------------------------------------
# Spill failure modes
# ---------------------------------------------------------------------------
# offload_heavy() detaches the reads BEFORE the write is attempted, so a failed
# write means they are gone. Continuing from there does not crash - rescue just
# runs on read-less windows and emits different numbers. On shared scratch a full
# disk is realistic, so both directions must fail loudly or recover, never degrade.


class _FakeRead:
    """The narrowest thing _SpillStore's collaborator has to be: an id and a payload."""

    def __init__(self, read_id, payload=None):
        self.id = read_id
        self.alleles = payload if payload is not None else {}

    def __eq__(self, other):
        return isinstance(other, _FakeRead) and (self.id, self.alleles) == (
            other.id,
            other.alleles,
        )

    def __repr__(self):  # pragma: no cover - diagnostics only
        return f"_FakeRead({self.id!r})"


class _FakeWR:
    """A stand-in for WindowResult with the shape _SpillStore actually reaches into.

    The store goes through ``_detach_reads``, which reads ``wr.window.reads`` to build
    the id-only stand-ins before calling ``offload_heavy``. A stub with a bare ``reads``
    attribute and no ``window`` stopped modelling the collaborator the moment those
    stand-ins existed, and would have gone on passing while the real store crashed.
    """

    def __init__(self, reads):
        self.window = type("W", (), {"reads": list(reads), "_pos_sets": None})()
        self.restored = False

    @property
    def reads(self):
        return self.window.reads

    def offload_heavy(self):
        out, self.window.reads = self.window.reads, []
        return out

    def restore_heavy(self, reads):
        self.window.reads = reads
        self.restored = True


def test_a_failed_spill_write_keeps_the_reads_in_memory(dataset, monkeypatch, tmp_path):
    """Disk full mid-run must cost memory, not correctness."""
    from strainphase.longitudinal import _SpillStore

    tmp, bams, vcfs = dataset
    real_open = open

    def exploding_open(path, mode="r", *a, **k):
        if "reads.pkl" in str(path) and "w" in mode:
            raise OSError(28, "No space left on device")
        return real_open(path, mode, *a, **k)

    monkeypatch.setattr("builtins.open", exploding_open)
    store = _SpillStore(str(tmp_path / "spill"))

    originals = [_FakeRead("r1", {1: "A"}), _FakeRead("r2", {1: "C"})]
    wrs = [_FakeWR(originals), _FakeWR(originals)]
    store.offload("s1", {"c1": wrs})

    assert all(w.restored for w in wrs), "reads were dropped instead of kept in memory"
    assert all(w.reads == originals for w in wrs), "reads came back wrong"
    assert ("s1", "c1") not in store._paths, (
        "a failed write must not register a path - restore() would then expect a file "
        "that does not exist"
    )


def test_an_unreadable_spill_file_raises_rather_than_silently_dropping_reads(tmp_path):
    """If the reads cannot come back, there is no correct way to continue."""
    from strainphase.longitudinal import _SpillStore

    store = _SpillStore(str(tmp_path / "spill"))

    wrs = [_FakeWR([_FakeRead("r1", {1: "A"})])]
    store.offload("s1", {"c1": wrs})
    path = store._paths[("s1", "c1")]
    with open(path, "wb") as fh:          # corrupt it
        fh.write(b"not a pickle")

    with pytest.raises(RuntimeError, match="could not be read back"):
        store.restore("s1", "c1", wrs)


def test_spill_leaves_id_only_stand_ins_and_round_trips_the_reads(tmp_path):
    """Spilling releases the PAYLOAD and keeps the row correspondence.

    This asserted `w.reads == []` after the offload, which is the behaviour that made
    the returned partition empty. What the store owes the caller is the ids, in gamma
    order, until the real reads come back over the top of them.
    """
    from strainphase.core import _ReadRef
    from strainphase.longitudinal import _SpillStore

    store = _SpillStore(str(tmp_path / "spill"))

    a = [_FakeRead("a", {1: "A"})]
    bc = [_FakeRead("b", {1: "C"}), _FakeRead("c", {1: "G"})]
    wrs = [_FakeWR(a), _FakeWR(bc)]
    store.offload("s1", {"c1": wrs})

    assert [[r.id for r in w.reads] for w in wrs] == [["a"], ["b", "c"]], (
        "the ids and their order must survive the offload"
    )
    assert all(isinstance(r, _ReadRef) for w in wrs for r in w.reads), (
        "the payload must be detached, not merely reachable"
    )
    assert not any(hasattr(r, "alleles") for w in wrs for r in w.reads)

    store.restore("s1", "c1", wrs)
    assert wrs[0].reads == a
    assert wrs[1].reads == bc


def test_the_real_cli_actually_writes_spill_files(dataset, monkeypatch):
    """End-to-end proof that spilling happens through `strainphase longitudinal`.

    This is the test that would have caught cmd_longitudinal dropping output_dir: it
    watches the spill directory being used during a real run rather than inspecting
    kwargs. The directory is cleaned up on success, so the check hooks the write.
    """
    import strainphase.longitudinal as L

    tmp, bams, vcfs = dataset
    seen_paths = []
    real_offload = L._SpillStore.offload

    def spy(self, sample_id, contig_map):
        real_offload(self, sample_id, contig_map)
        seen_paths.extend(self._paths.values())

    monkeypatch.setattr(L._SpillStore, "offload", spy)
    out, _results, _tables = _run(tmp, bams, vcfs, _cfg(spill_results_to_disk=True), "cli_spill")

    assert seen_paths, "no spill file was ever written - spilling is not active"
    assert all("spill" in p for p in seen_paths)
