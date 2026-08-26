#!/usr/bin/env python3
"""
Longitudinal integration script for haplotyper v3.

This script:
1. Runs haplotyping per contig, per sample (using haplotyper.process_contig)
2. Performs cross-timepoint rescue of low-abundance haplotypes (LongitudinalIntegrator)
3. Builds a lineage table by clustering similar haplotypes across samples
4. Writes:
   - lineages.tsv        (one row per (MAG, contig, window, sample, lineage))
   - longitudinal_summary.tsv
   - <sample>.rescued.tsv (per-sample haplotypes after rescue)

Recommended usage for efficiency:
    * Run ONE MAG per job using --mags <MAG_NAME>
    * Use --contig-filter to restrict to high-coverage / high-breadth contigs
    * Keep validate_results=False (default) for production

Example:
    python run_longitudinal.py \
        --samples bc2001,bc2002,... \
        --bams /ddn_scratch/.../mapping/{sample}.sorted.bam \
        --vcfs /ddn_scratch/.../variants/clair3/{sample}/pileup.vcf.gz \
        --reference /ddn_scratch/.../references/combined_bins.fasta \
        --output-dir /ddn_scratch/.../haplotypes/longitudinal/BF_MAG_01 \
        --mags BF_MAG_01 \
        --contig-filter good_contigs.tsv \
        --window-size 3000 \
        --max-reads 300 \
        --log-level INFO
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import shutil
import sys
from collections import defaultdict

import pysam  # noqa: F401

from strainphase.core import (
    Haplotype,
    HaplotyperConfig,
    LongitudinalIntegrator,
    WindowResult,
    link_windows,
    make_worker_pool,
    process_contig,
    variable_marker_positions,
)
from strainphase.lineages import build_lineages
from strainphase.window_groups import WindowHaplotype, group_all_windows

# -----------------------------------------------------------------------------#
# Helpers
# -----------------------------------------------------------------------------#


def _window_conditional_abundance(pi_vec, hap_idx: int) -> float | None:
    """Per-window abundance ``pi_k / (1 - pi_junk)``, or ``None`` when unmeasurable.

    Returning ``None`` rather than ``0.0`` is the point. A window with no ``pi`` vector,
    a short one, or one that is entirely junk (``pi_junk == 1``) has *no measurement* -
    it is not a measurement of zero. Collapsing the two let a junk-dominated window
    contribute a hard 0.0 carrying full window-read weight, and since the aggregate is a
    weighted median (a selection operator, which returns one input verbatim) that
    spurious 0.0 could be the value actually reported for the timepoint.
    """
    if pi_vec is None or len(pi_vec) <= hap_idx:
        return None
    pi_junk = float(pi_vec[-1])
    denom = 1.0 - pi_junk
    if denom <= 0:
        return None
    return max(0.0, min(1.0, float(pi_vec[hap_idx]) / denom))


def _window_haplotype_id(sample_id: str, contig_id: str, window_start: int, h_idx: int) -> str:
    """The step-0 haplotype id, ``sample_contig_windowstart_H<idx>``.

    ONE definition, because a second spelling of it is a silent no-op rather than a
    mismatch anyone notices. The step-1 mismatch rows and the WindowHaplotype objects
    step 3 vetoes against are built in different loops of build_window_tables, and while
    those loops spelled the id differently - one pipe-separated and track-prefixed, the
    other underscore-separated and window-keyed - the veto set could not match a single
    member id, so the "a sample whose own reads disagree vetoes the join outright"
    safeguard in build_lineages never once fired.
    """
    return f"{sample_id}_{contig_id}_{window_start}_H{h_idx}"


def load_allowed_contigs(path: str) -> set[str]:
    """
    Load an optional contig filter file.

    Expected formats:
      - Simple 1-column file: each line is a contig name
      - TSV with header containing a 'contig' column

    Returns a set of contig IDs to keep.
    """
    allowed: set[str] = set()
    with open(path) as f:
        first = f.readline().strip()
        if not first:
            return allowed

        cols = first.split("\t")
        if len(cols) == 1:
            # No header, first line is a contig name
            allowed.add(cols[0])
            for line in f:
                line = line.strip()
                if line:
                    allowed.add(line)
        else:
            # Assume header; require 'contig' column
            header = cols
            if "contig" not in header:
                raise ValueError(
                    f"--contig-filter file {path} has multiple columns but no 'contig' header"
                )
            idx = header.index("contig")
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) <= idx:
                    continue
                allowed.add(parts[idx])

    logging.info(f"Loaded {len(allowed)} contigs from filter {path}")
    return allowed


def parse_reference_contigs(
    fasta_path: str, allowed_contigs: set[str] | None = None
) -> dict[str, dict[str, int]]:
    """
    Parse reference .fai to get contig info grouped by MAG.

    Headers are assumed to look like:
        MAGNAME_contig_1
        MAGNAME_contig_2
        ...

    If allowed_contigs is provided, only those contigs are kept.
    """
    fai_path = fasta_path + ".fai"
    mags: dict[str, dict[str, int]] = defaultdict(dict)

    with open(fai_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if not parts:
                continue
            contig_name = parts[0]
            length = int(parts[1])

            if allowed_contigs is not None and contig_name not in allowed_contigs:
                continue

            if "_contig_" in contig_name:
                mag_name = contig_name.rsplit("_contig_", 1)[0]
            else:
                mag_name = contig_name

            mags[mag_name][contig_name] = length

    return dict(mags)


# -----------------------------------------------------------------------------#
# Read spilling
# -----------------------------------------------------------------------------#


class _ReadRef:
    """A read's identity, kept after its alleles have been released.

    ``window.reads`` is emptied as soon as a window's own sample is finished with it (see
    WindowResult.offload_heavy), but the WindowResults handed back to the caller still
    have to say WHICH read each gamma row belongs to. That correspondence IS the read
    partition, and it is the entire output for a caller scoring reads rather than
    haplotypes. Dropping it returned every window with zero reads against a gamma of
    50-odd rows, so a partition built from the return value came out empty and the
    pipeline looked like it had phased nothing.

    Holding the whole Read instead is not an option - two position-keyed dicts, ~90 KB on
    a variant-dense contig, times every sample resident at once - which is exactly what
    the offload exists to prevent. The id alone is ~1500x cheaper (48 bytes against the
    ~70 KB of a 600-marker read) and preserves the row correspondence exactly. Nothing
    else survives, on purpose: code that reaches for
    ``.alleles`` here is reading a released read, and an AttributeError naming this class
    is far better than silently seeing no alleles.
    """

    __slots__ = ("id",)

    def __init__(self, read_id: str) -> None:
        self.id = read_id

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"_ReadRef({self.id!r})"


def _detach_reads(wr: WindowResult) -> list:
    """Offload a window's reads, leaving id-only stand-ins in gamma-row order.

    Returns the detached Read objects so the caller can spill them; ``restore_heavy``
    lays the real ones back over the stand-ins. Idempotent, and order-independent across
    the several WindowResults that can share one Window after rescue.
    """
    refs = [r if isinstance(r, _ReadRef) else _ReadRef(r.id) for r in wr.window.reads]
    reads = wr.offload_heavy()
    wr.window.reads = refs
    return reads


class _SpillStore:
    """Parks per-sample read lists on disk between the phasing pass and the rescue pass.

    Only ``window.reads`` moves - see WindowResult.offload_heavy for why that is the
    field worth moving and why the WindowResult objects themselves must stay resident.
    An id-only stand-in is left in each read's place (``_detach_reads``) so the window
    still knows which read each gamma row is.

    Reads are spilled rather than re-read from the BAM because re-deriving them would
    have to reproduce the per-window subsample exactly (make_windows_lazy draws
    max_reads_per_window with config.get_rng()); any drift there would silently change
    which haplotypes are called. Round-tripping the objects cannot drift.

    ``_NullSpill`` is the same interface with every method a no-op, used when spilling is
    switched off or no output directory is available (tests, library callers).
    """

    def __init__(self, root: str):
        self.root = root
        self._paths: dict[tuple[str, str], str] = {}
        self._n = 0
        os.makedirs(root, exist_ok=True)

    @staticmethod
    def create(config, output_dir: str | None, mag_label: str):
        if not getattr(config, "spill_results_to_disk", True) or not output_dir:
            return _NullSpill()
        root = os.path.join(output_dir, "tmp", "spill", mag_label)
        try:
            return _SpillStore(root)
        except OSError as e:
            # Not a warning: the resource tiers are sized on the assumption that reads
            # are spilled. Falling back to holding them all makes an OOM likely, so say
            # so at ERROR rather than burying it in an INFO-level log.
            logging.error(
                f"Cannot create spill dir {root} ({e}); keeping every sample's reads in "
                f"memory. Expect much higher peak memory and possible OOM."
            )
            return _NullSpill()

    def offload(self, sample_id: str, contig_map: dict[str, list]) -> None:
        """Detach and persist every contig's reads for one sample.

        A failed write must NOT be shrugged off. offload_heavy() has already detached
        the reads by the time the write is attempted, so silently continuing would leave
        rescue running against read-less windows - which does not crash, it just quietly
        produces different numbers. On a shared scratch filesystem a full disk is a
        realistic event, so the reads go back into memory instead: the run then costs
        more memory (the old behaviour) but stays correct.
        """
        for contig_id, window_results in contig_map.items():
            payload = [_detach_reads(wr) for wr in window_results]
            if not any(payload):
                continue
            tmp_path = os.path.join(self.root, f"{self._n + 1:06d}.reads.pkl")
            try:
                with open(tmp_path, "wb") as fh:
                    pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
            except (OSError, pickle.PicklingError) as e:
                logging.error(
                    f"Spill write failed for {sample_id}/{contig_id} ({e}); keeping "
                    f"these reads in memory instead - peak memory will be higher"
                )
                for wr, reads in zip(window_results, payload):  # noqa: B905
                    if reads:
                        wr.restore_heavy(reads)
                continue
            # Registered only AFTER a successful write, so a key in _paths means
            # "a file really is on disk" and restore() can treat a miss as an error.
            self._n += 1
            self._paths[(sample_id, contig_id)] = tmp_path

    def restore(self, sample_id: str, contig_id: str, window_results: list) -> None:
        """Re-attach reads spilled for one sample+contig.

        Raises rather than degrading. If a spill file was written but cannot be read
        back, the reads are gone and there is no correct way to continue: rescue would
        run on empty windows and emit plausible-looking but wrong abundances. Failing
        loudly is the only safe option.
        """
        path = self._paths.get((sample_id, contig_id))
        if path is None:
            return  # nothing was ever spilled for this pair (no reads to begin with)
        try:
            with open(path, "rb") as fh:
                payload = pickle.load(fh)
        except (OSError, pickle.UnpicklingError) as e:
            raise RuntimeError(
                f"spilled reads for {sample_id}/{contig_id} could not be read back "
                f"from {path}: {e}. Refusing to continue - rescue would silently run "
                f"on read-less windows and produce wrong abundances."
            ) from e
        if len(payload) != len(window_results):
            raise RuntimeError(
                f"spill length mismatch for {sample_id}/{contig_id}: {len(payload)} "
                f"payloads vs {len(window_results)} windows. Refusing to continue."
            )
        for wr, reads in zip(window_results, payload):  # noqa: B905
            wr.restore_heavy(reads)

    def discard(self, sample_id: str, contig_id: str) -> None:
        path = self._paths.pop((sample_id, contig_id), None)
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
        self._paths.clear()


class _NullSpill:
    """No-op spill store: reads stay in memory (previous behaviour)."""

    def offload(self, sample_id: str, contig_map: dict[str, list]) -> None:
        pass

    def restore(self, sample_id: str, contig_id: str, window_results: list) -> None:
        pass

    def discard(self, sample_id: str, contig_id: str) -> None:
        pass

    def cleanup(self) -> None:
        pass


# -----------------------------------------------------------------------------#
# Core longitudinal logic
# -----------------------------------------------------------------------------#


def process_mag_longitudinal(
    mag_name: str | None,
    mag_contigs: dict[str, int],
    samples: list[str],
    bam_paths: dict[str, str],
    vcf_paths: dict[str, str],
    config: HaplotyperConfig,
    sv_sidecar_paths: dict[str, str] | None = None,
    output_dir: str | None = None,
) -> tuple[dict[str, dict[str, list[WindowResult]]], LongitudinalIntegrator | None]:
    """
    Process a single MAG across all samples with longitudinal rescue.

    Haplotypes are linked across windows after processing and after rescue.

    Reads are offloaded to ``<output_dir>/tmp/spill`` as each sample finishes
    (config.spill_results_to_disk) and reloaded one sample at a time during rescue.
    Without that, every sample's reads stay resident until the whole MAG is done, which
    on a 146-sample variant-dense MAG is what runs the job out of memory. Gamma is ~1000x
    smaller and stays resident throughout. The WindowResult objects themselves stay in RAM
    too - the rescue panel holds references to their Haplotype objects and mutates weights
    in place, so they must not be replaced.

    Returns:
        Tuple of:
        - {sample_id: {contig_id: [WindowResult, ...]}}
        - LongitudinalIntegrator instance (or None if single timepoint)

    The returned windows carry ``_ReadRef`` stand-ins rather than Read objects wherever
    the alleles have been released: ``window.reads[i].id`` still names the read gamma row
    ``i`` describes, which is what a caller scoring the read partition needs, but the
    alleles are gone and must not be read.
    """
    mag_label = mag_name or "<unknown>"
    logging.info(
        f"Processing MAG {mag_label} across {len(samples)} samples " f"({len(mag_contigs)} contigs)"
    )

    # ------------------ First pass: per-sample EM haplotyping ------------------
    # (process_contig now includes window linking)
    all_results: dict[str, dict[str, list[WindowResult]]] = {}

    # Build one worker pool for the whole MAG run so we don't pay spawn /
    # re-import / config-pickling cost per contig. Workers are initialized
    # once with `config`; process_contig forwards `pool` and uses it directly.
    n_workers = max(1, getattr(config, "n_workers", 1))
    worker_pool = make_worker_pool(n_workers, config) if n_workers > 1 else None

    spill = _SpillStore.create(config, output_dir, mag_label)

    for sample_id in samples:
        logging.info(f"  Sample {sample_id}: initial contig processing")
        all_results[sample_id] = {}

        for contig_id, contig_length in mag_contigs.items():
            try:
                results = process_contig(
                    bam_path=bam_paths[sample_id],
                    vcf_path=vcf_paths[sample_id],
                    contig_id=contig_id,
                    contig_length=contig_length,
                    config=config,
                    sample_id=sample_id,
                    pool=worker_pool,
                    sv_sidecar_path=(
                        sv_sidecar_paths.get(sample_id) if sv_sidecar_paths else None
                    ),
                )

                if results:
                    all_results[sample_id][contig_id] = results
                    n_haps = sum(len(wr.haplotypes) for wr in results)
                    # Count unique tracks
                    track_ids = {h.track_id for wr in results for h in wr.haplotypes if h.track_id}
                    logging.debug(
                        f"    {contig_id}: {len(results)} windows, {n_haps} haplotypes, "
                        f"{len(track_ids)} tracks"
                    )
            except Exception as e:
                logging.warning(f"    Error on contig {contig_id} in {sample_id}: {e}")
                continue

        # Park this sample's reads + gamma on disk before starting the next one.
        spill.offload(sample_id, all_results[sample_id])

    # ------------------ Second pass: cross-timepoint rescue -------------------
    integrator = None
    if len(samples) >= 2:
        logging.info(f"  Performing longitudinal rescue across {len(samples)} samples")
        integrator = LongitudinalIntegrator(config)

        for contig_id in mag_contigs.keys():
            # Collect results for this contig across samples
            results_by_timepoint: dict[str, list[WindowResult]] = {}
            for sample_id in samples:
                sample_contigs = all_results.get(sample_id, {})
                if contig_id in sample_contigs:
                    results_by_timepoint[sample_id] = sample_contigs[contig_id]

            if len(results_by_timepoint) >= 2:
                # Diagnostic: log window counts and junk statistics before rescue
                n_windows_per_sample = {s: len(wrs) for s, wrs in results_by_timepoint.items()}
                total_haps = sum(len(wr.haplotypes) for wrs in results_by_timepoint.values() for wr in wrs)

                # Count junk reads across all windows
                total_reads = 0
                total_junk_reads = 0
                for wrs in results_by_timepoint.values():
                    for wr in wrs:
                        n_reads = wr.gamma.shape[0]
                        junk_idx = wr.gamma.shape[1] - 1
                        junk_reads = (wr.gamma[:, junk_idx] > 0.5).sum()
                        total_reads += n_reads
                        total_junk_reads += junk_reads

                junk_pct = 100 * total_junk_reads / total_reads if total_reads > 0 else 0
                logging.info(
                    f"    Contig {contig_id}: windows={n_windows_per_sample}, "
                    f"haplotypes={total_haps}, junk_reads={total_junk_reads}/{total_reads} ({junk_pct:.1f}%)"
                )

                # Apply rescue. One sample is resident at a time: every OTHER sample
                # contributes only its haplotypes to the anchor panel, which survives
                # offloading. Iteration order matches the all-at-once call, so the
                # in-place weight updates propagate through the panel identically.
                for sample_id in samples:
                    originals = results_by_timepoint.get(sample_id)
                    if originals is None:
                        continue
                    spill.restore(sample_id, contig_id, originals)
                    rescued = integrator.rescue_low_abundance(
                        results_by_timepoint, only_sample=sample_id
                    )
                    window_results = rescued.get(sample_id)
                    if window_results is not None:
                        # Re-link windows, because rescue may have added haplotypes. This
                        # is the second link_windows call on these objects; it resets
                        # link_mismatches itself, so the rows are re-derived rather than
                        # duplicated.
                        all_results[sample_id][contig_id] = link_windows(window_results, config)
                    # `results_by_timepoint` deliberately keeps pointing at the PRE-rescue
                    # objects, so later samples see the same panel the all-at-once call
                    # gave them: weights mutated in place, newly rescued haplotypes not
                    # yet included. Their read ALLELES are finished with - the panel needs
                    # only haplotypes - so release them for good rather than re-spilling.
                    # The ids stay: a rescued WindowResult wraps the SAME Window object,
                    # so emptying it outright handed the caller a window with zero reads
                    # against a full gamma, i.e. an empty read partition.
                    for wr in originals:
                        _detach_reads(wr)
                    # Those rescued results share that Window, so they already hold the
                    # stand-ins and only their own flag has to agree. Deliberately NOT
                    # offload_heavy(), which would also clear `assignments` - the one
                    # place --keep-read-assignments' output survives the rescue pass.
                    for wr in all_results[sample_id].get(contig_id, ()):
                        wr.heavy_offloaded = True
                    spill.discard(sample_id, contig_id)

        # Log rescue statistics
        n_rescued = sum(1 for s in integrator.rescue_statistics if s.was_rescued)
        n_total = len(integrator.rescue_statistics)
        logging.info(f"  Rescue completed: {n_rescued}/{n_total} haplotypes rescued")

    # Log integrator status for debugging
    if integrator:
        logging.info(f"  Returning integrator with {len(integrator.rescue_statistics)} statistics records")
    else:
        logging.info(f"  No integrator (len(samples)={len(samples)})")

    if worker_pool is not None:
        worker_pool.close()
        worker_pool.join()

    # Anything still parked (contigs that never entered rescue, single-timepoint runs)
    # is finished with - the output tables need haplotypes and gamma, not reads.
    spill.cleanup()

    return all_results, integrator


def _group_marker_span(group) -> tuple[int, int]:
    """Marker footprint of a window group: min/max marker position over all its members."""
    pos = [p for m in group.members for p in m.consensus]
    return (min(pos), max(pos)) if pos else (0, 0)


def _median_member_span(group) -> float:
    """Median marker span of the INDIVIDUAL members.

    Distinguishes a group whose members each cover the whole window from one where every
    member covers a sliver and only the union looks wide - which is the difference between
    a haplotype that can chain into the next window and one that cannot.
    """
    spans = sorted(max(m.consensus) - min(m.consensus) for m in group.members if m.consensus)
    return float(spans[len(spans) // 2]) if spans else 0.0


def _read_counts(wr) -> tuple[int, int]:
    """``(resolved, junk)`` reads in this window. One definition, used everywhere.

    Both are reported because the choice of denominator is a real one and should not be
    baked in. Dividing by RESOLVED reads alone answers "of the reads that phased, what
    share is this haplotype" - which renormalises away the fact that a window where 10%
    of reads resolved is far weaker evidence than one where 90% did. Including junk
    answers "what share of the reads at this locus is this haplotype", which degrades
    gracefully instead: a poorly-resolving window pulls the estimate down rather than
    being silently rescaled up to look like a good one.
    """
    n = getattr(wr, "n_reads_examined", len(wr.window.reads))
    if wr.gamma is None or wr.gamma.size == 0:
        return n, 0
    junk = int((wr.gamma[:, wr.gamma.shape[1] - 1] >= 0.5).sum())
    return n - junk, junk


def _lineage_rows(lineages, mag_name: str) -> list[dict]:
    """ONE row per (lineage, sample) - the grain where identity, abundance and membership
    all fit in a single table.

    Lineage-level fields repeat down the rows; ``abundance`` is that sample's POOLED
    estimate (Sum reads / Sum denominator over the windows this lineage occupies in that
    sample, never an average of per-window ratios); ``haplotype_ids`` lists that sample's
    members so the row joins straight back to haplotypes.tsv.
    """
    rows: list[dict] = []
    for lin in lineages:
        ab = lin.abundance_by_sample()
        ms, me = lin.marker_span
        by_sample: dict[str, list] = {}
        for g in lin.groups:
            for m in g.members:
                by_sample.setdefault(m.sample, []).append((g, m))
        for sample_id, pairs in sorted(by_sample.items()):
            p = ab[sample_id]
            rows.append({
                "lineage_id": lin.lineage_id,
                "mag": mag_name,
                "contig": lin.contig,
                "sample": sample_id,
                "n_windows_lineage": lin.n_windows,
                "n_samples_lineage": len(lin.samples),
                "window_start": lin.window_start,
                "window_end": lin.window_end,
                "marker_start": ms,
                "marker_end": me,
                "marker_span_bp": me - ms,
                "n_windows": p.n_windows,
                "abundance": p.abundance,
                "abundance_all_reads": p.abundance_all_reads,
                "reads": p.reads,
                "total_reads": p.total_reads,
                "junk_reads": p.junk_reads,
                "window_group_ids": ",".join(g.group_id for g, _ in sorted(
                    pairs, key=lambda x: x[0].window_start)),
                "haplotype_ids": ",".join(m.haplotype_id for _, m in sorted(
                    pairs, key=lambda x: x[0].window_start)),
            })
    return rows


def _write_lineage_edges(edges, output_dir: str) -> None:
    """Every attempted continuation with its outcome, into ``<output_dir>/tmp``.

    Diagnostic, not a deliverable: which joins were refused and why is what distinguishes
    a recombination breakpoint from a coverage hole, but it is O(groups^2). It lives
    beside the run it describes rather than in a system temp directory, so it is findable,
    cleanable, and cannot collide between concurrent runs.
    """
    import csv as _csv

    if not edges:
        return
    tmp = os.path.join(output_dir, "tmp")
    os.makedirs(tmp, exist_ok=True)
    path = os.path.join(tmp, "lineage_edges.tsv")
    with open(path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(vars(edges[0]).keys()), delimiter="\t")
        w.writeheader()
        w.writerows(vars(e) for e in edges)
    from collections import Counter as _C
    logging.info(f"  lineage edges (temp, diagnostic): {len(edges)} -> {path} | "
                 f"{dict(_C(e.reason for e in edges))}")


def build_window_tables(
    output_dir: str,
    all_results: dict[str, dict[str, dict[str, list[WindowResult]]]],
    config: HaplotyperConfig,
    sample_order: list[str] | None = None,
) -> tuple[
    list[dict], list[dict], list[dict], list[dict], list[dict], dict[str, int], list[dict]
]:
    """Build the window-level tables plus the recorded cross-sample comparisons.

    Returns ``(haplotypes, windows_within_sample, windows_across_samples, edges,
    within_mismatches, edge_counts, lineages)``:

    ``haplotypes``
        One row per haplotype per window per sample. The raw unit; no merging applied.

    ``windows_within_sample``
        Windows merged WITHIN one sample, across windows - the ``link_windows`` output
        (the "horizontal" axis). One row per (sample, contig, entity).

    ``windows_across_samples``
        Windows merged ACROSS samples at ONE FIXED window - the "vertical" axis. One row
        per (contig, window, group).

    ``edges``
        Every cross-sample comparison that was attempted, passed or failed, with the
        reason. ``failed_no_evidence`` (a dropout) and ``failed_mismatch`` (a genuine
        genotypic difference, i.e. a candidate recombination breakpoint) must stay
        distinguishable; a discarded comparison cannot be told apart from one never made.

    ``within_mismatches``
        The step-1 rows where one sample's own reads disagreed across a window boundary.
        They are also step 3's veto set, which is why they carry the same haplotype ids
        as the rows above.

    ``edge_counts``
        Attempted continuations tallied by reason, for the run log.

    ``lineages``
        One row per (lineage, sample), produced here whenever ``config.build_lineages``
        is set - which it is by default. It is built in this function rather than by a
        separate driver because everything step 3 needs is already in memory: the groups,
        the contig-wide marker set, and step 1's mismatch verdicts.
    """
    haplotype_rows: list[dict] = []
    within_rows: list[dict] = []
    window_haps: list[WindowHaplotype] = []
    site_type_all: dict[int, str] = {}
    within_mismatch_rows: list[dict] = []
    mag_of_contig: dict[str, str] = {}

    for mag_name, mag_results in all_results.items():
        for sample_id, contig_results in mag_results.items():
            for contig_id, window_results in contig_results.items():
                mag_of_contig[contig_id] = mag_name
                # Group by the within-sample entity id assigned by link_windows.
                entities: dict[str, list[tuple[WindowResult, Haplotype, int]]] = defaultdict(list)

                # Step-1 mismatches, resolved through the SAME id helper the haplotype
                # rows and the WindowHaplotype objects use below, so they join back to
                # haplotypes.tsv and so step 3's veto set speaks the members' language.
                # Only genuine allele disagreements are here; dropouts are not recorded
                # (see WindowResult.link_mismatches).
                by_start = {wr.window.start: wr for wr in window_results}

                def _hid(win_start: int, h_idx: int, _bs=by_start,
                         _s=sample_id, _c=contig_id) -> str:
                    wr_ = _bs.get(win_start)
                    if wr_ is None or h_idx >= len(wr_.haplotypes):
                        return ""
                    return _window_haplotype_id(_s, _c, win_start, h_idx)

                for wr in window_results:
                    for m in getattr(wr, "link_mismatches", []):
                        within_mismatch_rows.append(
                            {
                                "mag": mag_name,
                                "contig": m["contig"],
                                "sample": sample_id,
                                "window_a": m["window_a"],
                                "window_b": m["window_b"],
                                "haplotype_a": _hid(m["window_a"], m["hap_a_idx"]),
                                "haplotype_b": _hid(m["window_b"], m["hap_b_idx"]),
                                "rate": m["rate"],
                                "n_shared": m["n_shared"],
                                "n_diff": m["n_diff"],
                                "used_fallback": m["used_fallback"],
                            }
                        )

                for wr in window_results:
                    site_type_all.update(wr.window.site_type)
                    n_reads_w = getattr(wr, "n_reads_examined", len(wr.window.reads))
                    junk_col = wr.gamma.shape[1] - 1
                    n_junk_w = int((wr.gamma[:, junk_col] >= 0.5).sum())
                    nonjunk = n_reads_w - n_junk_w

                    for h_idx, hap in enumerate(wr.haplotypes):
                        # Haplotypes with no CONFIDENT read are emitted too, with reads=0.
                        # supporting_reads counts gamma[:, k] >= assign_confidence_threshold
                        # (0.90); nothing prunes a haplotype on read support, so a real
                        # mixture component can hold pi weight and still have every one of
                        # its reads sitting at gamma 0.6-0.89 - two strains agreeing on all
                        # but a couple of markers at ~10x depth do exactly that. Skipping
                        # them kept them out of EVERY table, and since `abundance` below is
                        # pi_k / (1 - pi_junk), which sums to 1 over all k by construction,
                        # the published abundances then summed to less than 1 with no
                        # residual column and no log line - a reader could not tell mass
                        # was missing. Emitting the row is what makes the window audit.
                        raw_eid = hap.track_id or f"unlinked_{wr.window.start}"
                        eid = f"{sample_id}_{contig_id}_{raw_eid}"
                        entities[eid].append((wr, hap, h_idx))

                        abundance = _window_conditional_abundance(
                            getattr(wr, "pi", None), h_idx
                        )
                        # UNIFORM ID SCHEME (author's, 2026-07-28). Every id carries the
                        # scope it is unique within, so no id needs a companion column to
                        # be a key:
                        #   step 0  haplotype  sample_contig_window_H<idx>
                        #   step 1  track      sample_contig_T<idx>
                        #   step 2  group      contig_window_H<idx>
                        # Nothing here restarts per scope the way the raw counters do -
                        # link_windows assigns track_id per (sample, contig), so bare
                        # "T0001" recurred in every sample AND every contig, and joining
                        # on it inflated haplotypes.tsv x windows_within_sample.tsv 94x.
                        hap_id = _window_haplotype_id(
                            sample_id, contig_id, wr.window.start, h_idx
                        )
                        consensus_str = "|".join(
                            f"{p}:{b}" for p, b in sorted(hap.consensus.items())
                        )
                        # Footprint the haplotype's MARKERS actually occupy, as distinct
                        # from the window tile it sits in. A haplotype can carry 3 markers
                        # across 200 bp inside a 20 kb window, and reporting only the tile
                        # makes that indistinguishable from one spanning the whole window.
                        # That distinction is what separates "linking failed" from "there
                        # was no variation here to link on" - the latter is a fact about
                        # the biology, not a defect.
                        if hap.consensus:
                            hap_start = min(hap.consensus)
                            hap_end = max(hap.consensus)
                            hap_span = hap_end - hap_start
                        else:
                            hap_start = hap_end = hap_span = 0
                        win_bp = max(wr.window.end - wr.window.start, 1)
                        haplotype_rows.append(
                            {
                                "haplotype_id": hap_id,
                                "mag": mag_name,
                                "contig": contig_id,
                                "sample": sample_id,
                                "window_start": wr.window.start,
                                "window_end": wr.window.end,
                                "hap_start": hap_start,
                                "hap_end": hap_end,
                                "hap_span_bp": hap_span,
                                "hap_span_frac": round(hap_span / win_bp, 4),
                                "markers_per_kb": (
                                    round(len(hap.consensus) / (hap_span / 1000.0), 3)
                                    if hap_span else 0.0
                                ),
                                "within_sample_id": eid,
                                "abundance": abundance,
                                "reads": hap.supporting_reads,
                                "total_reads": nonjunk,
                                "junk_reads": n_junk_w,
                                "n_markers": len(hap.consensus),
                                "consensus": consensus_str,
                            }
                        )
                        window_haps.append(
                            WindowHaplotype(
                                sample=sample_id,
                                contig=contig_id,
                                window_start=wr.window.start,
                                window_end=wr.window.end,
                                haplotype_id=hap_id,
                                consensus=dict(hap.consensus),
                                reads=hap.supporting_reads,
                                total_reads=nonjunk,
                                junk_reads=n_junk_w,
                                abundance=abundance if abundance is not None else float("nan"),
                                # Step 1's chain id. Step 3 votes on this to join two
                                # window groups; without it every edge is
                                # `failed_no_votes` and NO lineage can span more than a
                                # single window. It was written to the TSV above but
                                # never onto the object step 3 actually reads.
                                within_sample_id=eid,
                            )
                        )

                for eid, members in entities.items():
                    track_reads = sum(h.supporting_reads for _, h, _ in members)
                    _counts = [_read_counts(wr) for wr, _, _ in members]
                    track_total = sum(c[0] for c in _counts)
                    track_junk = sum(c[1] for c in _counts)
                    track_all = track_total + track_junk
                    starts = [wr.window.start for wr, _, _ in members]
                    ends = [wr.window.end for wr, _, _ in members]
                    # marker footprint across every member, vs the tiles it nominally spans
                    mpos = [p for _, h, _ in members for p in h.consensus]
                    hs, he = (min(mpos), max(mpos)) if mpos else (0, 0)
                    within_rows.append(
                        {
                            "within_sample_id": eid,
                            "mag": mag_name,
                            "contig": contig_id,
                            "sample": sample_id,
                            "n_windows": len({s for s in starts}),
                            "window_min": min(starts),
                            "window_max": max(ends),
                            "span_bp": max(ends) - min(starts),
                            "hap_start": hs,
                            "hap_end": he,
                            "hap_span_bp": he - hs,
                            "n_markers": len(set(mpos)),
                            # POOLED READ COUNTS, not an average of per-window ratios.
                            # Sum(supporting) / Sum(non-junk) over the track's windows is
                            # a proper pooled estimate; a mean or median of the per-window
                            # abundances is not, because each window has its own
                            # denominator (median 9 non-junk reads, varying ~4x across one
                            # sample's windows) and 46% of windows hold a single haplotype
                            # whose abundance is 1.000 by construction.
                            #
                            # CAVEAT: adjacent windows overlap by 50%, so a read spanning
                            # the overlap is counted in both. Numerator and denominator
                            # inflate together so the ratio holds, but the overlap region
                            # is effectively double-weighted.
                            "reads": track_reads,
                            "total_reads": track_total,
                            "junk_reads": track_junk,
                            # share of the reads that PHASED (renormalises away how much
                            # of the window resolved)
                            "abundance": (track_reads / track_total) if track_total else float("nan"),
                            # share of ALL reads at these loci - does not hide a window
                            # where most reads went to junk
                            "abundance_all_reads": (track_reads / track_all) if track_all else float("nan"),
                            # The SAME id as haplotypes.tsv, so the member list joins
                            # back directly. It previously used the track-prefixed form,
                            # which matched nothing in that table; it now comes from the
                            # one helper, so it cannot drift again.
                            "haplotype_ids": ",".join(
                                _window_haplotype_id(sample_id, contig_id, wr.window.start, i)
                                for wr, _, i in members
                            ),
                        }
                    )

    # ---- vertical axis: group across samples at each fixed window ----
    groups, edges, edge_counts = group_all_windows(
        window_haps, config, sample_order=sample_order, site_type=site_type_all
    )

    # ---- step 3: chain those groups along the genome into lineages ----
    #
    # Run here rather than from a separate driver because everything it needs is already
    # in memory: the groups, the contig-wide marker set, and step 1's mismatch verdicts.
    # Driving it externally meant re-reading a 224 MB haplotypes.tsv and, historically,
    # meant the pipeline shipped a lineage table built by a different algorithm.
    lineage_rows: list[dict] = []
    if config.build_lineages:
        step1_mm = {frozenset((r["haplotype_a"], r["haplotype_b"]))
                    for r in within_mismatch_rows if r["haplotype_a"] and r["haplotype_b"]}
        by_contig: dict[str, list] = defaultdict(list)
        for g in groups:
            by_contig[g.contig].append(g)
        lineage_edges: list = []
        for contig_id_, cgroups in sorted(by_contig.items()):
            markers = variable_marker_positions(
                (m.consensus for g in cgroups for m in g.members), site_type_all, config)
            lins, ledges = build_lineages(
                cgroups, config, markers=markers, step1_mismatches=step1_mm,
                max_bad_frac=config.lineage_max_bad_frac,
                lineage_prefix=f"{contig_id_}_LIN")
            lineage_edges.extend(ledges)
            lineage_rows.extend(_lineage_rows(lins, mag_of_contig.get(contig_id_, "")))
        _write_lineage_edges(lineage_edges, output_dir)

    across_rows: list[dict] = []
    for g in groups:
        across_rows.append(
            {
                "window_group_id": g.group_id,
                "contig": g.contig,
                "window_start": g.window_start,
                "window_end": g.window_end,
                "n_members": g.n_members,
                "n_samples": g.n_samples,
                "hap_start": _group_marker_span(g)[0],
                "hap_end": _group_marker_span(g)[1],
                "hap_span_bp": _group_marker_span(g)[1] - _group_marker_span(g)[0],
                "median_member_span_bp": _median_member_span(g),
                "n_markers": len({p for m in g.members for p in m.consensus}),
                "samples": ",".join(sorted({m.sample for m in g.members})),
                "haplotype_ids": ",".join(m.haplotype_id for m in g.members),
                "method": config.cross_sample_method,
            }
        )

    # `edges` now holds ONLY mismatches - the other outcomes are counted, not stored.
    edge_rows = [
        {
            "contig": e.contig,
            "window_start": e.window_start,
            "sample_a": e.sample_a,
            "sample_b": e.sample_b,
            "haplotype_a": e.haplotype_a,
            "haplotype_b": e.haplotype_b,
            "reason": e.reason,
            "rate": e.rate,
            "n_shared": e.n_shared,
            "n_diff": e.n_diff,
        }
        for e in edges
    ]

    return (haplotype_rows, within_rows, across_rows, edge_rows,
            within_mismatch_rows, edge_counts, lineage_rows)


def write_window_tables(
    haplotype_rows: list[dict],
    within_rows: list[dict],
    across_rows: list[dict],
    edge_rows: list[dict],
    output_dir: str,
    within_mismatch_rows: list[dict] | None = None,
    lineage_rows: list[dict] | None = None,
) -> dict[str, str]:
    """Write the window-level tables, plus the two MISMATCH tables.

    A link that was not made for lack of shared markers is a measurement hole and is not
    reported - there is nothing to say about it, and at cohort scale those rows are what
    made the full comparison log unaffordable. A link that was not made because the
    alleles genuinely DISAGREE is a finding: a candidate recombination breakpoint, and
    the one negative the merge rules treat as absolute. Those are written, from both
    linking steps.
    """
    import csv as _csv

    os.makedirs(output_dir, exist_ok=True)
    tmp_dir = os.path.join(output_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    written: dict[str, str] = {}

    # The FULL comparison log is still not written: it was ~144 MB for a single MAG
    # (1.27M rows), ~30 GB across a 233-MAG cohort, and almost all of it is
    # failed_no_evidence. Only the mismatches are kept - 11% of rows on 000089747_1.
    # DELIVERABLES in output_dir; DIAGNOSTICS in output_dir/tmp beside the run they
    # describe, so they are findable and cleanable and cannot collide between concurrent
    # runs. The mismatch tables are diagnostics: they carry the cannot-link constraints
    # step 3 consumes internally, and at 2.3M rows for one MAG they are not a deliverable.
    tables = [
        ("haplotypes.tsv", haplotype_rows, output_dir),
        ("windows_within_sample.tsv", within_rows, output_dir),
        ("windows_across_samples.tsv", across_rows, output_dir),
        ("lineages.tsv", lineage_rows or [], output_dir),
        ("mismatches_within_sample.tsv", within_mismatch_rows or [], tmp_dir),
        ("mismatches_across_samples.tsv", edge_rows, tmp_dir),
    ]
    for name, rows, dest in tables:
        path = os.path.join(dest, name)
        with open(path, "w", newline="") as f:
            if rows:
                writer = _csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
                writer.writeheader()
                writer.writerows(rows)
            else:
                f.write("")
        written[name] = path
        logging.info(f"Wrote {len(rows)} rows to {path}")

    return written


# -----------------------------------------------------------------------------#
# CLI
# -----------------------------------------------------------------------------#


def main():
    parser = argparse.ArgumentParser(
        description="Longitudinal haplotype integration across samples",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--samples", required=True, help="Comma-separated list of sample names")
    parser.add_argument("--bams", required=True, help="BAM path template with {sample} placeholder")
    parser.add_argument("--vcfs", required=True, help="VCF path template with {sample} placeholder")
    parser.add_argument(
        "--sv-sidecars",
        help="Optional SV sidecar TSV path template with {sample} placeholder "
        "(from strainphase.sv_encoding). Enables co-phasing of structural "
        "variants as pseudo-SNVs.",
    )
    parser.add_argument("--reference", required=True, help="Combined reference FASTA")
    parser.add_argument("--output-dir", required=True, help="Output directory")

    # Optional / efficiency-related
    parser.add_argument(
        "--mags", help="Comma-separated list of MAGs to process (default: all MAGs in reference)"
    )
    parser.add_argument(
        "--contig-filter", help="TSV or text file listing contigs to include (see docstring)"
    )
    parser.add_argument("--window-size", type=int, default=20000, help="Window size for haplotyping")
    parser.add_argument(
        "--max-reads",
        type=int,
        # 300, matching `strainphase longitudinal` and README.md. This parser stood at
        # 500, and max_reads_per_window selects WHICH reads enter a window, upstream of
        # the graph, the EM and every abundance denominator - so any window holding
        # 301-500 reads was phased from a different read set depending on which of the
        # two documented entry points the operator happened to use.
        default=300,
        help="Max reads per window; above this reads are uniformly subsampled with the "
        "config seed. Applied to RAW reads before junk classification, so the effective "
        "non-junk ceiling is lower.",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes for within-MAG window processing "
        "(default: 1 = sequential). Set this to --cpus-per-task so reserved "
        "cores are actually used; useful for large/high-coverage MAGs.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed. Seeded by default: it drives both read subsampling above "
        "--max-reads and Louvain read clustering, so an unseeded run is not reproducible.",
    )
    parser.add_argument(
        "--validate-results",
        action="store_true",
        help="Enable internal consistency checks on WindowResult (slower)",
    )

    # Longitudinal hyperparameters exposed for transparency
    parser.add_argument(
        "--min-anchor-weight",
        type=float,
        default=0.15,
        help="Min mixture weight for a haplotype to serve as an anchor",
    )
    parser.add_argument(
        "--rescued-min-weight",
        type=float,
        default=0.02,
        help="Min mixture weight to assign to rescued haplotypes",
    )

    # Identity gates (shared across the three linking levels)
    parser.add_argument(
        "--lineage-merge-distance",
        type=float,
        default=0.01,
        help="Max mismatch RATE to group haplotypes (default: 0.01 = 1%%). "
        "--max-num-diff caps its reach: a pair still needs that many differences or "
        "fewer, so raising this above max_num_diff/min_shared stops changing anything.",
    )
    parser.add_argument(
        "--max-num-diff",
        type=int,
        default=1,
        help="Max ABSOLUTE mismatches. The rate gate is a floor, so it already forces 0 "
        "mismatches below n_shared=100; this cap is what binds at high n_shared, where a "
        "rate alone would tolerate 11 mismatches at n_shared=1172.",
    )
    parser.add_argument(
        "--min-shared-for-lineage",
        type=int,
        default=3,
        help="Min shared identity markers to compare two haplotypes across samples",
    )
    parser.add_argument(
        "--min-cosupported-span-frac",
        type=float,
        default=0.25,
        help="Min co-supported span between two haplotypes, as a fraction of their shared "
        "region. 0.25 rejects 16%% of adjacent-window pairs; 0.50 rejects 30%%.",
    )
    parser.add_argument(
        "--min-entity-overlap-bp",
        type=int,
        default=1000,
        help="Min physical overlap between two entities; below this the verdict is an "
        "explicit non-merge rather than 'unknown'",
    )
    parser.add_argument(
        "--min-read-window-overlap-bp",
        type=int,
        default=1000,
        help="A read must cover at least this many bp of a window to be counted in it",
    )
    parser.add_argument(
        "--min-read-read-overlap-bp",
        type=int,
        default=1000,
        help="Two reads must physically overlap by at least this much to be compared",
    )
    parser.add_argument(
        "--keep-read-assignments",
        action="store_true",
        help="Retain the per-read hard assignments on every WindowResult. Debug only: "
        "they are written to no output file and read by no other code, and on a "
        "many-sample MAG they are tens of millions of dicts held for nothing.",
    )
    parser.add_argument(
        "--no-spill",
        action="store_true",
        help="Keep every sample's reads in memory until the whole MAG finishes (the old "
        "behaviour). By default reads are parked in <output-dir>/tmp/spill after each "
        "sample is phased and reloaded one sample at a time for the rescue pass.",
    )
    parser.add_argument(
        "--window-batch-factor",
        type=int,
        default=4,
        help="Windows are dispatched to the worker pool in batches of workers * this. "
        "Lower it to cut peak memory on variant-dense contigs; raise it to cut "
        "scheduling overhead.",
    )
    parser.add_argument(
        "--cross-sample-method",
        choices=["clique", "reciprocal"],
        default="clique",
        help="How haplotypes are grouped across samples at a fixed window. 'clique' = "
        "complete linkage, no time axis, immune to irregular timepoint spacing. "
        "'reciprocal' = unique-best + mutual between consecutive samples; requires "
        "--samples to be in true chronological order.",
    )
    parser.add_argument(
        "--min-reads-for-rescue",
        type=int,
        default=5,
        help="Reads needed for a window to be BUILT (so rescue can populate it). De-novo "
        "phasing uses the higher --min-reads-per-window.",
    )
    parser.add_argument(
        "--min-reads-per-window",
        type=int,
        default=10,
        help="Reads needed to PHASE a window de novo. Below this no haplotype is invented "
        "and the trajectory carries a gap.",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Parse samples
    samples = [s.strip() for s in args.samples.split(",") if s.strip()]
    if not samples:
        logging.error("No valid samples provided in --samples")
        sys.exit(1)
    logging.info(f"Processing {len(samples)} samples: {samples}")

    # Build file paths
    bam_paths = {s: args.bams.replace("{sample}", s) for s in samples}
    vcf_paths = {s: args.vcfs.replace("{sample}", s) for s in samples}
    sv_sidecar_paths = None
    if getattr(args, "sv_sidecars", None):
        sv_sidecar_paths = {s: args.sv_sidecars.replace("{sample}", s) for s in samples}

    # Validate files exist
    for sample in samples:
        checks = [(bam_paths[sample], "BAM"), (vcf_paths[sample], "VCF")]
        if sv_sidecar_paths:
            checks.append((sv_sidecar_paths[sample], "SV sidecar"))
        for path, name in checks:
            if not os.path.exists(path):
                logging.error(f"{name} not found for {sample}: {path}")
                sys.exit(1)

    # Load optional contig filter
    allowed_contigs: set[str] | None = None
    if args.contig_filter:
        allowed_contigs = load_allowed_contigs(args.contig_filter)

    # Build config for haplotyper
    config = HaplotyperConfig(
        window_size=args.window_size,
        max_reads_per_window=args.max_reads,
        random_seed=args.seed,
        validate_results=args.validate_results,
        min_weight_for_anchor=args.min_anchor_weight,
        rescued_min_weight=args.rescued_min_weight,
        # Depth policy
        min_reads_per_window=args.min_reads_per_window,
        min_reads_for_rescue=args.min_reads_for_rescue,
        # Identity gates
        lineage_merge_distance=args.lineage_merge_distance,
        min_shared_for_lineage=args.min_shared_for_lineage,
        max_num_diff=args.max_num_diff,
        min_cosupported_span_frac=args.min_cosupported_span_frac,
        min_entity_overlap_bp=args.min_entity_overlap_bp,
        min_read_window_overlap_bp=args.min_read_window_overlap_bp,
        min_read_read_overlap_bp=args.min_read_read_overlap_bp,
        cross_sample_method=args.cross_sample_method,
        n_workers=max(1, args.workers),
        keep_read_assignments=args.keep_read_assignments,
        spill_results_to_disk=not args.no_spill,
        window_batch_factor=args.window_batch_factor,
    )

    # Get MAG -> contigs map, optionally filtered by allowed_contigs
    all_mags = parse_reference_contigs(args.reference, allowed_contigs)

    mag_names_raw = args.mags.strip() if args.mags else ""
    if mag_names_raw:
        mag_names = [m.strip() for m in mag_names_raw.split(",") if m.strip()]
        if not mag_names:
            logging.error("--mags was provided but contains no valid MAG names after stripping whitespace")
            sys.exit(1)
        mags_to_process = {m: all_mags[m] for m in mag_names if m in all_mags}
        missing = sorted(set(mag_names) - set(mags_to_process.keys()))
        if missing:
            logging.warning(f"Requested MAGs not found in reference: {missing}")
    else:
        mags_to_process = all_mags

    if not mags_to_process:
        logging.error("No MAGs to process after applying filters")
        sys.exit(1)

    logging.info(f"Processing {len(mags_to_process)} MAGs: {sorted(mags_to_process)}")

    # Process each MAG with longitudinal integration
    all_results: dict[str, dict[str, dict[str, list[WindowResult]]]] = {}
    all_integrators = []

    for mag_name, mag_contigs in mags_to_process.items():
        mag_results, integrator = process_mag_longitudinal(
            mag_name=mag_name,
            mag_contigs=mag_contigs,
            samples=samples,
            bam_paths=bam_paths,
            vcf_paths=vcf_paths,
            config=config,
            sv_sidecar_paths=sv_sidecar_paths,
            output_dir=args.output_dir,
        )
        all_results[mag_name] = mag_results
        if integrator:
            all_integrators.append(integrator)

    # ---- Window-level tables (the deliverables) ----
    # lineages.tsv comes back from here too, under config.build_lineages (default on):
    # composing the within-sample and across-sample axes was the open decision these
    # tables were built as the substrate for, and step 3 now makes it.
    logging.info("Building window tables across processed MAGs")
    (hap_rows, within_rows, across_rows, edge_rows, mismatch_rows,
     edge_counts, lineage_rows) = build_window_tables(
        args.output_dir, all_results, config, sample_order=samples
    )
    write_window_tables(
        hap_rows, within_rows, across_rows, edge_rows, args.output_dir, mismatch_rows,
        lineage_rows
    )

    n_within = len({(r["sample"], r["contig"], r["within_sample_id"]) for r in within_rows})
    n_across = len(across_rows)
    n_failed_mismatch = edge_counts.get("failed_mismatch", 0)
    n_failed_evidence = edge_counts.get("failed_no_evidence", 0)
    n_linked = edge_counts.get("linked", 0)
    logging.info(
        f"DONE: {len(mags_to_process)} MAGs | {len(hap_rows)} window-haplotypes | "
        f"{n_within} within-sample entities | {n_across} across-sample window groups "
        f"(method={config.cross_sample_method}) | comparisons: "
        f"{n_linked} linked, "
        f"{n_failed_evidence} no-evidence, {n_failed_mismatch} mismatch | "
        f"mismatches written: {len(mismatch_rows)} within-sample, "
        f"{n_failed_mismatch} across-sample | "
        f"lineages: {len({r['lineage_id'] for r in lineage_rows})} over "
        f"{len(lineage_rows)} (lineage, sample) rows"
    )



if __name__ == "__main__":
    main()
