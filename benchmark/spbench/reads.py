"""Extracting per-read alleles at variant sites, and building consensus
haplotypes from a read partition.

Both live here because the benchmark applies them *uniformly to every tool*.

The reason is fairness. Tools disagree wildly about how to report a haplotype:
Floria emits allele indices over its own SNP numbering, Strainy emits assembly
graph paths, devider emits an MSA, strainphase emits window-linked consensus
tracks. Scoring those native representations against each other means scoring
four different consensus-calling implementations as much as four phasing
algorithms.

So the benchmark takes each tool's read partition - the one output they all
genuinely produce - and derives the consensus haplotype from it with the same
code for everyone. A tool's score then reflects how well it grouped reads, which
is the thing it actually does.

Tools that also emit a native haplotype (strainphase does) are additionally
scored on it; those columns are labelled ``native`` in the results and are not
used for the headline cross-tool comparison.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from spbench.formats import Haplotype

logger = logging.getLogger(__name__)


def load_sites(vcf_path: str, contig: str) -> dict[int, tuple[str, str, str]]:
    """``pos -> (ref, alt, kind)`` for a contig. ``kind`` is snv/del/ins."""
    import pysam

    sites: dict[int, tuple[str, str, str]] = {}
    with pysam.VariantFile(vcf_path) as vcf:
        if contig not in vcf.header.contigs:
            return sites
        for record in vcf.fetch(contig):
            if record.filter.keys() and "PASS" not in record.filter.keys():
                continue
            if not record.alts:
                continue
            ref, alt = record.ref, record.alts[0]
            if len(ref) == 1 and len(alt) == 1:
                kind = "snv"
            elif len(ref) > len(alt):
                kind = "del"
            elif len(alt) > len(ref):
                kind = "ins"
            else:
                continue  # MNP: out of scope, as it is for the tools too
            sites[record.pos] = (ref, alt, kind)
    return sites


def read_alleles(
    bam_path: str,
    contig: str,
    sites: dict[int, tuple[str, str, str]],
    min_mapq: int = 20,
) -> dict[str, dict[int, tuple[str, int]]]:
    """``read_id -> {pos: (allele_string, base_quality)}``.

    The allele is the literal sequence observed, so it compares directly against
    the VCF REF/ALT strings and against the truth encoding. A sequencing error
    yields an allele equal to neither, which is the correct outcome: it should
    not silently round to the nearer of the two.
    """
    import pysam

    out: dict[str, dict[int, tuple[str, int]]] = {}
    if not sites:
        return out
    positions = sorted(sites)
    lo, hi = positions[0], positions[-1]

    with pysam.AlignmentFile(bam_path, "rb") as bam:
        if contig not in bam.references:
            return out
        for aln in bam.fetch(contig, max(0, lo - 1), hi + 1):
            if aln.is_unmapped or aln.mapping_quality < min_mapq:
                continue
            if aln.is_secondary or aln.is_supplementary:
                continue
            seq = aln.query_sequence
            quals = aln.query_qualities
            if seq is None:
                continue

            ref_to_query: dict[int, int] = {}
            deleted: set[int] = set()
            for qpos, rpos in aln.get_aligned_pairs(matches_only=False):
                if rpos is None:
                    continue
                if qpos is None:
                    deleted.add(rpos + 1)
                else:
                    ref_to_query[rpos + 1] = qpos

            insertions = _insertion_anchors(aln)

            calls: dict[int, tuple[str, int]] = {}
            start = aln.reference_start + 1
            end = aln.reference_end or start
            for pos in positions:
                if pos < start or pos > end:
                    continue
                ref, alt, kind = sites[pos]
                call = _call_site(
                    pos, ref, alt, kind, seq, quals, ref_to_query, deleted, insertions, end
                )
                if call is not None:
                    calls[pos] = call
            if calls:
                out[aln.query_name] = calls
    return out


def _insertion_anchors(aln) -> dict[int, int]:
    """``anchor reference position (1-based) -> inserted length``."""
    anchors: dict[int, int] = {}
    ref_pos = aln.reference_start  # 0-based
    for op, length in aln.cigartuples or []:
        if op in (0, 7, 8):  # M / = / X
            ref_pos += length
        elif op == 2:  # D
            ref_pos += length
        elif op == 1:  # I - anchored on the last consumed reference base
            anchors[ref_pos] = anchors.get(ref_pos, 0) + length
        # S/H/P consume no reference
    return anchors


def _call_site(
    pos: int,
    ref: str,
    alt: str,
    kind: str,
    seq: str,
    quals,
    ref_to_query: dict[int, int],
    deleted: set[int],
    insertions: dict[int, int],
    read_end: int,
) -> tuple[str, int] | None:
    qpos = ref_to_query.get(pos)
    if qpos is None:
        return None
    quality = int(quals[qpos]) if quals is not None and qpos < len(quals) else 30

    if kind == "snv":
        return seq[qpos], quality

    if kind == "del":
        del_len = len(ref) - len(alt)
        span = range(pos + 1, pos + del_len + 1)
        if read_end < pos + del_len:
            return None  # read does not span the deletion; no call, not a ref call
        if all(p in deleted for p in span):
            return alt, quality
        if not any(p in deleted for p in span):
            return ref, quality
        return None  # partial overlap: ambiguous, and saying so is the honest call

    # insertion
    ins_len = len(alt) - len(ref)
    observed = insertions.get(pos, 0)
    if observed == ins_len:
        return alt, quality
    if observed == 0:
        return ref, quality
    return None


def consensus_from_partition(
    assignments: dict[str, str],
    alleles: dict[str, dict[int, tuple[str, int]]],
    sample: str,
    contig: str,
    tool: str,
    min_depth: int = 3,
    min_fraction: float = 0.5,
) -> list[Haplotype]:
    """Majority-vote consensus per cluster, weighted by base quality.

    ``min_depth`` sites with fewer supporting reads are left uncalled rather
    than guessed. Leaving a site uncalled costs the tool nothing in the allele
    accuracy metric (which only scores shared positions) but does cost it span
    and site count, which is the right trade: a method should not be rewarded
    for confabulating alleles it has no reads for.
    """
    by_cluster: dict[str, list[str]] = defaultdict(list)
    for read_id, cluster in assignments.items():
        by_cluster[cluster].append(read_id)

    total_reads = sum(len(v) for v in by_cluster.values())
    haplotypes: list[Haplotype] = []

    for cluster, read_ids in sorted(by_cluster.items()):
        votes: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        depth: dict[int, int] = defaultdict(int)
        for read_id in read_ids:
            for pos, (allele, quality) in alleles.get(read_id, {}).items():
                # Weight by P(base correct); a Q10 base carries a tenth of the
                # vote of a Q30 one rather than an equal share.
                votes[pos][allele] += 1.0 - 10.0 ** (-quality / 10.0)
                depth[pos] += 1

        consensus: dict[int, str] = {}
        for pos, allele_votes in votes.items():
            if depth[pos] < min_depth:
                continue
            best_allele, best_weight = max(allele_votes.items(), key=lambda kv: kv[1])
            if best_weight / sum(allele_votes.values()) < min_fraction:
                continue
            consensus[pos] = best_allele

        if not consensus:
            continue

        haplotypes.append(
            Haplotype(
                hap_id=f"{tool}:{cluster}",
                sample=sample,
                contig=contig,
                alleles=consensus,
                start=min(consensus),
                end=max(consensus),
                abundance=(len(read_ids) / total_reads) if total_reads else None,
            )
        )

    return haplotypes
