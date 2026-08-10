"""Haplotype reconstruction accuracy: alleles, abundance, and how many.

Matching predicted haplotypes to true strains is the step where a benchmark can
quietly favour one tool, so the rule here is fixed and stated up front:

* Agreement between a predicted haplotype and a true strain is computed only
  over positions where **both** have a call. A tool that reports a short, highly
  confident haplotype is not penalised for the positions it declined to call -
  it is penalised by the separate span and contiguity columns.
* A pair must share at least ``min_shared_sites`` positions to be comparable at
  all. Without this floor a two-site fragment agreeing by luck would count as a
  recovered strain.
* Matching is a global optimum, not greedy: the assignment maximising total
  agreement is found with the Hungarian algorithm, so the result does not depend
  on the order haplotypes happen to appear in a file.
* One predicted haplotype matches at most one true strain and vice versa. Tools
  that shatter a strain into fragments therefore score high precision on the
  fragment that matches and pay for the rest in recall and in ``n_predicted``.

The abundance columns are reported only for tools that emit an abundance. Most
long-read phasers do not, and inventing one from cluster read counts would be
scoring them on a quantity they never claimed to estimate.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from spbench.formats import Haplotype, Truth


def agreement(
    predicted: dict[int, str], true: dict[int, str]
) -> tuple[float, int, int]:
    """Return ``(fraction_agreeing, n_agreeing, n_shared)`` over shared sites."""
    shared = predicted.keys() & true.keys()
    if not shared:
        return 0.0, 0, 0
    n_agree = sum(1 for pos in shared if predicted[pos] == true[pos])
    return n_agree / len(shared), n_agree, len(shared)


def switch_count(predicted: dict[int, str], truth: Truth, contig: str) -> tuple[int, int]:
    """Count haplotype switches: how often a reconstruction jumps lineage.

    Walking the predicted haplotype's sites in genomic order, each site is
    attributed to the set of true strains carrying that allele there. A switch
    is counted when the running set of strains consistent with everything seen
    so far becomes empty and has to be reseeded - that is, when no single true
    strain explains the reconstruction across the boundary.

    This is the n-strain generalisation of diploid switch error, and it measures
    the failure mode that matters most in a metagenome: a chimeric contig that
    looks like a real strain but never existed.

    Returns ``(n_switches, n_opportunities)``.
    """
    strain_alleles = {
        sid: hap.alleles for sid, hap in truth.strains.items() if hap.contig == contig
    }
    if not strain_alleles:
        return 0, 0

    positions = sorted(predicted)
    consistent: set[str] | None = None
    switches = 0
    opportunities = 0

    for pos in positions:
        allele = predicted[pos]
        carriers = {
            sid
            for sid, alleles in strain_alleles.items()
            if alleles.get(pos) == allele
        }
        if not carriers:
            # An allele no true strain carries is a base error, not a switch;
            # it is already counted in the Hamming column.
            continue
        if consistent is None:
            consistent = carriers
            continue
        opportunities += 1
        overlap = consistent & carriers
        if overlap:
            consistent = overlap
        else:
            switches += 1
            consistent = carriers

    return switches, opportunities


def match_haplotypes(
    predicted: list[Haplotype],
    truth: Truth,
    contig: str,
    min_shared_sites: int = 10,
) -> list[tuple[Haplotype, str, float, int]]:
    """Optimal one-to-one assignment of predicted haplotypes to true strains.

    Returns ``(haplotype, strain_id, agreement_fraction, n_shared)`` for each
    matched pair. Pairs below ``min_shared_sites`` are never returned.
    """
    true_ids = [sid for sid, hap in truth.strains.items() if hap.contig == contig]
    if not predicted or not true_ids:
        return []

    scores = np.zeros((len(predicted), len(true_ids)))
    shared_counts = np.zeros((len(predicted), len(true_ids)), dtype=int)
    for i, hap in enumerate(predicted):
        for j, sid in enumerate(true_ids):
            frac, n_agree, n_shared = agreement(hap.alleles, truth.strains[sid].alleles)
            shared_counts[i, j] = n_shared
            # Score by the count of agreeing sites, not the fraction: a long
            # haplotype agreeing at 5000 sites should outrank a 12-site fragment
            # that happens to agree perfectly, when both compete for the same
            # true strain.
            scores[i, j] = n_agree if n_shared >= min_shared_sites else 0.0

    rows, cols = linear_sum_assignment(-scores)
    matches = []
    for i, j in zip(rows, cols, strict=True):
        if shared_counts[i, j] < min_shared_sites:
            continue
        frac, _, n_shared = agreement(
            predicted[i].alleles, truth.strains[true_ids[j]].alleles
        )
        matches.append((predicted[i], true_ids[j], frac, n_shared))
    return matches


def _n50(spans: list[int]) -> int:
    """N50 of reconstructed haplotype spans - the contiguity of the output."""
    if not spans:
        return 0
    ordered = sorted(spans, reverse=True)
    half = sum(ordered) / 2.0
    running = 0
    for span in ordered:
        running += span
        if running >= half:
            return span
    return ordered[-1]


def haplotype_metrics(
    predicted: list[Haplotype],
    truth: Truth,
    sample: str,
    contig: str,
    match_threshold: float = 0.99,
    min_shared_sites: int = 10,
    detection_floor: float = 0.0,
) -> dict:
    """Score haplotype reconstruction for one sample and contig.

    ``match_threshold`` is the allele agreement above which a matched pair
    counts as a recovered strain. 0.99 is strict on purpose: at the SNV
    densities simulated here, 1% disagreement is hundreds of wrong alleles, well
    beyond sequencing error.

    ``detection_floor`` excludes true strains below a given abundance from the
    recall denominator. It exists for the sensitivity sweep in the report, where
    recall is recomputed at several floors; the headline numbers use 0.0, so
    every strain that is genuinely present counts.
    """
    in_sample = [hap for hap in predicted if hap.sample == sample and hap.contig == contig]
    present = [
        sid
        for sid in truth.strains
        if truth.strains[sid].contig == contig
        and truth.abundance.get((sample, sid), 0.0) > detection_floor
    ]

    matches = match_haplotypes(in_sample, truth, contig, min_shared_sites)
    good = [m for m in matches if m[2] >= match_threshold and m[1] in present]

    n_pred = len(in_sample)
    n_true = len(present)
    precision = len(good) / n_pred if n_pred else 0.0
    recall = len(good) / n_true if n_true else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if n_true and (precision + recall) > 0
        else 0.0
    )

    # Allele-level accuracy over matched pairs, weighted by shared-site count so
    # a long correct haplotype is not outvoted by a short noisy one.
    total_shared = sum(m[3] for m in matches)
    weighted_agreement = (
        sum(m[2] * m[3] for m in matches) / total_shared if total_shared else float("nan")
    )

    switches = 0
    opportunities = 0
    for hap in in_sample:
        s, o = switch_count(hap.alleles, truth, contig)
        switches += s
        opportunities += o

    metrics = {
        "sample": sample,
        "contig": contig,
        "n_predicted": n_pred,
        "n_true_present": n_true,
        "n_matched": len(good),
        "hap_precision": precision,
        "hap_recall": recall,
        "hap_f1": f1,
        "k_error": n_pred - n_true,
        "allele_accuracy": weighted_agreement,
        "hamming_error_rate": (
            1.0 - weighted_agreement if weighted_agreement == weighted_agreement else float("nan")
        ),
        "switch_error_rate": switches / opportunities if opportunities else float("nan"),
        "n_switches": switches,
        "span_n50": _n50([hap.end - hap.start for hap in in_sample if hap.end > hap.start]),
        "mean_sites_per_hap": (
            float(np.mean([hap.n_sites for hap in in_sample])) if in_sample else 0.0
        ),
    }
    metrics.update(_abundance_metrics(good, truth, sample, present))
    return metrics


def _abundance_metrics(
    good: list[tuple[Haplotype, str, float, int]],
    truth: Truth,
    sample: str,
    present: list[str],
) -> dict:
    """Abundance error over matched pairs, plus a whole-community error that
    charges a tool for the strains it missed entirely."""
    pairs = [
        (truth.abundance.get((sample, sid), 0.0), hap.abundance)
        for hap, sid, _, _ in good
        if hap.abundance is not None
    ]
    if not pairs:
        return {
            "abundance_mae": float("nan"),
            "abundance_pearson_r": float("nan"),
            "abundance_mae_with_missed": float("nan"),
            "reports_abundance": False,
        }

    true_values = np.array([p[0] for p in pairs])
    pred_values = np.array([p[1] for p in pairs])
    mae = float(np.mean(np.abs(true_values - pred_values)))

    if len(pairs) >= 2 and true_values.std() > 0 and pred_values.std() > 0:
        pearson = float(np.corrcoef(true_values, pred_values)[0, 1])
    else:
        pearson = float("nan")

    matched_ids = {sid for _, sid, _, _ in good}
    missed = [truth.abundance.get((sample, sid), 0.0) for sid in present if sid not in matched_ids]
    all_errors = np.concatenate([np.abs(true_values - pred_values), np.array(missed)])

    return {
        "abundance_mae": mae,
        "abundance_pearson_r": pearson,
        "abundance_mae_with_missed": float(np.mean(all_errors)) if all_errors.size else float("nan"),
        "reports_abundance": True,
    }
