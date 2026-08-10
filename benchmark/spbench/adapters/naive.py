"""A deliberately unsophisticated baseline.

Greedy seed-and-sweep clustering of reads by their allele profiles: take the
read with the most calls, absorb every read within a Hamming threshold of it,
emit that as a cluster, repeat. No probabilistic model, no graph, no
cross-sample information.

Every benchmark of a phasing method needs this row. Without it there is no way
to tell whether a method's score reflects the method or reflects the fact that
the simulated strains were far enough apart that any clustering would separate
them. If a published tool cannot beat this row on a given configuration, that
configuration is too easy to be evidence for anything.

It is pure Python and depends only on pysam, so it runs everywhere the harness
runs - including CI, where it is the reference against which regressions in the
harness itself are detected.
"""

from __future__ import annotations

from pathlib import Path

from spbench.adapters.base import Adapter, ToolInfo
from spbench.dataset import Dataset
from spbench.formats import UNASSIGNED
from spbench.reads import load_sites, read_alleles


class NaiveAdapter(Adapter):
    info = ToolInfo(
        name="naive-greedy",
        designed_for=(
            "Nothing - it is a floor, not a method. Greedy Hamming clustering of "
            "read allele profiles within a single sample."
        ),
        supports_cross_sample_ids=False,
        multi_sample=False,
    )

    def __init__(
        self,
        max_mismatch_frac: float = 0.02,
        min_shared_sites: int = 3,
        min_cluster_reads: int = 3,
    ) -> None:
        self.max_mismatch_frac = max_mismatch_frac
        self.min_shared_sites = min_shared_sites
        self.min_cluster_reads = min_cluster_reads

    def partition(
        self, dataset: Dataset, workdir: Path, threads: int
    ) -> dict[tuple[str, str], str]:
        assignments: dict[tuple[str, str], str] = {}
        for sample in dataset.samples:
            for contig in dataset.contigs:
                sites = load_sites(str(dataset.vcfs[sample]), contig)
                if not sites:
                    continue
                alleles = read_alleles(str(dataset.bams[sample]), contig, sites)
                profiles = {
                    read_id: {pos: allele for pos, (allele, _) in calls.items()}
                    for read_id, calls in alleles.items()
                }
                for read_id, cluster in self._cluster(profiles).items():
                    assignments[(sample, read_id)] = cluster
        return assignments

    def _cluster(self, profiles: dict[str, dict[int, str]]) -> dict[str, str]:
        remaining = dict(profiles)
        result: dict[str, str] = {}
        cluster_idx = 0

        while remaining:
            seed_id = max(remaining, key=lambda rid: len(remaining[rid]))
            seed = remaining[seed_id]
            members = [seed_id]
            for read_id, profile in remaining.items():
                if read_id == seed_id:
                    continue
                shared = seed.keys() & profile.keys()
                if len(shared) < self.min_shared_sites:
                    continue
                mismatches = sum(1 for pos in shared if seed[pos] != profile[pos])
                if mismatches / len(shared) <= self.max_mismatch_frac:
                    members.append(read_id)

            label = (
                f"cluster_{cluster_idx}"
                if len(members) >= self.min_cluster_reads
                else UNASSIGNED
            )
            for read_id in members:
                result[read_id] = label
                remaining.pop(read_id, None)
            if label != UNASSIGNED:
                cluster_idx += 1
            elif len(members) == 1 and not remaining:
                break

        return result
