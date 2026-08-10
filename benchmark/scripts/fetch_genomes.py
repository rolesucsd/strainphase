#!/usr/bin/env python3
"""Download the reference genomes used by the real-genome benchmark tier.

The synthetic tiers need no downloads: they generate their own reference, which
makes them fully reproducible offline. Random sequence is fine for the phasing
question - the methods operate on variant sites from a VCF, not on genome
composition - but it removes repeats, and repeats are where real phasing gets
hard. The real-genome tier exists to put them back.

Checksums
---------
``data/genomes.tsv`` carries a ``sha256`` column. Where it reads ``TBD``, the
checksum has not been recorded yet: run with ``--record`` once, then commit the
updated file. From then on every download is verified against it and a silently
changed upstream assembly becomes an error rather than a different answer.

Verification only means something if the recorded checksum came from a trusted
download, so record it once, deliberately, and review the diff.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import urllib.request
from pathlib import Path

DEFAULT_MANIFEST = Path(__file__).resolve().parent.parent / "data" / "genomes.tsv"
UNRECORDED = "TBD"


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while data := handle.read(chunk):
            digest.update(data)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"  fetching {url}")
    with urllib.request.urlopen(url, timeout=300) as response, open(partial, "wb") as out:
        while data := response.read(1 << 20):
            out.write(data)
    partial.replace(destination)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--outdir", default="genomes", help="Where to place the FASTAs")
    parser.add_argument(
        "--record",
        action="store_true",
        help="Write observed checksums back into the manifest for TBD entries",
    )
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    outdir = Path(args.outdir)
    rows = list(csv.DictReader(manifest_path.open(), delimiter="\t"))

    failures: list[str] = []
    changed = False

    for row in rows:
        if row.get("accession", "").startswith("#"):
            continue
        destination = outdir / row["filename"]
        if not destination.exists():
            try:
                download(row["url"], destination)
            except Exception as exc:  # noqa: BLE001 - report and continue
                failures.append(f"{row['accession']}: download failed: {exc}")
                continue
        else:
            print(f"  have {destination}")

        observed = sha256(destination)
        expected = row.get("sha256", UNRECORDED).strip()
        if expected == UNRECORDED:
            if args.record:
                row["sha256"] = observed
                changed = True
                print(f"    recorded sha256 {observed[:16]}...")
            else:
                print(
                    f"    sha256 {observed[:16]}... not yet recorded "
                    f"(rerun with --record to pin it)"
                )
        elif observed != expected:
            failures.append(
                f"{row['accession']}: checksum mismatch\n"
                f"    expected {expected}\n    observed {observed}"
            )
        else:
            print("    checksum ok")

    if changed:
        with manifest_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(rows[0].keys()), delimiter="\t"
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nUpdated {manifest_path}. Commit it so others verify against it.")

    if failures:
        print("\nFAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(f"\nGenomes ready in {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
