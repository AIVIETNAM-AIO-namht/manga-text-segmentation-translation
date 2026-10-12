"""Build the ``classical_baseline`` admitted view Spec 003's intake expects.

Spec 1 writes a *flat* tree: ``outputs/segmentation/<run>/<method>/<manga>/<stem>.png``
(+ sidecar beside it). Spec 003's contract reads masks from
``<root>/masks/<manga>/<stem>.png`` and sidecars from
``<root>/metadata/<manga>/<stem>.json`` (contracts/intake-validation.md,
"Input roots"; research R1 resolves ``classical_baseline`` to Spec 1's own
``adaptive`` output). Spec 004's intake additionally requires five fields Spec 1's
sidecar does not carry, and which are quoted verbatim from the committed
benchmark rather than recomputed (FR-054a).

So this step copies the 390 masks byte-for-byte and *transforms* the 390 Spec 1
sidecars into the richer schema — it is a relayout, not a raw copy.

Sources are read-only: the flat Spec 1 masks and sidecars are never modified,
moved or deleted (FR-013's read-only rule for Spec 1 output). This script only
adds two sibling directories under the method root.

Run:  python tools/relayout_classical_admitted_view.py
      python tools/relayout_classical_admitted_view.py --check
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Spec 1's method this identity resolves to (research R1), and the identity
#: name Spec 004 compares the sidecar's ``method`` field against.
SPEC1_METHOD = "adaptive"
IDENTITY = "classical_baseline"

#: Width, height — the aligned page size every downstream stage requires.
ALIGNED_SIZE = [1654, 1170]

MASKS = "masks"
METADATA = "metadata"


def _method_root(repo_root: Path) -> Path:
    return repo_root / "outputs" / "segmentation" / "default" / SPEC1_METHOD


def _benchmark(repo_root: Path) -> tuple[str, dict[str, str]]:
    """The page_list_identity and each page's input_image_identity, quoted verbatim."""
    page_list = json.loads(
        (repo_root / "benchmark" / "page-list.json").read_text(encoding="utf-8")
    )
    identities = {
        page["image_id"]: page["input_image_identity"]
        for page in page_list["pages"]
    }
    return page_list["page_list_identity"], identities


def _transform_sidecar(
    spec1: dict, *, image_id: str, page_identity: str, image_identity: str
) -> dict:
    """Spec 1 sidecar -> the schema Spec 003 and Spec 004 both accept.

    Everything Spec 1 recorded is kept verbatim; only the five fields the
    downstream contract requires are overridden or added. ``method`` becomes the
    identity name (Spec 004 compares it against the configured identity, and
    research R1 pins this identity to Spec 1's ``adaptive``), ``status`` becomes
    ``success`` (Spec 2's spelling), and the two identity fields are quoted from
    the committed benchmark (FR-054a).
    """
    document = dict(spec1)
    document["method"] = IDENTITY
    document["status"] = "success"
    document["output_size"] = list(ALIGNED_SIZE)
    document["page_list_identity"] = page_identity
    document["input_image_identity"] = image_identity
    document["admitted_view"] = {
        "derived_from": f"outputs/segmentation/default/{SPEC1_METHOD}/"
        f"{image_id}.json",
        "spec1_method": SPEC1_METHOD,
        "spec1_status": spec1.get("status"),
        "builder": "tools/relayout_classical_admitted_view.py",
    }
    return document


def build(repo_root: Path = REPO_ROOT) -> int:
    """Write the admitted view; return the number of pages laid out."""
    root = _method_root(repo_root)
    if not root.is_dir():
        raise SystemExit(f"Spec 1 method root is missing: {root}")
    page_identity, identities = _benchmark(repo_root)

    count = 0
    for image_id in sorted(identities):
        manga, stem = image_id.split("/", 1)
        source_mask = root / manga / f"{stem}.png"
        source_sidecar = root / manga / f"{stem}.json"
        if not source_mask.is_file() or not source_sidecar.is_file():
            raise SystemExit(
                f"Spec 1 output is incomplete for {image_id}: {source_mask}"
            )

        target_mask = root / MASKS / manga / f"{stem}.png"
        target_mask.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_mask, target_mask)
        if target_mask.read_bytes() != source_mask.read_bytes():
            raise SystemExit(f"mask copy is not byte-identical for {image_id}")

        target_sidecar = root / METADATA / manga / f"{stem}.json"
        target_sidecar.parent.mkdir(parents=True, exist_ok=True)
        document = _transform_sidecar(
            json.loads(source_sidecar.read_text(encoding="utf-8")),
            image_id=image_id,
            page_identity=page_identity,
            image_identity=identities[image_id],
        )
        target_sidecar.write_text(
            json.dumps(document, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        count += 1
    return count


def check(repo_root: Path = REPO_ROOT) -> list[str]:
    """Re-verify the laid-out view; return a list of problems (empty is good).

    Re-runs Spec 003's own intake on every page, so a pass means the view is
    accepted by the code that will consume it — not merely that files exist.
    """
    from manga_text_seg.intake import intake_mask
    from manga_text_seg.manifest import load_manifest

    root = _method_root(repo_root)
    page_identity, identities = _benchmark(repo_root)
    manifest = load_manifest(repo_root / "outputs" / "segmentation" / "default"
                             / "manifest.json")
    problems: list[str] = []

    for image_id in sorted(identities):
        manga, stem = image_id.split("/", 1)
        mask_path = root / MASKS / manga / f"{stem}.png"
        sidecar_path = root / METADATA / manga / f"{stem}.json"
        flat_mask = root / manga / f"{stem}.png"

        if not mask_path.is_file():
            problems.append(f"{image_id}: no admitted mask")
            continue
        if mask_path.read_bytes() != flat_mask.read_bytes():
            problems.append(f"{image_id}: admitted mask differs from its source")

        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        for field, expected in (
            ("image_id", image_id),
            ("method", IDENTITY),
            ("status", "success"),
            ("output_size", ALIGNED_SIZE),
            ("page_list_identity", page_identity),
            ("input_image_identity", identities[image_id]),
        ):
            if sidecar.get(field) != expected:
                problems.append(
                    f"{image_id}: {field}={sidecar.get(field)!r}, expected {expected!r}"
                )

        try:
            intake_mask(root, manifest, manga, stem)
        except Exception as error:  # MaskRejection and friends
            problems.append(f"{image_id}: Spec 003 intake refused it: {error}")

    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the existing view instead of building it",
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)

    if not args.check:
        count = build(args.repo_root)
        print(f"Laid out {count} page(s) under {_method_root(args.repo_root)}")

    problems = check(args.repo_root)
    if problems:
        print(f"{len(problems)} problem(s):", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print(
        "Admitted view verified: masks byte-identical, sidecars accepted by "
        "Spec 003 intake."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
