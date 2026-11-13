"""Create a frozen PACS membership manifest for the source-only DPL runner.

Usage (run from the repository root)::

    python helpers/dpl_manifest.py --data-root DATA --target-domain sketch \
        --output WORK/PACS_sketch.json --seed 0

Expected layout: DATA/pacs/images/DOMAIN/CLASS/IMAGE, with all four PACS
domains and all seven classes. This standard-library tool inventories and hashes
images and records class labels as preparation/evaluation metadata. The DPL
feature preparer subsequently separates source-training labels into its offline
evaluation files; the training runner does not consume this labeled manifest.
Validation and held-out images are inventoried but must not be used to train,
tune, or select a model.

This public helper uses a deterministic SHA256 ordering, NOT the historical
torch.randperm split. A newly generated manifest is therefore a new cohort; it
must not be described as reproducing the archived 4,851-sample diagnostic cohort
unless its membership is independently verified. Existing outputs are never
overwritten. No model weights or third-party Python packages are required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


DOMAINS = ("art_painting", "cartoon", "photo", "sketch")
CLASSES = ("dog", "elephant", "giraffe", "guitar", "horse", "house", "person")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".pgm", ".tif", ".tiff", ".webp"}
PROTOCOL_ID = "public-pacs-hash-split-v1"


def canonical_hash(value):
    """Match the canonical JSON fingerprint used by the DPL data preparer."""
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside_root(path, root):
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError(f"Path or symlink escapes the data root: {path}")
    return resolved


def _inventory(root):
    by_domain, seen = {}, set()
    for domain in DOMAINS:
        base = root / "pacs" / "images" / domain
        _inside_root(base, root)
        if not base.is_dir():
            raise ValueError(f"Expected a domain directory: {base}")
        folders = {entry.name for entry in base.iterdir() if entry.is_dir()}
        if folders != set(CLASSES):
            raise ValueError(f"Expected exactly the seven PACS class folders in {base}; "
                             f"missing={sorted(set(CLASSES) - folders)}, "
                             f"extra={sorted(folders - set(CLASSES))}")
        rows = []
        for label, class_name in enumerate(CLASSES):
            class_dir = base / class_name
            _inside_root(class_dir, root)
            class_rows = []
            for path in sorted(class_dir.rglob("*")):
                resolved = _inside_root(path, root)
                # pathlib does not consistently recurse through directory links
                # across Python versions, so reject them rather than omit images.
                if path.is_symlink() and path.is_dir():
                    raise ValueError(f"Symlinked image subdirectories are unsupported: {path}")
                if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                    continue
                relative = resolved.relative_to(root)
                if relative.parts[:4] != ("pacs", "images", domain, class_name):
                    raise ValueError(f"Resolved image belongs to a different domain or class: {path}")
                identity = relative.as_posix()
                if identity in seen:
                    raise ValueError(f"Repeated image identity or overlapping domain paths: {identity}")
                seen.add(identity)
                class_rows.append({"path": identity, "label": label, "domain": domain,
                                   "image_sha256": file_sha256(resolved),
                                   "size_bytes": resolved.stat().st_size})
            if not class_rows:
                raise ValueError(f"No supported images in class directory: {class_dir}")
            rows.extend(sorted(class_rows, key=lambda row: row["path"]))
        by_domain[domain] = rows
    return by_domain


def make_manifest(data_root, target_domain="sketch", seed=0,
                  validation_fraction=0.2, dataset="PACS"):
    """Inventory PACS and return a signed, preparation-only membership record."""
    if dataset != "PACS":
        raise ValueError("This public manifest helper currently supports PACS only")
    if target_domain not in DOMAINS:
        raise ValueError(f"Unknown PACS target domain: {target_domain}")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not isinstance(validation_fraction, (int, float))
            or not math.isfinite(validation_fraction)
            or not 0 < validation_fraction < 1):
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    root = Path(data_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("data_root must be a directory")
    inventory = _inventory(root)
    sources = [domain for domain in DOMAINS if domain != target_domain]
    clients = {}
    for domain in sources:
        ordered = sorted(inventory[domain], key=lambda row: (
            hashlib.sha256(f"{seed}:{row['path']}".encode()).hexdigest(), row["path"]))
        n_validation = int(validation_fraction * len(ordered))
        n_training = len(ordered) - n_validation
        if n_validation == 0 or n_training == 0:
            raise ValueError(f"Empty train/validation partition for source domain {domain}")
        train, validation = ordered[:n_training], ordered[n_training:]
        if {row["label"] for row in train} != set(range(len(CLASSES))):
            raise ValueError(f"Source training split does not cover all classes: {domain}")
        clients[domain] = {"train": train, "validation": validation}
    split_policy = {
        "algorithm": "ascending sha256(seed:relative_path), with relative_path as tie-breaker",
        "validation_fraction": validation_fraction,
        "partition": "first N-floor(fraction*N) source rows train; remaining source rows validation",
        "historical_split_equivalent": False,
        "note": "New public hash split, not the archived torch.randperm diagnostic membership.",
    }
    value = {"schema_version": 1, "protocol_id": PROTOCOL_ID,
             "protocol_sha256": canonical_hash({"dataset": dataset, "classes": CLASSES,
                                                 "domain_order": DOMAINS, "split_policy": split_policy}),
             "dataset": dataset, "data_root": str(root), "seed": seed,
             "classes": list(CLASSES), "domain_order": list(DOMAINS),
             "target_domain": target_domain, "source_domains": sources,
             "clients": clients, "target_test": inventory[target_domain],
             "inventory_sha256": canonical_hash(inventory), "split_policy": split_policy}
    value["signatures"] = {
        "source_train": canonical_hash({domain: clients[domain]["train"] for domain in sources}),
        "source_validation": canonical_hash({domain: clients[domain]["validation"] for domain in sources}),
        "target_test": canonical_hash(value["target_test"]),
    }
    value["manifest_sha256"] = canonical_hash(value)
    return value


def write_manifest(manifest, output):
    """Write a new manifest, refusing to replace an existing file or symlink."""
    path = Path(output).expanduser()
    serialized = json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=("PACS",), default="PACS")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--target-domain", choices=DOMAINS, default="sketch")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args(argv)
    try:
        manifest = make_manifest(args.data_root, args.target_domain, args.seed,
                                 args.validation_fraction, args.dataset)
        output = write_manifest(manifest, args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({"output": str(output), "protocol_id": PROTOCOL_ID,
                      "target_domain": args.target_domain,
                      "source_counts": {domain: {part: len(rows) for part, rows in splits.items()}
                                        for domain, splits in manifest["clients"].items()},
                      "heldout_count": len(manifest["target_test"]),
                      "manifest_sha256": manifest["manifest_sha256"],
                      "historical_split_equivalent": False}, indent=2))


if __name__ == "__main__":
    main()
