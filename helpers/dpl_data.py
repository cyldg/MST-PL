"""Prepare source-only, unlabeled weak/strong CLIP feature packs for DPL.

Only this preparation process reads the old labeled membership manifest. All GT
and source paths are written beneath evaluation_only, never into training input.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import random

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

CLIP_SHA256 = "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
DEFAULT_CLIP = str(Path.home() / ".cache" / "clip" / "ViT-L-14.pt")
SCHEMA = "dpl-unlabeled-feature-pack-v1"
FEATURE_WIDTH, FEATURE_BATCH_SIZE, SMOKE_CAP = 768, 16, 32
DOMAINS = {"PACS": ["art_painting", "cartoon", "photo", "sketch"],
           "VLCS": ["Caltech101", "LabelMe", "SUN09", "VOC2007"],
           "OfficeHome": ["Art", "Clipart", "Product", "RealWorld"],
           "DomainNet": ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]}
VLCS_PACKED = {"Caltech101": "CALTECH", "LabelMe": "LABELME", "SUN09": "SUN", "VOC2007": "PASCAL"}
CLIENT_FIELDS = {"weak", "strong", "sample_id", "client_id"}
VIEW_DEFINITION = {
    "weak": "RGB;PIL bilinear resize256x256;flip iff sha256(seed:relative_path)[0]<128;center crop224;ToTensor;Normalize(mean=.5,std=.5)",
    "strong": {"implementation": "lazy torchvision transforms, recovered read_un.py augmentation family",
               "RandomResizedCrop": {"size": 224, "scale": [0.2, 1.0], "ratio": [0.75, 4 / 3]},
               "RandomHorizontalFlip_p": 0.5,
               "ColorJitter": {"brightness": 0.5, "contrast": 0.5, "saturation": 0.5, "hue": 0.1, "apply_p": 0.8},
               "RandomGrayscale_p": 0.2, "RandomRotation_degrees": 15,
               "GaussianBlur": {"kernel_size": 23, "sigma": [0.1, 2.0], "apply_p": 0.5},
               "Normalize": {"mean": [0.5] * 3, "std": [0.5] * 3},
               "recovered_code_difference": "Use common .5/.5 normalization, replacing recovered strong-view CLIP mean/std to match weak preprocessing."},
    "strong_seed": "first63bits sha256(seed:relative_path:strong);fixed once/image;Python/numpy/torchCPU RNG restored",
    "sample_id": "sha256(dataset + NUL + relative_path);independent of GT, seed and variant",
    "feature_storage": "frozen image encoder;L2-normalized float32 weak and strong features",
    "model_precision": "fp32;CUDA matmul TF32 and cuDNN TF32 disabled",
    "view_refresh": "one weak and one deterministic strong view per source-train image, cached once",
}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _safe_relative(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts or str(path) == ".":
        raise ValueError("Expected an image path relative to the declared data root")
    return path


def _domain_path(dataset, domain, path):
    parts = _safe_relative(path).parts
    if dataset == "PACS":
        prefixes = [("pacs", "images", domain)]
    elif dataset == "VLCS":
        prefixes = [("VLCS", domain), ("VLCS", VLCS_PACKED[domain])]
    else:
        prefixes = [(dataset, domain)]
    if not any(parts[:len(prefix)] == prefix for prefix in prefixes):
        raise ValueError("Image path does not belong to the declared source domain")


def validate_membership_manifest(value):
    """Check membership evidence without importing a supervised-only contract."""
    dataset = value["dataset"]
    if dataset not in DOMAINS or value["domain_order"] != DOMAINS[dataset]:
        raise ValueError("Unknown dataset/domain order")
    target = value["target_domain"]
    sources = [d for d in DOMAINS[dataset] if d != target]
    if target not in DOMAINS[dataset] or value["source_domains"] != sources or set(value["clients"]) != set(sources):
        raise ValueError("Invalid source/held-out domain separation")
    classes = value["classes"]
    if not classes or len(set(classes)) != len(classes) or not all(isinstance(x, str) for x in classes):
        raise ValueError("A unique public class vocabulary is required")
    seen = set()
    for domain, partition, rows in [(d, p, value["clients"][d][p]) for d in sources for p in ("train", "validation")] + [(target, "target_test", value["target_test"])]:
        if not rows:
            raise ValueError("Empty declared dataset partition")
        for row in rows:
            _domain_path(dataset, domain, row["path"])
            if row["domain"] != domain or row["path"] in seen:
                raise ValueError("Repeated identity or source/target leakage in membership manifest")
            if type(row["label"]) is not int or not 0 <= row["label"] < len(classes):
                raise ValueError("Invalid GT metadata in preparation-only manifest")
            if len(row.get("image_sha256", "")) != 64:
                raise ValueError("Image content SHA256 missing")
            int(row["image_sha256"], 16)
            seen.add(row["path"])
    signatures = {
        "source_train": canonical_hash({d: value["clients"][d]["train"] for d in sources}),
        "source_validation": canonical_hash({d: value["clients"][d]["validation"] for d in sources}),
        "target_test": canonical_hash(value["target_test"])}
    if signatures != value["signatures"] or value["manifest_sha256"] != canonical_hash({k: v for k, v in value.items() if k != "manifest_sha256"}):
        raise ValueError("Source membership signatures or manifest fingerprint changed")
    return value


def sample_id(dataset, relative_path):
    return hashlib.sha256((dataset + "\0" + str(relative_path)).encode()).hexdigest()


def strong_seed(seed, relative_path):
    return int.from_bytes(hashlib.sha256(f"{seed}:{relative_path}:strong".encode()).digest()[:8], "big") >> 1


def selected_source_records(manifest, domain, smoke=False):
    """Labels are not consulted by identity construction or smoke selection."""
    rows = manifest["clients"][domain]["train"]
    selected = [(sample_id(manifest["dataset"], row["path"]), index, row) for index, row in enumerate(rows)]
    if smoke:
        selected = sorted(selected, key=lambda item: item[0])[:SMOKE_CAP]
    return selected


def weak_view(image, relative_path, seed):
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    image = image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
    if hashlib.sha256(f"{seed}:{relative_path}".encode()).digest()[0] < 128:
        image = ImageOps.mirror(image)
    image = image.crop((16, 16, 240, 240))
    values = np.asarray(image, dtype=np.float32).copy().transpose(2, 0, 1)
    return torch.from_numpy(values).div_(127.5).sub_(1.)


@contextmanager
def isolated_cpu_rng(seed):
    import numpy as np
    import torch
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed % (1 << 32))
        # Seed only a local CPU generator; torch.manual_seed would also change
        # CUDA streams. No CUDA RNG state is touched by image augmentation.
        torch.random.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)


def build_strong_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=.5),
        transforms.RandomApply([transforms.ColorJitter(brightness=.5, contrast=.5, saturation=.5, hue=.1)], p=.8),
        transforms.RandomGrayscale(p=.2), transforms.RandomRotation(degrees=15),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(.1, 2.0))], p=.5),
        transforms.ToTensor(), transforms.Normalize(mean=[.5] * 3, std=[.5] * 3)])


def strong_view(image, relative_path, seed, transform=None):
    transform = transform if transform is not None else build_strong_transform()
    with isolated_cpu_rng(strong_seed(seed, relative_path)):
        return transform(image.convert("RGB").copy())


class SourceViews:
    """Preparation-only Dataset; stored records intentionally contain no GT."""
    def __init__(self, data_root, dataset, records, seed):
        self.root, self.dataset, self.records, self.seed = Path(data_root).resolve(), dataset, records, int(seed)
        self.strong_transform = None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from PIL import Image
        row = self.records[index]
        path = (self.root / _safe_relative(row["path"])).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Source image symlink escapes data root")
        _domain_path(self.dataset, row["client_id"], path.relative_to(self.root))
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["image_sha256"]:
            raise ValueError("Source image content changed since membership freeze")
        with Image.open(io.BytesIO(raw)) as opened:
            image = opened.convert("RGB")
        if self.strong_transform is None:
            self.strong_transform = build_strong_transform()
        weak = weak_view(image, row["path"], self.seed)
        strong = strong_view(image, row["path"], self.seed, self.strong_transform)
        return weak, strong, row["sample_id"]


def load_verified_clip(checkpoint=None, device="cpu"):
    import clip
    path = Path(checkpoint or DEFAULT_CLIP).expanduser().resolve()
    if not path.is_file() or file_sha256(path) != CLIP_SHA256:
        raise ValueError("An existing exact official L14 checkpoint is required; no downloads")
    model, _ = clip.load(str(path), device="cpu", jit=False)
    model = model.float().eval().requires_grad_(False)
    signature = (int(model.ln_final.weight.numel()), int(model.visual.output_dim),
                 len(model.transformer.resblocks), len(model.visual.transformer.resblocks),
                 int(model.visual.conv1.weight.shape[0]))
    if signature != (768, 768, 12, 24, 1024):
        raise ValueError("Unexpected CLIP L14 dimensions")
    return model.to(device), CLIP_SHA256


def _save_npz(path, **values):
    import numpy as np
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)


def prepare(manifest_path, output, device="cpu", clip_checkpoint=None, workers=0, smoke=False, model_loader=None):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    if workers < 0:
        raise ValueError("workers must be nonnegative")
    if model_loader is not None and not smoke:
        raise ValueError("Injected/fake encoders may only prepare explicit diagnostic smoke packs")
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    input_hash = file_sha256(manifest_path)
    manifest = validate_membership_manifest(json.loads(manifest_path.read_text()))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Feature preparation requires a new empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    (output / "clients").mkdir()
    private = output / "evaluation_only"
    private.mkdir(mode=0o700)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    loader_fn = model_loader or load_verified_clip
    with isolated_cpu_rng(int(manifest["seed"])):
        model, weight_sha = loader_fn(clip_checkpoint, device)
    model = model.to(device).float().eval().requires_grad_(False)
    if model_loader is None and weight_sha != CLIP_SHA256:
        raise ValueError("Production feature encoder identity is not verified official L14")
    training = {"schema_version": SCHEMA, "dataset": manifest["dataset"], "classes": list(manifest["classes"]),
                "source_domains": list(manifest["source_domains"]), "heldout_domain": manifest["target_domain"],
                "seed": int(manifest["seed"]), "smoke": bool(smoke), "feature_dimension": FEATURE_WIDTH,
                "clip_sha256": weight_sha, "encoder_verified_official": model_loader is None,
                "clients": {}, "view_definition": VIEW_DEFINITION}
    gt_ids, gt_clients, gt_labels, audit_records = [], [], [], []
    try:
        for client_number, domain in enumerate(manifest["source_domains"]):
            chosen = selected_source_records(manifest, domain, smoke)
            records = [{"sample_id": sid, "client_id": domain, "path": row["path"],
                        "image_sha256": row["image_sha256"]} for sid, _, row in chosen]
            if not records:
                raise ValueError("Empty source-train client")
            dataset = SourceViews(manifest["data_root"], manifest["dataset"], records, manifest["seed"])
            iterator = DataLoader(dataset, batch_size=FEATURE_BATCH_SIZE, shuffle=False, drop_last=False,
                                  num_workers=workers, generator=torch.Generator(device="cpu").manual_seed(int(manifest["seed"])))
            weak_features, strong_features, ordered_ids = [], [], []
            with torch.inference_mode():
                for weak, strong, sample_ids in iterator:
                    encoded = []
                    for images in (weak, strong):
                        features = model.encode_image(images.to(device=device, dtype=torch.float32)).float()
                        norms = features.norm(dim=-1, keepdim=True)
                        if tuple(features.shape) != (len(images), FEATURE_WIDTH) or not torch.isfinite(features).all() or not torch.isfinite(norms).all() or bool((norms <= 0).any()):
                            raise ValueError("Invalid frozen image-encoder feature output")
                        encoded.append((features / norms).cpu().numpy().astype(np.float32, copy=False))
                    weak_features.append(encoded[0])
                    strong_features.append(encoded[1])
                    ordered_ids.extend(str(value) for value in sample_ids)
            expected_ids = [row["sample_id"] for row in records]
            if ordered_ids != expected_ids:
                raise ValueError("Feature rows changed source sample order")
            cache_path = output / "clients" / f"client_{client_number:03d}.npz"
            _save_npz(cache_path, weak=np.concatenate(weak_features), strong=np.concatenate(strong_features),
                      sample_id=np.asarray(ordered_ids, dtype=str), client_id=np.full(len(ordered_ids), domain, dtype=f"U{len(domain)}"))
            training["clients"][domain] = {"client_id": domain, "cache_path": str(cache_path.relative_to(output)),
                                            "cache_sha256": file_sha256(cache_path), "sample_count": len(ordered_ids)}
            for sid, source_index, row in chosen:
                gt_ids.append(sid); gt_clients.append(domain); gt_labels.append(row["label"])
                weak_key = hashlib.sha256(f"{manifest['seed']}:{row['path']}".encode()).digest()
                audit_records.append({"sample_id": sid, "client_id": domain, "source_manifest_index": source_index,
                                      "relative_path": row["path"], "image_sha256": row["image_sha256"],
                                      "weak_seed": int(manifest["seed"]), "weak_flip_digest_sha256": weak_key.hex(),
                                      "weak_flip": weak_key[0] < 128, "strong_seed": strong_seed(manifest["seed"], row["path"])})
        if len(set(gt_ids)) != len(gt_ids):
            raise ValueError("Sample identity repeated across clients")
        for row in audit_records:
            path = (Path(manifest["data_root"]) / row["relative_path"]).resolve()
            root = Path(manifest["data_root"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Source image escaped the data root during preparation")
            _domain_path(manifest["dataset"], row["client_id"], path.relative_to(root))
            if file_sha256(path) != row["image_sha256"]:
                raise ValueError("A source image changed during preparation")
        if file_sha256(manifest_path) != input_hash:
            raise ValueError("Preparation membership manifest changed during execution")
        _save_npz(private / "ground_truth.npz", sample_id=np.asarray(gt_ids, dtype=str),
                  client_id=np.asarray(gt_clients, dtype=str), label=np.asarray(gt_labels, dtype=np.int64))
        (private / "ground_truth.npz").chmod(0o600)
        audit = {"schema_version": SCHEMA, "prepared_utc": datetime.now(timezone.utc).isoformat(),
                 "input_manifest_path": str(manifest_path), "input_manifest_file_sha256": input_hash,
                 "input_membership_sha256": manifest["manifest_sha256"], "data_root": manifest["data_root"],
                 "source_train_only": True, "target_images_read": 0, "source_validation_images_read": 0,
                 "smoke_selection": f"first{SMOKE_CAP} per client after sorting opaque sample_id; no class stratification" if smoke else "every source train record",
                 "GT_use": "split off in preparation only; features and training inputs contain no per-sample GT",
                 "view_definition": VIEW_DEFINITION, "records": audit_records,
                 "encoder_verified_official": model_loader is None, "clip_sha256": weight_sha,
                 "preparation_module_sha256": file_sha256(__file__)}
        write_json(private / "audit_records.json", audit)
        (private / "audit_records.json").chmod(0o600)
        training["manifest_sha256"] = canonical_hash(training)
        write_json(output / "training_manifest.json", training)
        return training
    except Exception as error:
        write_json(private / "preparation_failure.json", {"error_type": type(error).__name__, "error": str(error),
                   "training_manifest_created": (output / "training_manifest.json").exists()})
        raise


def load_training_manifest(path, allow_smoke=False):
    value = json.loads(Path(path).read_text())
    expected = {"schema_version", "dataset", "classes", "source_domains", "heldout_domain", "seed", "smoke",
                "feature_dimension", "clip_sha256", "encoder_verified_official", "clients", "view_definition", "manifest_sha256"}
    if set(value) != expected or value["schema_version"] != SCHEMA or value["feature_dimension"] != FEATURE_WIDTH:
        raise ValueError("Invalid unlabeled training manifest fields")
    if value["manifest_sha256"] != canonical_hash({k: v for k, v in value.items() if k != "manifest_sha256"}):
        raise ValueError("Unlabeled training manifest changed")
    dataset = value["dataset"]
    if (dataset not in DOMAINS or value["heldout_domain"] not in DOMAINS[dataset]
            or value["source_domains"] != [d for d in DOMAINS[dataset] if d != value["heldout_domain"]]):
        raise ValueError("Training pack has an unexpected dataset/source/held-out topology")
    classes = value["classes"]
    if (not isinstance(classes, list) or len(classes) != {"PACS": 7, "VLCS": 5, "OfficeHome": 10, "DomainNet": 10}[dataset]
            or any(not isinstance(c, str) or not c for c in classes) or len(set(classes)) != len(classes)):
        raise ValueError("Training pack has an unexpected public class vocabulary")
    if value["view_definition"] != VIEW_DEFINITION:
        raise ValueError("Training pack preprocessing differs from this fixed-view reconstruction")
    if (type(value["seed"]) is not int or value["seed"] < 0
            or type(value["smoke"]) is not bool or type(value["encoder_verified_official"]) is not bool):
        raise ValueError("Invalid run seed/mode/encoder metadata")
    if value["smoke"] and not allow_smoke:
        raise ValueError("Diagnostic smoke feature packs cannot be used for a formal run")
    if not value["encoder_verified_official"] and not (allow_smoke and value["smoke"]):
        raise ValueError("Unverified fixture encoder cannot be used for a formal run")
    if value["encoder_verified_official"] and value["clip_sha256"] != CLIP_SHA256:
        raise ValueError("Training pack has an unexpected official encoder identity")
    if set(value["clients"]) != set(value["source_domains"]) or value["heldout_domain"] in value["source_domains"]:
        raise ValueError("Invalid source-client membership in training manifest")
    return value


def load_training_pack(path, allow_smoke=False):
    """The training-side loader opens only this manifest and listed client NPZs."""
    import numpy as np
    path = Path(path).resolve()
    manifest = load_training_manifest(path, allow_smoke)
    clients, identities = {}, set()
    for domain in manifest["source_domains"]:
        entry = manifest["clients"][domain]
        if set(entry) != {"client_id", "cache_path", "cache_sha256", "sample_count"} or entry["client_id"] != domain:
            raise ValueError("Invalid client cache entry")
        relative = _safe_relative(entry["cache_path"])
        cache = (path.parent / relative).resolve()
        if not cache.is_relative_to(path.parent) or relative.parts[0] != "clients":
            raise ValueError("Training cache must stay inside the clients directory")
        raw = cache.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["cache_sha256"]:
            raise ValueError("Training feature-cache hash changed")
        with np.load(io.BytesIO(raw), allow_pickle=False) as saved:
            if set(saved.files) != CLIENT_FIELDS:
                raise ValueError("Training NPZ has forbidden or missing fields")
            client = {key: saved[key] for key in saved.files}
        n = entry["sample_count"]
        for key in ("weak", "strong"):
            x = client[key]
            if x.dtype != np.float32 or x.shape != (n, FEATURE_WIDTH) or not np.isfinite(x).all() or not np.allclose(np.linalg.norm(x, axis=1), 1., atol=1e-4):
                raise ValueError("Invalid unlabeled feature shape/dtype/normalization")
        if client["sample_id"].dtype.kind != "U" or client["client_id"].dtype.kind != "U" or client["sample_id"].shape != (n,) or client["client_id"].shape != (n,):
            raise ValueError("Client identities must be Unicode arrays aligned with feature rows")
        if not np.all(client["client_id"] == domain) or len(set(client["sample_id"].tolist())) != n or any(s in identities for s in client["sample_id"]):
            raise ValueError("Invalid/repeated sample or client identities")
        for sid in client["sample_id"]:
            if len(sid) != 64:
                raise ValueError("Unexpected opaque sample identity")
            int(sid, 16)
        identities.update(client["sample_id"].tolist())
        clients[domain] = client
    return manifest, clients


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--clip-checkpoint")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    value = prepare(args.manifest, args.output, args.device, args.clip_checkpoint, args.workers, args.smoke)
    print(json.dumps({"schema_version": value["schema_version"], "smoke": value["smoke"],
                      "source_client_counts": {d: v["sample_count"] for d, v in value["clients"].items()},
                      "encoder_verified_official": value["encoder_verified_official"]}), flush=True)


if __name__ == "__main__":
    main()
