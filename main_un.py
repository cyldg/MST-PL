"""Unlabeled source-only DPL reconstruction; sealed labels are evaluated separately."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from FASTPT_un import DPLConfig, FederatedDPL, LABEL
from helpers.dpl_data import CLIP_SHA256, load_training_pack, load_verified_clip
from helpers.dpl_templates import source_texts
from prompt_CLIP_un import FrozenMSTTextCLIP
from helpers.dpl_trace import TraceReader, TraceRecorder, canonical_hash, file_hash, write_json


ROOT = Path(__file__).resolve().parent
POLICIES = ("full", "initial_fixed", "expansion_only")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def code_binding():
    paths = ("FASTPT_un.py", "main_un.py", "helpers/dpl_data.py",
             "helpers/dpl_trace.py", "helpers/dpl_templates.py",
             "prompt_CLIP_un.py", "evaluate_dpl.py", "configs/dpl_protocol.json")
    return {name: file_hash(ROOT / name) for name in sorted(paths)}


def tensor_tree_hash(value):
    """Content identity, independent of torch.save container timestamps/storage IDs."""
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            x = item.detach().cpu().contiguous()
            digest.update(str((str(x.dtype), tuple(x.shape))).encode())
            digest.update(x.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item):
                digest.update(str(key).encode()); visit(item[key])
        elif isinstance(item, (list, tuple)):
            for entry in item:
                visit(entry)
        else:
            digest.update(json.dumps(item, sort_keys=True, allow_nan=False).encode())
    visit(value)
    return digest.hexdigest()


def initialization_hash(checkpoint):
    return tensor_tree_hash({key: checkpoint[key] for key in
                             ("global_prompt", "domain_prompts", "domain_classifier",
                              "states", "initial_states", "MST_transforms", "feature_manifest")})


class Observer:
    def __init__(self, recorder=None):
        self.recorder = recorder
        self.mst_exposures = []

    def __call__(self, event, client, round, stage, **payload):
        if event == "usage" and stage == "mst_alignment":
            self.mst_exposures.append({"client": client, "round": round,
                                       **{key: value.tolist() for key, value in payload.items()}})
            return
        if self.recorder is None:
            return
        if event == "state":
            mask = payload["selected"].numpy()
            labels = payload["labels"].numpy().copy()
            labels[~mask] = -1
            self.recorder.write_state(client, round, stage, mask, labels,
                                      payload["candidate"].numpy(), payload["confidence"].numpy(),
                                      None if payload["view1_label"] is None else payload["view0_label"].numpy(),
                                      None if payload["view1_label"] is None else payload["view1_label"].numpy())
        elif event == "probe":
            self.recorder.write_probe(client, round, **{key: value.numpy() for key, value in payload.items()})
        elif event == "usage":
            mapped = {"global_class_ce": "global", "domain_uce": "domain"}[stage]
            self.recorder.write_usage(client, round, mapped,
                                      payload["usage_counts"].numpy(), payload["pseudo_labels_used"].numpy())
        else:
            raise ValueError("Unknown core observer event")


def run_policies(manifest, clients, bridge, domain_text, handcrafted, config, output,
                 device="cpu", implementation_verified=False):
    """Shared r0 then independent policies. This function has no ground-truth input."""
    started = time.monotonic()
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("A new output directory is required; traces cannot be overwritten/resumed")
    output.mkdir(parents=True, exist_ok=True)
    binding = code_binding()
    metadata = {"implementation_label": LABEL, "paper_fidelity_verified": False,
                "real_clip_verified": bool(implementation_verified),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "dataset": manifest["dataset"], "heldout_domain": manifest["heldout_domain"],
                "source_domains": manifest["source_domains"], "classes": manifest["classes"],
                "num_classes": len(manifest["classes"]), "seed": int(manifest["seed"]),
                "smoke": bool(manifest["smoke"]), "rounds": config.rounds,
                "training_manifest_sha256": manifest["manifest_sha256"],
                "code_sha256": binding, "resolved_config": asdict(config),
                "target_images_read": 0, "selection": "fixed_final_no_accuracy_selection",
                "scope": "source_pseudo_label_diagnostics;no_target_accuracy_evaluation",
                "results_written_to_manuscript": False}
    write_json(output / "run_metadata.json", metadata)
    init_observer = Observer()
    base = FederatedDPL(bridge, clients, domain_text, handcrafted, config,
                        seed=manifest["seed"], device=device, observer=init_observer)
    counts = base.initialize_pseudo_labels()
    print(json.dumps({"event": "PLIF", "selected": counts}), flush=True)
    base.prepare_style()
    shared = base.checkpoint()
    shared_identity = initialization_hash(shared)
    torch.save(shared, output / "shared_r0.pt")
    write_json(output / "MST_alignment_usage.json", init_observer.mst_exposures)
    write_json(output / "MST_events.json", base.mst_events)
    write_json(output / "initialization.json", {"selected": counts,
               "shared_r0_file_sha256": file_hash(output / "shared_r0.pt"),
               "initialization_content_sha256": shared_identity,
               "MST_seconds_including_PLIF": time.monotonic() - started})
    del base
    final_locks = {}
    for policy in POLICIES:
        policy_started = time.monotonic()
        destination = output / policy
        destination.mkdir()
        recorder = TraceRecorder(destination / "trace", {s: list(clients[s]["sample_ids"]) for s in clients},
                                 {**metadata, "variant": policy, "shared_initialization_sha256": shared_identity})
        engine = FederatedDPL(bridge, clients, domain_text, handcrafted, replace(config, variant=policy),
                              seed=manifest["seed"], device=device, observer=Observer(recorder))
        engine.load_checkpoint(shared)
        if initialization_hash(engine.checkpoint()) != shared_identity:
            raise ValueError("Policy initialization or MST differs from the shared r0")
        engine.emit_initial_observations()
        with (destination / "rounds.jsonl").open("w", buffering=1) as stream:
            while engine.complete_rounds < config.rounds:
                event = engine.run_round()
                event["elapsed_seconds"] = time.monotonic() - policy_started
                stream.write(json.dumps(event, allow_nan=False) + "\n")
                checkpoint = engine.checkpoint()
                # Q is immutable and is stored once in shared_r0 and in final.pt.
                snapshot = {k: v for k, v in checkpoint.items() if k != "MST_transforms"}
                snapshot["MST_reference_sha256"] = shared_identity
                torch.save(snapshot, destination / f"round_{engine.complete_rounds:02d}_state.pt")
                print(json.dumps({"event": "round_complete", "variant": policy,
                                  "round": engine.complete_rounds, "selected": event["selected"],
                                  "elapsed_seconds": event["elapsed_seconds"]}), flush=True)
        reader = TraceReader(destination / "trace")
        missing = reader.missing_events()
        if missing:
            raise RuntimeError(f"Incomplete trace: {missing}")
        reader.verify_unchanged()
        final = engine.checkpoint()
        torch.save(final, destination / "final.pt")
        reloaded = torch.load(destination / "final.pt", map_location="cpu", weights_only=True)
        if tensor_tree_hash(reloaded) != tensor_tree_hash(final):
            raise ValueError("Final checkpoint roundtrip changed content")
        budget = {"rounds": engine.complete_rounds, "global_epochs": config.global_epochs,
                  "domain_epochs": config.domain_epochs, "MST_epochs": config.mst_epochs,
                  "max_batches": config.max_batches, "optimizer_updates": dict(engine.updates),
                  "samples_seen": dict(engine.samples_seen), "MST_transforms": len(engine.transforms),
                  "seconds": time.monotonic() - policy_started,
                  "class_CE_exposure": "actual per-ID counts in trace; budgets do not imply equal selected sizes"}
        write_json(destination / "budget.json", budget)
        lock = {"variant": policy, "complete": True, "rounds": engine.complete_rounds,
                "final_file_sha256": file_hash(destination / "final.pt"),
                "final_content_sha256": tensor_tree_hash(final),
                "trace_index_sha256": file_hash(destination / "trace" / "index.json"),
                "shared_initialization_sha256": shared_identity,
                "budget_sha256": file_hash(destination / "budget.json"),
                "training_manifest_sha256": manifest["manifest_sha256"],
                "selection": "fixed_final", "offline_label_evaluation_performed": False}
        write_json(destination / "final_lock.json", lock)
        final_locks[policy] = lock
        del engine, final, reloaded, checkpoint, snapshot
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if code_binding() != binding:
        raise ValueError("Implementation changed while policies were running")
    completion = {"complete": True, "smoke": manifest["smoke"],
                  "real_clip_verified": bool(implementation_verified), "policies": final_locks,
                  "training_manifest_sha256": manifest["manifest_sha256"],
                  "total_seconds": time.monotonic() - started,
                  "offline_label_evaluation_performed": False, "results_written_to_manuscript": False}
    write_json(output / "training_complete.json", completion)
    return completion


def resolved_config(smoke=False):
    protocol = json.loads((ROOT / "configs" / "dpl_protocol.json").read_text())
    required = {"protocol_id": "mstpl-dpl-unlabeled-paper-v1", "clip_sha256": CLIP_SHA256,
                "rounds": 20, "global_epochs": 5, "domain_epochs": 5, "mst_epochs": 50,
                "batch_size": 16, "prompt_tokens_each": 4, "alpha_UCE": .5, "beta_CV": .5,
                "thresholds": {"initial": .9, "purification": .88, "expansion": .92},
                "paper_fidelity_verified": False}
    if any(protocol.get(key) != value for key, value in required.items()):
        raise ValueError("DPL protocol changed; review the reconstruction contract before running")
    return DPLConfig(rounds=1 if smoke else 20, global_epochs=1 if smoke else 5,
                     domain_epochs=1 if smoke else 5, mst_epochs=1 if smoke else 50,
                     max_batches=1 if smoke else 0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clip-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.threads < 1:
        raise ValueError("CPU thread count must be positive")
    torch.set_num_threads(args.threads)
    manifest, packed = load_training_pack(args.training_manifest, allow_smoke=args.smoke)
    if bool(manifest["smoke"]) != args.smoke or manifest["encoder_verified_official"] is not True:
        raise ValueError("Run mode must match a real-encoder feature pack")
    config = resolved_config(args.smoke)
    seed_all(manifest["seed"])
    model, sha = load_verified_clip(args.clip_checkpoint, args.device)
    if sha != manifest["clip_sha256"]:
        raise ValueError("Training backbone differs from the feature pack")
    bridge = FrozenMSTTextCLIP(model, manifest["classes"]).to(args.device)
    texts, captions = source_texts({**manifest, "target_domain": manifest["heldout_domain"]})
    domain_text = {s: bridge.handcrafted_features(texts[s]) for s in manifest["source_domains"]}
    handcrafted = {s: bridge.handcrafted_features([captions[s]])[0] for s in manifest["source_domains"]}
    clients = {s: {"sample_ids": packed[s]["sample_id"].tolist(),
                   "weak": torch.from_numpy(packed[s]["weak"]),
                   "strong": torch.from_numpy(packed[s]["strong"])} for s in manifest["source_domains"]}
    before = file_hash(args.training_manifest)
    result = run_policies(manifest, clients, bridge, domain_text, handcrafted, config, args.output,
                          device=args.device, implementation_verified=True)
    load_training_pack(args.training_manifest, allow_smoke=args.smoke)
    if file_hash(args.training_manifest) != before:
        raise ValueError("Training feature pack changed during execution")
    print(json.dumps({"event": "all_policies_complete", "smoke": args.smoke,
                      "seconds": result["total_seconds"]}), flush=True)
    return result


if __name__ == "__main__":
    main()
