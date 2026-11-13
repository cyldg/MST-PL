"""Ground-truth-free, single-writer traces of DPL state and CE usage.

The recorder accepts predictions, selection masks and actually used pseudo
labels only. Ground truth belongs exclusively to evaluate_trace.py. There is no
resume/overwrite API: a run starts in a new, empty trace directory.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 1
STATE_STAGES = ("plif", "global_before_plp", "plp", "ple")
USAGE_STAGES = ("global", "domain")
MAX_ROUND = 20


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _json_metadata(metadata):
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON dictionary")
    try:
        copied = json.loads(json.dumps(metadata, allow_nan=False))
    except (ValueError, TypeError) as error:
        raise ValueError("metadata must contain finite JSON values, not sample arrays") from error
    def check(value):
        if isinstance(value, dict):
            for key, item in value.items():
                normalized = key.lower().replace("-", "_").replace(" ", "_")
                ground_truth_key = (normalized in {"gt", "gold", "gold_label", "gold_labels", "true_label", "true_labels"}
                                    or normalized.startswith(("ground_truth", "groundtruth", "gt_")))
                if ground_truth_key and item is not False and item is not None:
                    raise ValueError("Ground-truth values/paths do not belong in a training trace")
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)
    check(copied)
    return copied


def _rounds(metadata):
    value = metadata.get("rounds", metadata.get("total_rounds", MAX_ROUND))
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_ROUND:
        raise ValueError("metadata rounds must be an integer in[1,20]")
    if "rounds" in metadata and "total_rounds" in metadata and metadata["rounds"] != metadata["total_rounds"]:
        raise ValueError("Conflicting round metadata")
    return value


def _num_classes(metadata):
    values = [metadata[key] for key in ("num_classes", "class_count") if key in metadata]
    if "classes" in metadata:
        if not isinstance(metadata["classes"], list) or not metadata["classes"]:
            raise ValueError("metadata classes must be a nonempty class vocabulary")
        values.append(len(metadata["classes"]))
    if not values:
        return None
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in values) or len(set(values)) != 1:
        raise ValueError("Invalid/conflicting class-count metadata")
    return values[0]


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in[{low},{high}]")
    return int(value)


def int_vector(value, count, name, minimum=-1, classes=None):
    array = np.asarray(value)
    if array.ndim != 1 or len(array) != count or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a length-{count} integer vector")
    converted = array.astype(np.int64)
    if not np.array_equal(array, converted) or (converted < minimum).any():
        raise ValueError(f"{name} contains invalid/out-of-range integers")
    if classes is not None and (converted >= classes).any():
        raise ValueError(f"{name} exceeds the recorded class vocabulary")
    return converted.copy()


def confidence_vector(value, count, name):
    try:
        array = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{name} must be a numeric confidence vector") from error
    if array.ndim != 1 or len(array) != count or not np.isfinite(array).all() or ((array < 0) | (array > 1)).any():
        raise ValueError(f"{name} must contain {count} finite probabilities in[0,1]")
    return array.copy()


def mask_vector(value, count):
    array = np.asarray(value)
    if array.ndim != 1 or len(array) != count or array.dtype.kind not in "biu" or not np.isin(array, [0, 1]).all():
        raise ValueError("selected must be a full-cohort Boolean/0-1 vector")
    return array.astype(bool, copy=True)


def expected_state(last, rounds):
    if last is None:
        return 0, "plif"
    r, stage = last
    if stage in ("plif", "ple"):
        if stage == "ple" and r == rounds:
            return None
        return r + 1, "global_before_plp"
    return (r, "plp") if stage == "global_before_plp" else (r, "ple")


def _state_payload(payload, count, classes, stage, previous):
    required = {"selected", "pseudo_label", "candidate_label", "confidence"}
    optional = {"view0_label", "view1_label"}
    if set(payload) - required - optional or not required.issubset(payload):
        raise ValueError("Unexpected/missing state arrays; GT fields are never accepted")
    selected = mask_vector(payload["selected"], count)
    pseudo = int_vector(payload["pseudo_label"], count, "pseudo_label", classes=classes)
    candidate = int_vector(payload["candidate_label"], count, "candidate_label", minimum=0, classes=classes)
    confidence = confidence_vector(payload["confidence"], count, "confidence")
    if (pseudo[~selected] != -1).any() or (pseudo[selected] < 0).any():
        raise ValueError("Inactive pseudo labels must be-1; active labels must be valid classes")
    has_views = ("view0_label" in payload, "view1_label" in payload)
    if has_views[0] != has_views[1]:
        raise ValueError("Both view labels or neither must be supplied")
    result = {"selected": selected, "pseudo_label": pseudo, "candidate_label": candidate, "confidence": confidence}
    if has_views[0]:
        for key in optional:
            result[key] = int_vector(payload[key], count, key, minimum=0, classes=classes)
    if stage == "plif":
        if not np.array_equal(pseudo[selected], candidate[selected]):
            raise ValueError("PLIF stored labels must agree with admitted candidate labels")
    else:
        old_mask, old_label = previous["selected"], previous["pseudo_label"]
        if stage == "global_before_plp":
            if not np.array_equal(selected, old_mask) or not np.array_equal(pseudo, old_label):
                raise ValueError("Global training cannot silently update the stored label state")
        elif stage == "plp":
            if (selected & ~old_mask).any():
                raise ValueError("PLP can remove/relabel existing samples but cannot expand")
        elif stage == "ple":
            if (old_mask & ~selected).any() or not np.array_equal(pseudo[old_mask], old_label[old_mask]):
                raise ValueError("PLE cannot remove or relabel previously retained samples")
            added = selected & ~old_mask
            if added.any():
                if not has_views[0]:
                    raise ValueError("New PLE admissions require both recorded view predictions")
                if (not np.array_equal(pseudo[added], candidate[added])
                        or not np.array_equal(result["view0_label"][added], result["view1_label"][added])
                        or not np.array_equal(candidate[added], result["view0_label"][added])):
                    raise ValueError("New PLE pseudo labels must satisfy recorded candidate/view agreement")
    return result


def _probe_payload(payload, count, classes):
    if set(payload) != {"global_pred", "composite_pred", "global_confidence", "composite_confidence"}:
        raise ValueError("Unexpected/missing probe arrays")
    result = {key: int_vector(payload[key], count, key, minimum=0, classes=classes)
              for key in ("global_pred", "composite_pred")}
    result.update({key: confidence_vector(payload[key], count, key)
                   for key in ("global_confidence", "composite_confidence")})
    return result


def _usage_payload(payload, count, classes, reference):
    if set(payload) != {"usage_counts", "pseudo_labels_used"}:
        raise ValueError("Unexpected/missing usage arrays")
    counts = int_vector(payload["usage_counts"], count, "usage_counts", minimum=0)
    labels = int_vector(payload["pseudo_labels_used"], count, "pseudo_labels_used", classes=classes)
    used = counts > 0
    selected = np.asarray(reference["selected"], dtype=bool)
    if ((used & ~selected).any()
            or not np.array_equal(labels[used], reference["pseudo_label"][used])):
        raise ValueError("CE usage must use selected samples and their phase-correct stored labels")
    if np.any((~used) & (labels != -1) & (labels != reference["pseudo_label"])):
        raise ValueError("Unused labels must be-1 or the recorded phase-correct label")
    return {"usage_counts": counts, "pseudo_labels_used": labels}


def usage_reference(round_number, stage):
    if stage == "global":
        return (0, "plif") if round_number == 1 else (round_number - 1, "ple")
    return round_number, "plp"


def _safe_path(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Invalid relative trace path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Trace path escapes its directory")
    return path


def _save_npz(path, values):
    if path.exists():
        raise FileExistsError(f"Trace files cannot be overwritten: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)


def _read_npz(path):
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key].copy() for key in payload.files}


class TraceRecorder:
    def __init__(self, output, client_ids, metadata):
        self.output = Path(output).resolve()
        self._metadata = _json_metadata(metadata)
        self.rounds, self.num_classes = _rounds(self._metadata), _num_classes(self._metadata)
        if not isinstance(client_ids, dict) or not client_ids:
            raise ValueError("client_ids must be a nonempty client-to-ID mapping")
        parsed = {}
        for client, values in client_ids.items():
            if not isinstance(client, str) or not client or isinstance(values, (str, bytes)):
                raise ValueError("Client identities and sample IDs must be nonempty strings")
            ids = list(values)
            if not ids or any(not isinstance(v, str) or not v for v in ids) or len(set(ids)) != len(ids):
                raise ValueError("Every client needs a nonempty vector of unique string sample IDs")
            parsed[client] = ids
        if self.output.exists() and any(self.output.iterdir()):
            raise FileExistsError("TraceRecorder requires a new/empty independent directory")
        self.output.mkdir(parents=True, exist_ok=True)
        clients = {}
        for i, (client, ids) in enumerate(parsed.items()):
            directory = f"clients/{i:02d}_{hashlib.sha256(client.encode()).hexdigest()[:12]}"
            relative = f"{directory}/ids.npz"
            cohort_hash = canonical_hash({"client_id": client, "sample_ids": ids})
            _save_npz(self.output / relative, {"sample_id": np.asarray(ids, dtype=str),
                                               "client_id": np.repeat(np.asarray(client, dtype=str), len(ids))})
            clients[client] = {"directory": directory, "count": len(ids), "ids_file": relative,
                               "ids_file_sha256": file_hash(self.output / relative), "cohort_sha256": cohort_hash}
        self.manifest = {"schema_version": SCHEMA_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
                         "metadata": self._metadata, "rounds": self.rounds, "num_classes": self.num_classes,
                         "client_order": list(parsed), "clients": clients,
                         "identity": "ordered(client_id,sample_id); IDs may repeat across distinct clients",
                         "state_stages": list(STATE_STAGES), "usage_stages": list(USAGE_STAGES),
                         "ground_truth_interface": False,
                         "usage_definition": "Actual hard-pseudo-label class-CE loss-term exposures per original sample ID; include repeated views/latent augmentations, exclude domain-classifier CE and soft CV loss.",
                         "recorder_scope": "single writer; no resume; no threshold/policy decisions; GT evaluation happens only offline"}
        write_json(self.output / "manifest.json", self.manifest)
        self._manifest_hash = file_hash(self.output / "manifest.json")
        self._events, self._states, self._probes, self._usages, self._last = [], {}, {}, {}, {}
        self._index_hash = None
        self._write_index()

    def _write_index(self):
        data = {"schema_version": SCHEMA_VERSION, "manifest_sha256": self._manifest_hash,
                "event_count": len(self._events), "events": self._events,
                "last_event_sha256": self._events[-1]["event_sha256"] if self._events else self._manifest_hash}
        data["index_sha256"] = canonical_hash(data)
        write_json(self.output / "index.json", data)
        self._index_hash = file_hash(self.output / "index.json")

    def _check(self, client):
        if client not in self.manifest["clients"]:
            raise ValueError("Unknown client identity")
        if file_hash(self.output / "manifest.json") != self._manifest_hash or file_hash(self.output / "index.json") != self._index_hash:
            raise ValueError("Trace metadata/index changed outside its recorder")
        info = self.manifest["clients"][client]
        if file_hash(self.output / info["ids_file"]) != info["ids_file_sha256"]:
            raise ValueError("Client cohort IDs changed")
        return info

    def _state(self, client, round_number, stage):
        event = self._states.get((client, round_number, stage))
        if event is None:
            raise ValueError(f"Required reference state missing: {(client, round_number, stage)}")
        path = self.output / event["path"]
        if file_hash(path) != event["file_sha256"]:
            raise ValueError("A prior trace state was modified")
        values = _read_npz(path)
        return {key: value for key, value in values.items() if key not in {"kind", "client_id", "round", "stage", "cohort_sha256"}}

    def _commit(self, kind, client, round_number, stage, payload):
        info = self.manifest["clients"][client]
        relative = f"{info['directory']}/{kind}/r{round_number:02d}_{stage}.npz"
        header = {"kind": np.asarray(kind), "client_id": np.asarray(client), "round": np.asarray(round_number, dtype=np.int64),
                  "stage": np.asarray(stage), "cohort_sha256": np.asarray(info["cohort_sha256"])}
        _save_npz(self.output / relative, dict(header, **payload))
        event = {"event_id": len(self._events), "kind": kind, "client_id": client, "round": round_number,
                 "stage": stage, "path": relative, "file_sha256": file_hash(self.output / relative),
                 "cohort_sha256": info["cohort_sha256"], "created_utc": datetime.now(timezone.utc).isoformat(),
                 "prev_event_sha256": self._events[-1]["event_sha256"] if self._events else self._manifest_hash}
        event["event_sha256"] = canonical_hash(event)
        self._events.append(event)
        self._write_index()
        return event

    def write_state(self, client, round, stage, selected, pseudo_label, candidate_label, confidence,
                    view0_label=None, view1_label=None):
        info = self._check(client)
        round = _integer(round, "round", 0, self.rounds)
        if stage not in STATE_STAGES or (round, stage) != expected_state(self._last.get(client), self.rounds):
            raise ValueError("State stages are duplicated, skipped or out of order")
        previous_key = self._last.get(client)
        previous = self._state(client, *previous_key) if previous_key is not None else None
        payload = {"selected": selected, "pseudo_label": pseudo_label, "candidate_label": candidate_label, "confidence": confidence}
        if view0_label is not None:
            payload["view0_label"] = view0_label
        if view1_label is not None:
            payload["view1_label"] = view1_label
        payload = _state_payload(payload, info["count"], self.num_classes, stage, previous)
        event = self._commit("state", client, round, stage, payload)
        self._states[(client, round, stage)] = event
        self._last[client] = (round, stage)

    def write_probe(self, client, round, global_pred, composite_pred, global_confidence, composite_confidence):
        info = self._check(client)
        round = _integer(round, "round", 0, self.rounds)
        if (client, round) in self._probes:
            raise ValueError("Duplicate client/round probe")
        required_state = (round, "plif" if round == 0 else "ple")
        if self._last.get(client) != required_state:
            raise ValueError("Probe must be written at its client's current completed round boundary")
        self._state(client, *required_state)
        payload = _probe_payload({"global_pred": global_pred, "composite_pred": composite_pred,
                                  "global_confidence": global_confidence, "composite_confidence": composite_confidence},
                                 info["count"], self.num_classes)
        self._probes[(client, round)] = self._commit("probe", client, round, "fixed_view_end_round", payload)

    def write_usage(self, client, round, stage, usage_counts, pseudo_labels_used):
        info = self._check(client)
        round = _integer(round, "round", 1, self.rounds)
        if stage not in USAGE_STAGES or (client, round, stage) in self._usages:
            raise ValueError("Invalid or duplicated CE usage stage")
        reference = self._state(client, *usage_reference(round, stage))
        payload = _usage_payload({"usage_counts": usage_counts, "pseudo_labels_used": pseudo_labels_used},
                                 info["count"], self.num_classes, reference)
        self._usages[(client, round, stage)] = self._commit("usage", client, round, stage, payload)


class TraceReader:
    """Read-only integrity/shape/phase verifier, with no GT input."""
    def __init__(self, trace):
        self.root = Path(trace).resolve()
        self.manifest_hash = file_hash(self.root / "manifest.json")
        self.index_hash = file_hash(self.root / "index.json")
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        index = json.loads((self.root / "index.json").read_text())
        if self.manifest["schema_version"] != SCHEMA_VERSION or index["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Unsupported trace schema")
        if index["manifest_sha256"] != self.manifest_hash or index["index_sha256"] != canonical_hash({k: v for k, v in index.items() if k != "index_sha256"}):
            raise ValueError("Trace manifest/index integrity failed")
        self.rounds = _rounds(self.manifest["metadata"])
        self.num_classes = _num_classes(self.manifest["metadata"])
        _json_metadata(self.manifest["metadata"])
        if (self.manifest["rounds"] != self.rounds or self.manifest["num_classes"] != self.num_classes
                or self.manifest["ground_truth_interface"] is not False):
            raise ValueError("Trace metadata scope mismatch")
        self.ids, self.states, self.probes, self.usages = {}, {}, {}, {}
        for client in self.manifest["client_order"]:
            info = self.manifest["clients"][client]
            path = _safe_path(self.root, info["ids_file"])
            if file_hash(path) != info["ids_file_sha256"]:
                raise ValueError("Client ID-file integrity failed")
            saved = _read_npz(path)
            ids, clients = saved["sample_id"], saved["client_id"]
            if (set(saved) != {"sample_id", "client_id"} or ids.ndim != 1 or clients.ndim != 1
                    or ids.dtype.kind != "U" or clients.dtype.kind != "U" or len(ids) != info["count"]
                    or len(clients) != len(ids) or not np.all(clients == client) or len(set(ids.tolist())) != len(ids)
                    or any(not value for value in ids.tolist())
                    or canonical_hash({"client_id": client, "sample_ids": ids.tolist()}) != info["cohort_sha256"]):
                raise ValueError("Invalid or reordered cohort identities")
            self.ids[client] = ids
        if set(self.ids) != set(self.manifest["clients"]) or len(self.ids) != len(self.manifest["client_order"]):
            raise ValueError("Duplicate/missing client metadata")
        self.events = index["events"]
        if index["event_count"] != len(self.events):
            raise ValueError("Trace event count mismatch")
        chain, paths, last, previous_arrays = self.manifest_hash, set(), {}, {}
        for i, event in enumerate(self.events):
            if (event["event_id"] != i or event["prev_event_sha256"] != chain
                    or event["event_sha256"] != canonical_hash({k: v for k, v in event.items() if k != "event_sha256"})):
                raise ValueError("Trace event hash chain failed")
            chain = event["event_sha256"]
            if event["path"] in paths or event["client_id"] not in self.ids:
                raise ValueError("Duplicate event path or unknown client")
            paths.add(event["path"])
            client, round, stage, kind = (event[k] for k in ("client_id", "round", "stage", "kind"))
            _integer(round, "round", 0, self.rounds)
            payload = self.arrays(event)
            n = len(self.ids[client])
            if kind == "state":
                if stage not in STATE_STAGES or (round, stage) != expected_state(last.get(client), self.rounds):
                    raise ValueError("Trace state order failed")
                validated = _state_payload(payload, n, self.num_classes, stage, previous_arrays.get(client))
                self.states[(client, round, stage)] = event
                last[client], previous_arrays[client] = (round, stage), validated
            elif kind == "probe":
                key = (client, round)
                if (stage != "fixed_view_end_round" or key in self.probes
                        or last.get(client) != (round, "plif" if round == 0 else "ple")):
                    raise ValueError("Probe has no phase-correct completed state")
                _probe_payload(payload, n, self.num_classes)
                self.probes[key] = event
            elif kind == "usage":
                key = (client, round, stage)
                ref = (client, *usage_reference(round, stage))
                if round < 1 or stage not in USAGE_STAGES or key in self.usages or ref not in self.states:
                    raise ValueError("Usage has no phase-correct label reference")
                _usage_payload(payload, n, self.num_classes, self.arrays(self.states[ref]))
                self.usages[key] = event
            else:
                raise ValueError("Unknown trace event kind")
        if index["last_event_sha256"] != chain:
            raise ValueError("Trace chain tail mismatch")
        indexed = {"manifest.json", "index.json"} | paths | {v["ids_file"] for v in self.manifest["clients"].values()}
        actual = {p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file()}
        if actual != indexed:
            raise ValueError("Unindexed or missing files in trace directory")

    def arrays(self, event):
        path = _safe_path(self.root, event["path"])
        if file_hash(path) != event["file_sha256"]:
            raise ValueError("Trace array-file integrity failed")
        saved = _read_npz(path)
        for key in ("kind", "client_id", "round", "stage", "cohort_sha256"):
            if key not in saved or saved[key].ndim != 0 or saved[key].item() != event[key]:
                raise ValueError("Trace NPZ header differs from indexed event")
        if event["cohort_sha256"] != self.manifest["clients"][event["client_id"]]["cohort_sha256"]:
            raise ValueError("Trace event belongs to a different cohort")
        return {k: v for k, v in saved.items() if k not in {"kind", "client_id", "round", "stage", "cohort_sha256"}}

    def missing_events(self):
        missing = []
        for client in self.ids:
            for r, stage in [(0, "plif")] + [(r, s) for r in range(1, self.rounds + 1) for s in STATE_STAGES[1:]]:
                if (client, r, stage) not in self.states:
                    missing.append({"kind": "state", "client_id": client, "round": r, "stage": stage})
            for r in range(self.rounds + 1):
                if (client, r) not in self.probes:
                    missing.append({"kind": "probe", "client_id": client, "round": r})
            for r in range(1, self.rounds + 1):
                for stage in USAGE_STAGES:
                    if (client, r, stage) not in self.usages:
                        missing.append({"kind": "usage", "client_id": client, "round": r, "stage": stage})
        return missing

    def verify_unchanged(self):
        if file_hash(self.root / "manifest.json") != self.manifest_hash or file_hash(self.root / "index.json") != self.index_hash:
            raise ValueError("Trace changed during offline evaluation")
        for info in self.manifest["clients"].values():
            if file_hash(_safe_path(self.root, info["ids_file"])) != info["ids_file_sha256"]:
                raise ValueError("Trace IDs changed during offline evaluation")
        for event in self.events:
            if file_hash(_safe_path(self.root, event["path"])) != event["file_sha256"]:
                raise ValueError("Trace arrays changed during offline evaluation")
