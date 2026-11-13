"""Offline, ground-truth-isolated evaluation of a frozen DPL training trace."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from helpers.dpl_trace import TraceReader, file_hash, write_json


STATE_NAMES = ("unselected", "correct", "wrong")


def _divide(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def _load_ground_truth(path, reader):
    path = Path(path).resolve()
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"sample_id", "client_id", "label"}:
            raise ValueError("GT NPZ must have exactly sample_id, client_id and label arrays")
        ids, clients, labels = (data[k].copy() for k in ("sample_id", "client_id", "label"))
    if (ids.ndim != 1 or clients.ndim != 1 or labels.ndim != 1
            or ids.dtype.kind != "U" or clients.dtype.kind != "U" or labels.dtype.kind not in "iu"
            or len(ids) != len(clients) or len(ids) != len(labels)):
        raise ValueError("GT requires aligned1D Unicode IDs and integer labels")
    converted = labels.astype(np.int64)
    if not np.array_equal(labels, converted) or (converted < 0).any():
        raise ValueError("GT labels must be nonnegative int64-compatible class IDs")
    if reader.num_classes is not None and (converted >= reader.num_classes).any():
        raise ValueError("GT class label exceeds the recorded vocabulary")
    keys = list(zip(clients.tolist(), ids.tolist()))
    if any(not c or not s for c, s in keys) or len(set(keys)) != len(keys):
        raise ValueError("Duplicate/empty GT client/sample identity")
    expected = {(client, sample) for client, values in reader.ids.items() for sample in values.tolist()}
    if set(keys) != expected or len(keys) != len(expected):
        raise ValueError("GT identities must match the entire source cohort exactly, without extras or omissions")
    lookup = dict(zip(keys, converted.tolist()))
    return {client: np.asarray([lookup[(client, sample)] for sample in values.tolist()], dtype=np.int64)
            for client, values in reader.ids.items()}


def _counts(state, truth, cohort):
    n = int(cohort.sum())
    selected = state["selected"] & cohort
    correct = selected & (state["pseudo_label"] == truth)
    active, good = int(selected.sum()), int(correct.sum())
    wrong = active - good
    return {"N": n, "A": active, "C": good, "W": wrong,
            "precision_denominator": active, "coverage_denominator": n, "mass_denominator": n,
            "precision": _divide(good, active), "coverage": _divide(active, n),
            "correct_mass": _divide(good, n), "wrong_mass": _divide(wrong, n)}


def _state_codes(state, truth):
    return np.where(~state["selected"], 0, np.where(state["pseudo_label"] == truth, 1, 2))


def _transitions(previous, current, truth, cohort):
    before, after = _state_codes(previous, truth), _state_codes(current, truth)
    matrix = np.bincount((before[cohort] * 3 + after[cohort]), minlength=9).reshape(3, 3)
    row_counts, n = matrix.sum(axis=1), int(cohort.sum())
    result = {"N": n, "joint_denominator": n}
    for i, source in enumerate(STATE_NAMES):
        result[f"from_{source}_denominator"] = int(row_counts[i])
        for j, destination in enumerate(STATE_NAMES):
            count = int(matrix[i, j])
            result[f"{source}_to_{destination}_count"] = count
            result[f"{source}_to_{destination}_conditional_fraction"] = _divide(count, int(row_counts[i]))
            result[f"{source}_to_{destination}_joint_fraction"] = _divide(count, n)
    persistent = cohort & (before == 2) & (after == 2)
    unchanged = int(np.sum(persistent & (previous["pseudo_label"] == current["pseudo_label"])))
    result["wrong_to_wrong_same_label_count"] = unchanged
    result["wrong_to_wrong_different_wrong_label_count"] = int(persistent.sum()) - unchanged
    added = int(matrix[0, 1] + matrix[0, 2])
    result["new_admission_denominator"] = added
    result["new_admission_wrong_fraction"] = _divide(int(matrix[0, 2]), added)
    return result


def _probe_counts(probe, truth, cohort):
    n = int(cohort.sum())
    result = {"N": n, "prediction_error_denominator": n}
    for name in ("global", "composite"):
        correct = int(np.sum((probe[f"{name}_pred"] == truth) & cohort))
        result.update({f"{name}_correct_count": correct, f"{name}_wrong_count": n - correct,
                       f"{name}_accuracy": _divide(correct, n), f"{name}_error_rate": _divide(n - correct, n)})
    return result


def _probe_transition(previous, current, truth, cohort, name):
    was_correct, now_correct = previous[f"{name}_pred"] == truth, current[f"{name}_pred"] == truth
    n, correct_before = int(cohort.sum()), int(np.sum(was_correct & cohort))
    wrong_before = n - correct_before
    result = {"N": n, "from_correct_denominator": correct_before, "from_wrong_denominator": wrong_before}
    for before, left in ((False, "wrong"), (True, "correct")):
        denominator = correct_before if before else wrong_before
        for after, right in ((False, "wrong"), (True, "correct")):
            count = int(np.sum(cohort & (was_correct == before) & (now_correct == after)))
            result[f"{left}_to_{right}_count"] = count
            result[f"{left}_to_{right}_conditional_fraction"] = _divide(count, denominator)
    persistent = cohort & ~was_correct & ~now_correct
    same = int(np.sum(persistent & (previous[f"{name}_pred"] == current[f"{name}_pred"])))
    result["wrong_to_wrong_same_prediction_count"] = same
    result["wrong_to_wrong_different_wrong_prediction_count"] = int(persistent.sum()) - same
    return result


def _usage_counts(usage, truth, cohort):
    counts, labels = usage["usage_counts"], usage["pseudo_labels_used"]
    used = (counts > 0) & cohort
    wrong = used & (labels != truth)
    total = sum(int(x) for x in counts[used])
    wrong_total = sum(int(x) for x in counts[wrong])
    n = int(cohort.sum())
    return {"N": n, "unique_used_sample_count": int(used.sum()), "unique_wrong_used_sample_count": int(wrong.sum()),
            "CE_exposure_count": total, "correct_CE_exposure_count": total - wrong_total,
            "wrong_CE_exposure_count": wrong_total, "wrong_CE_fraction_denominator": total,
            "wrong_CE_exposure_fraction": _divide(wrong_total, total),
            "wrong_CE_exposures_per_source_sample": _divide(wrong_total, n)}


def _write_csv(path, rows, empty_columns):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else empty_columns)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(trace, ground_truth, output):
    trace, ground_truth, output = Path(trace).resolve(), Path(ground_truth).resolve(), Path(output).resolve()
    if ground_truth == trace or ground_truth.is_relative_to(trace):
        raise ValueError("Ground truth must remain outside the training trace directory")
    if output == trace or output.is_relative_to(trace):
        raise ValueError("Offline GT evaluation output must be separate from the training trace")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Offline evaluator output must be new and empty")
    reader = TraceReader(trace)
    gt_hash = file_hash(ground_truth)
    truth_by_client = _load_ground_truth(ground_truth, reader)
    state_rows, class_rows, transition_rows, probe_rows, probe_transition_rows, usage_rows = [], [], [], [], [], []
    cohorts_by_client = {}
    state_order = {"plif": 0, "global_before_plp": 1, "plp": 2, "ple": 3}
    for client, ids in reader.ids.items():
        truth = truth_by_client[client]
        initial_event = reader.states.get((client, 0, "plif"))
        if initial_event is None:
            continue
        initial = reader.arrays(initial_event)
        cohorts = {"all_source_samples": np.ones(len(ids), dtype=bool),
                   "initially_selected_S0": initial["selected"].copy(),
                   "initially_unselected": ~initial["selected"]}
        cohorts_by_client[client] = cohorts
        ordered = sorted((key for key in reader.states if key[0] == client), key=lambda x: (x[1], state_order[x[2]]))
        last = None
        for key in ordered:
            _, round_number, stage = key
            state = reader.arrays(reader.states[key])
            for cohort_name, cohort in cohorts.items():
                state_rows.append({"client_id": client, "round": round_number, "stage": stage, "cohort": cohort_name,
                                   **_counts(state, truth, cohort)})
            for label in np.unique(truth):
                class_rows.append({"client_id": client, "round": round_number, "stage": stage,
                                   "true_class": int(label), **_counts(state, truth, truth == label)})
            if last is not None:
                previous_key, previous = last
                for cohort_name, cohort in cohorts.items():
                    transition_rows.append({"client_id": client, "transition_scope": "adjacent_stage",
                        "from_round": previous_key[1], "from_stage": previous_key[2],
                        "to_round": round_number, "to_stage": stage, "cohort": cohort_name,
                        **_transitions(previous, state, truth, cohort)})
            if stage == "ple":
                previous_key = (client, 0, "plif") if round_number == 1 else (client, round_number - 1, "ple")
                previous = reader.arrays(reader.states[previous_key])
                for cohort_name, cohort in cohorts.items():
                    transition_rows.append({"client_id": client, "transition_scope": "round_end_to_round_end",
                        "from_round": previous_key[1], "from_stage": previous_key[2],
                        "to_round": round_number, "to_stage": stage, "cohort": cohort_name,
                        **_transitions(previous, state, truth, cohort)})
            last = (key, state)
        previous_probe = None
        for _, round_number in sorted(key for key in reader.probes if key[0] == client):
            probe = reader.arrays(reader.probes[(client, round_number)])
            for cohort_name, cohort in cohorts.items():
                probe_rows.append({"client_id": client, "round": round_number, "cohort": cohort_name,
                                   **_probe_counts(probe, truth, cohort)})
            if previous_probe is not None:
                previous_round, previous = previous_probe
                for model in ("global", "composite"):
                    for cohort_name, cohort in cohorts.items():
                        probe_transition_rows.append({"client_id": client, "from_round": previous_round,
                            "to_round": round_number, "prediction_model": model, "cohort": cohort_name,
                            **_probe_transition(previous, probe, truth, cohort, model)})
            previous_probe = (round_number, probe)
        for key in sorted((key for key in reader.usages if key[0] == client), key=lambda x: (x[1], 0 if x[2] == "global" else 1)):
            usage = reader.arrays(reader.usages[key])
            for cohort_name, cohort in cohorts.items():
                usage_rows.append({"client_id": client, "round": key[1], "stage": key[2], "cohort": cohort_name,
                                   **_usage_counts(usage, truth, cohort)})
    round_usage_rows = []
    for client, cohorts in cohorts_by_client.items():
        for cohort in cohorts:
            cumulative_total, cumulative_wrong = 0, 0
            rounds = sorted({r["round"] for r in usage_rows if r["client_id"] == client and r["cohort"] == cohort})
            for round_number in rounds:
                rows = [r for r in usage_rows if r["client_id"] == client and r["cohort"] == cohort and r["round"] == round_number]
                total, wrong = sum(r["CE_exposure_count"] for r in rows), sum(r["wrong_CE_exposure_count"] for r in rows)
                cumulative_total += total
                cumulative_wrong += wrong
                round_usage_rows.append({"client_id": client, "round": round_number, "cohort": cohort,
                    "recorded_usage_stages": len(rows), "N": rows[0]["N"],
                    "CE_exposure_count": total, "wrong_CE_exposure_count": wrong,
                    "wrong_CE_fraction_denominator": total, "wrong_CE_exposure_fraction": _divide(wrong, total),
                    "cumulative_CE_exposure_count": cumulative_total, "cumulative_wrong_CE_exposure_count": cumulative_wrong})
    # These are source-cohort micro totals, never silent means of client rates.
    aggregate_states = []
    grouped = defaultdict(list)
    for row in state_rows:
        grouped[(row["round"], row["stage"], row["cohort"])].append(row)
    for (round_number, stage, cohort), rows in sorted(grouped.items(), key=lambda x: (x[0][0], state_order[x[0][1]], x[0][2])):
        n, a, c, w = (sum(r[k] for r in rows) for k in ("N", "A", "C", "W"))
        aggregate_states.append({"round": round_number, "stage": stage, "cohort": cohort, "clients_recorded": len(rows),
                                "N": n, "A": a, "C": c, "W": w,
                                "precision_denominator": a, "coverage_denominator": n, "mass_denominator": n,
                                "precision": _divide(c, a), "coverage": _divide(a, n),
                                "correct_mass": _divide(c, n), "wrong_mass": _divide(w, n)})
    reader.verify_unchanged()
    if file_hash(ground_truth) != gt_hash:
        raise ValueError("Ground-truth input changed during evaluation")
    missing = reader.missing_events()
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "state_metrics": state_rows, "class_state_metrics": class_rows, "state_transitions": transition_rows,
        "probe_metrics": probe_rows, "probe_transitions": probe_transition_rows,
        "CE_usage_metrics": usage_rows, "round_CE_usage_metrics": round_usage_rows,
        "aggregate_state_metrics": aggregate_states,
    }
    for name, rows in tables.items():
        _write_csv(output / f"{name}.csv", rows, ["client_id", "round", "stage", "cohort"])
    result = {"schema_version": 1, "status": "complete" if not missing else "partial_trace",
              "offline_evaluation_finished": True, "trace_complete": not missing,
              "completed_utc": datetime.now(timezone.utc).isoformat(),
              "trace_directory": str(trace), "trace_manifest_sha256": reader.manifest_hash,
              "trace_index_file_sha256": reader.index_hash, "trace_event_count": len(reader.events),
              "metadata": reader.manifest["metadata"], "configured_rounds": reader.rounds,
              "source_cohort_sizes": {client: len(ids) for client, ids in reader.ids.items()},
              "ground_truth_path": str(ground_truth), "ground_truth_npz_sha256": gt_hash,
              "ground_truth_identity_match": "strict one-to-one (client_id,sample_id), all and only source cohort",
              "rate_units": "fractions0_to1 unless explicitly exposures_per_source_sample, which may exceed1",
              "state_names": list(STATE_NAMES), "missing_events": missing,
              "limits": ["GT is loaded only by this offline evaluator, never by TraceRecorder",
                         "Use these diagnostics after training; do not tune thresholds, seeds, folds or checkpoints against their GT outcomes",
                         "Fixed-initial labels have constant stored-label precision by construction; use full fixed-cohort probe errors to examine changing model predictions",
                         "Cumulative wrong exposures are nondecreasing by definition; their growth alone is not proof of confirmation bias",
                         "Counts and transitions are empirical diagnostics, not a pseudo-label convergence theorem or a guarantee against future error accumulation"],
              "input_hashes_reverified": True, "results_written_to_manuscript": False,
              **tables,
              "csv_sha256": {name: file_hash(output / f"{name}.csv") for name in tables}}
    write_json(output / "results.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = evaluate(args.trace, args.ground_truth, args.output)
    print(json.dumps({"status": result["status"], "trace_complete": result["trace_complete"],
                      "state_rows": len(result["state_metrics"]), "probe_rows": len(result["probe_metrics"]),
                      "usage_rows": len(result["CE_usage_metrics"])}, indent=2))
