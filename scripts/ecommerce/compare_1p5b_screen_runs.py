#!/usr/bin/env python3
"""Aggregate fixed-case screen results with deterministic paired bootstrap intervals."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_mean(values: Sequence[float], seed: int, resamples: int) -> Dict[str, float]:
    if not values:
        raise ValueError("cannot bootstrap an empty sequence")
    generator = random.Random(seed)
    size = len(values)
    estimates = [sum(values[generator.randrange(size)] for _ in range(size)) / size for _ in range(resamples)]
    return {
        "estimate": sum(values) / size,
        "ci95_low": percentile(estimates, 0.025),
        "ci95_high": percentile(estimates, 0.975),
    }


def _metric_value(row: Mapping[str, Any], metric: str) -> float:
    if metric == "task_success_rate":
        return float(bool(row["passed"]))
    if metric == "eligible_auto_resolution_rate":
        return float(bool(row.get("auto_resolved")))
    return float(bool(row.get("checks", {}).get(metric)))


def _metric_case_ids(
    run_maps: Mapping[str, Mapping[str, Mapping[str, Any]]],
    case_ids: Sequence[str],
    metric: str,
) -> List[str]:
    if metric != "eligible_auto_resolution_rate":
        return list(case_ids)
    labels = list(run_maps)
    eligible_ids = []
    for case_id in case_ids:
        flags = [bool(run_maps[label][case_id].get("auto_resolution_eligible")) for label in labels]
        if len(set(flags)) != 1:
            raise ValueError(f"auto-resolution eligibility differs across runs for case {case_id}")
        if flags[0]:
            eligible_ids.append(case_id)
    if not eligible_ids:
        raise ValueError("no cases are eligible for auto-resolution")
    return eligible_ids


def compare_runs(
    cases: Iterable[Mapping[str, Any]],
    runs: Sequence[Tuple[str, Iterable[Mapping[str, Any]]]],
    seed: int,
    resamples: int,
) -> Dict[str, Any]:
    case_list = list(cases)
    case_ids = [str(case["case_id"]) for case in case_list]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("cases contain duplicate case_id values")
    tiers = {str(case["case_id"]): str(case.get("tier", "unknown")) for case in case_list}
    run_maps: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    for label, rows in runs:
        row_map = {str(row["case_id"]): row for row in rows}
        if set(row_map) != set(case_ids):
            raise ValueError(f"run {label} does not contain exactly the fixed case set")
        run_maps[label] = row_map

    metric_names = [
        "task_success_rate",
        "parse_success",
        "tool_selection_valid",
        "arguments_valid",
        "forbidden_tool_absent",
        "observation_outcomes_valid",
        "answer_requirements_met",
        "state_assertions_met",
        "facts_faithful",
        "within_step_limit",
        "eligible_auto_resolution_rate",
    ]
    report: Dict[str, Any] = {
        "schema_version": "1.1",
        "case_count": len(case_ids),
        "bootstrap_seed": seed,
        "bootstrap_resamples": resamples,
        "runs": {},
        "paired_task_success_deltas": {},
        "paired_metric_deltas": {},
    }
    for run_index, (label, row_map) in enumerate(run_maps.items()):
        run_report: Dict[str, Any] = {"metrics": {}, "tier_metrics": {}}
        for metric_index, metric in enumerate(metric_names):
            metric_ids = _metric_case_ids(run_maps, case_ids, metric)
            values = [_metric_value(row_map[case_id], metric) for case_id in metric_ids]
            run_report["metrics"][metric] = {
                "case_count": len(metric_ids),
                **bootstrap_mean(
                values, seed + run_index * 1000 + metric_index, resamples
                ),
            }
        for tier_index, tier in enumerate(sorted(set(tiers.values()))):
            tier_ids = [case_id for case_id in case_ids if tiers[case_id] == tier]
            values = [_metric_value(row_map[case_id], "task_success_rate") for case_id in tier_ids]
            run_report["tier_metrics"][tier] = {
                "case_count": len(tier_ids),
                **bootstrap_mean(values, seed + run_index * 1000 + 100 + tier_index, resamples),
            }
        report["runs"][label] = run_report

    labels = list(run_maps)
    for left_index, left in enumerate(labels):
        for right_index in range(left_index + 1, len(labels)):
            right = labels[right_index]
            comparison = f"{right}_minus_{left}"
            report["paired_metric_deltas"][comparison] = {}
            for metric_index, metric in enumerate(metric_names):
                metric_ids = _metric_case_ids(run_maps, case_ids, metric)
                differences = [
                    _metric_value(run_maps[right][case_id], metric)
                    - _metric_value(run_maps[left][case_id], metric)
                    for case_id in metric_ids
                ]
                result = {
                    "case_count": len(metric_ids),
                    **bootstrap_mean(
                        differences,
                        seed + 10000 + left_index * 1000 + right_index * 100 + metric_index,
                        resamples,
                    ),
                }
                report["paired_metric_deltas"][comparison][metric] = result
                if metric == "task_success_rate":
                    report["paired_task_success_deltas"][comparison] = result
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, help="LABEL=per_sample.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--resamples", type=int, default=10000)
    args = parser.parse_args()
    parsed_runs: List[Tuple[str, List[Dict[str, Any]]]] = []
    for spec in args.run:
        label, separator, path = spec.partition("=")
        if not separator or not label or not path:
            raise ValueError(f"invalid --run value: {spec}")
        parsed_runs.append((label, load_jsonl(Path(path))))
    report = compare_runs(load_jsonl(args.cases), parsed_runs, args.seed, args.resamples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(report, output_file, ensure_ascii=False, indent=2, sort_keys=True)
        output_file.write("\n")
    print(json.dumps({"run_count": len(parsed_runs), "case_count": report["case_count"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
