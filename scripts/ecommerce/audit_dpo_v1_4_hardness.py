#!/usr/bin/env python3
"""Audit frozen-SFT DPO hardness by behavior bucket and error type."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Mapping, Sequence

from build_dpo_v1_4_quality import BUCKETS, behavior_bucket, load_jsonl_dir


def percentile(values: Sequence[float], probability: float) -> float:
    """Return a deterministic linearly interpolated percentile."""
    if not values:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between zero and one")
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _metadata(row: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = row.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("DPO row is missing metadata")
    return metadata


def _margin(row: Mapping[str, Any]) -> float:
    hardness = _metadata(row).get("sft_hardness", {})
    if not isinstance(hardness, Mapping) or "mean_logp_margin" not in hardness:
        raise ValueError("DPO row is missing metadata.sft_hardness.mean_logp_margin")
    return float(hardness["mean_logp_margin"])


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty DPO collection")
    margins = [_margin(row) for row in rows]
    parents = [str(_metadata(row).get("parent_id", "")) for row in rows]
    if any(not parent for parent in parents):
        raise ValueError("DPO row is missing metadata.parent_id")
    error_counts = Counter(str(_metadata(row).get("primary_error", "unknown")) for row in rows)
    target_counts = Counter(
        f"{_metadata(row).get('chosen_target_kind', 'unknown')}"
        f"->{_metadata(row).get('rejected_target_kind', 'unknown')}"
        for row in rows
    )
    return {
        "count": len(rows),
        "unique_parent_count": len(set(parents)),
        "margin": {
            "minimum": min(margins),
            "p10": percentile(margins, 0.10),
            "p25": percentile(margins, 0.25),
            "p50": percentile(margins, 0.50),
            "p75": percentile(margins, 0.75),
            "p90": percentile(margins, 0.90),
            "maximum": max(margins),
            "mean": mean(margins),
        },
        "hard_candidate_counts": {
            "margin_le_0": sum(value <= 0.0 for value in margins),
            "margin_le_0p25": sum(value <= 0.25 for value in margins),
            "margin_le_0p5": sum(value <= 0.5 for value in margins),
        },
        "hard_candidate_fractions": {
            "margin_le_0": sum(value <= 0.0 for value in margins) / len(margins),
            "margin_le_0p25": sum(value <= 0.25 for value in margins) / len(margins),
            "margin_le_0p5": sum(value <= 0.5 for value in margins) / len(margins),
        },
        "primary_error_counts": dict(sorted(error_counts.items())),
        "target_transition_counts": dict(sorted(target_counts.items())),
    }


def audit_dataset(input_root: Path) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "schema_version": "1.0",
        "status": "diagnostic_only_not_training_acceptance",
        "input_root": str(input_root),
        "splits": {},
    }
    for split in ("train", "validation"):
        rows = load_jsonl_dir(input_root / split)
        grouped: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[behavior_bucket(row)].append(row)
        report["splits"][split] = {
            "overall": summarize_rows(rows),
            "by_bucket": {
                bucket: summarize_rows(grouped[bucket])
                for bucket in BUCKETS
                if grouped[bucket]
            },
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_dataset(args.input_root)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
