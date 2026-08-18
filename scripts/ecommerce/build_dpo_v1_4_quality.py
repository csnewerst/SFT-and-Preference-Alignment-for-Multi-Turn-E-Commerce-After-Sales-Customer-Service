#!/usr/bin/env python3
"""Select a quality-first DPO v1.4 set from frozen-SFT-scored candidates."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence


BUCKETS = ("must_continue", "must_stop", "wrong_action", "parameter", "response")


def load_jsonl_dir(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for file_path in sorted(path.rglob("*.jsonl")):
        with file_path.open("r", encoding="utf-8") as input_file:
            rows.extend(json.loads(line) for line in input_file if line.strip())
    if not rows:
        raise ValueError(f"no JSONL rows found under {path}")
    return rows


def behavior_bucket(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("DPO row is missing metadata")
    level = str(metadata.get("preference_level", ""))
    chosen_kind = str(metadata.get("chosen_target_kind", ""))
    rejected_kind = str(metadata.get("rejected_target_kind", ""))
    if level == "parameter" and chosen_kind == rejected_kind == "action":
        return "parameter"
    if level == "response" and chosen_kind == rejected_kind == "response":
        return "response"
    if level == "decision":
        if chosen_kind == "action" and rejected_kind == "response":
            return "must_continue"
        if chosen_kind == "response" and rejected_kind == "action":
            return "must_stop"
        if chosen_kind == rejected_kind == "action":
            return "wrong_action"
    raise ValueError(
        f"unsupported preference structure: level={level}, chosen={chosen_kind}, rejected={rejected_kind}"
    )


def _hardness(row: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = row.get("metadata", {})
    hardness = metadata.get("sft_hardness", {}) if isinstance(metadata, Mapping) else {}
    required = ("chosen_mean_logp", "rejected_mean_logp", "mean_logp_margin")
    if not isinstance(hardness, Mapping) or any(key not in hardness for key in required):
        raise ValueError("every v1.4 candidate must contain frozen-SFT metadata.sft_hardness scores")
    return hardness


def _identity(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata", {})
    if isinstance(metadata, Mapping):
        for key in ("sample_id", "pair_id"):
            if metadata.get(key):
                return str(metadata[key])
    return hashlib.sha256(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _parent(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata", {})
    if not isinstance(metadata, Mapping) or not metadata.get("parent_id"):
        raise ValueError("every v1.4 candidate must contain metadata.parent_id")
    return str(metadata["parent_id"])


def quotas(total: int, fractions: Mapping[str, float]) -> Dict[str, int]:
    if total < 1 or set(fractions) != set(BUCKETS):
        raise ValueError("invalid total or bucket fractions")
    if abs(sum(float(fractions[name]) for name in BUCKETS) - 1.0) > 1e-9:
        raise ValueError("bucket fractions must sum to 1")
    exact = {name: total * float(fractions[name]) for name in BUCKETS}
    result = {name: int(exact[name]) for name in BUCKETS}
    remainder = total - sum(result.values())
    order = sorted(BUCKETS, key=lambda name: (-(exact[name] - result[name]), name))
    for name in order[:remainder]:
        result[name] += 1
    return result


def select_quality_rows(
    rows: Sequence[Mapping[str, Any]],
    target_count: int,
    fractions: Mapping[str, float],
    max_pairs_per_parent: int,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if max_pairs_per_parent < 1:
        raise ValueError("max_pairs_per_parent must be positive")
    target_by_bucket = quotas(target_count, fractions)
    by_bucket: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        _hardness(row)
        by_bucket[behavior_bucket(row)].append(row)
    for bucket in BUCKETS:
        by_bucket[bucket].sort(
            key=lambda row: (float(_hardness(row)["mean_logp_margin"]), _identity(row))
        )

    selected: List[Dict[str, Any]] = []
    parent_counts: Counter[str] = Counter()
    selected_by_bucket: Counter[str] = Counter()
    scarcity_order = sorted(
        BUCKETS,
        key=lambda bucket: (len(by_bucket[bucket]) / max(target_by_bucket[bucket], 1), bucket),
    )
    for bucket in scarcity_order:
        for source_row in by_bucket[bucket]:
            if selected_by_bucket[bucket] >= target_by_bucket[bucket]:
                break
            parent_id = _parent(source_row)
            if parent_counts[parent_id] >= max_pairs_per_parent:
                continue
            row = copy.deepcopy(dict(source_row))
            metadata = row.setdefault("metadata", {})
            metadata["dpo_v1_4_bucket"] = bucket
            selected.append(row)
            selected_by_bucket[bucket] += 1
            parent_counts[parent_id] += 1
        if selected_by_bucket[bucket] != target_by_bucket[bucket]:
            raise ValueError(
                f"insufficient {bucket} candidates after parent cap: "
                f"selected {selected_by_bucket[bucket]}, required {target_by_bucket[bucket]}"
            )

    selected.sort(key=lambda row: hashlib.sha256(_identity(row).encode("utf-8")).hexdigest())
    margins = [float(_hardness(row)["mean_logp_margin"]) for row in selected]
    return selected, {
        "input_count": len(rows),
        "selected_count": len(selected),
        "bucket_counts": dict(sorted(selected_by_bucket.items())),
        "unique_parent_count": len(parent_counts),
        "max_pairs_for_one_parent": max(parent_counts.values(), default=0),
        "mean_logp_margin": mean(margins),
        "nonpositive_margin_fraction": sum(value <= 0 for value in margins) / len(margins),
        "minimum_margin": min(margins),
        "maximum_margin": max(margins),
    }


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output_file:
        for row in materialized:
            output_file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {"count": len(materialized), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def build_dataset(
    input_root: Path,
    output_root: Path,
    config: Mapping[str, Any],
    profile: str,
) -> Dict[str, Any]:
    size = config["size_policy"]
    if profile == "screen":
        split_targets = {"train": int(size["screen_train"]), "validation": int(size["screen_validation"])}
        fractions = config.get("screen_bucket_fractions", config.get("bucket_fractions"))
    elif profile == "formal_target":
        split_targets = {"train": int(size["formal_train"]), "validation": int(size["formal_validation"])}
        fractions = config.get("formal_provisional_bucket_fractions", config.get("bucket_fractions"))
    else:
        raise ValueError(f"unsupported profile: {profile}")
    manifest: Dict[str, Any] = {
        "schema_version": "1.0",
        "dataset_id": config["dataset_id"],
        "status": "quality_selected_training_data_not_formal_evaluation",
        "profile": profile,
        "config_sha256": hashlib.sha256(
            json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "splits": {},
    }
    for split, target in split_targets.items():
        selected, audit = select_quality_rows(
            load_jsonl_dir(input_root / split),
            target,
            fractions,
            int(size["max_pairs_per_parent_per_split"]),
        )
        artifact = _write_jsonl(output_root / split / "records.jsonl", selected)
        manifest["splits"][split] = {**audit, "artifact": artifact}
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", choices=("screen", "formal_target"), default="screen")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(json.dumps(build_dataset(args.input_root, args.output_root, config, args.profile), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
