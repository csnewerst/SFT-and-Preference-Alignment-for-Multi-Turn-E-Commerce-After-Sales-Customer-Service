#!/usr/bin/env python3
"""Create deterministic screen/gate splits from pre-freeze rollout artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


ARTIFACT_NAMES = ("cases", "private_oracle", "evaluator_cases", "source_manifest")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict) or not row.get("case_id"):
                    raise ValueError(f"{path}:{line_number} must contain a case_id object")
                rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rank(seed: int, case_id: str) -> str:
    return hashlib.sha256(f"{seed}:{case_id}".encode("utf-8")).hexdigest()


def _allocate(sizes: Mapping[str, int], count: int) -> Dict[str, int]:
    total = sum(sizes.values())
    if count < 0 or count > total:
        raise ValueError("allocation count is outside the available population")
    exact = {key: count * size / total for key, size in sizes.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    remaining = count - sum(quotas.values())
    order = sorted(sizes, key=lambda key: (-(exact[key] - quotas[key]), key))
    for key in order[:remaining]:
        quotas[key] += 1
    return quotas


def stratified_case_ids(
    cases: Sequence[Mapping[str, Any]], screen_count: int, seed: int
) -> Tuple[List[str], List[str]]:
    if not 0 < screen_count < len(cases):
        raise ValueError("screen_count must be between zero and the total case count")
    case_ids = [str(row["case_id"]) for row in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("case_id values must be unique")
    parents = [str(row.get("source_ref", {}).get("parent_id", "")) for row in cases]
    nonempty_parents = [parent for parent in parents if parent]
    if len(nonempty_parents) != len(set(nonempty_parents)):
        raise ValueError("parent_id values must be unique before a case-level split")

    strata: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for row in cases:
        key = (str(row.get("tier", "unknown")), str(row.get("category", "unknown")))
        strata[key].append(str(row["case_id"]))

    tier_sizes: Counter = Counter()
    for (tier, _category), values in strata.items():
        tier_sizes[tier] += len(values)
    tier_quotas = _allocate(tier_sizes, screen_count)
    quotas: Dict[Tuple[str, str], int] = {}
    for tier, tier_quota in tier_quotas.items():
        category_sizes = {category: len(values) for (row_tier, category), values in strata.items() if row_tier == tier}
        for category, quota in _allocate(category_sizes, tier_quota).items():
            quotas[(tier, category)] = quota

    screen = set()
    for key, values in strata.items():
        ranked = sorted(values, key=lambda case_id: (_rank(seed, case_id), case_id))
        screen.update(ranked[: quotas[key]])
    gate = set(case_ids) - screen
    if len(screen) != screen_count or screen & gate:
        raise AssertionError("invalid deterministic split")
    return sorted(screen), sorted(gate)


def _slice(rows: Sequence[Mapping[str, Any]], case_ids: set[str]) -> List[Mapping[str, Any]]:
    selected = [row for row in rows if str(row["case_id"]) in case_ids]
    if len(selected) != len(case_ids):
        raise ValueError("artifact case_id set does not match cases.jsonl")
    return selected


def _summary(cases: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "case_count": len(cases),
        "tier_counts": dict(sorted(Counter(str(row.get("tier")) for row in cases).items())),
        "category_counts": dict(sorted(Counter(str(row.get("category")) for row in cases).items())),
    }


def write_split(input_dir: Path, output_dir: Path, screen_count: int, seed: int) -> Dict[str, Any]:
    artifacts = {name: _read_jsonl(input_dir / f"{name}.jsonl") for name in ARTIFACT_NAMES}
    cases = artifacts["cases"]
    expected_ids = {str(row["case_id"]) for row in cases}
    for name, rows in artifacts.items():
        if {str(row["case_id"]) for row in rows} != expected_ids:
            raise ValueError(f"{name}.jsonl case IDs do not match cases.jsonl")

    screen_list, gate_list = stratified_case_ids(cases, screen_count, seed)
    split_ids = {"screen": set(screen_list), "gate": set(gate_list)}
    manifest: Dict[str, Any] = {
        "schema_version": "1.0",
        "status": "development_split_not_formal_test",
        "source_dir": str(input_dir),
        "seed": seed,
        "source_case_count": len(cases),
        "splits": {},
    }
    for split_name, ids in split_ids.items():
        split_dir = output_dir / split_name
        paths: Dict[str, Any] = {}
        split_cases = _slice(cases, ids)
        for name, rows in artifacts.items():
            path = split_dir / f"{name}.jsonl"
            _write_jsonl(path, _slice(rows, ids))
            paths[name] = {"path": str(path.relative_to(output_dir)).replace("\\", "/"), "sha256": _sha256(path)}
        manifest["splits"][split_name] = {**_summary(split_cases), "artifacts": paths}

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(manifest, output_file, ensure_ascii=False, indent=2, sort_keys=True)
        output_file.write("\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--screen-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260809)
    args = parser.parse_args()
    print(json.dumps(write_split(args.input_dir, args.output_dir, args.screen_count, args.seed), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
