#!/usr/bin/env python3
"""Build count-matched response-only and multigranularity DPO ablation sets."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


LEVELS = ("decision", "parameter", "response")


def _load_dir(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for file_path in sorted(path.rglob("*.jsonl")):
        with file_path.open("r", encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(f"{file_path}:{line_number} must be an object")
                    rows.append(row)
    if not rows:
        raise ValueError(f"no JSONL rows found under {path}")
    return rows


def _level(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("DPO row is missing metadata")
    level = str(metadata.get("preference_level") or metadata.get("pair_level") or "")
    if level not in LEVELS:
        raise ValueError(f"unsupported preference level: {level!r}")
    return level


def _identity(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
    for key in ("sample_id", "pair_id", "source_record_id", "parent_id"):
        if metadata.get(key):
            return f"{key}:{metadata[key]}"
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _rank(row: Mapping[str, Any], seed: int) -> str:
    return hashlib.sha256(f"{seed}:{_identity(row)}".encode("utf-8")).hexdigest()


def _matched_multigranularity(rows: Sequence[Mapping[str, Any]], count: int, seed: int) -> List[Mapping[str, Any]]:
    by_level: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_level[_level(row)].append(row)
    exact = {level: count * len(by_level[level]) / len(rows) for level in LEVELS}
    quotas = {level: int(exact[level]) for level in LEVELS}
    remaining = count - sum(quotas.values())
    order = sorted(LEVELS, key=lambda level: (-(exact[level] - quotas[level]), level))
    for level in order[:remaining]:
        quotas[level] += 1
    selected: List[Mapping[str, Any]] = []
    for level in LEVELS:
        candidates = sorted(by_level[level], key=lambda row: (_rank(row, seed), _identity(row)))
        if len(candidates) < quotas[level]:
            raise ValueError(f"not enough {level} rows for matched subset")
        selected.extend(candidates[: quotas[level]])
    return sorted(selected, key=lambda row: (_rank(row, seed), _identity(row)))


def _write(path: Path, rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output_file:
        for row in materialized:
            output_file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "path": str(path),
        "count": len(materialized),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "level_counts": dict(sorted(Counter(_level(row) for row in materialized).items())),
    }


def build_variants(input_root: Path, output_root: Path, seed: int = 20260809) -> Dict[str, Any]:
    manifest: Dict[str, Any] = {
        "schema_version": "1.0",
        "status": "training_ablation_data_not_formal_evaluation",
        "seed": seed,
        "input_root": str(input_root),
        "variants": {},
    }
    for split in ("train", "validation"):
        rows = _load_dir(input_root / split)
        response_rows = [row for row in rows if _level(row) == "response"]
        matched_rows = _matched_multigranularity(rows, len(response_rows), seed)
        outputs = {
            "response_only_matched": response_rows,
            "multigranularity_matched": matched_rows,
            "multigranularity_full": rows,
        }
        for variant, selected in outputs.items():
            manifest["variants"].setdefault(variant, {})[split] = _write(
                output_root / variant / split / "records.jsonl", selected
            )

    manifest_path = output_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(manifest, output_file, ensure_ascii=False, indent=2, sort_keys=True)
        output_file.write("\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260809)
    args = parser.parse_args()
    print(json.dumps(build_variants(args.input_root, args.output_root, args.seed), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
