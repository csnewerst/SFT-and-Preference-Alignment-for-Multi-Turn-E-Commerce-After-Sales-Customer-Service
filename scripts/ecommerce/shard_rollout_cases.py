#!/usr/bin/env python3
"""Deterministically shard rollout cases and losslessly merge their traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


def load_jsonl(path: Path) -> list[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def _case_id(row: Mapping[str, Any]) -> str:
    case_id = row.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("every row must contain a non-empty string case_id")
    return case_id


def _assert_unique(rows: Sequence[Mapping[str, Any]], label: str) -> None:
    case_ids = [_case_id(row) for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError(f"duplicate case_id in {label}")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def prepare_shards(cases_path: Path, output_dir: Path, shard_count: int) -> Dict[str, Any]:
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    rows = load_jsonl(cases_path)
    _assert_unique(rows, str(cases_path))
    shards = [rows[index::shard_count] for index in range(shard_count)]
    artifacts = []
    for index, shard in enumerate(shards):
        artifact = _write_jsonl(output_dir / f"cases-{index:02d}.jsonl", shard)
        artifacts.append({"index": index, "count": len(shard), **artifact})
    manifest = {
        "schema_version": "1.0",
        "source": {
            "path": str(cases_path),
            "count": len(rows),
            "sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        },
        "shard_count": shard_count,
        "shards": artifacts,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def merge_traces(cases_path: Path, trace_paths: Sequence[Path], output_path: Path) -> Dict[str, Any]:
    cases = load_jsonl(cases_path)
    _assert_unique(cases, str(cases_path))
    traces: list[Dict[str, Any]] = []
    for path in trace_paths:
        traces.extend(load_jsonl(path))
    _assert_unique(traces, "trace shards")
    trace_by_id = {_case_id(row): row for row in traces}
    expected = [_case_id(row) for row in cases]
    missing = sorted(set(expected) - set(trace_by_id))
    extra = sorted(set(trace_by_id) - set(expected))
    if missing or extra:
        raise ValueError(f"trace coverage mismatch: missing={missing[:5]}, extra={extra[:5]}")
    artifact = _write_jsonl(output_path, (trace_by_id[case_id] for case_id in expected))
    return {"case_count": len(cases), "trace_count": len(traces), "artifact": artifact}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--cases", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--shard-count", type=int, default=4)
    merge = subparsers.add_parser("merge")
    merge.add_argument("--cases", type=Path, required=True)
    merge.add_argument("--trace", type=Path, action="append", required=True)
    merge.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        report = prepare_shards(args.cases, args.output_dir, args.shard_count)
    else:
        report = merge_traces(args.cases, args.trace, args.output)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
