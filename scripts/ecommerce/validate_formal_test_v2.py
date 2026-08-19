#!/usr/bin/env python3
"""Validate that a formal test candidate is frozen and disjoint from every development set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Set


ARTIFACTS = ("cases", "evaluator_cases", "private_oracle", "source_manifest")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def identities(rows: Iterable[Mapping[str, Any]], *, require_case_id: bool = False) -> Dict[str, Set[str]]:
    result = {"case_id": set(), "parent_id": set(), "source_record_id": set(), "message_sha256": set()}
    for row in rows:
        if require_case_id and not row.get("case_id"):
            raise ValueError("formal test rows must contain case_id")
        for container in (row, row.get("source_ref", {}), row.get("metadata", {})):
            if not isinstance(container, Mapping):
                continue
            for key in result:
                if key == "message_sha256":
                    continue
                if container.get(key):
                    result[key].add(str(container[key]))
        messages = row.get("messages")
        if isinstance(messages, list):
            normalized = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            result["message_sha256"].add(hashlib.sha256(normalized.encode("utf-8")).hexdigest())
    return result


def validate(
    candidate_dir: Path,
    development_dirs: Iterable[Path],
    minimum_cases: int,
    reference_jsonl: Iterable[Path] = (),
) -> Dict[str, Any]:
    manifest_path = candidate_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "2.0" or manifest.get("status") != "formal_frozen_test_unopened":
        raise ValueError("formal test manifest must be schema 2.0 with status formal_frozen_test_unopened")
    if manifest.get("sealed") is not True:
        raise ValueError("formal test manifest must record sealed=true")

    artifacts = {name: candidate_dir / f"{name}.jsonl" for name in ARTIFACTS}
    cases = load_jsonl(artifacts["cases"])
    if len(cases) < minimum_cases:
        raise ValueError(f"formal test has {len(cases)} cases; minimum is {minimum_cases}")
    candidate_ids = identities(cases, require_case_id=True)
    if len(candidate_ids["case_id"]) != len(cases):
        raise ValueError("formal test case_id values must be unique")
    for name, path in artifacts.items():
        rows = load_jsonl(path)
        if {str(row["case_id"]) for row in rows} != candidate_ids["case_id"]:
            raise ValueError(f"{name}.jsonl case IDs do not match cases.jsonl")
        expected_hash = manifest.get("artifacts", {}).get(name, {}).get("sha256")
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_hash != actual_hash:
            raise ValueError(f"{name}.jsonl hash does not match the sealed manifest")

    checked = []
    for development_dir in development_dirs:
        development_cases = load_jsonl(development_dir / "cases.jsonl")
        development_ids = identities(development_cases)
        for key in candidate_ids:
            overlap = sorted(candidate_ids[key] & development_ids[key])
            if overlap:
                raise ValueError(f"{key} leakage against {development_dir}: {overlap[:5]}")
        checked.append(str(development_dir))

    reference_files_checked = []
    for reference_path in reference_jsonl:
        reference_ids = identities(load_jsonl(reference_path))
        for key in candidate_ids:
            overlap = sorted(candidate_ids[key] & reference_ids[key])
            if overlap:
                raise ValueError(f"{key} leakage against {reference_path}: {overlap[:5]}")
        reference_files_checked.append(str(reference_path))
    return {
        "status": "passed",
        "candidate_dir": str(candidate_dir),
        "case_count": len(cases),
        "minimum_cases": minimum_cases,
        "development_dirs_checked": checked,
        "reference_jsonl_checked": reference_files_checked,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--development-dir", type=Path, action="append", required=True)
    parser.add_argument(
        "--reference-jsonl",
        type=Path,
        action="append",
        default=[],
        help="Training or preference JSONL to check using top-level/source_ref/metadata identities",
    )
    parser.add_argument("--minimum-cases", type=int, default=600)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(
        args.candidate_dir,
        args.development_dir,
        args.minimum_cases,
        reference_jsonl=args.reference_jsonl,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
