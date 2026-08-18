#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.tool_utils import load_local_json_datasets


ALLOWED_ROLES = {"system", "human", "user", "gpt", "assistant", "function_call", "observation"}


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{index} is not valid JSONL: {exc}") from exc
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def _validate_tools(tools: Any, path: Path, row_index: int) -> None:
    if not isinstance(tools, list) or not tools:
        raise ValueError(f"{path}:{row_index} tools must be a non-empty list")
    for tool_index, tool in enumerate(tools, start=1):
        if not isinstance(tool, dict):
            raise ValueError(f"{path}:{row_index} tool #{tool_index} must be an object")
        for key in ("name", "description", "parameters"):
            if key not in tool:
                raise ValueError(f"{path}:{row_index} tool #{tool_index} missing {key}")


def _validate_conversations(conversations: Any, path: Path, row_index: int, preference_row: bool) -> None:
    if not isinstance(conversations, list) or not conversations:
        raise ValueError(f"{path}:{row_index} conversations must be a non-empty list")

    saw_user = False
    saw_assistant = False
    saw_function_call = False
    saw_observation = False

    for message_index, message in enumerate(conversations, start=1):
        if not isinstance(message, dict):
            raise ValueError(f"{path}:{row_index} conversation #{message_index} must be an object")
        role = message.get("from")
        value = message.get("value")
        if role not in ALLOWED_ROLES:
            raise ValueError(f"{path}:{row_index} conversation #{message_index} has unsupported role {role!r}")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{path}:{row_index} conversation #{message_index} value must be a non-empty string")
        if role in {"human", "user"}:
            saw_user = True
        if role in {"gpt", "assistant"}:
            saw_assistant = True
        if role == "function_call":
            saw_function_call = True
        if role == "observation":
            saw_observation = True

    if not saw_user:
        raise ValueError(f"{path}:{row_index} must contain at least one user turn")
    if not preference_row and not saw_assistant:
        raise ValueError(f"{path}:{row_index} SFT/OPD rows must contain at least one assistant turn")
    if saw_function_call and not saw_observation:
        raise ValueError(f"{path}:{row_index} tool rows must include an observation turn")


def _validate_row(row: Dict[str, Any], path: Path, row_index: int) -> None:
    if "conversations" not in row:
        raise ValueError(f"{path}:{row_index} missing conversations")

    preference_row = "chosen" in row or "rejected" in row
    _validate_conversations(row["conversations"], path, row_index, preference_row)

    if preference_row:
        if not isinstance(row.get("chosen"), str) or not row["chosen"].strip():
            raise ValueError(f"{path}:{row_index} chosen must be a non-empty string")
        if not isinstance(row.get("rejected"), str) or not row["rejected"].strip():
            raise ValueError(f"{path}:{row_index} rejected must be a non-empty string")
    elif "chosen" in row or "rejected" in row:
        raise ValueError(f"{path}:{row_index} preference fields are only allowed together")

    if "tools" in row and row["tools"] is not None:
        _validate_tools(row["tools"], path, row_index)


def _collect_split_files(root: Path) -> Dict[str, List[str]]:
    split_files: Dict[str, List[str]] = {}
    for split in ("train", "validation"):
        files = []
        split_dir = root / split
        if split_dir.is_dir():
            files.extend(sorted(str(p) for p in split_dir.rglob("*.jsonl")))
        files.extend(sorted(str(p) for p in root.glob(f"{split}*.jsonl")))
        if files:
            split_files[split] = sorted(dict.fromkeys(files))
    return split_files


def validate_task(root: Path, cache_dir: Path) -> None:
    split_files = _collect_split_files(root)
    if not split_files:
        raise ValueError(f"No jsonl files found under {root}")

    for split_name, files in split_files.items():
        for file_name in files:
            rows = _load_jsonl(Path(file_name))
            for index, row in enumerate(rows, start=1):
                _validate_row(row, Path(file_name), index)

    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_local_json_datasets(split_files, cache_dir=str(cache_dir))
    for split_name, split_dataset in dataset.items():
        print(f"{root.name}/{split_name}: {len(split_dataset)} rows, columns={split_dataset.column_names}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate ecommerce smoke data.")
    parser.add_argument("--root", type=Path, default=ROOT / "data" / "ecommerce" / "processed")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache" / "datasets")
    args = parser.parse_args()

    for task_dir in (args.root / "sft", args.root / "dpo"):
        validate_task(task_dir, args.cache_dir)
    print("ecommerce smoke data validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
