#!/usr/bin/env python3
"""Audit training-formatted token lengths before selecting 1.5B limits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _rows(path: Path) -> Iterable[Dict[str, Any]]:
    for file_path in sorted(path.rglob("*.jsonl")):
        with file_path.open("r", encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(f"{file_path}:{line_number} must be an object")
                    yield row


def _percentile(values: List[int], percentile: float) -> int:
    ordered = sorted(values)
    if not ordered:
        return 0
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def summarize(values: List[int], limits: Iterable[int]) -> Dict[str, Any]:
    if not values:
        raise ValueError("cannot summarize empty token lengths")
    return {
        "count": len(values),
        "min": min(values),
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values),
        "limit_exceedance": {
            str(limit): {
                "count": sum(value > limit for value in values),
                "rate": sum(value > limit for value in values) / len(values),
            }
            for limit in limits
        },
    }


def _encoded_length(encoded: Any) -> int:
    if isinstance(encoded, Mapping):
        if "input_ids" not in encoded:
            raise ValueError("tokenizer mapping result is missing input_ids")
        encoded = encoded["input_ids"]
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError("expected one tokenized conversation")
        encoded = encoded[0]
    return len(encoded)


def _messages(row: Mapping[str, Any], tool_format: str) -> List[Dict[str, str]]:
    from training.tool_utils import FunctionCall, get_tool_utils

    system_prompt = str(row.get("system_prompt") or "")
    messages: List[Dict[str, str]] = []
    for turn in row.get("conversations", []):
        role = str(turn.get("from", ""))
        value = str(turn.get("value", ""))
        if role == "system":
            system_prompt = value
        elif role in {"human", "user"}:
            messages.append({"role": "user", "content": value})
        elif role == "observation":
            messages.append({"role": "user", "content": f"Observation: {value}"})
        elif role in {"gpt", "assistant"}:
            messages.append({"role": "assistant", "content": value})
        elif role == "function_call":
            call = json.loads(value)
            formatted = get_tool_utils(tool_format).function_formatter(
                [FunctionCall(call["name"], json.dumps(call["arguments"], ensure_ascii=False))]
            )
            messages.append({"role": "assistant", "content": formatted})
    tools = row.get("tools")
    if tools:
        parsed = json.loads(tools) if isinstance(tools, str) else tools
        if parsed:
            tools_text = get_tool_utils(tool_format).tool_formatter(parsed)
            system_prompt += ("\n\n" if system_prompt else "") + tools_text
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
    return messages


def audit(model_path: Path, sft_root: Path, dpo_root: Path, limits: List[int]) -> Dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False, use_fast=False)
    sft_lengths: List[int] = []
    for row in _rows(sft_root):
        token_ids = tokenizer.apply_chat_template(_messages(row, "default"), tokenize=True, add_generation_prompt=False)
        sft_lengths.append(_encoded_length(token_ids))

    prompt_lengths: List[int] = []
    chosen_lengths: List[int] = []
    rejected_lengths: List[int] = []
    chosen_full_lengths: List[int] = []
    rejected_full_lengths: List[int] = []
    for row in _rows(dpo_root):
        prompt_ids = tokenizer.apply_chat_template(_messages(row, "default"), tokenize=True, add_generation_prompt=True)
        chosen_ids = tokenizer.encode(str(row["chosen"]), add_special_tokens=False)
        rejected_ids = tokenizer.encode(str(row["rejected"]), add_special_tokens=False)
        prompt_length = _encoded_length(prompt_ids)
        prompt_lengths.append(prompt_length)
        chosen_lengths.append(len(chosen_ids))
        rejected_lengths.append(len(rejected_ids))
        chosen_full_lengths.append(prompt_length + len(chosen_ids))
        rejected_full_lengths.append(prompt_length + len(rejected_ids))

    if _percentile(sft_lengths, 0.50) < 20 or _percentile(prompt_lengths, 0.50) < 20:
        raise ValueError("token audit produced implausibly short training prompts; check tokenizer return type and schema")

    return {
        "schema_version": "1.0",
        "model_path": str(model_path),
        "tokenizer_config_sha256": hashlib.sha256((model_path / "tokenizer_config.json").read_bytes()).hexdigest(),
        "sft": summarize(sft_lengths, limits),
        "dpo_prompt": summarize(prompt_lengths, limits),
        "dpo_chosen": summarize(chosen_lengths, limits),
        "dpo_rejected": summarize(rejected_lengths, limits),
        "dpo_prompt_plus_chosen": summarize(chosen_full_lengths, limits),
        "dpo_prompt_plus_rejected": summarize(rejected_full_lengths, limits),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--sft-root", type=Path, required=True)
    parser.add_argument("--dpo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, action="append", default=[1024, 1536, 2048, 3072, 4096, 4608])
    args = parser.parse_args()
    report = audit(args.model_path, args.sft_root, args.dpo_root, sorted(set(args.limit)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(report, output_file, ensure_ascii=False, indent=2, sort_keys=True)
        output_file.write("\n")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
