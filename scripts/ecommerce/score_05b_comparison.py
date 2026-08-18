#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[2]
ACTION_RE = re.compile(r"Action:\s*([a-z_]+)")
ORDER_ID_RE = re.compile(r"EC-[A-Z0-9-]+")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        value = json.load(input_file)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _action_name(output: str) -> str | None:
    match = ACTION_RE.search(output)
    if match:
        return match.group(1)
    try:
        fenced = output.strip().removeprefix("```json").removesuffix("```").strip()
        value = json.loads(fenced)
    except (ValueError, TypeError):
        return None
    if isinstance(value, dict) and isinstance(value.get("action"), str):
        return value["action"]
    return None


def score(comparison_path: Path, prompts_path: Path) -> Dict[str, Any]:
    prompt_config = _load_json(prompts_path)
    prompts = {row["id"]: row for row in prompt_config["prompts"]}
    rows = [json.loads(line) for line in comparison_path.read_text(encoding="utf-8").splitlines() if line]
    stage_rows: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        expected = prompts[row["prompt_id"]]
        output = str(row["output"])
        action = _action_name(output)
        if expected.get("expected_behavior") == "ask_order_id":
            passed = action is None and "订单号" in output and not ORDER_ID_RE.search(output)
        else:
            passed = action == expected.get("expected_tool")
        detail = dict(row)
        detail.update({"parsed_action": action, "passed": passed})
        stage_rows[row["stage"]].append(detail)
    stages = {}
    for stage, details in sorted(stage_rows.items()):
        passed = sum(row["passed"] for row in details)
        stages[stage] = {
            "passed": passed,
            "total": len(details),
            "accuracy": passed / len(details) if details else 0.0,
            "details": details,
        }
    return {"metric": "first_action_exact", "stages": stages}


def main() -> int:
    parser = argparse.ArgumentParser(description="Score fixed-prompt first-action behavior.")
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, default=ROOT / "configs" / "ecommerce" / "smoke_prompts_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = score(args.comparison, args.prompts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(summary, output_file, ensure_ascii=False, indent=2, sort_keys=True)
        output_file.write("\n")
    compact = {
        stage: {key: value for key, value in result.items() if key != "details"}
        for stage, result in summary["stages"].items()
    }
    print(json.dumps(compact, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
