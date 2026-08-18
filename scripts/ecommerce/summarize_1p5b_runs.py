#!/usr/bin/env python3
"""Aggregate immutable 1.5B run artifacts into a machine-readable comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List


def _json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _number(value: str) -> float:
    return float(value.strip())


def hardware_summary(path: Path, gpu_index: str | None) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    if gpu_index is not None:
        rows = [row for row in rows if row["index"].strip() == gpu_index]
    if not rows:
        raise ValueError(f"no GPU samples found in {path} for GPU {gpu_index}")
    active = [row for row in rows if _number(row["memory_used_mib"]) > 0] or rows
    return {
        "sample_count": len(rows),
        "gpu_index": rows[0]["index"].strip(),
        "gpu_name": rows[0]["name"].strip(),
        "peak_utilization_pct": max(_number(row["utilization_gpu_pct"]) for row in rows),
        "mean_active_utilization_pct": mean(_number(row["utilization_gpu_pct"]) for row in active),
        "peak_memory_mib": max(_number(row["memory_used_mib"]) for row in rows),
        "peak_power_w": max(_number(row["power_draw_w"]) for row in rows),
        "peak_temperature_c": max(_number(row["temperature_c"]) for row in rows),
    }


def summarize_run(run_dir: Path) -> Dict[str, Any]:
    manifest = _json(run_dir / "manifest.json")
    adapter_dir = run_dir / "adapter"
    train = _json(adapter_dir / "train_results.json")
    evaluation = _json(adapter_dir / "eval_results.json")
    adapter_config = _json(adapter_dir / "adapter_config.json")
    gpu_value = manifest.get("runtime", {}).get("cuda_visible_devices")
    gpu_index = str(gpu_value).split(",", 1)[0] if gpu_value is not None else None
    adapter_path = adapter_dir / "adapter_model.safetensors"
    return {
        "run_id": manifest["run_id"],
        "git_commit": manifest["git"]["commit"],
        "git_dirty": manifest["git"]["dirty"],
        "command": manifest["command"],
        "lora": {
            "rank": adapter_config.get("r"),
            "alpha": adapter_config.get("lora_alpha"),
            "target_modules": sorted(adapter_config.get("target_modules", [])),
            "adapter_bytes": adapter_path.stat().st_size,
        },
        "train": train,
        "evaluation": evaluation,
        "hardware": hardware_summary(run_dir / "hardware.csv", gpu_index),
    }


def summarize(run_dirs: Iterable[Path]) -> Dict[str, Any]:
    runs = [summarize_run(path) for path in run_dirs]
    if len({run["run_id"] for run in runs}) != len(runs):
        raise ValueError("run IDs must be unique")
    return {"schema_version": "1.0", "run_count": len(runs), "runs": runs}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(report, output_file, ensure_ascii=False, indent=2, sort_keys=True)
        output_file.write("\n")
    print(json.dumps({"run_count": report["run_count"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
