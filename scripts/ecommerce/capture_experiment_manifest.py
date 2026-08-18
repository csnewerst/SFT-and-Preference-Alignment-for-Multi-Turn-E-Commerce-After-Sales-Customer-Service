#!/usr/bin/env python3
"""Capture immutable inputs and environment metadata for an experiment run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[2]


def _run(command: List[str]) -> str:
    try:
        return subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for path in paths:
        resolved = path.resolve()
        if resolved.is_dir():
            children = sorted(child for child in resolved.rglob("*") if child.is_file())
            digest = hashlib.sha256()
            total_bytes = 0
            for child in children:
                relative = str(child.relative_to(resolved)).replace("\\", "/")
                file_hash = _sha256(child)
                digest.update(f"{relative}\0{file_hash}\n".encode("utf-8"))
                total_bytes += child.stat().st_size
            result.append(
                {
                    "path": str(path),
                    "type": "directory",
                    "file_count": len(children),
                    "bytes": total_bytes,
                    "sha256": digest.hexdigest(),
                }
            )
            continue
        if not resolved.is_file():
            raise FileNotFoundError(path)
        result.append(
            {"path": str(path), "type": "file", "bytes": resolved.stat().st_size, "sha256": _sha256(resolved)}
        )
    return result


def capture(output_dir: Path, run_id: str, config: Path, inputs: Iterable[Path], command: str) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    config_payload = json.loads(config.read_text(encoding="utf-8"))
    git_prefix = ["git", "-c", f"safe.directory={ROOT}"]
    git_commit = _run([*git_prefix, "rev-parse", "HEAD"])
    git_status = _run([*git_prefix, "status", "--short"])
    nvidia_smi = _run([
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,driver_version",
        "--format=csv,noheader",
    ])
    try:
        import torch

        torch_info: Dict[str, Any] = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cudnn_version": torch.backends.cudnn.version(),
        }
    except Exception as exc:  # pragma: no cover - depends on runtime image
        torch_info = {"unavailable": str(exc)}

    manifest = {
        "schema_version": "1.0",
        "run_id": run_id,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "config_path": str(config),
        "config_sha256": _sha256(config),
        "config": config_payload,
        "inputs": _files(inputs),
        "git": {"commit": git_commit, "dirty": bool(git_status), "status": git_status},
        "runtime": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "executable": sys.executable,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch": torch_info,
            "nvidia_smi": nvidia_smi,
        },
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(manifest, output_file, ensure_ascii=False, indent=2, sort_keys=True)
        output_file.write("\n")
    (output_dir / "command.sh").write_text(command.rstrip() + "\n", encoding="utf-8", newline="\n")
    (output_dir / "git_status.txt").write_text(git_status + "\n", encoding="utf-8", newline="\n")
    packages = sorted(
        f"{distribution.metadata.get('Name', 'unknown')}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
    )
    (output_dir / "environment.txt").write_text("\n".join(packages) + "\n", encoding="utf-8", newline="\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", default=[])
    parser.add_argument("--command", required=True)
    args = parser.parse_args()
    manifest = capture(args.output_dir, args.run_id, args.config, args.input, args.command)
    print(json.dumps({"run_id": manifest["run_id"], "git_commit": manifest["git"]["commit"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
