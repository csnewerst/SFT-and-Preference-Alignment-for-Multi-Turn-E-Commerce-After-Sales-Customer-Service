import gc
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.tool_utils import load_local_json_datasets


ROOT = Path(__file__).resolve().parents[1]


TOOLS = [
    {
        "name": "query_order_status",
        "description": "Query the current order status.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    }
]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_smoke_fixture(root: Path) -> None:
    sft_row = {
        "conversations": [
            {"from": "human", "value": "请查询订单 EC-1001 的状态。"},
            {"from": "function_call", "value": '{"name":"query_order_status","arguments":{"order_id":"EC-1001"}}'},
            {"from": "observation", "value": '{"order_id":"EC-1001","status":"shipped"}'},
            {"from": "gpt", "value": "订单 EC-1001 已发货。"},
        ],
        "tools": TOOLS,
    }
    dpo_row = {
        "conversations": [{"from": "human", "value": "订单 EC-1001 到哪了？"}],
        "chosen": "我先为你查询订单状态。",
        "rejected": "订单已经签收。",
        "tools": TOOLS,
    }
    for split in ("train", "validation"):
        _write_jsonl(root / "sft" / f"{split}.jsonl", [sft_row])
        _write_jsonl(root / "dpo" / f"{split}.jsonl", [dpo_row])


def test_ecommerce_smoke_data_loads_with_optional_tools(tmp_path):
    data_root = tmp_path / "processed"
    _build_smoke_fixture(data_root)
    sft_dir = data_root / "sft"
    dpo_dir = data_root / "dpo"

    with tempfile.TemporaryDirectory(prefix="ec-cache-") as cache_dir:
        sft = load_local_json_datasets(
            {
                "train": [str(p) for p in sorted(sft_dir.glob("train*.jsonl"))],
                "validation": [str(p) for p in sorted(sft_dir.glob("validation*.jsonl"))],
            },
            cache_dir=cache_dir,
        )
        dpo = load_local_json_datasets(
            {
                "train": [str(p) for p in sorted(dpo_dir.glob("train*.jsonl"))],
                "validation": [str(p) for p in sorted(dpo_dir.glob("validation*.jsonl"))],
            },
            cache_dir=cache_dir,
        )
        assert len(sft["train"]) > 0
        assert len(sft["validation"]) > 0
        assert "tools" in sft["train"].column_names
        assert len(dpo["train"]) > 0
        assert len(dpo["validation"]) > 0
        assert "chosen" in dpo["train"].column_names
        del sft, dpo
        gc.collect()


def test_ecommerce_smoke_validator_script_passes(tmp_path):
    data_root = tmp_path / "processed"
    _build_smoke_fixture(data_root)
    script = ROOT / "scripts" / "ecommerce" / "validate_ecommerce_smoke_data.py"
    with tempfile.TemporaryDirectory(prefix="ec-validator-cache-") as cache_dir:
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--root",
                str(data_root),
                "--cache-dir",
                cache_dir,
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    assert "validation passed" in result.stdout
