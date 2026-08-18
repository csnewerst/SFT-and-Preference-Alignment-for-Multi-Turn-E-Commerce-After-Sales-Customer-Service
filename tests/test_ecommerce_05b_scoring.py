import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "ecommerce"))

from score_05b_comparison import score


def test_scores_first_action_and_missing_order_behavior(tmp_path):
    prompts = {
        "prompts": [
            {"id": "ask", "expected_behavior": "ask_order_id"},
            {"id": "query", "expected_tool": "query_order_status"},
        ]
    }
    rows = [
        {"stage": "sft", "prompt_id": "ask", "output": "请先提供脱敏后的订单号。"},
        {"stage": "sft", "prompt_id": "query", "output": 'Action: query_order_status\nAction Input: {"order_id":"EC-X"}'},
        {"stage": "initial", "prompt_id": "ask", "output": '{"order_id":"EC-FAKE"}'},
        {"stage": "initial", "prompt_id": "query", "output": '```json\n{"action":"create_after_sales_request"}\n```'},
    ]
    prompts_path = tmp_path / "prompts.json"
    comparison_path = tmp_path / "comparison.jsonl"
    prompts_path.write_text(json.dumps(prompts, ensure_ascii=False), encoding="utf-8")
    comparison_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")

    summary = score(comparison_path, prompts_path)

    assert summary["stages"]["sft"]["accuracy"] == 1.0
    assert summary["stages"]["initial"]["accuracy"] == 0.0
