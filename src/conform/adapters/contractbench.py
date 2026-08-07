from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from conform.core import ContractResult, timed_result


EXPECTED_EPISODES = 99
EXPECTED_SUCCESSES = 56
RESULT_DIR = "huggingface__Qwen__Qwen3.5-9B"


def _result(
    contract_id: str,
    status: str,
    observed: Any,
    expected: Any,
    severity: int,
    consequence: str,
    evidence: list[str],
    started: float,
) -> ContractResult:
    return timed_result(
        contract_id,
        "contractbench",
        status,
        observed,
        expected,
        severity,
        consequence,
        evidence,
        started,
    )


def _records(results_root: Path) -> list[tuple[Path, dict[str, Any], dict[str, Any]]]:
    rows = []
    for reward_path in sorted((results_root / RESULT_DIR).rglob("reward.json")):
        trace_path = reward_path.with_name("agent_trace.json")
        if not trace_path.exists():
            continue
        rows.append(
            (
                reward_path,
                json.loads(reward_path.read_text(encoding="utf-8")),
                json.loads(trace_path.read_text(encoding="utf-8")),
            )
        )
    return rows


def _assistant_messages(trace: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        message
        for message in trace.get("messages", [])
        if message.get("role") == "assistant"
        and (
            str(message.get("content") or "").strip()
            or bool(message.get("tool_calls"))
        )
    ]


def audit_contractbench(
    root: Path, specs: list[dict[str, Any]], data_root: Path | None = None
) -> list[ContractResult]:
    del specs
    release = data_root or root.parent / "contractbench"
    rows = _records(release / "results")
    if not rows:
        raise FileNotFoundError("pinned ContractBench result slice is missing")
    results: list[ContractResult] = []

    started = time.time()
    unlogged = [
        path
        for path, _reward, trace in rows
        if trace.get("total_output_tokens", 0) > 0 and not _assistant_messages(trace)
    ]
    results.append(
        _result(
            "CBEN-LOG-01",
            "discrepant" if unlogged else "conformant",
            {
                "episodes": len(rows),
                "output_without_assistant_message": len(unlogged),
            },
            {"output_without_assistant_message": 0},
            1 if unlogged else 0,
            (
                "Generated output is not fully represented in the released trace; "
                "the number of corrected successes is not identifiable."
            ),
            [
                "released-results/selected-9b-slice",
                f"{root / 'tests/test_docker_runner_loop.py'}:77",
            ],
            started,
        )
    )

    started = time.time()
    success_rows = [
        (path, trace)
        for path, reward, trace in rows
        if bool(reward.get("success"))
    ]
    unrecoverable_success = [
        path for path, trace in success_rows if not _assistant_messages(trace)
    ]
    results.append(
        _result(
            "CBEN-LOG-02",
            "conformant" if not unrecoverable_success else "discrepant",
            {
                "successes": len(success_rows),
                "unrecoverable_successes": len(unrecoverable_success),
            },
            {"unrecoverable_successes": 0},
            0 if not unrecoverable_success else 2,
            "The released slice has no success event lacking a trace.",
            ["released-results/selected-9b-slice"],
            started,
        )
    )

    started = time.time()
    schema_failures = []
    for path, reward, trace in rows:
        reward_ok = {
            "reward",
            "task_id",
            "suite",
            "success",
        }.issubset(reward)
        trace_ok = {
            "task",
            "run_id",
            "turns",
            "total_output_tokens",
            "messages",
        }.issubset(trace)
        if not (reward_ok and trace_ok):
            schema_failures.append(path)
    results.append(
        _result(
            "CBEN-SCHEMA-01",
            "conformant" if not schema_failures else "discrepant",
            {"valid": len(rows) - len(schema_failures), "total": len(rows)},
            {"valid": len(rows), "total": len(rows)},
            0 if not schema_failures else 1,
            "Paired reward and trajectory records satisfy the declared fields.",
            ["released-results/selected-9b-slice"],
            started,
        )
    )

    started = time.time()
    released_successes = sum(bool(reward.get("success")) for _, reward, _ in rows)
    results.append(
        _result(
            "CBEN-METRIC-01",
            (
                "conformant"
                if released_successes == EXPECTED_SUCCESSES
                else "discrepant"
            ),
            {"episodes": len(rows), "successes": released_successes},
            {"episodes": EXPECTED_EPISODES, "successes": EXPECTED_SUCCESSES},
            2 if released_successes != EXPECTED_SUCCESSES else 0,
            (
                "The released record-level aggregate does not reproduce the "
                "reported baseline count. Missing trace content prevents a "
                "defensible corrected performance estimate."
            ),
            ["released-results/selected-9b-slice", "reported-table-baseline"],
            started,
        )
    )

    started = time.time()
    direct_rate = released_successes / len(rows)
    stored_rate = sum(float(reward.get("success", False)) for _, reward, _ in rows) / len(
        rows
    )
    results.append(
        _result(
            "CBEN-METRIC-02",
            "conformant" if abs(direct_rate - stored_rate) < 1e-12 else "discrepant",
            stored_rate,
            direct_rate,
            0 if abs(direct_rate - stored_rate) < 1e-12 else 2,
            "The released slice is internally aggregatable under its own labels.",
            ["released-results/selected-9b-slice"],
            started,
        )
    )
    return results
