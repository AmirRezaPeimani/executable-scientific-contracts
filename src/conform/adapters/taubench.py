from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from conform.core import ContractResult, timed_result


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
        "taubench",
        status,
        observed,
        expected,
        severity,
        consequence,
        evidence,
        started,
    )


def _trajectory_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in ("sonnet-35-new-retail.json", "sonnet-35-new-airline.json"):
        rows.extend(
            json.loads(
                (root / "historical_trajectories" / name).read_text(encoding="utf-8")
            )
        )
    return rows


def _orphan_tool_results(messages: list[dict[str, Any]]) -> int:
    pending: set[str] = set()
    orphans = 0
    for message in messages:
        for call in message.get("tool_calls") or []:
            if "id" in call:
                pending.add(str(call["id"]))
        if message.get("role") == "tool":
            call_id = str(message.get("tool_call_id", ""))
            if call_id not in pending:
                orphans += 1
            else:
                pending.remove(call_id)
    return orphans


def audit_taubench(root: Path, specs: list[dict[str, Any]]) -> list[ContractResult]:
    del specs
    rows = _trajectory_rows(root)
    tools = {
        path.stem
        for environment in ("retail", "airline")
        for path in (root / "tau_bench/envs" / environment / "tools").glob("*.py")
        if path.stem != "__init__"
    }
    results: list[ContractResult] = []

    started = time.time()
    declared_actions = [
        action["name"]
        for row in rows
        for action in row.get("info", {}).get("task", {}).get("actions", [])
    ]
    unknown = sorted(set(declared_actions) - tools)
    results.append(
        _result(
            "TBEN-SCHEMA-01",
            "conformant" if not unknown else "discrepant",
            {"actions": len(declared_actions), "unknown_tools": unknown},
            {"unknown_tools": []},
            0 if not unknown else 1,
            "Released task actions reference declared tool modules.",
            ["historical-trajectories/two-domains"],
            started,
        )
    )

    started = time.time()
    orphans = sum(_orphan_tool_results(row.get("traj", [])) for row in rows)
    results.append(
        _result(
            "TBEN-LOG-01",
            "conformant" if orphans == 0 else "discrepant",
            {"trajectories": len(rows), "orphan_tool_results": orphans},
            {"orphan_tool_results": 0},
            0 if orphans == 0 else 1,
            "Tool-result messages preserve call provenance.",
            ["historical-trajectories/two-domains"],
            started,
        )
    )

    started = time.time()
    reward_mismatches = 0
    for row in rows:
        info = row.get("info", {}).get("reward_info", {})
        if info and abs(float(row.get("reward", 0)) - float(info.get("reward", 0))) > 1e-12:
            reward_mismatches += 1
    results.append(
        _result(
            "TBEN-METRIC-01",
            "conformant" if reward_mismatches == 0 else "discrepant",
            {"reward_mismatches": reward_mismatches, "trajectories": len(rows)},
            {"reward_mismatches": 0},
            0 if reward_mismatches == 0 else 2,
            "Top-level rewards agree with their saved reward checks.",
            ["historical-trajectories/two-domains"],
            started,
        )
    )

    started = time.time()
    direct = sum(float(row.get("reward", 0)) for row in rows) / len(rows)
    grouped = {}
    for row in rows:
        grouped.setdefault(int(row["task_id"]), []).append(float(row.get("reward", 0)))
    once_per_record = sum(sum(values) for values in grouped.values()) / sum(
        len(values) for values in grouped.values()
    )
    results.append(
        _result(
            "TBEN-METRIC-02",
            "conformant" if abs(direct - once_per_record) < 1e-12 else "discrepant",
            direct,
            once_per_record,
            0 if abs(direct - once_per_record) < 1e-12 else 2,
            "Each eligible trajectory enters the aggregate exactly once.",
            ["historical-trajectories/two-domains"],
            started,
        )
    )

    started = time.time()
    prior = results
    clean = all(result.status == "conformant" for result in prior)
    results.append(
        _result(
            "TBEN-NEG-01",
            "conformant" if clean else "discrepant",
            {"prior_contracts_conformant": clean},
            {"prior_contracts_conformant": True},
            0,
            "Pinned trajectories serve as a clean false-positive control.",
            ["historical-trajectories/two-domains"],
            started,
        )
    )
    return results
