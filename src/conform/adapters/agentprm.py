from __future__ import annotations

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
        "agentprm",
        status,
        observed,
        expected,
        severity,
        consequence,
        evidence,
        started,
    )


def audit_agentprm(root: Path, specs: list[dict[str, Any]]) -> list[ContractResult]:
    del specs
    target_file = root / "scripts/dataproc/compute_prm_target.py"
    preference_file = root / "scripts/dataproc/compute_prm_preference_target.py"
    rollout_file = root / "scripts/dataproc/rollout_alfworld.py"
    source = target_file.read_text(encoding="utf-8")
    preference_source = preference_file.read_text(encoding="utf-8")
    rollout_source = rollout_file.read_text(encoding="utf-8")
    results: list[ContractResult] = []

    started = time.time()
    implemented_forward = "gamma ** t" in source
    results.append(
        _result(
            "APRM-TEMP-01",
            "discrepant" if implemented_forward else "conformant",
            "gamma**t" if implemented_forward else "terminal-relative exponent",
            "gamma**(T-1-t)",
            1 if implemented_forward else 0,
            (
                "Per-state target values change on controlled non-unit-discount "
                "fixtures; released trajectories are unavailable, so no result "
                "ranking consequence is assigned."
            ),
            [f"{target_file}:55"],
            started,
        )
    )

    started = time.time()
    gamma = 1.0
    implemented = [gamma**t for t in range(4)]
    expected = [gamma ** (3 - t) for t in range(4)]
    results.append(
        _result(
            "APRM-TEMP-02",
            "conformant" if implemented == expected else "discrepant",
            implemented,
            expected,
            0,
            "Unit discount is an exact negative control.",
            [f"{target_file}:55"],
            started,
        )
    )

    started = time.time()
    gamma = 0.9
    implemented = [gamma**t for t in range(4)]
    expected = [gamma ** (3 - t) for t in range(4)]
    late_non_decreasing = all(
        expected[index] <= expected[index + 1] for index in range(3)
    )
    implementation_satisfies = all(
        implemented[index] <= implemented[index + 1] for index in range(3)
    )
    results.append(
        _result(
            "APRM-TEMP-03",
            "conformant" if implementation_satisfies else "discrepant",
            {
                "weights": implemented,
                "late_non_decreasing": implementation_satisfies,
            },
            {"weights": expected, "late_non_decreasing": late_non_decreasing},
            1 if not implementation_satisfies else 0,
            "The direction of temporal credit is reversed on the exact fixture.",
            [f"{target_file}:55", f"{preference_file}:55"],
            started,
        )
    )

    started = time.time()
    duplicated = (
        '"action": env_data["observation"][t]' in rollout_source
        or "'action': env_data['observation'][t]" in rollout_source
    )
    results.append(
        _result(
            "APRM-SCHEMA-01",
            "discrepant" if duplicated else "conformant",
            "action copied from observation" if duplicated else "distinct action",
            "recorded policy action",
            1 if duplicated else 0,
            (
                "The saved history value is wrong; no released affected trace is "
                "present locally to establish a metric consequence."
            ),
            [f"{rollout_file}:66"],
            started,
        )
    )

    started = time.time()
    outcome = [1.0, 0.6, 0.2, 0.0]
    implemented_values = [reward * gamma**t for t, reward in enumerate(outcome)]
    corrected_values = [
        reward * gamma ** (len(outcome) - 1 - t)
        for t, reward in enumerate(outcome)
    ]
    max_delta = max(
        abs(left - right)
        for left, right in zip(implemented_values, corrected_values)
    )
    results.append(
        _result(
            "APRM-METRIC-01",
            "discrepant" if max_delta > 1e-12 else "conformant",
            {"values": implemented_values, "mean": sum(implemented_values) / 4},
            {"values": corrected_values, "mean": sum(corrected_values) / 4},
            1 if max_delta > 1e-12 else 0,
            (
                "Corrected fixture values differ. Severity remains value-level "
                "because released result traces are not available."
            ),
            [f"{target_file}:55"],
            started,
        )
    )
    return results
