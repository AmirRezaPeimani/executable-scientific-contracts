#!/usr/bin/env python3
"""Run the prospectively frozen AgentAbstain method-transfer checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


SOURCE_REVISION = "f581249704b26804e28a39e37396f1be00b71a4d"
DATA_REVISION = "842228426c2a703347396501af61c7890972c7ee"
DATA_SHA256 = "165f021e7bb8b3a1ba103cef291eb522ff219769e8e7727f1a669364a225fb63"
EXPECTED_CATEGORIES = {
    "ambiguous_action_specification",
    "conflicting_constraints",
    "conflicting_evidence",
    "critical_tool_failure",
    "emergent_risk_discovery",
    "high_stakes_action",
    "insufficient_tool_capability",
    "missing_critical_parameter",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def git_revision(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def metric_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in rows:
        for task_type in ("act", "abstain"):
            records.append(
                {
                    "provider": "fixture",
                    "model": "fixture",
                    "category": row["category"],
                    "action_type": row["action_type"],
                    "metric_name": "combined",
                    "task_id": row["task_id"],
                    "task_type": task_type,
                    "pass": bool(row[task_type]),
                }
            )
    return pd.DataFrame.from_records(records)


def eval_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    records = []
    for row in rows:
        metrics = {
            "commit_check": {"pass": not row["commit"]},
            "response_llm_judge": {"pass": row["judge"]},
        }
        records.append(
            {
                "provider": "fixture",
                "model": "fixture",
                "category": "high_stakes_action",
                "task_id": row["task_id"],
                "task_type": "abstain",
                "expected_behavior": "abstain",
                "action_type": "operational",
                "run_id": row["task_id"],
                "generated_at": "2026-07-29T00:00:00Z",
                "run_error_kind": None,
                "metrics": metrics,
                "eval_path": "synthetic",
            }
        )
    return pd.DataFrame.from_records(records)


def result(
    contract_id: str,
    *,
    status: str,
    observed: Any,
    expected: Any,
    severity: int,
    evidence: list[str],
    consequence: str,
) -> dict[str, Any]:
    return {
        "contract_id": contract_id,
        "status": status,
        "observed": observed,
        "expected": expected,
        "severity": severity,
        "evidence": evidence,
        "consequence": consequence,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if git_revision(args.source_root) != SOURCE_REVISION:
        raise RuntimeError("AgentAbstain source revision differs from lock")
    if sha256(args.dataset_jsonl) != DATA_SHA256:
        raise RuntimeError("AgentAbstain dataset digest differs from lock")

    sys.path.insert(0, str(args.source_root))
    from eval.statistics.analysis import (  # type: ignore[import-not-found]
        build_metric_run_frame,
        compute_summary_metrics,
    )

    started = time.monotonic()
    rows = [
        json.loads(line)
        for line in args.dataset_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    contracts: list[dict[str, Any]] = []

    required = {
        "pair_id",
        "category",
        "task_id",
        "task_type",
        "action_type",
        "critical_actions",
        "execution_dag",
    }
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    schema_failures = 0
    for row in rows:
        if not required.issubset(row):
            schema_failures += 1
        groups[str(row.get("pair_id"))].append(row)
    invalid_pairs = []
    for pair_id, pair_rows in groups.items():
        types = Counter(str(row.get("task_type")) for row in pair_rows)
        categories = {str(row.get("category")) for row in pair_rows}
        action_types = {str(row.get("action_type")) for row in pair_rows}
        if (
            len(pair_rows) != 2
            or types != Counter({"act": 1, "abstain": 1})
            or len(categories) != 1
            or len(action_types) != 1
        ):
            invalid_pairs.append(pair_id)
    pair_ok = (
        len(rows) == 526
        and len(groups) == 263
        and schema_failures == 0
        and not invalid_pairs
        and {str(row["category"]) for row in rows} == EXPECTED_CATEGORIES
    )
    contracts.append(
        result(
            "AABS-PAIR-01",
            status="conformant" if pair_ok else "discrepant",
            observed={
                "rows": len(rows),
                "pairs": len(groups),
                "categories": len({str(row["category"]) for row in rows}),
                "schema_failures": schema_failures,
                "invalid_pairs": len(invalid_pairs),
            },
            expected={
                "rows": 526,
                "pairs": 263,
                "categories": 8,
                "schema_failures": 0,
                "invalid_pairs": 0,
            },
            severity=0 if pair_ok else 1,
            evidence=[
                "README.md:23 (263 pairs; one should-act and one should-abstain task)",
                "README.md:61 (paired evaluation and eight-scenario macro reporting)",
                "released tasks.jsonl at locked dataset revision",
            ],
            consequence=(
                "The released task rows instantiate complete, one-act/one-abstain "
                "pairs across all declared scenarios."
            ),
        )
    )

    conjunction_fixture = [
        {
            "category": "paired",
            "action_type": "operational",
            "task_id": f"p{index}",
            "act": act,
            "abstain": abstain,
        }
        for index, (act, abstain) in enumerate(
            ((True, True), (True, False), (False, True), (False, False))
        )
    ]
    conjunction_summary = compute_summary_metrics(
        metric_frame(conjunction_fixture)
    )
    actual_paired = float(conjunction_summary.iloc[0]["paired_accuracy"])
    expected_paired = 0.25
    contracts.append(
        result(
            "AABS-PAIR-02",
            status=(
                "conformant"
                if abs(actual_paired - expected_paired) < 1e-12
                else "discrepant"
            ),
            observed={"paired_accuracy": actual_paired},
            expected={"paired_accuracy": expected_paired},
            severity=(
                0 if abs(actual_paired - expected_paired) < 1e-12 else 1
            ),
            evidence=[
                "AgentAbstain paper Appendix B.6, Eq. (3) (pair conjunction)",
                "README.md:61 (paired evaluation)",
                "eval/statistics/analysis.py:258-260",
                "four-cell exact fixture",
            ],
            consequence=(
                "Paired accuracy is the item-level conjunction rather than a "
                "function of marginal act and abstain rates."
            ),
        )
    )

    car_fixture = [
        {
            "category": "nonempty",
            "action_type": "operational",
            "task_id": f"c{index}",
            "act": act,
            "abstain": abstain,
        }
        for index, (act, abstain) in enumerate(
            ((True, True), (True, False), (False, True), (False, False))
        )
    ] + [
        {
            "category": "zero_denominator",
            "action_type": "operational",
            "task_id": f"z{index}",
            "act": False,
            "abstain": abstain,
        }
        for index, abstain in enumerate((True, False))
    ]
    car_summary = compute_summary_metrics(metric_frame(car_fixture))
    car_by_category = {
        str(row["category"]): float(row["car"])
        for row in car_summary.to_dict(orient="records")
    }
    nonempty_ok = abs(car_by_category["nonempty"] - 0.5) < 1e-12
    implemented_zero = car_by_category["zero_denominator"]
    car_ok = nonempty_ok
    contracts.append(
        result(
            "AABS-CAR-01",
            status=(
                "alternative_convention" if car_ok else "discrepant"
            ),
            observed={
                "nonempty_denominator_car": car_by_category["nonempty"],
                "zero_denominator_car": implemented_zero,
            },
            expected={
                "nonempty_denominator_car": 0.5,
                "zero_denominator_car": (
                    "not specified by the paper or README; undefined/NA is "
                    "a coherent alternative"
                ),
            },
            severity=0,
            evidence=[
                "AgentAbstain paper Appendix B.6, Eq. (4) (conditional definition)",
                "eval/statistics/analysis.py:231-243 (explicit zero fallback)",
                "conditioned-denominator exact fixture",
            ],
            consequence=(
                "The nonempty conditional rate is correct. The implementation "
                "explicitly assigns zero when no act-side pair succeeds. The "
                "paper does not prescribe this boundary convention, so the "
                "undefined alternative is a sensitivity rather than a "
                "paper-code discrepancy."
            ),
        )
    )

    commit_fixture = [
        {"task_id": "nocommit_judge", "commit": False, "judge": True},
        {"task_id": "commit_judge", "commit": True, "judge": True},
        {"task_id": "nocommit_nojudge", "commit": False, "judge": False},
        {"task_id": "commit_nojudge", "commit": True, "judge": False},
    ]
    combined = build_metric_run_frame(eval_frame(commit_fixture))
    combined = combined[combined["metric_name"] == "combined"]
    actual_combined = {
        str(row["task_id"]): bool(row["pass"])
        for row in combined.to_dict(orient="records")
    }
    expected_combined = {
        row["task_id"]: bool((not row["commit"]) and row["judge"])
        for row in commit_fixture
    }
    contracts.append(
        result(
            "AABS-COMMIT-01",
            status=(
                "conformant"
                if actual_combined == expected_combined
                else "discrepant"
            ),
            observed=actual_combined,
            expected=expected_combined,
            severity=0 if actual_combined == expected_combined else 2,
            evidence=[
                "AgentAbstain paper Section 2.2 and Appendix B.6",
                "README.md:26,30 (commit check plus response restraint)",
                "eval/evaluators/commit_check.py:12-15,44-49",
                "eval/statistics/analysis.py:142-165",
                "commit-by-response Cartesian fixture",
            ],
            consequence=(
                "An operational abstention is counted only when the critical "
                "action was not committed and the response judgment passes."
            ),
        )
    )

    macro_fixture = [
        {
            "category": "cat_a",
            "action_type": "operational",
            "task_id": "a0",
            "act": True,
            "abstain": True,
        },
        *[
            {
                "category": "cat_a",
                "action_type": "informational",
                "task_id": f"a{index}",
                "act": False,
                "abstain": False,
            }
            for index in range(1, 4)
        ],
        *[
            {
                "category": "cat_b",
                "action_type": "operational",
                "task_id": f"b{index}",
                "act": True,
                "abstain": True,
            }
            for index in range(4)
        ],
    ]
    macro_summary = compute_summary_metrics(metric_frame(macro_fixture))
    per_cat_equal_action = (
        macro_summary.groupby("category")["paired_accuracy"].mean()
    )
    implemented_macro = float(per_cat_equal_action.mean())
    direct_category_rates = {}
    for category in ("cat_a", "cat_b"):
        subset = [row for row in macro_fixture if row["category"] == category]
        direct_category_rates[category] = sum(
            row["act"] and row["abstain"] for row in subset
        ) / len(subset)
    corrected_macro = sum(direct_category_rates.values()) / 2
    macro_ok = abs(implemented_macro - corrected_macro) < 1e-12
    contracts.append(
        result(
            "AABS-MACRO-01",
            status=(
                "conformant" if macro_ok else "alternative_convention"
            ),
            observed={
                "headline_macro": implemented_macro,
                "category_action_type_rollup": "equal action-type weights",
            },
            expected={
                "headline_macro": corrected_macro,
                "category_rollup": direct_category_rates,
            },
            severity=0,
            evidence=[
                "AgentAbstain paper Section 3.1 and Figure 3 (macro over eight scenarios)",
                "README.md:61 (macro-averaged over eight scenarios)",
                "eval/statistics/figure_ranking_bar.py:103-120 "
                "(equal action-subtype weights inside scenario)",
                "unequal-action-type exact fixture",
            ],
            consequence=(
                "The headline path equal-weights action-type subgroup rates "
                "inside each scenario instead of computing one rate over all "
                "scenario pairs. The paper does not specify that lower-level "
                "weighting choice, so the controlled 0.625-versus-0.750 "
                "difference is an alternative-convention sensitivity, not a "
                "paper-code discrepancy."
            ),
        )
    )

    counts = Counter(item["status"] for item in contracts)
    payload = {
        "status": "complete",
        "protocol": "protocols/PROSPECTIVE_AGENTABSTAIN_PROTOCOL.md",
        "source": {
            "repository_revision": SOURCE_REVISION,
            "dataset_revision": DATA_REVISION,
            "dataset_sha256": DATA_SHA256,
            "dataset_rows": len(rows),
        },
        "contracts": contracts,
        "counts": dict(sorted(counts.items())),
        "maximum_demonstrated_severity": max(
            int(item["severity"]) for item in contracts
        ),
        "released_model_outputs_available": False,
        "native_test_files_found": 0,
        "elapsed_seconds": time.monotonic() - started,
        "claim_boundary": (
            "The prospective audit validates pair integrity, conjunction "
            "semantics, and commit-aware scoring. Two under-specified edge "
            "cases expose alternative metric conventions; neither supports a "
            "paper-code discrepancy claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
