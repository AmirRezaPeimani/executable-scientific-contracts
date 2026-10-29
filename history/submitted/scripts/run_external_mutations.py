#!/usr/bin/env python3
"""Run the frozen externally sourced mutation-transfer set."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


def wilson(successes: int, total: int) -> list[float]:
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = proportion + z * z / (2 * total)
    radius = z * math.sqrt(
        proportion * (1 - proportion) / total
        + z * z / (4 * total * total)
    )
    return [
        (centre - radius) / denominator,
        (centre + radius) / denominator,
    ]


def exact_sequence(observed: list[float], expected: list[float]) -> bool:
    return len(observed) == len(expected) and all(
        abs(left - right) < 1e-12
        for left, right in zip(observed, expected)
    )


def run_case(
    mutation_id: str,
    family: str,
    fixture: str,
    clean: Any,
    mutant: Any,
    oracle: Callable[[Any], bool],
    transformation: str,
) -> dict[str, Any]:
    clean_passes = bool(oracle(clean))
    mutant_passes = bool(oracle(mutant))
    return {
        "id": mutation_id,
        "external_family": family,
        "fixture": fixture,
        "transformation": transformation,
        "clean_passes": clean_passes,
        "mutant_passes": mutant_passes,
        "detected": clean_passes and not mutant_passes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--execution",
        required=True,
        choices=("first", "corrected"),
        help="reproduce the frozen first execution or the fixture-corrected rerun",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    gamma = 0.9
    horizon = 4
    target = [gamma ** (horizon - 1 - t) for t in range(horizon)]
    target_oracle = lambda values: exact_sequence(values, target)

    rewards = [1.0, 0.0, 1.0, 0.0, 1.0]
    threshold = 1.0 - 1e-6 if args.execution == "first" else 1.0
    metric_expected = 3 / 5
    metric_oracle = lambda value: abs(value - metric_expected) < 1e-12

    pair_cells = [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ]
    paired_expected = 0.25
    paired_oracle = lambda value: abs(value - paired_expected) < 1e-12

    logging_clean = {
        "messages": [
            {"role": "tool", "content": "found"},
            {"role": "assistant", "content": "DONE"},
        ]
    }
    logging_oracle = lambda value: bool(
        value["messages"]
        and value["messages"][-1].get("role") == "assistant"
        and value["messages"][-1].get("content")
    )

    schema_clean = {
        "required_arguments": ["record_id"],
        "arguments": {"record_id": "R-1"},
    }
    schema_oracle = lambda value: all(
        argument in value["arguments"]
        for argument in value["required_arguments"]
    )

    commit_cells = [
        (False, True),
        (True, True),
        (False, False),
        (True, False),
    ]
    commit_expected = [True, False, False, False]
    commit_oracle = lambda values: values == commit_expected

    cases = [
        run_case(
            "XM-01",
            "arithmetic_operator_replacement",
            "discounted_terminal_target",
            target,
            [gamma ** (horizon - 1 + t) for t in range(horizon)],
            target_oracle,
            "horizon - 1 - t -> horizon - 1 + t",
        ),
        run_case(
            "XM-02",
            "relational_operator_replacement",
            "thresholded_pass_rate",
            sum(reward >= threshold for reward in rewards) / len(rewards),
            sum(reward > threshold for reward in rewards) / len(rewards),
            metric_oracle,
            "reward >= threshold -> reward > threshold",
        ),
        run_case(
            "XM-03",
            "constant_value_replacement",
            "terminal_relative_index",
            target,
            [gamma ** (horizon - t) for t in range(horizon)],
            target_oracle,
            "terminal offset 1 -> 0",
        ),
        run_case(
            "XM-04",
            "unary_operator_replacement",
            "signed_target_values",
            target,
            [-value for value in target],
            target_oracle,
            "value -> -value",
        ),
        run_case(
            "XM-05",
            "logical_operator_replacement",
            "paired_success",
            sum(act and abstain for act, abstain in pair_cells) / 4,
            sum(act or abstain for act, abstain in pair_cells) / 4,
            paired_oracle,
            "act and abstain -> act or abstain",
        ),
        run_case(
            "XM-06",
            "statement_deletion",
            "terminal_response_logging",
            logging_clean,
            {"messages": logging_clean["messages"][:-1]},
            logging_oracle,
            "delete final response append",
        ),
        run_case(
            "XM-07",
            "statement_deletion",
            "required_tool_arguments",
            schema_clean,
            {
                "required_arguments": ["record_id"],
                "arguments": {},
            },
            schema_oracle,
            "delete required-argument insertion",
        ),
        run_case(
            "XM-08",
            "logical_operator_replacement",
            "commit_aware_abstention",
            [(not commit) and judge for commit, judge in commit_cells],
            [(not commit) or judge for commit, judge in commit_cells],
            commit_oracle,
            "not commit and judge -> not commit or judge",
        ),
    ]

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_family[case["external_family"]].append(case)
    family_summary = {}
    for family, family_cases in sorted(by_family.items()):
        detected = sum(bool(case["detected"]) for case in family_cases)
        family_summary[family] = {
            "detected": detected,
            "total": len(family_cases),
        }

    detected = sum(bool(case["detected"]) for case in cases)
    payload = {
        "protocol": "protocols/EXTERNAL_MUTATION_TRANSFER_PROTOCOL.md",
        "execution": args.execution,
        "external_sources": [
            "Just 2014, Major mutation framework, Section 2.2",
            "PIT public mutator catalog",
        ],
        "mutations": cases,
        "overall": {
            "detected": detected,
            "total": len(cases),
            "rate": detected / len(cases),
            "wilson_95": wilson(detected, len(cases)),
        },
        "by_family": family_summary,
        "claim_boundary": (
            "Operator-transfer evidence for eight prospectively frozen "
            "single-site mutants; not independent authorship or a prevalence sample."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["overall"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
