#!/usr/bin/env python3
"""Additional path-relevant benign transformations, separate from first run.

This supplemental diagnostic keeps the first frozen challenge untouched. It
tests equivalent rewrites on the exact expression exercised by each contract.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from agentabstain_analysis import digest, extraction, load_analysis, metric_frame, rank, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = (args.source_root / "eval/statistics/analysis.py").read_text()
    ranking_full = (args.source_root / "eval/statistics/figure_ranking_bar.py").read_text()
    ranking, _anchors = extraction(ranking_full, ["combined", "per_cat", "macro"], inner="main")
    fixture = [{"category": "truth", "action_type": "operational", "task_id": str(i), "act": a, "abstain": b} for i, (a, b) in enumerate([(True, True), (True, False), (False, True), (False, False)])]
    transforms = [
        ("paired_operand_order", '(complete_pairs["act"] & complete_pairs["abstain"]).mean()', '(complete_pairs["abstain"] & complete_pairs["act"]).mean()'),
        ("conditional_mean_to_sum_count", 'car_value = conditioned["abstain"].mean()', 'car_value = conditioned["abstain"].sum() / len(conditioned)'),
        ("combined_operand_order", 'combined_pass = commit_pass and judge_pass', 'combined_pass = judge_pass and commit_pass'),
        ("informational_parentheses", 'combined_pass = judge_pass if isinstance(judge_pass, bool) else pd.NA', 'combined_pass = (judge_pass) if isinstance(judge_pass, bool) else pd.NA')
    ]
    write_json(args.output.with_name("benign_supplement_freeze.json"), {"script_sha256": digest(Path(__file__)), "transforms": transforms, "expected": {"paired_accuracy": .25, "car": .5}, "frozen_before_this_supplement": True, "prior_challenge_already_observed": True})
    checks = []
    original = load_analysis(source)
    from agentabstain_analysis import eval_frame
    combined_fixture = eval_frame([{"task_id": str(i), "commit_pass": a, "judge_pass": b} for i, (a, b) in enumerate([(True, True), (True, False), (False, True), (False, False)])] + [{"task_id": "info", "action_type": "informational", "commit_pass": False, "judge_pass": True}])
    for name, old, new in transforms:
        assert source.count(old) == 1
        mutated = source.replace(old, new)
        local = load_analysis(mutated)
        actual = local["compute_summary_metrics"](metric_frame(fixture))
        baseline = original["compute_summary_metrics"](metric_frame(fixture))
        pd.testing.assert_frame_equal(actual, baseline)
        pd.testing.assert_frame_equal(local["build_metric_run_frame"](combined_fixture), original["build_metric_run_frame"](combined_fixture))
        checks.append({"id": name, "decision": "pass", "source_sha256": digest(source), "executed_sha256": digest(mutated), "old": old, "new": new})
    benign_rank = ranking.replace('per_cat.groupby(["provider", "model"])[["act", "abstain", "paired", "car"]]', 'per_cat.groupby(["model", "provider"])[["act", "abstain", "paired", "car"]]')
    a = rank(original["compute_summary_metrics"](metric_frame(fixture)), ranking)
    b = rank(original["compute_summary_metrics"](metric_frame(fixture)), benign_rank)
    assert float(a.iloc[0]["paired"]) == float(b.iloc[0]["paired"])
    checks.append({"id": "ranking_group_key_order", "decision": "pass", "source_sha256": digest(ranking), "executed_sha256": digest(benign_rank)})
    write_json(args.output, {"supplemental": True, "does_not_replace_first_execution": True, "checks": checks, "false_alarms": 0, "cost_cad": 0})
    print(json.dumps({"path_relevant_benign_controls": len(checks), "false_alarms": 0}))


if __name__ == "__main__":
    main()
