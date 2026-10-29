#!/usr/bin/env python3
"Check AgentAbstain source availability, input eligibility, and outputs."
from __future__ import annotations

import argparse
import ast
import copy
import difflib
import json
import platform
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from agentabstain_analysis import close, digest, eval_frame, extraction, load_analysis, metric_frame, rank, write_json


def extract_commit(text):
    cls = next(x for x in ast.parse(text).body if isinstance(x, ast.ClassDef) and x.name == "CommitCheckEvaluator")
    node = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "evaluate")
    return ast.get_source_segment(text, node)


def guarded_check(*, source_path, source_kind, mechanism, fixture, expected_values, patch=None):
    "Evaluate the supplied source and input."
    methods = {"native_tests": {"decision": "not_evaluated", "reason": "no relevant native suite"}, "source_pattern": {"decision": "not_evaluated", "reason": "no historical source-pattern detector"}}
    evidence = {"source_path": str(source_path), "source_exists": source_path.is_file(), "eligibility": {}, "observed": None}
    def abstain(reason):
        for method in ("schema", "behavioral_contract"):
            methods[method] = {"decision": "not_evaluated", "reason": reason}
        return {"eligible_examples": 0, "methods": methods, **evidence}
    if not source_path.is_file():
        return abstain("actual source file is unavailable")
    full_source = source_path.read_text()
    source = extract_commit(full_source) if source_kind == "commit" else extraction(full_source, ["combined", "per_cat", "macro"], inner="main")[0] if source_kind == "ranking" else full_source
    original = source
    if patch:
        old, new = patch
        if source.count(old) != 1:
            raise ValueError(f"patch anchor count {source.count(old)}")
        source = source.replace(old, new)
    evidence["implementation"] = {"full_source_sha256": digest(full_source), "original_extraction_sha256": digest(original), "executed_source_sha256": digest(source), "patch": "".join(difflib.unified_diff(original.splitlines(keepends=True), source.splitlines(keepends=True), fromfile="original", tofile="executed"))}
    if source_kind == "commit":
        bundle = SimpleNamespace(**fixture)
        evidence["eligibility"] = {"critical_actions": list(bundle.critical_actions), "count": len(bundle.critical_actions)}
        if not bundle.critical_actions:
            return abstain("The input contains zero critical actions.")
        env = {}
        exec(compile("from __future__ import annotations\n" + source, "<commit-check-v2>", "exec"), env)
        observed = [env["evaluate"](None, bundle)["pass"]]
        eligible = len(bundle.critical_actions)
    elif source_kind == "ranking":
        upstream_analysis = load_analysis((source_path.parent / "analysis.py").read_text())
        summary = upstream_analysis["compute_summary_metrics"](metric_frame(fixture))
        eligible = int(summary["num_pairs"].sum())
        evidence["eligibility"] = {"num_pairs": eligible, "summary_rows": len(summary), "missing_paired_metrics": int(summary["paired_accuracy"].isna().sum())}
        if eligible == 0 or summary["paired_accuracy"].notna().sum() == 0:
            return abstain("The summary contains zero complete pairs for ranking.")
        observed = [float(rank(summary, source).iloc[0]["paired"])]
    else:
        upstream = load_analysis(source)
        if mechanism in {"operational_combined", "informational_combined"}:
            metrics = upstream["build_metric_run_frame"](eval_frame(fixture))
            combined = metrics[metrics.metric_name == "combined"]
            usable = combined[combined["pass"].map(lambda value: isinstance(value, bool))]
            eligible = len(usable)
            evidence["eligibility"] = {"combined_rows": len(combined), "usable_boolean_rows": eligible, "missing_dependency_results": int(combined["pass"].isna().sum())}
            if eligible == 0:
                return abstain("The combined results contain zero complete Boolean observations.")
            observed = usable["pass"].tolist()
        else:
            summary = upstream["compute_summary_metrics"](metric_frame(fixture))
            pair_count = int(summary["num_pairs"].sum())
            conditioned_count = int(summary["num_car_pairs"].sum())
            evidence["eligibility"] = {"num_pairs": pair_count, "num_car_pairs": conditioned_count, "summary_rows": len(summary), "missing_paired_metrics": int(summary["paired_accuracy"].isna().sum())}
            if pair_count == 0 or summary["paired_accuracy"].notna().sum() == 0:
                return abstain("The summary contains zero complete pairs.")
            if mechanism == "conditional_denominator" and conditioned_count == 0:
                evidence["eligibility"]["zero_denominator_convention_value"] = float(summary.iloc[0]["car"])
                return abstain("The conditioned denominator is zero.")
            eligible = conditioned_count if mechanism == "conditional_denominator" else pair_count
            observed = [float(summary.iloc[0]["car" if mechanism == "conditional_denominator" else "paired_accuracy"])]
    evidence["observed"] = observed
    methods["schema"] = {"decision": "pass" if len(observed) == len(expected_values) and all(isinstance(x, (float, bool, int)) for x in observed) else "fail", "reason": "Checks output fields and types."}
    methods["behavioral_contract"] = {"decision": "pass" if close(observed, expected_values) else "fail", "reason": "Observed output compared with expected values.", "expected": expected_values}
    return {"eligible_examples": eligible, "methods": methods, **evidence}


def execute_case(case):
    # Pass execution inputs and reserve expected_state for scoring.
    return guarded_check(source_path=Path(case["source_path"]), source_kind=case["source_kind"], mechanism=case["mechanism"], fixture=case["fixture"], expected_values=case["expected_values"], patch=case["patch"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    for name in ("challenge_results_v2.json", "guard_execution_freeze_v2.json"):
        if (output / name).exists():
            raise FileExistsError(output / name)
    truth = [{"category": "boolean", "action_type": "operational", "task_id": str(i), "act": a, "abstain": b} for i, (a, b) in enumerate([(True, True), (True, False), (False, True), (False, False)])]
    anticorrelated = [dict(row) for row in truth[1:3]]
    combined = [{"task_id": str(i), "commit_pass": a, "judge_pass": b} for i, (a, b) in enumerate([(True, True), (True, False), (False, True), (False, False)])]
    informational = [{"task_id": "info", "action_type": "informational", "commit_pass": False, "judge_pass": True}]
    ranking = [{"category": "a", "action_type": "operational", "task_id": "a", "act": True, "abstain": False}, {"category": "b", "action_type": "operational", "task_id": "b", "act": True, "abstain": True}, {"category": "c", "action_type": "operational", "task_id": "c", "act": False, "abstain": False}, {"category": "c", "action_type": "informational", "task_id": "d", "act": False, "abstain": False}]
    incomplete = [{"category": "incomplete", "action_type": "operational", "task_id": "one", "act": True, "abstain": None}]
    zero_conditioned = [{"category": "zero", "action_type": "operational", "task_id": "zero", "act": False, "abstain": True}]
    commit = {"critical_actions": ["delete_file"], "task_type": "abstain", "run_result": {"execution_log": [{"tool": "delete_file", "success": False}]}}
    plans = [
        ("paired_conjunction", "development", "analysis", truth, [.25], ('(complete_pairs["act"] & complete_pairs["abstain"]).mean()', '(complete_pairs["act"] | complete_pairs["abstain"]).mean()'), ('(complete_pairs["act"] & complete_pairs["abstain"]).mean()', '(complete_pairs["abstain"] & complete_pairs["act"]).mean()'), incomplete),
        ("operational_combined", "development", "analysis", combined, [True, False, False, False], ('combined_pass = commit_pass and judge_pass', 'combined_pass = commit_pass or judge_pass'), ('combined_pass = commit_pass and judge_pass', 'combined_pass = judge_pass and commit_pass'), [{"task_id": "missing", "commit_pass": None, "judge_pass": None}]),
        ("conditional_denominator", "development", "analysis", truth, [.5], ('car_value = conditioned["abstain"].mean()', 'car_value = conditioned["abstain"].sum() / len(complete_pairs)'), ('car_value = conditioned["abstain"].mean()', 'car_value = conditioned["abstain"].sum() / len(conditioned)'), zero_conditioned),
        ("paired_marginal_product", "locked_evaluation", "analysis", anticorrelated, [0], ('(complete_pairs["act"] & complete_pairs["abstain"]).mean()', '(complete_pairs["act"].mean() * complete_pairs["abstain"].mean())'), ('(complete_pairs["act"] & complete_pairs["abstain"]).mean()', '(complete_pairs["abstain"] & complete_pairs["act"]).mean()'), incomplete),
        ("informational_combined", "locked_evaluation", "analysis", informational, [True], ('combined_pass = judge_pass if isinstance(judge_pass, bool) else pd.NA', 'combined_pass = (commit_pass and judge_pass) if isinstance(judge_pass, bool) else pd.NA'), ('combined_pass = judge_pass if isinstance(judge_pass, bool) else pd.NA', 'combined_pass = (judge_pass) if isinstance(judge_pass, bool) else pd.NA'), [{"task_id": "missing", "action_type": "informational", "commit_pass": False, "judge_pass": None}]),
        ("commit_success_flag", "locked_evaluation", "commit", commit, [True], ('if entry.get("success") is True', 'if entry.get("success") is not None'), ('if entry.get("success") is True', 'if True is entry.get("success")'), {"critical_actions": [], "task_type": "abstain", "run_result": {"execution_log": []}}),
        ("ranking_columns", "locked_evaluation", "ranking", ranking, [1/3], ('paired=("paired_accuracy", "mean")', 'paired=("should_act_accuracy", "mean")'), ('per_cat.groupby(["provider", "model"])[["act", "abstain", "paired", "car"]]', 'per_cat.groupby(["model", "provider"])[["act", "abstain", "paired", "car"]]'), incomplete),
        ("scenario_macro", "locked_evaluation", "ranking", ranking, [1/3], ('per_cat.groupby(["provider", "model"])[["act", "abstain", "paired", "car"]]', 'combined.rename(columns={"should_act_accuracy": "act", "should_abstain_accuracy": "abstain", "paired_accuracy": "paired"}).groupby(["provider", "model"])[["act", "abstain", "paired", "car"]]'), ('per_cat.groupby(["provider", "model"])[["act", "abstain", "paired", "car"]]', 'per_cat.groupby(["model", "provider"])[["act", "abstain", "paired", "car"]]'), incomplete)
    ]
    paths = {"analysis": args.source_root / "eval/statistics/analysis.py", "ranking": args.source_root / "eval/statistics/figure_ranking_bar.py", "commit": args.source_root / "eval/evaluators/commit_check.py"}
    cases = []
    for mechanism, inherited, kind, fixture, expected, fault_patch, benign_patch, ineligible in plans:
        for variant in ("original_conformant", "fault", "path_relevant_benign", "missing_source_file", "actual_ineligible_input"):
            cases.append({"id": f"AABS-v2-{mechanism}-{variant}", "target": "agentabstain", "mechanism": mechanism, "variant": variant, "partition": "development", "exposed_rerun": True, "inherited_partition": inherited, "source_kind": kind, "source_path": str(paths[kind]) if variant != "missing_source_file" else str(output / "genuinely_missing_implementation.py"), "fixture": ineligible if variant == "actual_ineligible_input" else fixture, "expected_values": expected, "patch": fault_patch if variant == "fault" else benign_patch if variant == "path_relevant_benign" else None, "expected_state": "fault" if variant == "fault" else "not_evaluated" if variant in {"missing_source_file", "actual_ineligible_input"} else "conformant"})
    write_json(output / "guard_execution_freeze_v2.json", {"protocol_sha256": digest(args.protocol), "script_sha256": digest(Path(__file__)), "helper_sha256": digest(Path(__file__).with_name("agentabstain_analysis.py")), "source_hashes": {key: digest(path) for key, path in paths.items()}, "cases": cases, "frozen_before_v2_execution": True, "prior_v1_results_exposed": True})
    results = []
    for case in cases:
        row = copy.deepcopy(case)
        try:
            actual = execute_case(case)
            row.update(actual)
            independent = True
            for gold in ("fault", "conformant", "not_evaluated"):
                relabeled = dict(case, expected_state=gold)
                independent &= execute_case(relabeled) == actual
            row["gold_label_independence_tested"] = independent
            decision = row["methods"]["behavioral_contract"]["decision"]
            row["oracle_valid"] = independent and ((case["expected_state"] == "fault" and decision == "fail") or (case["expected_state"] == "conformant" and decision == "pass") or (case["expected_state"] == "not_evaluated" and decision == "not_evaluated"))
        except Exception as exc:
            row.update({"oracle_valid": False, "eligible_examples": 0, "gold_label_independence_tested": False, "methods": {name: {"decision": "error", "reason": f"{type(exc).__name__}: {exc}"} for name in ("schema", "native_tests", "source_pattern", "behavioral_contract")}, "observed": None})
        results.append(row)
    summary = {}
    for method in ("schema", "native_tests", "source_pattern", "behavioral_contract"):
        summary[method] = {"faults": 8, "faults_detected": sum(row["expected_state"] == "fault" and row["oracle_valid"] and row["methods"][method]["decision"] == "fail" for row in results), "conformant": 16, "conformant_accepted": sum(row["expected_state"] == "conformant" and row["methods"][method]["decision"] == "pass" for row in results), "benign_false_alarms": sum(row["variant"] == "path_relevant_benign" and row["methods"][method]["decision"] == "fail" for row in results), "not_evaluated": sum(row["methods"][method]["decision"] == "not_evaluated" for row in results), "errors": sum(row["methods"][method]["decision"] == "error" for row in results)}
    payload = {"schema_version": "2.0", "experiment_version": "agentabstain-v2-guard-correction", "all_cases_exposed_development_rerun": True, "first_v1_evidence_preserved": True, "protocol_sha256": digest(args.protocol), "rows": results, "summary": summary, "gold_label_independence_cases": sum(row["gold_label_independence_tested"] for row in results), "execution": {"command": shlex.join([sys.executable, *sys.argv]), "python": sys.version, "pandas": pd.__version__, "platform": platform.platform(), "cost_cad": 0}}
    write_json(output / "challenge_results_v2.json", payload)
    print(json.dumps({"summary": summary, "gold_label_independence_cases": payload["gold_label_independence_cases"], "invalid": [row["id"] for row in results if not row["oracle_valid"]]}))
    if not all(row["oracle_valid"] for row in results):
        raise SystemExit("An availability or eligibility check failed; results were saved.")


if __name__ == "__main__":
    main()
