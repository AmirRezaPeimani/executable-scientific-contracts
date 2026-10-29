#!/usr/bin/env python3
"""Run AgentAbstain checks and compute weighting bounds from the released task counts."""
from __future__ import annotations

import argparse
import ast
import csv
import difflib
import hashlib
import json
import math
import platform
import shlex
import subprocess
import sys
import time
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


def digest(value):
    if isinstance(value, Path):
        value = value.read_bytes()
    if isinstance(value, str):
        value = value.encode()
    return hashlib.sha256(value).hexdigest()


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def extraction(source, names, *, inner=None):
    tree = ast.parse(source)
    nodes = tree.body
    if inner:
        nodes = next(x for x in nodes if isinstance(x, (ast.FunctionDef, ast.ClassDef)) and x.name == inner).body
    picked = []
    for node in nodes:
        name = node.name if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else None
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        if name in names:
            picked.append(node)
    if len(picked) != len(names):
        raise ValueError(f"missing source blocks: {names}")
    blocks = [ast.get_source_segment(source, node) for node in picked]
    return "\n\n".join(blocks), [{"name": names[i], "line_start": node.lineno, "line_end": node.end_lineno, "sha256": digest(blocks[i])} for i, node in enumerate(picked)]


def load_analysis(source):
    ns = {}
    exec(compile(source, "<pinned-agentabstain-analysis>", "exec"), ns)
    return ns


def rank(summary, rank_source):
    ns = {"summary": summary}
    exec(compile(rank_source, "<pinned-agentabstain-ranking>", "exec"), ns)
    return ns["macro"]


def metric_frame(rows):
    records = []
    for row in rows:
        for task_type in ("act", "abstain"):
            if row.get(task_type) is None:
                continue
            records.append({"provider": "fixture", "model": "fixture", "category": row["category"], "action_type": row["action_type"], "metric_name": "combined", "task_id": row["task_id"], "task_type": task_type, "pass": bool(row[task_type])})
    return pd.DataFrame(records)


def eval_frame(rows):
    return pd.DataFrame([{"provider": "fixture", "model": "fixture", "category": row.get("category", "fixture"), "action_type": row.get("action_type", "operational"), "task_id": row["task_id"], "task_type": "abstain", "run_id": row["task_id"], "generated_at": "2026-10-29T00:00:00Z", "run_error_kind": None, "metrics": {"commit_check": {"pass": row["commit_pass"]}, "response_llm_judge": {"pass": row["judge_pass"]}}, "eval_path": "synthetic controlled fixture"} for row in rows])


def close(a, b):
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    if a is None or b is None:
        return a is b
    return math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--dataset-jsonl", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("case_analysis_v1.json", "challenge_results_v1.json"):
        if (args.output_dir / name).exists():
            raise FileExistsError(f"refusing to overwrite existing result {name}; choose a new output directory")
    protocol = json.loads(args.protocol.read_text())
    revision = subprocess.check_output(["git", "-C", str(args.source_root), "rev-parse", "HEAD"], text=True).strip()
    if revision != protocol["source_revision"] or digest(args.dataset_jsonl) != protocol["dataset_sha256"]:
        raise ValueError("Source or dataset differs from the protocol.")
    paths = {"analysis": args.source_root / "eval/statistics/analysis.py", "ranking": args.source_root / "eval/statistics/figure_ranking_bar.py", "commit": args.source_root / "eval/evaluators/commit_check.py"}
    sources = {name: path.read_text() for name, path in paths.items()}
    rank_source, rank_anchors = extraction(sources["ranking"], ["combined", "per_cat", "macro"], inner="main")
    cls = next(x for x in ast.parse(sources["commit"]).body if isinstance(x, ast.ClassDef) and x.name == "CommitCheckEvaluator")
    method = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "evaluate")
    commit_source = ast.get_source_segment(sources["commit"], method)
    freeze = {"protocol_sha256": digest(args.protocol), "script_sha256": digest(Path(__file__)), "source_revision": revision, "dataset_sha256": digest(args.dataset_jsonl), "sources": {key: {"path": str(paths[key]), "sha256": digest(src)} for key, src in sources.items()}, "ranking_extraction": rank_anchors, "commit_extraction": {"line_start": method.lineno, "line_end": method.end_lineno, "sha256": digest(commit_source)}, "frozen_at_unix": time.time(), "evaluation_started": False}
    write_json(args.output_dir / "execution_freeze_v1.json", freeze)
    # Record the protocol and script hash.
    started = time.monotonic()
    ns = load_analysis(sources["analysis"])
    rows = [json.loads(line) for line in args.dataset_jsonl.read_text().splitlines() if line.strip()]
    groups = defaultdict(list)
    for row in rows:
        groups[str(row["pair_id"])].append(row)
    invalid = [key for key, vals in groups.items() if len(vals) != 2 or Counter(r["task_type"] for r in vals) != {"act": 1, "abstain": 1} or len({r["category"] for r in vals}) != 1 or len({r["action_type"] for r in vals}) != 1]
    if invalid:
        raise ValueError(f"dataset has invalid pairs: {invalid}")
    counts = defaultdict(Counter)
    for vals in groups.values():
        counts[vals[0]["category"]][vals[0]["action_type"]] += 1
    scenarios, constructed = [], []
    for category, types in sorted(counts.items()):
        total, k = sum(types.values()), len(types)
        weights = {key: Fraction(1, k) - Fraction(num, total) for key, num in types.items()}
        bound = sum(x for x in weights.values() if x > 0)
        rates = {key: int(weight > 0) for key, weight in weights.items()}
        equal = sum(Fraction(x, k) for x in rates.values())
        pair = sum(Fraction(types[key] * rates[key], total) for key in types)
        assert equal - pair == bound
        scenarios.append({"category": category, "counts": dict(sorted(types.items())), "num_pairs": total, "sharp_gap": float(bound), "sharp_gap_exact": str(bound), "max_gap_pp": 100 * float(bound), "attaining_rates": rates, "attaining_equal": float(equal), "attaining_pair_weighted": float(pair)})
        for key, num in sorted(types.items()):
            for i in range(num):
                constructed.append({"category": category, "action_type": key, "task_id": f"{category}_{key}_{i}", "act": bool(rates[key]), "abstain": bool(rates[key])})
    composed_summary = ns["compute_summary_metrics"](metric_frame(constructed))
    composed_rank = rank(composed_summary, rank_source)
    actual_equal = float(composed_rank.iloc[0]["paired"])
    actual_pair = sum(sum(row["act"] and row["abstain"] for row in constructed if row["category"] == cat) / sum(row["category"] == cat for row in constructed) for cat in counts) / len(counts)
    macro_bound = sum(Fraction(s["sharp_gap_exact"]) for s in scenarios) / len(scenarios)
    assert close(actual_equal - actual_pair, float(macro_bound))
    missing_sensitivity = []
    for removed in protocol["sensitivity"]["missing_pair_counts"]:
        per_scenario = []
        for category, types in sorted(counts.items()):
            cases = []
            for subtype, count in types.items():
                remaining = dict(types)
                remaining[subtype] -= min(removed, count)
                remaining = {key: value for key, value in remaining.items() if value > 0}
                total, k = sum(remaining.values()), len(remaining)
                bound = sum(max(Fraction(1, k) - Fraction(n, total), 0) for n in remaining.values()) if k else None
                cases.append({"removed_from": subtype, "actual_removed": min(removed, count), "counts_remaining": remaining, "gap": float(bound) if bound is not None else None})
            per_scenario.append({"category": category, "cases": cases, "max_gap": max(x["gap"] for x in cases if x["gap"] is not None)})
        missing_sensitivity.append({"requested_pairs_removed_per_scenario": removed, "scenario_deletion_assumption": "from one subtype per scenario; maximize over subtype choice", "scenarios": per_scenario, "macro_max_gap": sum(s["max_gap"] for s in per_scenario) / len(per_scenario)})

    truth = [{"category": "boolean", "action_type": "operational", "task_id": str(i), "act": a, "abstain": b} for i, (a, b) in enumerate([(True, True), (True, False), (False, True), (False, False)])]
    anticorrelated = [{"category": "boolean", "action_type": "operational", "task_id": str(i), "act": a, "abstain": b} for i, (a, b) in enumerate([(True, False), (False, True)])]
    combined_rows = [{"task_id": str(i), "commit_pass": a, "judge_pass": b} for i, (a, b) in enumerate([(True, True), (True, False), (False, True), (False, False)])]
    rank_rows = [{"category": "a", "action_type": "operational", "task_id": "a", "act": True, "abstain": False}, {"category": "b", "action_type": "operational", "task_id": "b", "act": True, "abstain": True}, {"category": "c", "action_type": "operational", "task_id": "c", "act": False, "abstain": False}, {"category": "c", "action_type": "informational", "task_id": "d", "act": False, "abstain": False}]
    historical = [{"category": "a", "action_type": "operational", "task_id": "a0", "act": True, "abstain": True}, *[{"category": "a", "action_type": "informational", "task_id": f"a{i}", "act": False, "abstain": False} for i in range(1, 4)], *[{"category": "b", "action_type": "operational", "task_id": f"b{i}", "act": True, "abstain": True} for i in range(4)]]
    historical_equal = float(rank(ns["compute_summary_metrics"](metric_frame(historical)), rank_source).iloc[0]["paired"])
    historical_pair = (1 / 4 + 1) / 2
    assert close(historical_equal, .75) and close(historical_pair, .625)

    def observe(mechanism, source):
        if mechanism == "commit_success_flag":
            env = {}
            exec(compile("from __future__ import annotations\n" + source, "<pinned-commit-method>", "exec"), env)
            bundle = SimpleNamespace(critical_actions=["delete_file"], task_type="abstain", run_result={"execution_log": [{"tool": "delete_file", "success": False}]})
            return [env["evaluate"](None, bundle)["pass"]]
        if mechanism in {"ranking_columns", "scenario_macro"}:
            summary = ns["compute_summary_metrics"](metric_frame(rank_rows))
            return [float(rank(summary, source).iloc[0]["paired"])]
        local = load_analysis(source)
        if mechanism == "operational_combined":
            out = local["build_metric_run_frame"](eval_frame(combined_rows))
            return out[out.metric_name == "combined"]["pass"].tolist()
        if mechanism == "informational_combined":
            out = local["build_metric_run_frame"](eval_frame([{"task_id": "info", "action_type": "informational", "commit_pass": False, "judge_pass": True}]))
            return out[out.metric_name == "combined"]["pass"].tolist()
        fixtures = anticorrelated if mechanism == "paired_marginal_product" else truth
        summary = local["compute_summary_metrics"](metric_frame(fixtures))
        return [float(summary.iloc[0]["car" if mechanism == "conditional_denominator" else "paired_accuracy"])]

    plans = [
        ("paired_conjunction", "development", "analysis", '(complete_pairs["act"] & complete_pairs["abstain"]).mean()', '(complete_pairs["act"] | complete_pairs["abstain"]).mean()', [.25], "pair_conjunction_to_disjunction"),
        ("operational_combined", "development", "analysis", "combined_pass = commit_pass and judge_pass", "combined_pass = commit_pass or judge_pass", [True, False, False, False], "combined_conjunction_to_disjunction"),
        ("conditional_denominator", "development", "analysis", 'car_value = conditioned["abstain"].mean()', 'car_value = conditioned["abstain"].sum() / len(complete_pairs)', [.5], "conditional_wrong_denominator"),
        ("paired_marginal_product", "locked_evaluation", "analysis", '(complete_pairs["act"] & complete_pairs["abstain"]).mean()', '(complete_pairs["act"].mean() * complete_pairs["abstain"].mean())', [0], "pair_conjunction_to_marginal_product"),
        ("informational_combined", "locked_evaluation", "analysis", 'combined_pass = judge_pass if isinstance(judge_pass, bool) else pd.NA', 'combined_pass = (commit_pass and judge_pass) if isinstance(judge_pass, bool) else pd.NA', [True], "informational_metric_gated_by_commit"),
        ("commit_success_flag", "locked_evaluation", "commit", 'if entry.get("success") is True', 'if entry.get("success") is not None', [True], "count_failed_critical_tool_calls"),
        ("ranking_columns", "locked_evaluation", "ranking", 'paired=("paired_accuracy", "mean")', 'paired=("should_act_accuracy", "mean")', [1/3], "headline_uses_act_column"),
        ("scenario_macro", "locked_evaluation", "ranking", 'per_cat.groupby(["provider", "model"])[["act", "abstain", "paired", "car"]]', 'per_cat.assign(category="pooled").groupby(["provider", "model"])[["act", "abstain", "paired", "car"]]', [.5], "headline_pools_scenarios")
    ]
    # Pool the original subtype-summary rows.
    pool_old = '''per_cat.groupby(["provider", "model"])[["act", "abstain", "paired", "car"]]'''
    pool_new = '''combined.rename(columns={"should_act_accuracy": "act", "should_abstain_accuracy": "abstain", "paired_accuracy": "paired"}).groupby(["provider", "model"])[["act", "abstain", "paired", "car"]]'''
    plans[-1] = ("scenario_macro", "locked_evaluation", "ranking", pool_old, pool_new, [1/3], "headline_pools_scenarios")
    challenges = []
    original_by_kind = {"analysis": sources["analysis"], "ranking": rank_source, "commit": commit_source}
    benign_by_kind = {"analysis": ('combined_pass = commit_pass and judge_pass', 'combined_pass = judge_pass and commit_pass'), "commit": ('if entry.get("success") is True', 'if True is entry.get("success")'), "ranking": ('ascending=False', 'ascending=(False)')}

    def add(mechanism, partition, kind, variant, implementation, expected, state, family):
        original = original_by_kind[kind]
        patch = "".join(difflib.unified_diff(original.splitlines(keepends=True), implementation.splitlines(keepends=True), fromfile="original", tofile=variant)) if implementation is not None else ""
        result = {"id": f"AABS-{mechanism}-{variant}", "target": "agentabstain", "mechanism": mechanism, "variant": variant, "partition": partition, "expected_state": state, "oracle_valid": False, "eligible_examples": len(expected) if expected and implementation is not None else 0, "implementation": {"source": str(paths[kind]), "source_sha256": digest(sources[kind]), "extracted_original_sha256": digest(original), "executed_source_sha256": digest(implementation) if implementation is not None else None, "patch_sha256": digest(patch) if patch else None, "patch": patch, "mutation_family": family}, "methods": {}, "observed": None}
        unavailable = state == "not_evaluated"
        if unavailable:
            for method in protocol["methods"]:
                result["methods"][method] = {"decision": "not_evaluated", "reason": "missing implementation evidence" if implementation is None else "zero eligible complete pairs"}
            result["oracle_valid"] = True
            challenges.append(result)
            return
        try:
            observed = observe(mechanism, implementation)
            result["observed"] = observed
            manifest = not close(observed, expected)
            result["oracle_valid"] = manifest if state == "fault" else not manifest
            result["methods"]["behavioral_contract"] = {"decision": "fail" if manifest else "pass", "reason": "Implementation output compared with expected values.", "expected": expected}
            schema_ok = isinstance(observed, list) and len(observed) == len(expected) and all(isinstance(x, (bool, int, float)) and math.isfinite(float(x)) for x in observed)
            result["methods"]["schema"] = {"decision": "pass" if schema_ok else "fail", "reason": "Checks numeric and Boolean fields and output shape."}
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            for method in ("schema", "behavioral_contract"):
                result["methods"][method] = {"decision": "error", "reason": result["error"]}
        result["methods"]["native_tests"] = {"decision": "not_evaluated", "reason": "no relevant native test suite found at pinned source revision"}
        result["methods"]["source_pattern"] = {"decision": "not_evaluated", "reason": "historical public AgentAbstain evaluator has no source-pattern method"}
        challenges.append(result)

    for mechanism, partition, kind, old, new, expected, family in plans:
        original = original_by_kind[kind]
        if original.count(old) != 1:
            raise ValueError(f"patch match count for {family}: {original.count(old)}")
        benign_old, benign_new = benign_by_kind[kind]
        assert original.count(benign_old) >= 1
        for variant, implementation, state in [("original_conformant", original, "conformant"), ("fault", original.replace(old, new), "fault"), ("benign", original.replace(benign_old, benign_new), "conformant"), ("missing", None, "not_evaluated"), ("ineligible", original, "not_evaluated")]:
            add(mechanism, partition, kind, variant, implementation, [] if variant == "ineligible" else expected, state, family if variant == "fault" else None)
    # Save every result before checking validity.
    summary = {}
    for method in protocol["methods"]:
        selected = [row for row in challenges if row["oracle_valid"]]
        summary[method] = {"faults_detected": sum(row["expected_state"] == "fault" and row["methods"][method]["decision"] == "fail" for row in selected), "faults_eligible": sum(row["expected_state"] == "fault" for row in selected), "conformant_accepted": sum(row["expected_state"] == "conformant" and row["methods"][method]["decision"] == "pass" for row in selected), "benign_false_alarms": sum(row["variant"] == "benign" and row["methods"][method]["decision"] == "fail" for row in selected), "not_evaluated": sum(row["methods"][method]["decision"] == "not_evaluated" for row in challenges), "errors": sum(row["methods"][method]["decision"] == "error" for row in challenges)}
    write_json(args.output_dir / "challenge_results_v1.json", {"schema_version": "2.0", "protocol_sha256": digest(args.protocol), "rows": challenges, "summary": summary, "excluded_unmanifest_or_error": [row["id"] for row in challenges if not row["oracle_valid"]]})
    empty_fixture = [{"category": "incomplete", "action_type": "operational", "task_id": "one", "act": True, "abstain": None}]
    incomplete = ns["compute_summary_metrics"](metric_frame(empty_fixture))
    zero_car = ns["compute_summary_metrics"](metric_frame([{"category": "zero", "action_type": "operational", "task_id": "zero", "act": False, "abstain": True}]))
    primitive = {"four_boolean_pairs": ns["compute_summary_metrics"](metric_frame(truth)).iloc[0]["paired_accuracy"], "nonempty_car": ns["compute_summary_metrics"](metric_frame(truth)).iloc[0]["car"], "zero_conditioned_denominator_car": zero_car.iloc[0]["car"], "incomplete_pair_num_pairs": int(incomplete.iloc[0]["num_pairs"]), "incomplete_pair_metric_is_missing": bool(pd.isna(incomplete.iloc[0]["paired_accuracy"])), "combined_operational": observe("operational_combined", sources["analysis"]), "informational_ignores_commit": observe("informational_combined", sources["analysis"]), "failed_call_is_not_commit": observe("commit_success_flag", commit_source)}
    payload = {"schema_version": "1.0", "source": freeze, "dataset": {"rows": len(rows), "pairs": len(groups), "scenarios": len(counts), "invalid_pairs": len(invalid)}, "historical_fixture": {"equal_subtype_macro": historical_equal, "pair_weighted_macro": historical_pair, "synthetic": True, "actual_final_ranking_path_executed": True}, "primitive_checks": primitive, "subtype_sensitivity": {"scenarios": scenarios, "macro_sharp_gap": float(macro_bound), "macro_sharp_gap_exact": str(macro_bound), "macro_max_gap_pp": float(macro_bound) * 100, "constructed_attainment": {"equal_subtype_macro": actual_equal, "pair_weighted_macro": actual_pair, "gap": actual_equal - actual_pair, "synthetic": True, "actual_final_ranking_path_executed": True}, "missing_pair_sensitivity": missing_sensitivity}, "claim_boundary": "Maximum weighting difference for the released subtype counts, attained with constructed endpoint outcomes.", "execution": {"command": shlex.join([sys.executable, *sys.argv]), "python": sys.version, "pandas": pd.__version__, "platform": platform.platform(), "cost_cad": 0, "elapsed_seconds": time.monotonic() - started}}
    write_json(args.output_dir / "case_analysis_v1.json", payload)
    with (args.output_dir / "subtype_sensitivity_v1.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["category", "action_type", "pairs", "scenario_pairs", "sharp_gap", "max_gap_pp", "attaining_rate"])
        writer.writeheader()
        for scenario in scenarios:
            for key, count in scenario["counts"].items():
                writer.writerow({"category": scenario["category"], "action_type": key, "pairs": count, "scenario_pairs": scenario["num_pairs"], "sharp_gap": scenario["sharp_gap"], "max_gap_pp": scenario["max_gap_pp"], "attaining_rate": scenario["attaining_rates"][key]})
    # Save the extracted computation blocks.
    (args.output_dir / "ranking_extraction.py").write_text(rank_source + "\n")
    (args.output_dir / "commit_extraction.py").write_text("from __future__ import annotations\n" + commit_source + "\n")
    print(json.dumps({"dataset": payload["dataset"], "macro_max_gap_pp": payload["subtype_sensitivity"]["macro_max_gap_pp"], "challenge_summary": summary, "excluded": [row["id"] for row in challenges if not row["oracle_valid"]]}))
    if any(not row["oracle_valid"] for row in challenges):
        raise SystemExit("A mutation produced no output difference or an execution failed; results were saved.")


if __name__ == "__main__":
    main()
