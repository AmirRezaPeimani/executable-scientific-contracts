#!/usr/bin/env python3
"""Reproduce the nine published source cases with unchanged v2 checkers.

Preparation binds explicitly acquired inputs and freezes their hashes before
execution. Reproduction does not create another untouched evaluation set.
"""
from __future__ import annotations
import argparse
import datetime as dt
import difflib
from fractions import Fraction
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import traceback

BUNDLED_ANALYSIS = Path(__file__).resolve().parent
BUNDLED_STUDY = BUNDLED_ANALYSIS.parent
REPOSITORY_ROOT = BUNDLED_STUDY.parent.parent
LEGACY_ADAPTER = REPOSITORY_ROOT / "src/conform/adapters/agentprm.py"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read(path):
    return json.loads(Path(path).read_text())


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def trajectory(length, score, name):
    return {"task": f"fresh-reward-range-{name}", "gamefile": "synthetic",
            "trajectory": [{"observation": f"{name}-obs-{t}", "candidate_actions": [f"{name}-act-{t}"],
                            "action": f"{name}-act-{t}", "reason": "controlled fresh terminal outcome",
                            "score": score if t == length - 1 else 0, "agent_log": []} for t in range(length)]}


def verify_inputs(inputs):
    """Check acquisition identities without scanning data or running cases."""
    pins = {name: read(BUNDLED_STUDY / f"protocols/{name}_v1.json")["source_revision"]
            for name in ("agentprm", "toolace", "agentabstain")}
    revisions, hashes = {}, {}
    for target, expected in pins.items():
        source = Path(inputs[f"{target}_source_root"])
        actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip()
        if actual != expected or dirty:
            raise ValueError(f"{target}: acquire a clean checkout at {expected}")
        revisions[target] = actual
    aprm_pin = read(BUNDLED_STUDY / "protocols/agentprm_v1.json")
    for relative, expected in aprm_pin["source_hashes"].items():
        path = Path(inputs["agentprm_source_root"]) / relative
        if digest(path) != expected:
            raise ValueError(f"AgentPRM source differs: {relative}")
        hashes[str(path)] = expected
    aprm_record = read(BUNDLED_STUDY / "results/agentprm/challenge_results_v2.json")
    repaired = next(row for row in aprm_record["rows"] if row["variant"] == "repaired")
    for relative, expected in repaired["implementation"]["source_files"].items():
        path = Path(inputs["agentprm_results"]) / "sources/repaired" / relative
        if digest(path) != expected:
            raise ValueError(f"Prior AgentPRM repair differs: {relative}")
        hashes[str(path)] = expected
    toolace_case = read(BUNDLED_STUDY / "results/toolace/case_analysis_v1.json")
    loader_relative = "verl/verl/utils/dataset/sft_dataset_wo_apply_template.py"
    original_loader = Path(inputs["toolace_source_root"]) / loader_relative
    if digest(original_loader) != toolace_case["implementation"]["loader_file_sha256"]:
        raise ValueError("Acquired ToolACE loader differs from the published source")
    hashes[str(original_loader)] = digest(original_loader)
    position = Path(inputs["toolace_source_root"]) / "verl/verl/utils/model.py"
    hashes[str(position)] = digest(position)
    toolace_record = read(BUNDLED_STUDY / "results/toolace/challenge_results_verified_v2.json")
    repaired = next(row for row in toolace_record["rows"] if row["id"] == "TACE-REPAIR")
    path = Path(inputs["toolace_results"]) / repaired["implementation"]["source"]
    if digest(path) != repaired["implementation"]["source_sha256"]:
        raise ValueError("Prior ToolACE repaired loader differs from the published source")
    hashes[str(path)] = digest(path)
    aabs_case = read(BUNDLED_STUDY / "results/agentabstain/case_analysis_v1.json")
    for kind, relative in (("analysis", "eval/statistics/analysis.py"), ("ranking", "eval/statistics/figure_ranking_bar.py")):
        path = Path(inputs["agentabstain_source_root"]) / relative
        expected = aabs_case["source"]["sources"][kind]["sha256"]
        if digest(path) != expected:
            raise ValueError(f"Acquired AgentAbstain source differs: {relative}")
        hashes[str(path)] = expected
    tokenizer = Path(inputs["tokenizer"])
    tokenizer_pin = read(BUNDLED_STUDY / "metadata/toolace_tokenizer_acquisition.json")
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        path = tokenizer / name
        if digest(path) != tokenizer_pin["assets"][name]:
            raise ValueError(f"Tokenizer asset differs: {name}")
        hashes[str(path)] = digest(path)
    for name in ("config.json", "generation_config.json", "special_tokens_map.json", "added_tokens.json"):
        path = tokenizer / name
        if path.is_file():
            hashes[str(path)] = digest(path)
    if not LEGACY_ADAPTER.is_file():
        raise FileNotFoundError("Run this script from the complete repository checkout; its original adapter is missing")
    return revisions, hashes


def prepare(w, inputs):
    folder = w / "results/fresh_holdout_v1"
    protocol_path = w / "protocols/fresh_holdout_v1.json"
    if protocol_path.exists() or folder.exists():
        raise FileExistsError("Fresh holdout has already been prepared; never overwrite it")
    revisions, input_hashes = verify_inputs(inputs)
    folder.mkdir(parents=True)
    frozen = [BUNDLED_ANALYSIS / f"{name}.py" for name in ("agentprm_guarded_v2", "agentprm_analysis", "toolace_evaluator_v2", "toolace_analysis", "agentabstain_guards_v2", "agentabstain_analysis")]
    frozen.append(LEGACY_ADAPTER)
    prior_protocols = [BUNDLED_STUDY / f"protocols/{name}.json" for name in ("agentprm_guarded_v2", "toolace_confirmation_v2", "agentabstain_v2")]
    cases, diffs = [], []
    def emit_source(path, original, changed, label):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(changed)
        diff = "".join(difflib.unified_diff(original.splitlines(True), changed.splitlines(True), fromfile="frozen_base", tofile=label))
        diffpath = folder / "patches" / f"{label}-{path.name}.diff"
        diffpath.parent.mkdir(exist_ok=True)
        diffpath.write_text(diff)
        diffs.append({"source": str(path), "source_sha256": digest(path), "parent_content_sha256": hashlib.sha256(original.encode()).hexdigest(), "diff": str(diffpath), "diff_sha256": digest(diffpath)})
    # Family 1: terminal-outcome range conversion, not discount indexing.
    files = ["scripts/dataproc/compute_prm_target.py", "scripts/dataproc/compute_prm_preference_target.py", "scripts/dataproc/rollout_alfworld.py", "src/agent_prm/utils/alfworld.py"]
    sourcebase = Path(inputs["agentprm_results"]) / "sources/repaired"
    original_reward = "outcome_reward = 2 * trajectory[-1]['score'] - 1"
    reward_sources = {"reference": original_reward, "fault": "outcome_reward = trajectory[-1]['score']", "benign": "outcome_reward = trajectory[-1]['score'] * 2 - 1"}
    fixture_paths = []
    for length, score, name in [(3, 0, "failure-length3"), (5, 1, "success-length5")]:
        path = folder / f"fixtures/agentprm-{name}.json"
        save(path, trajectory(length, score, name))
        fixture_paths.append(str(path))
    for variant, replacement in reward_sources.items():
        root = folder / "sources/agentprm" / variant
        for relative in files:
            original = (sourcebase / relative).read_text()
            changed = original
            if relative in files[:2]:
                assert original.count(original_reward) == 1
                changed = original.replace(original_reward, replacement)
            emit_source(root / relative, original, changed, f"agentprm-{variant}-{Path(relative).parent.name}")
        cases.append({"id": f"FRESH-APRM-REWARD-RANGE-{variant}", "target": "agentprm", "family": "terminal_outcome_range_remapping", "variant": variant,
                      "source_root": str(root), "fixture_paths": fixture_paths, "gamma": .95,
                      "frozen_evaluator_mechanism_argument": "terminal_discount", "expected_state": "fault" if variant == "fault" else "conformant",
                      "expected_eligible_examples": 16, "reference_rule": "For terminal score s in {0,1}, signed reward is 2s-1; return at distance d is (2s-1)*(19/20)^d. Two unchanged pinned caller functions execute."})
    # Family 2: response token identity/order, not crop or mask coordinates.
    loader_relative = "verl/verl/utils/dataset/sft_dataset_wo_apply_template.py"
    loader_base = (Path(inputs["toolace_results"]) / "implementation_variants/TACE-REPAIR" / loader_relative).read_text()
    old = "input_ids = torch.cat((prompt_ids, response_ids), dim=-1)"
    assert loader_base.count(old) == 1
    content_variants = {"reference": old, "fault": "input_ids = torch.cat((prompt_ids, torch.flip(response_ids, dims=[0])), dim=-1)", "benign": "input_ids = torch.cat([prompt_ids, response_ids], dim=-1)"}
    new_grid = [{"prompt_tokens": p, "response_tokens": r, "maximum": m} for p in (3, 5, 11) for r in (3, 6) for m in (12, 20)]
    for variant, replacement in content_variants.items():
        path = folder / "sources/toolace" / variant / loader_relative
        emit_source(path, loader_base, loader_base.replace(old, replacement), f"toolace-{variant}")
        cases.append({"id": f"FRESH-TACE-RESPONSE-ORDER-{variant}", "target": "toolace", "family": "response_token_order_corruption", "variant": variant,
                      "source_path": str(path), "fixtures": new_grid, "expected_state": "fault" if variant == "fault" else "conformant", "expected_eligible_examples": len(new_grid),
                      "reference_rule": "Tokenize real prompt and response+EOS separately; concatenate in original order, retain configured left-crop, pad. Compare all input IDs and shifted labels. No mask transformation is injected."})
    # Family 3: mean -> median at the specified OUTER scenario reduction.
    # Every scenario has exactly one pair, so subtype weighting is irrelevant.
    ranking_base = (Path(inputs["agentabstain_source_root"]) / "eval/statistics/figure_ranking_bar.py").read_text()
    rank_old = 'per_cat.groupby(["provider", "model"])[["act", "abstain", "paired", "car"]]\n        .mean()'
    assert ranking_base.count(rank_old) == 1
    rank_variants = {"reference": rank_old, "fault": rank_old.replace(".mean()", ".median()"), "benign": rank_old.replace(".mean()", '.agg("mean")')}
    scenarios = ["ambiguous_action_specification", "conflicting_constraints", "conflicting_evidence", "critical_tool_failure", "emergent_risk_discovery", "high_stakes_action", "insufficient_tool_capability", "missing_critical_parameter"]
    rank_fixture = [{"category": category, "action_type": "operational", "task_id": f"fresh-scenario-{i}", "act": i >= 5, "abstain": i >= 5} for i, category in enumerate(scenarios)]
    for variant, replacement in rank_variants.items():
        path = folder / "sources/agentabstain" / variant / "eval/statistics/figure_ranking_bar.py"
        emit_source(path, ranking_base, ranking_base.replace(rank_old, replacement), f"agentabstain-{variant}")
        analysis_base = (Path(inputs["agentabstain_source_root"]) / "eval/statistics/analysis.py").read_text()
        emit_source(path.parent / "analysis.py", analysis_base, analysis_base, f"agentabstain-helper-{variant}")
        cases.append({"id": f"FRESH-AABS-SCENARIO-STATISTIC-{variant}", "target": "agentabstain", "family": "outer_macro_mean_to_order_statistic", "variant": variant,
                      "source_path": str(path), "fixture": rank_fixture, "expected_state": "fault" if variant == "fault" else "conformant", "expected_eligible_examples": 8,
                      "expected_values": [3 / 8], "reference_rule": "Eight equally weighted scenario paired scores [0,0,0,0,0,1,1,1] have arithmetic mean3/8. Replacing outer mean by median gives0, contradicting source-selected scenario mean, not an unspecified subtype convention."})
    source_files = list((folder / "sources").rglob("*.py"))
    fixture_files = list((folder / "fixtures").glob("*.json"))
    protocol = {"schema_version": "1.0", "frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "purpose": "Reproduction of the nine published source cases; no existing evaluator, reference contract, case family or expected value changes.",
                "scope": "Small researcher-constructed source challenge; not independent human validation, new repositories, population generalization or model-performance evidence.",
                "runner_sha256": digest(Path(__file__)), "evaluator_and_import_hashes": {str(path): digest(path) for path in frozen},
                "execution_role": "reproduction_of_published_cases", "execution_inputs": inputs,
                "acquired_revisions": revisions, "input_file_hashes": input_hashes,
                "bundled_analysis": str(BUNDLED_ANALYSIS), "bundled_legacy_adapter": str(LEGACY_ADAPTER),
                "prior_frozen_protocol_hashes": {str(path): digest(path) for path in prior_protocols},
                "isolated_source_and_fixture_hashes": {str(path): digest(path) for path in source_files + fixture_files},
                "source_diffs": diffs, "cases": cases,
                "method_policy": "Use each target's existing four method decisions unchanged. Missing relevant native/pattern coverage remains not_evaluated. An error is not detection. A separately manifested fault missed by checker remains an explicit miss.",
                "primary_metrics": ["independently_manifested_faults", "faults_detected", "faults_missed", "reference_controls_accepted", "benign_false_alarms", "errors", "unmanifest_faults"],
                "freeze_rule": "Verify every runner/evaluator/helper/source/fixture hash before first execution; no patching based on these outputs; save all outputs before interpretation.",
                "environment_versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "pandas", "pyarrow", "numpy")},
                "paid_cost_cad": 0, "model_weights_or_training": False}
    save(protocol_path, protocol)
    print(json.dumps({"status": "FROZEN_NOT_EXECUTED", "protocol": str(protocol_path), "protocol_sha256": digest(protocol_path), "cases": len(cases), "new_families": 3}, indent=2))


def run(w):
    folder = w / "results/fresh_holdout_v1"
    protocol_path = w / "protocols/fresh_holdout_v1.json"
    protocol = read(protocol_path)
    output = folder / "challenge_results_v1.json"
    if output.exists() or (folder / "execution_started.json").exists():
        raise FileExistsError("First fresh evaluation already exists")
    revisions, _ = verify_inputs(protocol["execution_inputs"])
    if revisions != protocol["acquired_revisions"]:
        raise RuntimeError("Acquired revisions changed after preparation")
    required = {**protocol["evaluator_and_import_hashes"], **protocol["prior_frozen_protocol_hashes"], **protocol["isolated_source_and_fixture_hashes"], **protocol["input_file_hashes"], str(Path(__file__)): protocol["runner_sha256"]}
    changed = [{"path": path, "expected": expected, "actual": digest(path) if Path(path).is_file() else None} for path, expected in required.items() if not Path(path).is_file() or digest(path) != expected]
    if changed:
        save(folder / "freeze_verification_failure.json", changed)
        raise RuntimeError("Frozen source/evaluator mismatch; no evaluation executed")
    save(folder / "execution_started.json", {"started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "protocol_sha256": digest(protocol_path), "hashes_verified": len(required), "command": sys.argv})
    aprm = load(BUNDLED_ANALYSIS / "agentprm_guarded_v2.py", "fresh_aprm_checker")
    aabs = load(BUNDLED_ANALYSIS / "agentabstain_guards_v2.py", "fresh_aabs_checker")
    tool = load(BUNDLED_ANALYSIS / "toolace_evaluator_v2.py", "fresh_tool_checker")
    from transformers import AutoTokenizer
    inputs = protocol["execution_inputs"]
    tokenizer = AutoTokenizer.from_pretrained(inputs["tokenizer"], local_files_only=True)
    position_source = (Path(inputs["toolace_source_root"]) / "verl/verl/utils/model.py").read_text()
    legacy = LEGACY_ADAPTER
    rows = []
    for case in protocol["cases"]:
        row = {"id": case["id"], "target": case["target"], "mechanism": case["family"], "variant": case["variant"], "partition": "locked_evaluation", "expected_state": case["expected_state"], "oracle_valid": False, "eligible_examples": 0,
               "freshness": "new_cases_after_v2_evaluator_freeze", "independent_human_validation": False, "implementation": {"source_root": case.get("source_root"), "source_path": case.get("source_path")}}
        try:
            # Independently establish source behavior and reference validity
            # before invoking the unchanged checker; never infer gold from it.
            witnesses = []
            if case["target"] == "agentprm":
                for relative in ("scripts/dataproc/compute_prm_target.py", "scripts/dataproc/compute_prm_preference_target.py"):
                    module = aprm.V1.module(Path(case["source_root"]) / relative)
                    for path in case["fixture_paths"]:
                        data = read(path)
                        observed = module.process_file(path, case["gamma"])
                        values = {item["state"]["observation"]: item["qestimate"] for item in observed.values()}
                        terminal = Fraction(2 * data["trajectory"][-1]["score"] - 1)
                        length = len(data["trajectory"])
                        expected = [float(terminal * Fraction(19, 20) ** (length - 1 - i)) for i in range(length)]
                        actual = [values[step["observation"]] for step in data["trajectory"]]
                        witnesses.append({"source_path": relative, "fixture": Path(path).name, "actual": actual, "reference": expected, "different": any(not math.isclose(a, b, abs_tol=1e-12) for a, b in zip(actual, expected))})
                checked = aprm.evaluate(Path(case["source_root"]), case["frozen_evaluator_mechanism_argument"], [Path(p) for p in case["fixture_paths"]], case["gamma"], legacy)
            elif case["target"] == "toolace":
                path = Path(case["source_path"])
                call, _ = tool.loader_callable(path.read_text(), position_source)
                for fixture in case["fixtures"]:
                    prompt = " x" * fixture["prompt_tokens"]
                    response = " y" * (fixture["response_tokens"] - 1)
                    maximum = fixture["maximum"]
                    actual = tool.execute_loader(call, tokenizer, [prompt], [response], 0, maximum)
                    reference = tokenizer(prompt, add_special_tokens=False)["input_ids"] + tokenizer(response + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
                    reference = reference[-maximum:]
                    reference += [tokenizer.pad_token_id] * (maximum - len(reference))
                    witnesses.append({"fixture": fixture, "actual_input_ids": actual["input_ids"].tolist(), "reference_input_ids": reference, "different": actual["input_ids"].tolist() != reference})
                checked = tool.assess(path, case["fixtures"], tokenizer, position_source)
                checked["eligible_examples"] = checked["guard"]["eligible_examples"]
            else:
                path = Path(case["source_path"])
                ns = aabs.load_analysis((path.parent / "analysis.py").read_text())
                summary = ns["compute_summary_metrics"](aabs.metric_frame(case["fixture"]))
                extracted = aabs.extraction(path.read_text(), ["combined", "per_cat", "macro"], inner="main")[0]
                actual = float(aabs.rank(summary, extracted).iloc[0]["paired"])
                exact = sum(Fraction(int(pair["act"] and pair["abstain"])) for pair in case["fixture"]) / len(case["fixture"])
                witnesses.append({"actual": actual, "reference": float(exact), "reference_exact": str(exact), "scenario_pairs": len(case["fixture"]), "different": not math.isclose(actual, float(exact), abs_tol=1e-12)})
                checked = aabs.guarded_check(source_path=path, source_kind="ranking", mechanism="scenario_macro", fixture=case["fixture"], expected_values=case["expected_values"])
            manifested = any(x["different"] for x in witnesses)
            row["oracle_valid"] = manifested if case["expected_state"] == "fault" else not manifested
            row["independent_manifestation"] = {"different_from_reference": manifested, "witnesses": witnesses}
            row["eligible_examples"] = checked["eligible_examples"]
            row["expected_coverage_matches"] = row["eligible_examples"] == case["expected_eligible_examples"]
            row["methods"] = checked["methods"]
            row["observed"] = checked
        except Exception as exc:
            row["execution_error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
            row["methods"] = {name: {"decision": "error", "reason": f"{type(exc).__name__}: {exc}"} for name in ("schema", "native_tests", "source_pattern", "behavioral_contract")}
        save(folder / "raw" / f"{case['id']}.json", row)
        rows.append(row)
    methods = {}
    for method in ("schema", "native_tests", "source_pattern", "behavioral_contract"):
        faults = [r for r in rows if r["expected_state"] == "fault" and r["oracle_valid"]]
        methods[method] = {"independently_manifested_faults": len(faults), "faults_detected": sum(r["methods"][method]["decision"] == "fail" for r in faults),
                           "faults_missed": sum(r["methods"][method]["decision"] == "pass" for r in faults), "faults_uncovered": sum(r["methods"][method]["decision"] == "not_evaluated" for r in faults),
                           "reference_controls": sum(r["variant"] == "reference" for r in rows), "reference_controls_accepted": sum(r["variant"] == "reference" and r["methods"][method]["decision"] == "pass" for r in rows),
                           "benign_controls": sum(r["variant"] == "benign" for r in rows), "benign_false_alarms": sum(r["variant"] == "benign" and r["methods"][method]["decision"] == "fail" for r in rows),
                           "errors": sum(r["methods"][method]["decision"] == "error" for r in rows), "unmanifest_faults": sum(r["expected_state"] == "fault" and not r["oracle_valid"] for r in rows)}
    published = BUNDLED_STUDY / "results/fresh_holdout_v1/challenge_results_v1.json"
    summary_matches = methods == read(published)["summary"]
    result = {"schema_version": "2.0", "protocol": str(protocol_path), "protocol_sha256": digest(protocol_path), "frozen_evaluators_unchanged": True,
              "execution_role": "reproduction_of_published_cases", "published_summary_matches": summary_matches,
              "published_result_sha256": digest(published),
              "rows": rows, "summary": methods, "execution_finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "limitations": ["One new source family per target selected by the same researcher after viewing existing evaluators.", "No unseen repository, independent human author, population accuracy or historical model-performance claim.", "All misses, errors, uncovered cases and nonmanifested faults remain in the saved record."], "cost_cad": 0}
    save(output, result)
    lines = ["# Reproduction of the published source-family challenge", "", "The nine published cases were rerun with hash-verified, unchanged evaluators and imports. No new source family or untouched evaluation set is claimed.", "", "These are three researcher-constructed source families with paired reference and benign controls, not independent human validation or an accuracy estimate.", "", "| Method | Manifested faults | Detected | Missed | Uncovered | Benign false alarms | Errors |", "|---|---:|---:|---:|---:|---:|---:|"]
    for method, metrics in methods.items():
        lines.append(f"| {method} | {metrics['independently_manifested_faults']} | {metrics['faults_detected']} | {metrics['faults_missed']} | {metrics['faults_uncovered']} | {metrics['benign_false_alarms']}/{metrics['benign_controls']} | {metrics['errors']} |")
    lines += ["", "Every case has a separate actual-source manifestation witness and raw output. Missing native test coverage is unavailable, not a failed test. No existing result, protocol or evaluator was changed.", ""]
    (folder / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps({"rows": len(rows), "summary": methods, "published_summary_matches": summary_matches, "unmanifest_or_errors": [r["id"] for r in rows if not r["oracle_valid"] or "execution_error" in r]}, indent=2))
    if not summary_matches:
        raise SystemExit("Reproduction summary differs; all outputs were preserved")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("prepare", "run"), required=True)
    parser.add_argument("--workspace", type=Path, required=True, help="New external output directory; bundled evidence is never overwritten")
    names = ("agentprm-source-root", "toolace-source-root", "agentabstain-source-root", "agentprm-results", "toolace-results", "tokenizer")
    for name in names:
        parser.add_argument("--" + name, type=Path, help="Explicit acquired input; required for prepare and recorded in its protocol")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    if workspace == REPOSITORY_ROOT or workspace.is_relative_to(REPOSITORY_ROOT):
        parser.error("--workspace must be outside the repository so bundled evidence stays unchanged")
    if args.phase == "prepare":
        missing = ["--" + name for name in names if getattr(args, name.replace("-", "_")) is None]
        if missing:
            parser.error("prepare requires " + ", ".join(missing))
        inputs = {name.replace("-", "_"): str(getattr(args, name.replace("-", "_")).resolve()) for name in names}
        prepare(workspace, inputs)
    else:
        if any(getattr(args, name.replace("-", "_")) is not None for name in names):
            parser.error("run uses the exact input paths recorded by prepare; supply only --workspace")
        run(workspace)


if __name__ == "__main__":
    main()
