#!/usr/bin/env python3
"""Check AgentPRM targets, preference exports, and query history.

Import target and preference functions. Extract the rollout and utility
functions to run them with a deterministic agent and environment."""
from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

PIN = "e4714717f7f4bd4671848670c4ed54d0169f603a"
TARGET = "scripts/dataproc/compute_prm_target.py"
PREFERENCE = "scripts/dataproc/compute_prm_preference_target.py"
ROLLOUT = "scripts/dataproc/rollout_alfworld.py"
UTILS = "src/agent_prm/utils/alfworld.py"
FILES = [TARGET, PREFERENCE, ROLLOUT, UTILS]
TOL = 1e-12


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def extract(source: str, names: list[str], filename: str, env: dict) -> list[dict]:
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError("missing extracted function")
    provenance = []
    for node in nodes:
        segment = ast.get_source_segment(source, node)
        provenance.append({"name": node.name, "first_line": node.lineno,
                           "last_line": node.end_lineno,
                           "source_sha256": hashlib.sha256(segment.encode()).hexdigest()})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec"), env)
    return provenance


def module(path: Path):
    name = "agentprm_camera_" + digest(path)[:16] + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def variant(root: Path, source_text: dict, name: str, temporal: str, history: str,
            benign: str = "") -> Path:
    dst = root / name
    for relative, text in source_text.items():
        if relative in (TARGET, PREFERENCE):
            old = "outcome_reward, gamma, t)\n        return Q_target"
            if text.count(old) != 1:
                raise ValueError("caller patch anchor changed")
            text = text.replace(old, f"outcome_reward, gamma, {temporal})\n        return Q_target")
            if benign == "equivalent_power":
                text = text.replace("(gamma ** t)", "pow(gamma, t)")
        if relative == ROLLOUT:
            old = '"action": env_data["observation"][t]} for t'
            if text.count(old) != 1:
                raise ValueError("history patch anchor changed")
            text = text.replace(old, f'"action": {history}}} for t')
        path = dst / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return dst


def trajectory(length: int, outcome: int, branch: str = "a", shared: bool = False) -> dict:
    rows = []
    for t in range(length):
        rows.append({"observation": "shared-start" if shared and t == 0 else f"{branch}-obs-{t}",
                     "candidate_actions": ["left", "right"] if shared and t == 0 else [f"{branch}-act-{t}"],
                     "action": branch if shared and t == 0 else f"{branch}-act-{t}",
                     "reason": "controlled action", "score": outcome if t == length - 1 else 0,
                     "agent_log": []})
    return {"task": "shared-task" if shared else f"task-{branch}", "gamefile": "synthetic", "trajectory": rows}


def save_fixture(path: Path, data: dict) -> Path:
    dump(path, data)
    return path


def state_values(output: dict, data: dict) -> list[float]:
    values = {entry["state"]["observation"]: entry["qestimate"] for entry in output.values()}
    return [values[x["observation"]] for x in data["trajectory"]]


def expected_values(length: int, gamma: float, outcome: int) -> list[float]:
    # Compute reference targets by repeated multiplication from the terminal state.
    signed = 2 * outcome - 1
    result = [float(signed)]
    for _ in range(length - 1):
        result.append(result[-1] * gamma)
    return list(reversed(result))


def temporal_observation(root: Path, fixtures: list[tuple[Path, dict]], gamma: float) -> dict:
    errors, all_values = [], {}
    for relative in (TARGET, PREFERENCE):
        obj = module(root / relative)
        rows = []
        for path, data in fixtures:
            values = state_values(obj.process_file(str(path), gamma), data)
            expected = expected_values(len(data["trajectory"]), gamma, data["trajectory"][-1]["score"])
            errors.extend(abs(a - b) for a, b in zip(values, expected))
            rows.append({"fixture": path.name, "signed_values": values, "expected": expected})
        all_values[relative] = rows
    return {"max_abs_error": max(errors, default=0), "paths": all_values}


class FakeEnvironment:
    def __init__(self, length=4):
        self.length = length
        self.batch_env = SimpleNamespace(envs=[SimpleNamespace(max_episode_steps=length)])
        self.actions = []
        self.index = 0

    def reset(self):
        self.index = 0
        self.actions = []
        return ["room-0\n\nYour task is to: complete controlled sequence"], {
            "extra.gamefile": ["/tmp/json_2.1.1/controlled/game.tw-pddl"],
            "admissible_commands": [["act-0"]]}

    def step(self, actions):
        assert actions == [f"act-{self.index}"]
        self.actions.extend(actions)
        self.index += 1
        return [f"room-{self.index}"], [int(self.index == self.length)], [self.index == self.length], {
            "admissible_commands": [[f"act-{self.index}"]]}


class FakeAgent:
    def reset(self):
        self.queries = []

    def predict_reason_action_batch(self, queries, num_responses):
        self.queries.append(copy.deepcopy(queries[0]))
        return [[{"action": f"act-{len(self.queries) - 1}", "reason": "controlled"}]]

    def get_log(self):
        return None

    def name(self):
        return "controlled-recording-agent"


def history_observation(root: Path, length=4) -> dict:
    env = {"Optional": Optional, "re": re, "tqdm": lambda x, **kwargs: x}
    records = extract((root / UTILS).read_text(), ["parse_gamefile_string", "extract_task_from_observation"], str(root / UTILS), env)
    records += extract((root / ROLLOUT).read_text(), ["rollout_agent_env"], str(root / ROLLOUT), env)
    world, agent = FakeEnvironment(length), FakeAgent()
    rollouts, summaries = env["rollout_agent_env"](world, agent)
    if rollouts is None:
        raise RuntimeError("upstream rollout returned failure")
    saved = [r["action"] for r in rollouts[0]["trajectory"]]
    history = [r["observation_action_history"] for r in agent.queries]
    mismatches = sum(entry["action"] != world.actions[t]
                     for h in history for t, entry in enumerate(h))
    return {"query_histories": history, "executed_actions": world.actions,
            "saved_actions": saved, "saved_actions_match_execution": saved == world.actions,
            "history_entries": sum(map(len, history)), "history_action_mismatches": mismatches,
            "summaries": summaries, "unchanged_ast_extractions": records}


def legacy_decisions(root: Path, legacy_path: Path) -> dict:
    # Run the original adapter and capture its result.
    env = {"Path": Path, "Any": Any, "time": time, "ContractResult": dict,
           "timed_result": lambda cid, target, status, observed, expected, severity, consequence, evidence, started:
           {"id": cid, "status": status, "observed": observed}}
    extract(legacy_path.read_text(), ["_result", "audit_agentprm"], str(legacy_path), env)
    rows = env["audit_agentprm"](root, [])
    return {row["id"]: row for row in rows}


def challenge_rows(variants: dict, output: Path, fixtures, protocol: dict, legacy: Path) -> list[dict]:
    rows = []
    definitions = protocol["challenges"]
    native = {"decision": "not_evaluated", "reason": "No native test covers the selected target or query-history behavior."}
    for spec in definitions:
        root = variants.get(spec["variant"])
        row = {"id": spec["id"], "target": "agentprm", "mechanism": spec["mechanism"],
               "variant": spec["variant"], "partition": spec["partition"],
               "expected_state": spec["expected_state"], "oracle_valid": False,
               "eligible_examples": 0, "implementation": {}, "methods": {}, "observed": {}}
        methods = row["methods"]
        methods["native_tests"] = native.copy()
        if root is None or spec.get("ineligible"):
            reason = "required implementation missing" if root is None else "At gamma=1, both exponent directions produce identical targets."
            for name in ("schema", "source_pattern", "behavioral_contract"):
                methods[name] = {"decision": "not_evaluated", "reason": reason}
            row["oracle_valid"] = True
            rows.append(row)
            continue
        row["implementation"] = {"revision": PIN,
             "source_files": {relative: digest(root / relative) for relative in FILES},
             "base_source_files": protocol["source_hashes"],
             "patch_sha256": hashlib.sha256(json.dumps({p: (root / p).read_text() for p in FILES}, sort_keys=True).encode()).hexdigest(),
             "patch_record": str(root.relative_to(output))}
        try:
            if spec["mechanism"] == "query_history_action":
                obs = history_observation(root)
                failed = obs["history_action_mismatches"] > 0 or not obs["saved_actions_match_execution"]
                eligible = obs["history_entries"]
            else:
                obs = temporal_observation(root, fixtures, protocol["primary_gamma"])
                failed = obs["max_abs_error"] > TOL
                eligible = sum(len(d["trajectory"]) for _, d in fixtures) * 2
            row["observed"] = obs
            row["eligible_examples"] = eligible
            row["oracle_valid"] = ((failed and spec["expected_state"] == "fault") or
                                    (not failed and spec["expected_state"] == "conformant"))
            methods["behavioral_contract"] = {"decision": "fail" if failed else "pass",
                                              "reason": "Implementation output compared with repeated-multiplication targets or recorded environment actions."}
            # Check output fields and types.
            if spec["mechanism"] == "query_history_action":
                schema_valid = (all(isinstance(x,str) for x in obs["saved_actions"])
                    and all(isinstance(h,list) and all(set(("action","observation")).issubset(x)
                            and isinstance(x["action"],str) and isinstance(x["observation"],str)
                            for x in h) for h in obs["query_histories"]))
            else:
                schema_valid = all(isinstance(x["signed_values"],list) and bool(x["signed_values"])
                    and all(isinstance(v,(int,float)) and math.isfinite(v) for v in x["signed_values"])
                    for path_rows in obs["paths"].values() for x in path_rows)
            methods["schema"] = {"decision": "pass" if schema_valid else "fail", "reason": "Checks required fields and numeric, string, and list types."}
            old = legacy_decisions(root, legacy)
            ids = ["APRM-SCHEMA-01"] if spec["mechanism"] == "query_history_action" else ["APRM-TEMP-01", "APRM-TEMP-02", "APRM-TEMP-03", "APRM-METRIC-01"]
            methods["source_pattern"] = {"decision": "fail" if any(old[k]["status"] == "discrepant" for k in ids) else "pass",
                                         "reason": "Runs the original adapter and its formula calculations for the applicable checks."}
            row["observed"]["legacy_checks"] = {key: old[key] for key in ids}
        except Exception as exc:
            row["observed"]["error"] = repr(exc)
            for name in ("schema", "source_pattern", "behavioral_contract"):
                methods.setdefault(name, {"decision": "error", "reason": repr(exc)})
        rows.append(row)
    return rows


def summarize_challenges(rows: list[dict]) -> dict:
    out = {}
    for method in ("schema", "native_tests", "source_pattern", "behavioral_contract"):
        out[method] = {}
        for part in ("development", "locked_evaluation", "all"):
            subset = [r for r in rows if part == "all" or r["partition"] == part]
            eligible = [r for r in subset if r["oracle_valid"] and r["eligible_examples"] > 0]
            faults = [r for r in eligible if r["expected_state"] == "fault"]
            conformant = [r for r in eligible if r["expected_state"] == "conformant"]
            repairs = [r for r in conformant if r["variant"] == "repaired"]
            benign = [r for r in conformant if "benign" in r["variant"]]
            out[method][part] = {"faults": len(faults),
                "faults_detected": sum(r["methods"][method]["decision"] == "fail" for r in faults),
                "repairs": len(repairs), "repairs_accepted": sum(r["methods"][method]["decision"] == "pass" for r in repairs),
                "conformant": len(conformant), "conformant_accepted": sum(r["methods"][method]["decision"] == "pass" for r in conformant),
                "benign": len(benign), "benign_false_alarms": sum(r["methods"][method]["decision"] == "fail" for r in benign),
                "not_evaluated": sum(r["methods"][method]["decision"] == "not_evaluated" for r in subset),
                "errors": sum(r["methods"][method]["decision"] == "error" for r in subset)}
    return out


def scientific_analysis(variants: dict, out: Path, protocol: dict) -> dict:
    original, repaired = module(variants["original"] / TARGET), module(variants["repaired"] / TARGET)
    preference_original = module(variants["original"] / PREFERENCE)
    preference_repaired = module(variants["repaired"] / PREFERENCE)
    tokens, sweep = [], []
    for length in protocol["lengths"]:
        for outcome in protocol["terminal_outcomes"]:
            data = trajectory(length, outcome, f"sweep-{length}-{outcome}")
            path = save_fixture(out / "fixtures" / f"sweep-{length}-{outcome}.json", data)
            for gamma in protocol["gammas"]:
                old = state_values(original.process_file(str(path), gamma), data)
                new = state_values(repaired.process_file(str(path), gamma), data)
                ref = expected_values(length, gamma, outcome)
                assert max(abs(x-y) for x, y in zip(new, ref)) < TOL
                assert abs(sum(old) / length - sum(new) / length) < TOL
                for t, (a, b) in enumerate(zip(old, new)):
                    tokens.append({"length": length, "terminal_score": outcome, "gamma": gamma, "position": t,
                                   "original_signed_return": a, "repaired_signed_return": b,
                                   "original_exported_target": .5*(a+1), "repaired_exported_target": .5*(b+1),
                                   "signed_delta": b-a, "exported_delta": .5*(b-a)})
                sweep.append({"length": length, "terminal_score": outcome, "gamma": gamma,
                              "mean_signed_original": sum(old)/length, "mean_signed_repaired": sum(new)/length,
                              "mean_abs_signed_distortion": sum(abs(a-b) for a,b in zip(old,new))/length,
                              "max_abs_signed_distortion": max(abs(a-b) for a,b in zip(old,new)),
                              "max_abs_exported_distortion": .5*max(abs(a-b) for a,b in zip(old,new))})
    # Same initial state; different actions and valid successful continuations.
    # At t=0, the repair discounts by the remaining trajectory length.
    preference_rows, preference_raw = [], []
    lengths = protocol["preference_continuation_lengths"]
    for gamma in protocol["gammas"]:
        for short in lengths:
            for long in lengths:
                if short >= long:
                    continue
                paths, datas = [], []
                for branch, length in (("left", short), ("right", long)):
                    d = trajectory(length, 1, branch, shared=True)
                    p = out / "fixtures" / f"pref-{short}-{long}-{branch}.json"
                    if not p.exists(): save_fixture(p, d)
                    paths.append(p); datas.append(d)
                values, sets = [], []
                for obj in (preference_original, preference_repaired):
                    q = obj.merge_results([obj.process_file(str(p), gamma) for p in paths])
                    state_map = obj.create_state_action_map(q)
                    preferences = obj.create_preferences(state_map, protocol["primary_delta"])
                    pairs = {(p["reason_action_chosen"]["action"], p["reason_action_rejected"]["action"])
                             for p in preferences if p["state"]["observation"] == "shared-start"}
                    sets.append(pairs)
                    values.append({entry["reason_action"]["action"]: entry["qestimate"]
                                   for entry in q.values() if entry["state"]["observation"] == "shared-start"})
                expected_margin = gamma ** (short-1) - gamma ** (long-1)
                assert values[0] == {"left": 1., "right": 1.}
                assert abs(values[1]["left"] - gamma**(short-1)) < TOL
                assert abs(values[1]["right"] - gamma**(long-1)) < TOL
                assert sets[1] == ({("left", "right")} if expected_margin > protocol["primary_delta"] else set())
                preference_rows.append({"gamma": gamma, "delta": protocol["primary_delta"],
                                        "short_length": short, "long_length": long,
                                        "original_pairs": len(sets[0]), "repaired_pairs": len(sets[1]),
                                        "added_pairs": len(sets[1]-sets[0]), "removed_pairs": len(sets[0]-sets[1]),
                                        "short_repaired_signed": values[1]["left"],
                                        "long_repaired_signed": values[1]["right"], "signed_margin": expected_margin})
                preference_raw.append({"gamma": gamma, "lengths": [short,long], "values": values,
                                       "preference_sets": [sorted(x) for x in sets]})
    # Repeated state-action pairs test the count-weighted merge.
    merge_fixtures = []
    for index, outcome in enumerate([1,1,0]):
        d = trajectory(4, outcome, "merge")
        p = save_fixture(out / "fixtures" / f"merge-{index}.json", d)
        merge_fixtures.append((p,d))
    merged = repaired.merge_results([repaired.process_file(str(p), protocol["primary_gamma"]) for p,_ in merge_fixtures])
    merge_values = state_values(merged, merge_fixtures[0][1])
    merge_ref = [x/3 for x in expected_values(4, protocol["primary_gamma"], 1)]
    assert max(abs(x-y) for x,y in zip(merge_values, merge_ref)) < TOL
    assert all(x["count"] == 3 for x in merged.values())
    write_csv(out / "state_targets_v1.csv", tokens)
    write_csv(out / "target_sensitivity_v1.csv", sweep)
    write_csv(out / "preference_sensitivity_v1.csv", preference_rows)
    dump(out / "preference_raw_v1.json", preference_raw)
    history_old = history_observation(variants["original"])
    history_new = history_observation(variants["repaired"])
    assert history_old["history_action_mismatches"] == 6
    assert history_new["history_action_mismatches"] == 0
    assert history_old["saved_actions_match_execution"] and history_new["saved_actions_match_execution"]
    primary = [r for r in sweep if r["gamma"] == protocol["primary_gamma"]]
    pref_primary = [r for r in preference_rows if r["gamma"] == protocol["primary_gamma"]]
    return {"source_revision": PIN, "scope": "Controlled trajectories with a deterministic agent and environment.",
            "primary_gamma": protocol["primary_gamma"], "primary_delta": protocol["primary_delta"],
            "state_sweep_cells": len(sweep), "state_rows": len(tokens),
            "mean_identity_max_abs_error": max(abs(r["mean_signed_original"]-r["mean_signed_repaired"]) for r in sweep),
            "primary_target_sensitivity": primary,
            "preference_comparisons": len(preference_rows), "primary_preference_comparisons": len(pref_primary),
            "primary_preference_added_cases": sum(r["added_pairs"]>0 for r in pref_primary),
            "primary_preference_removed_cases": sum(r["removed_pairs"]>0 for r in pref_primary),
            "preference_scope": "Two distinct actions from the same initial state, followed by successful continuations of different lengths.",
            "merge_check": {"input_outcomes": [1,1,0], "counts": sorted(x["count"] for x in merged.values()), "values": merge_values, "reference": merge_ref},
            "query_history": {"original": history_old, "repaired": history_new}}


def top_level_smoke(variants: dict, out: Path, python: str, protocol: dict) -> list[dict]:
    "Run target and preference exports through the upstream command line."
    import pyarrow.parquet as pq
    fixture_dir = out / "cli_fixtures"
    fixture_dir.mkdir()
    for branch, length in (("left",4),("right",8)):
        save_fixture(fixture_dir / f"{branch}.json", trajectory(length,1,branch,True))
    records = []
    for name in ("original", "repaired"):
        for relative in (TARGET, PREFERENCE):
            label = f"{name}-{Path(relative).stem}"
            destination = out / "top_level" / label
            config = out / "configs" / f"{label}.yaml"
            config.parent.mkdir(exist_ok=True)
            text = (f"rolloutdirs:\n  - {fixture_dir}\nmax_files_per_dir: null\noutputdir: {destination}\n"
                    f"gamma: {protocol['primary_gamma']}\ntrain_split: 0.85\ndelta: {protocol['primary_delta']}\n")
            config.write_text(text)
            cmd = [python, str(variants[name] / relative), "--config", str(config), "--seed", "42"]
            run = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            (out / "logs" / f"{label}.stdout.log").write_text(run.stdout)
            (out / "logs" / f"{label}.stderr.log").write_text(run.stderr)
            record = {"command": cmd, "returncode": run.returncode, "variant": name,
                      "path": relative, "output": str(destination), "records": []}
            if run.returncode == 0:
                records_all = []
                for file in sorted(destination.glob("*.parquet")):
                    table = pq.read_table(file)
                    records_all.extend(table.to_pylist())
                    record["records"].append({"file": str(file), "sha256": digest(file), "rows": len(table)})
                if relative == TARGET:
                    # Match exported rows to process_file outputs by state/action.
                    obj = module(variants[name] / TARGET)
                    q = obj.merge_results([obj.process_file(str(p),protocol["primary_gamma"]) for p in sorted(fixture_dir.glob("*.json"))])
                    reference = {(x["state"]["observation"],x["reason_action"]["action"]):.5*(x["qestimate"]+1) for x in q.values()}
                    assert len(records_all) == len(reference)
                    assert all(abs(r["qestimate"]-reference[(r["state"]["observation"],r["reason_action"]["action"])])<TOL for r in records_all)
                    record["affine_export_verified"] = True
                else:
                    obj = module(variants[name] / PREFERENCE)
                    q = obj.merge_results([obj.process_file(str(p),protocol["primary_gamma"]) for p in sorted(fixture_dir.glob("*.json"))])
                    expected = obj.create_preferences(obj.create_state_action_map(q),protocol["primary_delta"])
                    assert sorted(json.dumps(x,sort_keys=True) for x in records_all)==sorted(json.dumps(x,sort_keys=True) for x in expected)
                    record["preference_export_verified"] = True
                    record["exported_preferences"] = len(records_all)
            records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    for folder in ("logs","fixtures","sources"):(out/folder).mkdir()
    protocol = json.loads(args.protocol.read_text())
    source_root = args.source_root.resolve()
    revision = subprocess.check_output(["git","-C",str(source_root),"rev-parse","HEAD"],text=True).strip()
    assert revision == protocol["source_revision"] == PIN
    assert not subprocess.check_output(["git","-C",str(source_root),"status","--porcelain"],text=True).strip()
    texts = {relative:(source_root/relative).read_text() for relative in FILES}
    assert {p:digest(source_root/p) for p in FILES} == protocol["source_hashes"]
    assert digest(Path(__file__)) == protocol["analysis_sha256"]
    started = time.time()
    command = [sys.executable]+sys.argv
    dump(out/"execution_environment_v1.json",{"command":command,"cwd":os.getcwd(),"python":sys.version,
         "platform":platform.platform(),"source_revision":revision,"source_hashes":protocol["source_hashes"],
         "analysis_sha256":digest(Path(__file__)),"protocol_sha256":digest(args.protocol),"seed":42,
         "started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())})
    variants = {}
    for name,spec in protocol["variants"].items():
        variants[name]=variant(out/"sources",texts,name,spec["temporal_argument"],spec["history_expression"],spec.get("benign",""))
    fixtures=[]
    for i,(length,outcome) in enumerate([(4,1),(8,0)]):
        data=trajectory(length,outcome,f"challenge-{i}")
        fixtures.append((save_fixture(out/"fixtures"/f"challenge-{i}.json",data),data))
    analysis=scientific_analysis(variants,out,protocol)
    analysis["actual_cli_export_checks"]=top_level_smoke(variants,out,sys.executable,protocol)
    dump(out/"case_analysis_v1.json",analysis)
    # Record the script and protocol hashes before evaluation.
    dump(out/"locked_execution_manifest_v1.json",{"analysis_sha256":digest(Path(__file__)),
          "protocol_sha256":digest(args.protocol),"first_locked_execution_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
          "claims":"Additional transformations and inputs from the same repository."})
    rows=challenge_rows(variants,out,fixtures,protocol,Path(protocol["legacy_adapter"]))
    payload={"schema_version":"2.0","rows":rows,"summary":summarize_challenges(rows),
             "finite_set_only":True,"all_oracles_valid":all(r["oracle_valid"] for r in rows),
             "elapsed_seconds":time.time()-started}
    dump(out/"challenge_results_v1.json",payload)
    dump(out/"source_integrity_after_v1.json",{"unchanged":{p:digest(source_root/p)==protocol["source_hashes"][p] for p in FILES},"source_revision":revision})
    print(json.dumps({"output":str(out),"case_analysis":{"state_sweep_cells":analysis["state_sweep_cells"],"preference_added":analysis["primary_preference_added_cases"]},"challenge_summary":payload["summary"],"all_oracles_valid":payload["all_oracles_valid"]},indent=2))


if __name__ == "__main__":
    main()
