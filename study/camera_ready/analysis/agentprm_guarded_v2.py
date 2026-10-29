#!/usr/bin/env python3
"""Preserved-v2 guard confirmation on exposed AgentPRM source challenges.

Version 1 stays unchanged. This evaluates availability from real paths and
discriminating eligibility from executed inputs, without an expected label.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location("agentprm_frozen_v1", HERE.with_name("agentprm_analysis.py"))
V1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V1)


def entry(decision, reason):
    return {"decision":decision,"reason":reason}


def structural_q_schema(output):
    if not isinstance(output,dict) or not output:
        return False
    for value in output.values():
        if not isinstance(value,dict) or not {"state","reason_action","qestimate","count"}.issubset(value):
            return False
        if not isinstance(value["state"],dict) or not isinstance(value["reason_action"],dict):
            return False
        if not isinstance(value["qestimate"],(int,float)) or not math.isfinite(value["qestimate"]):
            return False
        if not isinstance(value["count"],int) or value["count"] < 1:
            return False
    return True


def evaluate(source_root, mechanism, fixture_paths, gamma, legacy_path):
    """No expected-state/label argument exists; guards use actual evidence."""
    source_root = Path(source_root)
    missing = [p for p in V1.FILES if not (source_root/p).is_file()]
    unavailable = [str(p) for p in fixture_paths if not Path(p).is_file()]
    evidence = {"source_root":str(source_root),"required_source_files":V1.FILES,
                "missing_source_files":missing,"missing_fixture_paths":unavailable,
                "gamma":gamma,"executed_implementation_probes":0}
    if missing or unavailable:
        return {"eligible_examples":0,"evidence":evidence,"observed":{},
                "methods":{m:entry("not_evaluated","Required source/input file is unavailable.")
                           for m in ["schema","native_tests","source_pattern","behavioral_contract"]}}
    fixtures=[(Path(p),json.loads(Path(p).read_text())) for p in fixture_paths]
    evidence["source_hashes"]={p:V1.digest(source_root/p) for p in V1.FILES}
    evidence["fixture_hashes"]={str(p):V1.digest(p) for p,_ in fixtures}
    if mechanism == "query_history_action":
        observed=V1.history_observation(source_root)
        evidence["executed_implementation_probes"]=1
        eligible=observed["history_entries"]
        schema=(isinstance(observed["saved_actions"],list)
                and all(isinstance(a,str) for a in observed["saved_actions"])
                and all(isinstance(h,list) and all(isinstance(x,dict)
                    and {"action","observation"}.issubset(x)
                    and isinstance(x["action"],str) and isinstance(x["observation"],str)
                    for x in h) for h in observed["query_histories"]))
        fault=observed["history_action_mismatches"]>0 or not observed["saved_actions_match_execution"]
        evidence["eligibility_rule"]="Actual rollout supplied at least one previous action to a subsequent query."
    else:
        actual={};raw={};errors=[];eligible=0;schema=True
        for relative in [V1.TARGET,V1.PREFERENCE]:
            obj=V1.module(source_root/relative);values=[];raw[relative]=[]
            for path,data in fixtures:
                output=obj.process_file(str(path),gamma)
                evidence["executed_implementation_probes"]+=1
                schema &= structural_q_schema(output)
                row_values=V1.state_values(output,data)
                target=V1.expected_values(len(data["trajectory"]),gamma,data["trajectory"][-1]["score"])
                # Eligibility derives from actual input length/discount. Unit
                # discount is still executed and preserved before abstaining.
                if len(data["trajectory"])>1 and gamma!=1:
                    eligible+=len(data["trajectory"])
                errors.extend(abs(a-b) for a,b in zip(row_values,target))
                values.append({"fixture":path.name,"signed_values":row_values,
                               "trajectory_length":len(data["trajectory"]),
                               "terminal_score":data["trajectory"][-1]["score"]})
                raw[relative].append(output)
            actual[relative]=values
        observed={"paths":actual,"raw_outputs":raw,"max_abs_error":max(errors,default=0)}
        fault=observed["max_abs_error"]>1e-12
        evidence["eligibility_rule"]="Executed trajectory has more than one state and non-unit discount; otherwise exponent direction is undiscriminated."
    if eligible==0:
        methods={m:entry("not_evaluated","Actual executed inputs provide no eligible discriminating examples.")
                 for m in ["schema","native_tests","source_pattern","behavioral_contract"]}
    else:
        legacy=V1.legacy_decisions(source_root,legacy_path)
        ids=["APRM-SCHEMA-01"] if mechanism=="query_history_action" else ["APRM-TEMP-01","APRM-TEMP-02","APRM-TEMP-03","APRM-METRIC-01"]
        methods={"schema":entry("pass" if schema else "fail","Executed required-field/type/finite-number validator over actual outputs."),
                 "native_tests":entry("not_evaluated","No relevant native dataproc/history tests at the pinned revision."),
                 "source_pattern":entry("fail" if any(legacy[k]["status"]=="discrepant" for k in ids) else "pass","Executed unchanged v1 public adapter for relevant IDs."),
                 "behavioral_contract":entry("fail" if fault else "pass","Actual implementation compared against the frozen terminal/history expectation.")}
        observed["legacy_checks"]={k:legacy[k] for k in ids}
    return {"eligible_examples":eligible,"evidence":evidence,"observed":observed,"methods":methods}


def independent_oracle_valid(expected_state, result, mechanism):
    if result["eligible_examples"]==0:
        return expected_state=="not_evaluated"
    observed=result["observed"]
    if mechanism=="query_history_action":
        bad=any(x["action"]!=f"act-{i}" for h in observed["query_histories"] for i,x in enumerate(h))
    else:
        # Structural terminal anchoring is independent of the full-value
        # expectation used by the behavioral checker.
        bad=any(abs(x["signed_values"][-1]-(2*x["terminal_score"]-1))>1e-12
                for path_rows in observed["paths"].values() for x in path_rows)
    return bad if expected_state=="fault" else not bad


def mean_scope(results, gamma):
    paths=[]
    folder=results/"scope_inputs_v2"
    folder.mkdir()
    for length in [2,4]:
        data=V1.trajectory(length,1,"shared-prefix")
        path=folder/f"shared-prefix-length-{length}.json"
        V1.dump(path,data);paths.append(path)
    result={"scope":"Complete trajectory visit mean is invariant to exponent reversal. An unweighted mean after deduplicating and count-merging state-action evidence need not be invariant.",
            "gamma":gamma,"fixture_lengths":[2,4],"fixture_paths":[str(x) for x in paths],"versions":{}}
    for name in ["original","repaired"]:
        obj=V1.module(results/"sources"/name/V1.TARGET)
        outputs=[obj.process_file(str(p),gamma) for p in paths]
        visit_values=[x["qestimate"] for q in outputs for x in q.values()]
        merged=obj.merge_results(copy.deepcopy(outputs))
        values=[x["qestimate"] for x in merged.values()]
        weighted=sum(x["qestimate"]*x["count"] for x in merged.values())/sum(x["count"] for x in merged.values())
        result["versions"][name]={"complete_visit_mean_signed":sum(visit_values)/len(visit_values),
              "unweighted_unique_state_action_mean_signed":sum(values)/len(values),
              "count_weighted_merged_mean_signed":weighted,"unique_state_action_entries":len(values),
              "visits":len(visit_values),"raw_merged":merged}
    a,b=result["versions"]["original"],result["versions"]["repaired"]
    result["complete_visit_mean_difference_signed"]=b["complete_visit_mean_signed"]-a["complete_visit_mean_signed"]
    result["unique_state_action_mean_difference_signed"]=b["unweighted_unique_state_action_mean_signed"]-a["unweighted_unique_state_action_mean_signed"]
    result["unique_state_action_mean_difference_exported"]=result["unique_state_action_mean_difference_signed"]/2
    assert abs(result["complete_visit_mean_difference_signed"])<1e-12
    assert abs(result["unique_state_action_mean_difference_signed"])>1e-4
    V1.dump(results/"mean_scope_companion_v2.json",result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--protocol",type=Path,required=True)
    parser.add_argument("--results-dir",type=Path,required=True)
    args=parser.parse_args();protocol=json.loads(args.protocol.read_text());results=args.results_dir
    assert V1.digest(HERE)==protocol["evaluator_sha256"]
    assert V1.digest(HERE.with_name("agentprm_analysis.py"))==protocol["inherited_evaluator_sha256"]
    assert not (results/"challenge_results_v2.json").exists()
    V1.dump(results/"guarded_evaluation_freeze_v2.json",{"evaluator_sha256":V1.digest(HERE),"protocol_sha256":V1.digest(args.protocol),
        "started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"reuse":"All prior challenge sources/inputs are exposed; no new heldout claim."})
    old=json.loads((results/"challenge_results_v1.json").read_text());rows=[]
    legacy=Path(protocol["legacy_adapter"])
    fixture_paths=[results/"fixtures/challenge-0.json",results/"fixtures/challenge-1.json"]
    for prior in old["rows"]:
        if prior["variant"]=="missing":source=results/"sources/deliberately_unavailable"
        else:source=results/"sources"/prior["variant"]
        gamma=1. if prior["id"].endswith("INELIGIBLE") else .95
        result=evaluate(source,prior["mechanism"],fixture_paths,gamma,legacy)
        row={"id":prior["id"]+"-V2","target":"agentprm","mechanism":prior["mechanism"],"variant":prior["variant"],
            "partition":"development","inherited_partition":prior["partition"],"exposed_rerun":True,
            "expected_state":prior["expected_state"],"oracle_valid":independent_oracle_valid(prior["expected_state"],result,prior["mechanism"]),
            "eligible_examples":result["eligible_examples"],"implementation":prior["implementation"],
            "methods":result["methods"],"observed":result["observed"],"guard_evidence":result["evidence"]}
        rows.append(row)
    scope=mean_scope(results,.95)
    V1.dump(results/"challenge_results_v2.json",{"schema_version":"2.0","version":"guarded-confirmation-v2",
        "rows":rows,"summary":V1.summarize_challenges(rows),"all_oracles_valid":all(x["oracle_valid"] for x in rows),
        "protocol_sha256":V1.digest(args.protocol),"evaluator_sha256":V1.digest(HERE),
        "scope_companion":"mean_scope_companion_v2.json","limitations":["Reuses exposed v1 inputs and source variants; this is confirmation and guard validation, not untouched heldout evidence.","Finite selected mechanisms do not establish population detection accuracy."]})
    print(json.dumps({"rows":len(rows),"all_oracles_valid":all(x["oracle_valid"] for x in rows),"mean_scope_difference_signed":scope["unique_state_action_mean_difference_signed"]},indent=2))


if __name__=="__main__":main()
