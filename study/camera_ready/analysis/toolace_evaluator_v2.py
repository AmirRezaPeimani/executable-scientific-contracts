#!/usr/bin/env python3
"""Transparent confirmation rerun with executable schema/eligibility guards.

All challenge cases were exposed by v1; none is claimed as a new holdout.
The original protocol, first runner and every v1 output remain unchanged.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import datetime as dt
import json
import sys
import torch
import pandas as pd
from transformers import AutoTokenizer
from toolace_analysis import digest, execute_loader, loader_callable, oracle, save, save_csv, LOADER


def schema_check(result, maximum):
    required = {"input_ids", "attention_mask", "position_ids", "loss_mask"}
    if not isinstance(result, dict) or set(result) != required:
        return {"pass": False, "reason": "Required tensor fields differ"}
    for name, value in result.items():
        if not isinstance(value, torch.Tensor):
            return {"pass": False, "reason": f"{name} is not a tensor"}
        if value.dtype != torch.int64 or value.ndim != 1 or value.numel() != maximum:
            return {"pass": False, "reason": f"{name} dtype/rank/length differs"}
    return {"pass": True, "reason": "Four required int64 one-dimensional tensors have configured length"}


def availability_guard(source_path, fixtures):
    if not source_path.is_file():
        return {"decision": "not_evaluated", "reason": "Required implementation source is absent", "source_exists": False, "eligible_examples": len(fixtures)}
    if not fixtures:
        return {"decision": "not_evaluated", "reason": "Selection produced zero eligible examples", "source_exists": True, "eligible_examples": 0}
    return {"decision": "evaluate", "reason": "Source exists and selected fixtures are nonempty", "source_exists": True, "eligible_examples": len(fixtures)}


def assess(source_path, fixtures, tokenizer, position_source):
    guard = availability_guard(source_path, fixtures)
    if guard["decision"] != "evaluate":
        return {"guard": guard, "outcomes": [], "methods": {
            k: {"decision": "not_evaluated", "reason": guard["reason"]}
            for k in ("schema", "native_tests", "source_pattern", "behavioral_contract")}}
    changed = source_path.read_text()
    try:
        call, _ = loader_callable(changed, position_source)
    except Exception as exc:
        return {"guard": guard, "outcomes": [], "methods": {
            k: {"decision": "error", "reason": f"Loader extraction failed: {type(exc).__name__}: {exc}"}
            for k in ("schema", "native_tests", "source_pattern", "behavioral_contract")}}
    outcomes = []
    for fixture in fixtures:
        p, r, maximum = fixture["prompt_tokens"], fixture["response_tokens"], fixture["maximum"]
        prompt, response = " x" * p, " y" * (r - 1)
        pp = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        rr = tokenizer(response + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        try:
            out = execute_loader(call, tokenizer, [prompt], [response], 0, maximum)
            schema = schema_check(out, maximum)
            if not schema["pass"]:
                outcomes.append({"fixture": fixture, "schema": schema, "behavior": "not_evaluated", "reason": "Malformed output cannot be interpreted as a valid token-level observation"})
                continue
            expected_ids = (pp + rr)[-maximum:]
            expected_ids += [tokenizer.pad_token_id] * (maximum - len(expected_ids))
            labels_equal = out["input_ids"][1:].tolist() == expected_ids[1:]
            inputs_equal = out["input_ids"].tolist() == expected_ids
            truth = oracle(len(pp), len(rr), maximum)
            full = int((out["loss_mask"] != truth).sum().item())
            consumed = int((out["loss_mask"][:-1] != truth[:-1]).sum().item())
            outcomes.append({"fixture": fixture, "schema": schema, "behavior": "pass" if not consumed and labels_equal and inputs_equal else "fail",
                             "full_mask_mismatches": full, "consumed_mask_mismatches": consumed,
                             "shifted_label_identity_equal": labels_equal, "retained_input_identity_equal": inputs_equal,
                             "coded_supervised_predictions": int(out["loss_mask"][:-1].sum().item()),
                             "oracle_supervised_predictions": int(truth[:-1].sum().item())})
        except Exception as exc:
            outcomes.append({"fixture": fixture, "schema": {"pass": False, "reason": "Loader raised before schema observation"},
                             "behavior": "error", "exception": {"type": type(exc).__name__, "message": str(exc)}})
    errors = [o for o in outcomes if o["behavior"] == "error"]
    schema_bad = [o for o in outcomes if not o["schema"]["pass"] and o["behavior"] != "error"]
    if errors:
        schema_decision, behavioral_decision = "error", "error"
    else:
        schema_decision = "fail" if schema_bad else "pass"
        if schema_bad:
            behavioral_decision = "not_evaluated"
        else:
            behavioral_decision = "fail" if any(o["behavior"] == "fail" for o in outcomes) else "pass"
    source_defect = "input_ids = input_ids[-self.max_length :]" in changed and "min(prompt_length, loss_mask.size(0))" in changed
    masks_prompt = "loss_mask[: min(prompt_length, loss_mask.size(0)) - 1] = 0" in changed
    masks_terminal = "loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0" in changed
    return {"guard": guard, "outcomes": outcomes, "methods": {
        "schema": {"decision": schema_decision, "reason": f"Executed required-key/tensor/dtype/rank/length predicates on {len(outcomes)} observed outputs; malformed outputs={len(schema_bad)}; errors={len(errors)}"},
        "native_tests": {"decision": "not_evaluated", "reason": "No relevant native test for pinned custom loader identified; coverage unavailable, not a native failure"},
        "source_pattern": {"decision": "fail" if source_defect or not (masks_prompt and masks_terminal) else "pass",
                           "reason": "Historical coordinate and mask-source predicates executed on this exact variant",
                           "predicates": {"source_defect": source_defect, "masks_prompt": masks_prompt, "masks_terminal": masks_terminal}},
        "behavioral_contract": {"decision": behavioral_decision,
                                "reason": f"Exact retained token identities plus trainer-consumed mask[:-1] against frozen independent prediction-index oracle; failures={sum(o['behavior'] == 'fail' for o in outcomes)}; errors={len(errors)}"}}}


def main():
    p = argparse.ArgumentParser()
    for name in ("source-root", "tokenizer", "results", "protocol"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    dest = a.results / "challenge_results_verified_v2.json"
    if dest.exists():
        raise FileExistsError(dest)
    protocol = json.loads(a.protocol.read_text())
    if digest(Path(__file__).read_bytes()) != protocol["runner_sha256"]:
        raise ValueError("Runner changed after confirmation protocol freeze")
    original = a.results / "challenge_results_v1.json"
    if digest(original.read_bytes()) != protocol["first_challenge_sha256"]:
        raise ValueError("First challenge record differs from confirmation freeze")
    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    position_source = (a.source_root / "verl/verl/utils/model.py").read_text()
    fixtures = protocol["controlled_fixtures"]
    old = json.loads(original.read_text())
    census = pd.read_csv(a.results / "official_loader_census_v1.csv")
    eligible_census = census[census.overlength_at_official_8192]
    rows, observation_rows = [], []
    for inherited in old["rows"]:
        row = dict(inherited)
        ident = row["id"]
        if ident == "TACE-MISSING-EVIDENCE":
            source_path = a.results / "deliberately_absent_source.py"
            selected = fixtures
            selector = {"kind": "fixed controlled grid", "selected": len(fixtures)}
        elif ident == "TACE-ZERO-ELIGIBLE":
            source_path = a.source_root / LOADER
            # Actual census filter is the same eligibility guard exercised
            # for valid variants, not a preassigned not_evaluated label.
            selected = [{"prompt_tokens": int(x.prompt_tokens), "response_tokens": int(x.response_tokens_including_eos), "maximum": 8192} for x in eligible_census.itertuples()]
            selector = {"kind": "official census sequence_tokens > 8192", "source_rows": len(census),
                        "selected": len(selected), "source_csv_sha256": digest((a.results / "official_loader_census_v1.csv").read_bytes())}
        else:
            source_path = a.results / row["implementation"]["source"]
            if digest(source_path.read_bytes()) != row["implementation"]["source_sha256"]:
                raise ValueError(f"Variant source changed: {ident}")
            selected = fixtures
            selector = {"kind": "fixed controlled grid", "selected": len(fixtures)}
        observed = assess(source_path, selected, tokenizer, position_source)
        row["inherited_partition"] = row["partition"]
        row["partition"] = "development"
        row["evaluation_status"] = "exposed_confirmation_rerun_after_v1_inspection"
        row["eligible_examples"] = len(selected) if observed["guard"]["source_exists"] else 0
        row["methods"] = observed["methods"]
        failures = sum(x["behavior"] == "fail" for x in observed["outcomes"])
        errors = sum(x["behavior"] == "error" for x in observed["outcomes"])
        row["oracle_valid"] = (failures > 0 and errors == 0) if row["expected_state"] == "fault" else (observed["methods"]["behavioral_contract"]["decision"] == "pass" if row["expected_state"] == "conformant" else observed["guard"]["decision"] == "not_evaluated")
        row["observed"] = {"availability_guard": observed["guard"], "selection": selector,
                           "inputs": len(observed["outcomes"]), "inputs_disagreeing_with_oracle": failures,
                           "errors": errors, "full_returned_mask_is_not_scoring_unit": True}
        rows.append(row)
        for item in observed["outcomes"]:
            observation_rows.append({"id": ident, **item["fixture"], "schema_pass": item["schema"]["pass"],
                                     "behavior": item["behavior"], "full_mask_mismatches": item.get("full_mask_mismatches"),
                                     "consumed_mask_mismatches": item.get("consumed_mask_mismatches"),
                                     "shifted_label_identity_equal": item.get("shifted_label_identity_equal"),
                                     "retained_input_identity_equal": item.get("retained_input_identity_equal"),
                                     "coded_supervised_predictions": item.get("coded_supervised_predictions"),
                                     "oracle_supervised_predictions": item.get("oracle_supervised_predictions")})
    # Independent falsification of predicates: malformed returns cannot pass
    # merely because the producing function returned without an exception.
    clean = {k: torch.zeros(8, dtype=torch.int64) for k in ("input_ids", "attention_mask", "position_ids", "loss_mask")}
    controls = [("valid", clean, True), ("missing_field", {k: v for k, v in clean.items() if k != "loss_mask"}, False),
                ("wrong_type", {**clean, "loss_mask": [0] * 8}, False),
                ("wrong_dtype", {**clean, "loss_mask": torch.zeros(8, dtype=torch.float32)}, False),
                ("wrong_rank", {**clean, "loss_mask": torch.zeros((1, 8), dtype=torch.int64)}, False),
                ("wrong_length", {**clean, "loss_mask": torch.zeros(7, dtype=torch.int64)}, False)]
    predicate_tests = [{"id": name, "expected": expected, "observed": schema_check(obj, 8)["pass"]} for name, obj, expected in controls]
    assert all(x["expected"] == x["observed"] for x in predicate_tests)
    summary = {"finite_source_faults": sum(r["expected_state"] == "fault" and r["oracle_valid"] for r in rows),
               "behavioral_faults_detected": sum(r["expected_state"] == "fault" and r["oracle_valid"] and r["methods"]["behavioral_contract"]["decision"] == "fail" for r in rows),
               "repairs_accepted": sum(r["variant"] == "repair" and r["methods"]["behavioral_contract"]["decision"] == "pass" for r in rows),
               "benign_false_alarms": sum(r["variant"] == "benign" and r["methods"]["behavioral_contract"]["decision"] == "fail" for r in rows),
               "not_evaluated": sum(r["methods"]["behavioral_contract"]["decision"] == "not_evaluated" for r in rows),
               "execution_errors": sum(r["methods"]["behavioral_contract"]["decision"] == "error" for r in rows)}
    save_csv(a.results / "controlled_boundary_verified_v2.csv", observation_rows)
    save(dest, {"schema_version": "2.0", "rows": rows, "summary": summary,
                "protocol_sha256": digest(a.protocol.read_bytes()), "protocol": str(a.protocol),
                "confirmation_freeze": protocol, "schema_predicate_tests": predicate_tests,
                "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "command": sys.argv,
                "case_analysis_v1_sha256": digest((a.results / "case_analysis_v1.json").read_bytes()),
                "first_evidence_preserved": True,
                "corrections": ["Execute schema predicates rather than preassign pass", "Invoke real source availability and concrete census eligibility guards", "Score trainer-consumed positions plus exact shifted token identities", "All reused cases labeled exposed confirmation/development; inherited partition retained only as history"]})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
