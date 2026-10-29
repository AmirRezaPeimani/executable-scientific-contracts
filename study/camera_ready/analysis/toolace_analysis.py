#!/usr/bin/env python3
"""Run ToolACE conversion, sequence-length counts, and loader checks.

The loader and position-ID extracts retain the Bytedance Apache-2.0 notice.
The Tool-RL-Box converter uses the MIT license."""
from __future__ import annotations

import argparse
import ast
import contextlib
import csv
import datetime as dt
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
import types
from collections import Counter
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoTokenizer

PIN = "1156d649235235e686372956e99bfc50e4b1e3f6"
LOADER = "verl/verl/utils/dataset/sft_dataset_wo_apply_template.py"
TRAINER = "verl/verl/trainer/fsdp_sft_trainer.py"


def digest(x):
    return hashlib.sha256(x if isinstance(x, bytes) else str(x).encode()).hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def extract(source, name, cls=None):
    tree = ast.parse(source)
    nodes = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body if cls else tree.body
    node = next(n for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    return node, {"start_line": node.lineno, "end_line": node.end_lineno,
                  "source_sha256": digest(ast.get_source_segment(source, node))}


def loader_callable(source, position_source):
    method, meta = extract(source, "__getitem__", "SFTDataset_wo_tp")
    position, _ = extract(position_source, "compute_position_id_with_mask")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[position, method], type_ignores=[]), "pinned_toolace_extract", "exec"), namespace)
    return namespace["__getitem__"], meta


def execute_loader(call, tokenizer, prompts, responses, index, maximum=8192):
    # Supply storage fields and call the upstream __getitem__ method.
    obj = types.SimpleNamespace(tokenizer=tokenizer, prompts=prompts, responses=responses,
                                max_length=maximum, truncation="left")
    return call(obj, index)


def oracle(prompt_len, response_len, maximum):
    "Compute the expected mask from prediction positions."
    total = prompt_len + response_len
    crop = max(0, total - maximum)
    kept = min(total, maximum)
    expected = torch.zeros(maximum, dtype=torch.long)
    for prediction_index in range(max(0, kept - 1)):
        original_target = crop + prediction_index + 1
        expected[prediction_index] = int(prompt_len <= original_target < total)
    return expected


def convert_official(source, rows, tokenizer, output_dir, label):
    """Run the converter with local data, tokenizer, and output paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets_module = types.ModuleType("datasets")
    datasets_module.load_dataset = lambda path: {"train": rows}
    captured = []
    real_write = pd.DataFrame.to_parquet
    def output_write(frame, path, *args, **kwargs):
        captured.append(frame.copy(deep=True))
        return real_write(frame, output_dir / f"{label}.parquet", *args, **kwargs)
    exception = None
    namespace = {"__name__": "__official_converter__", "__file__": "data/convert_toolace.py"}
    with (output_dir / f"{label}.stdout.log").open("w") as stdout, (output_dir / f"{label}.stderr.log").open("w") as stderr:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), \
             patch.dict(sys.modules, {"datasets": datasets_module}), \
             patch.object(AutoTokenizer, "from_pretrained", return_value=tokenizer), \
             patch.object(pd.DataFrame, "to_parquet", output_write):
            try:
                exec(compile(source, "data/convert_toolace.py", "exec"), namespace)
            except Exception as exc:
                exception = {"type": type(exc).__name__, "message": str(exc)}
                traceback.print_exc()
    # Save partial rows after an exception and the full frame after a completed write.
    frame = captured[0] if captured else pd.DataFrame(namespace.get("convert_data", []))
    return frame, {"label": label, "complete": exception is None, "exception": exception,
                   "converted_rows": len(frame), "source_sha256": digest(source),
                   "boundary_redirections": ["local datasets.load_dataset storage adapter", "real pinned tokenizer path", "workspace parquet output path"]}


def repaired_source(source):
    needle = "        loss_mask = attention_mask.clone()\n        if prompt_length > 1:"
    replacement = ("        loss_mask = attention_mask.clone()\n"
                   "        retained_prompt_length = max(0, prompt_length - max(0, sequence_length - self.max_length))\n"
                   "        if retained_prompt_length > 1:")
    assert source.count(needle) == 1
    source = source.replace(needle, replacement)
    source = source.replace("loss_mask[: min(prompt_length, loss_mask.size(0)) - 1] = 0",
                            "loss_mask[: min(retained_prompt_length, loss_mask.size(0)) - 1] = 0")
    return source


def challenges(args, tokenizer, source, position_source):
    fixed = repaired_source(source)
    versions = [
        ("TACE-ORIGINAL", "uncropped_mask_origin", "original", "development", "fault", source),
        ("TACE-REPAIR", "uncropped_mask_origin", "repair", "development", "conformant", fixed),
        ("TACE-CROP-SIGN", "crop_offset_sign", "source_fault", "locked_evaluation", "fault", fixed.replace("prompt_length - max(0, sequence_length - self.max_length)", "prompt_length + max(0, sequence_length - self.max_length)")),
        ("TACE-FIRST-RESPONSE", "first_response_boundary", "source_fault", "locked_evaluation", "fault", fixed.replace("min(retained_prompt_length, loss_mask.size(0)) - 1", "min(retained_prompt_length, loss_mask.size(0))")),
        ("TACE-PADDING", "padding_supervision", "source_fault", "locked_evaluation", "fault", fixed.replace("loss_mask = attention_mask.clone()", "loss_mask = torch.ones_like(attention_mask)")),
        ("TACE-LAST-POSITION", "final_position_supervision", "source_fault", "locked_evaluation", "fault", fixed.replace("        loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0", "        pass  # mutation removes final-position masking")),
        ("TACE-BENIGN-MIN", "equivalent_mask_expression", "benign", "locked_evaluation", "conformant", fixed.replace("min(retained_prompt_length, loss_mask.size(0))", "min(loss_mask.size(0), retained_prompt_length)")),
        ("TACE-BENIGN-RENAME", "local_variable_rename", "benign", "locked_evaluation", "conformant", fixed.replace("retained_prompt_length", "remaining_prompt_length")),
    ]
    rows, boundary = [], []
    extracts = args.output_dir / "implementation_variants"
    native_matches = list(args.source_root.glob("**/test*sft*"))
    native = {"decision": "not_evaluated", "reason": "No native test covers the selected loader behavior.",
              "name_matches": [str(x.relative_to(args.source_root)) for x in native_matches]}
    for ident, mechanism, variant, partition, expected_state, changed in versions:
        path = extracts / ident / LOADER
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(changed)
        call, meta = loader_callable(changed, position_source)
        outcomes = []
        error = None
        for p in (1, 2, 7, 8, 9, 10, 16):
            for r in (1, 2, 4, 9):
                for maximum in (8, 16):
                    prompt, response = " x" * p, " y" * (r - 1)
                    p_actual = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
                    r_actual = len(tokenizer(response + tokenizer.eos_token, add_special_tokens=False)["input_ids"])
                    try:
                        out = execute_loader(call, tokenizer, [prompt], [response], 0, maximum)
                        truth = oracle(p_actual, r_actual, maximum)
                        equal = bool(torch.equal(out["loss_mask"].cpu(), truth))
                        # Apply the two trainer indexing expressions to the loader tensors.
                        labels = out["input_ids"].unsqueeze(0)[:, 1:].contiguous()
                        effective = out["loss_mask"].unsqueeze(0)[:, :-1].reshape(-1)
                        mismatch = int((out["loss_mask"] != truth).sum().item())
                        outcomes.append({"id": ident, "prompt_tokens": p_actual,
                                         "response_tokens_including_eos": r_actual, "maximum": maximum,
                                         "original_crop_start": max(0, p_actual + r_actual - maximum),
                                         "coded_supervised_predictions": int(effective.sum().item()),
                                         "oracle_supervised_predictions": int(truth[:-1].sum().item()),
                                         "mask_mismatches": mismatch, "matches_oracle": equal,
                                         "label_count_after_shift": labels.numel()})
                    except Exception as exc:
                        error = {"type": type(exc).__name__, "message": str(exc)}
                        break
        if error:
            behavior = {"decision": "error", "reason": repr(error)}
        else:
            behavior = {"decision": "pass" if all(o["matches_oracle"] for o in outcomes) else "fail",
                        "reason": f"Expected prediction mask compared with outputs on {len(outcomes)} controlled inputs."}
        # Combine the original TACE-COORD-01 and TACE-MASK-01 source checks.
        source_defect = "input_ids = input_ids[-self.max_length :]" in changed and "min(prompt_length, loss_mask.size(0))" in changed
        masks_prompt = "loss_mask[: min(prompt_length, loss_mask.size(0)) - 1] = 0" in changed
        masks_terminal = "loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0" in changed
        pattern = {"decision": "fail" if source_defect or not (masks_prompt and masks_terminal) else "pass",
                   "reason": "Conjunction of the original TACE-COORD-01 and TACE-MASK-01 source checks.",
                   "coordinate_fault_pattern": source_defect, "prompt_mask_pattern": masks_prompt,
                   "terminal_mask_pattern": masks_terminal}
        manifested = any(not o["matches_oracle"] for o in outcomes)
        rows.append({"id": ident, "target": "toolace", "mechanism": mechanism,
                     "variant": variant, "partition": partition, "expected_state": expected_state,
                     "oracle_valid": not error and (manifested if expected_state == "fault" else not manifested),
                     "eligible_examples": len(outcomes),
                     "implementation": {"source": str(path.relative_to(args.output_dir)), "source_sha256": digest(changed),
                                        "parent_sha256": digest(source if variant in ("original", "repair") else fixed),
                                        "patch_sha256": digest("".join(__import__("difflib").unified_diff(source.splitlines(True), changed.splitlines(True)))),
                                        "extraction": meta},
                     "methods": {"schema": {"decision": "pass" if not error else "error", "reason": "Checks output keys, tensor types, and shapes."},
                                 "native_tests": native, "source_pattern": pattern, "behavioral_contract": behavior},
                     "observed": {"inputs": len(outcomes), "inputs_disagreeing_with_oracle": sum(not o["matches_oracle"] for o in outcomes), "error": error}})
        boundary.extend(outcomes)
    for ident, reason in [("TACE-MISSING-EVIDENCE", "Required pinned source is deliberately unavailable"),
                          ("TACE-ZERO-ELIGIBLE", "The selected inputs contain zero sequences above the length limit.")]:
        rows.append({"id": ident, "target": "toolace", "mechanism": "evidence_eligibility", "variant": "control",
                     "partition": "locked_evaluation", "expected_state": "not_evaluated", "oracle_valid": True,
                     "eligible_examples": 0, "implementation": {"source": None, "source_sha256": None, "patch_sha256": None},
                     "methods": {k: {"decision": "not_evaluated", "reason": reason} for k in ("schema", "native_tests", "source_pattern", "behavioral_contract")},
                     "observed": {"reason": reason}})
    save_csv(args.output_dir / "controlled_boundary_sweep_v1.csv", boundary)
    summary = {"finite_source_faults": sum(x["expected_state"] == "fault" and x["oracle_valid"] for x in rows),
               "behavioral_faults_detected": sum(x["expected_state"] == "fault" and x["oracle_valid"] and x["methods"]["behavioral_contract"]["decision"] == "fail" for x in rows),
               "repairs_accepted": sum(x["variant"] == "repair" and x["methods"]["behavioral_contract"]["decision"] == "pass" for x in rows),
               "benign_false_alarms": sum(x["variant"] == "benign" and x["methods"]["behavioral_contract"]["decision"] == "fail" for x in rows),
               "not_evaluated": sum(x["methods"]["behavioral_contract"]["decision"] == "not_evaluated" for x in rows),
               "execution_errors": sum(x["methods"]["behavioral_contract"]["decision"] == "error" for x in rows)}
    save(args.output_dir / "challenge_results_v1.json", {"schema_version": "2.0", "protocol_sha256": digest(args.protocol.read_bytes()), "rows": rows, "summary": summary})
    return summary


def main():
    parser = argparse.ArgumentParser()
    for name in ("source-root", "data", "tokenizer", "output-dir", "protocol"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--historical-tokenizer", type=Path, default=Path("models/qwen2.5-0.5b-instruct-hf"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "case_analysis_v1.json").exists():
        raise FileExistsError("Refusing to overwrite completed ToolACE analysis")
    protocol = json.loads(args.protocol.read_text())
    rev = subprocess.check_output(["git", "-C", str(args.source_root), "rev-parse", "HEAD"], text=True).strip()
    if rev != PIN:
        raise RuntimeError("Upstream revision differs from the protocol.")
    source = (args.source_root / LOADER).read_text()
    position_source = (args.source_root / "verl/verl/utils/model.py").read_text()
    converter_source = (args.source_root / "data/convert_toolace.py").read_text()
    trainer_source = (args.source_root / TRAINER).read_text()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    raw = json.loads(args.data.read_text())
    original_call, loader_meta = loader_callable(source, position_source)
    repaired_call, _ = loader_callable(repaired_source(source), position_source)
    all_candidates = [(i, j) for i, row in enumerate(raw) for j, item in enumerate(row["conversations"])
                      if item["from"] == "assistant" and item["value"].startswith("[")]
    frame, official = convert_official(converter_source, raw, tokenizer, args.output_dir / "conversion", "official_full_first_execution")
    save(args.output_dir / "conversion_first_execution.json", official)
    # Save the original converter exception.
    if not official["complete"]:
        save(args.output_dir / "case_analysis_v1.json", {"status": "blocked_converter_exception", "official_conversion": official,
             "protocol_sha256": digest(args.protocol.read_bytes()), "candidate_targets": len(all_candidates)})
        return
    baseline_path = Path(__file__).resolve().parents[1] / "baseline/toolace_original_scan/audit_toolace_truncation.py"
    spec = importlib.util.spec_from_file_location("historical_toolace_scanner", baseline_path)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    # Follow converter iteration order and system-schema parsing, including
    # final-character removal, to match converted rows.
    output_ids, failed_schema_rows, candidates_in_failed_schema = [], [], []
    baseline_records = []
    with (args.output_dir / "baseline_render.stderr.log").open("w") as log:
        for i, row in enumerate(raw):
            raw_tools = row["system"].split("Here is a list of functions in JSON format that you can invoke:\n")[-1].split("\nShould you decide to return the function call(s).")[0]
            try:
                json.loads(raw_tools.strip()[:-1].strip())
            except Exception:
                failed_schema_rows.append(i)
                candidates_in_failed_schema.extend((i, j) for j, x in enumerate(row["conversations"]) if x["from"] == "assistant" and x["value"].startswith("["))
                continue
            try:
                system, translations = baseline.convert_system(row["system"])
            except Exception as exc:
                print(i, repr(exc), file=log)
                continue
            messages = [{"role": "system", "content": system}]
            for j, item in enumerate(row["conversations"]):
                role, value = item["from"], item["value"]
                if role == "assistant" and value.startswith("["):
                    output_ids.append((i, j))
                    rendered, names = baseline.render_target(value, translations)
                    prompt = tokenizer.apply_chat_template(__import__("copy").deepcopy(messages), add_generation_prompt=True, tokenize=False)
                    baseline_records.append((prompt, rendered, names))
                    messages.append({"role": role, "content": rendered})
                else:
                    messages.append({"role": role, "content": value})
    if len(frame) != len(output_ids):
        raise RuntimeError(f"Official output identities do not reconcile: {len(frame)} vs {len(output_ids)}")
    comparisons, census = [], []
    prompts, responses = frame["prompt"].tolist(), frame["response"].tolist()
    original_disagree, repaired_disagree = [], []
    for k, ((row_index, message_index), (base_prompt, base_response, names)) in enumerate(zip(output_ids, baseline_records)):
        prompt, response = prompts[k], responses[k]
        comparisons.append({"converter_index": k, "row_index": row_index, "message_index": message_index,
                            "prompt_equal": prompt == base_prompt, "response_equal": response == base_response,
                            "official_prompt_sha256": digest(prompt), "baseline_prompt_sha256": digest(base_prompt),
                            "official_response_sha256": digest(response), "baseline_response_sha256": digest(base_response),
                            "empty_response": response == ""})
        p = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        r = len(tokenizer(response + tokenizer.eos_token, add_special_tokens=False)["input_ids"])
        result = execute_loader(original_call, tokenizer, prompts, responses, k)
        repaired = execute_loader(repaired_call, tokenizer, prompts, responses, k)
        truth = oracle(p, r, 8192)
        orig_mismatch = int((result["loss_mask"] != truth).sum().item())
        repair_mismatch = int((repaired["loss_mask"] != truth).sum().item())
        original_disagree.append(orig_mismatch)
        repaired_disagree.append(repair_mismatch)
        supervised = result["loss_mask"][:-1].bool()
        labels = result["input_ids"][1:]
        eos_predictions = int(((labels == tokenizer.eos_token_id) & supervised).sum().item())
        # Parse the tool schema and check tool names in the uncropped prompt.
        census.append({"converter_index": k, "row_index": row_index, "message_index": message_index,
                       "empty_response": response == "", "prompt_tokens": p,
                       "response_tokens_including_eos": r, "sequence_tokens": p + r,
                       "overlength_at_official_8192": p + r > 8192,
                       "coded_supervised_predictions": int(supervised.sum().item()),
                       "oracle_supervised_predictions": int(truth[:-1].sum().item()),
                       "eos_supervised_predictions": eos_predictions,
                       "original_mask_mismatches": orig_mismatch, "repair_mask_mismatches": repair_mismatch,
                       "all_parsed_names_in_prompt": all(name in prompt for name in names)})
        if k % 1000 == 0:
            print(f"ToolACE loader census {k}/{len(frame)}", flush=True)
    save_csv(args.output_dir / "conversion_row_comparison_v1.csv", comparisons)
    save_csv(args.output_dir / "official_loader_census_v1.csv", census)
    matched = sum(x["prompt_equal"] and x["response_equal"] for x in comparisons)
    empty = [x for x in census if x["empty_response"]]
    historical = args.historical_tokenizer
    official_assets = {}
    for path in args.tokenizer.iterdir():
        if path.is_file() and path.name != "acquisition.json":
            old = historical / path.name
            official_assets[path.name] = {"official_sha256": digest(path.read_bytes()),
                                          "historical_sha256": digest(old.read_bytes()) if old.exists() else None,
                                          "identical": old.exists() and old.read_bytes() == path.read_bytes()}
    tokenizer_keys = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
    tokenizer_equivalent = all(official_assets[k]["identical"] for k in tokenizer_keys)
    challenge_summary = challenges(args, tokenizer, source, position_source)
    parsed = [x for x in census if not x["empty_response"]]
    result = {
        "schema_version": "2.0", "status": "complete", "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": str(args.protocol), "protocol_sha256": digest(args.protocol.read_bytes()),
        "source_revision": rev, "source_dirty": bool(subprocess.check_output(["git", "-C", str(args.source_root), "status", "--porcelain"], text=True).strip()),
        "dataset_sha256": digest(args.data.read_bytes()), "official_conversion": official,
        "tokenizer_acquisition": json.loads((args.tokenizer / "acquisition.json").read_text()),
        "tokenizer_equivalence": {"relevant_assets_byte_identical": tokenizer_equivalent, "assets": official_assets},
        "coverage": {"source_conversations": len(raw), "raw_candidate_targets": len(all_candidates),
                     "system_schema_failed_rows": len(failed_schema_rows), "candidate_targets_in_schema_failed_rows": len(candidates_in_failed_schema),
                     "official_converted_targets": len(frame), "nonempty_parsed_targets": len(parsed),
                     "empty_response_targets": len(empty), "total_candidate_exclusions_from_parsed_scan": len(all_candidates) - len(parsed),
                     "mixed_unit_failure_events": len(failed_schema_rows) + len(empty),
                     "reconciliation": "raw candidates = schema-excluded candidates + empty converted responses + nonempty parsed responses; mixed-unit failure events = schema-failed conversations + empty converted responses"},
        "baseline_conversion_comparison": {"compared_rows": len(frame), "exact_prompt_and_response_matches": matched,
                                            "mismatches": len(frame) - matched, "row_evidence": "conversion_row_comparison_v1.csv"},
        "official_8192_census": {"converted_rows": len(census), "overlength_targets": sum(x["overlength_at_official_8192"] for x in census),
                                 "maximum_sequence_tokens": max(x["sequence_tokens"] for x in census),
                                 "maximum_nonempty_sequence_tokens": max(x["sequence_tokens"] for x in parsed),
                                 "zero_supervised_predictions": sum(x["coded_supervised_predictions"] == 0 for x in census),
                                 "original_masks_disagreeing_with_oracle": sum(x > 0 for x in original_disagree),
                                 "repair_masks_disagreeing_with_oracle": sum(x > 0 for x in repaired_disagree),
                                 "empty_response_eos_only_supervised": sum(x["coded_supervised_predictions"] == 1 and x["eos_supervised_predictions"] == 1 for x in empty),
                                 "all_parsed_names_present": all(x["all_parsed_names_in_prompt"] for x in parsed)},
        "implementation": {"loader_file_sha256": digest(source), "loader_extract": loader_meta,
                           "converter_file_sha256": digest(converter_source), "trainer_file_sha256": digest(trainer_source),
                           "trainer_shift_evidence": {"mask_expression": 'batch.pop("loss_mask")[:, :-1].reshape(-1)',
                                                      "labels_expression": 'input_ids[:, 1:].contiguous()',
                                                      "lines": [382, 390], "executed": "Trainer indexing expressions evaluated on loader tensors."}},
        "challenge_summary": challenge_summary,
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                        "transformers": transformers.__version__, "pandas": pd.__version__, "numpy": np.__version__,
                        "command": sys.argv, "paid_cost_cad": 0},
        "limitations": ["Counts cover the listed source transformations.",
                        "Controlled lengths 8 and 16 test crop behavior. The released-data scan uses 8192.",
                        "The mask comparison evaluates preprocessing outputs.",
                        "Empty responses are supervised on EOS only.",
                        "Native test coverage is unavailable for these loader behaviors."]}
    save(args.output_dir / "case_analysis_v1.json", result)
    print(json.dumps({"coverage": result["coverage"], "official_8192_census": result["official_8192_census"], "challenge_summary": challenge_summary}, indent=2))


if __name__ == "__main__":
    main()
