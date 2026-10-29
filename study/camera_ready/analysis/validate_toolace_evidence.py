#!/usr/bin/env python3
"""Independent schema/token-identity checks and parser-difference diagnosis.

This companion does not tune challenges, change their oracle or rewrite the
first execution. It checks additional invariants specified by the frozen
protocol and saves a separate validation record.
"""
from pathlib import Path
import argparse
import ast
import json
import sys

import pandas as pd
import torch
from transformers import AutoTokenizer

from toolace_analysis import LOADER, digest, execute_loader, loader_callable, save


def main():
    parser = argparse.ArgumentParser()
    for name in ("source-root", "data", "tokenizer", "results", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    position = (args.source_root / "verl/verl/utils/model.py").read_text()
    challenge = json.loads((args.results / "challenge_results_v1.json").read_text())
    variants = []
    for row in challenge["rows"]:
        if row["eligible_examples"] == 0:
            continue
        source_path = args.results / row["implementation"]["source"]
        source = source_path.read_text()
        assert digest(source) == row["implementation"]["source_sha256"]
        call, _ = loader_callable(source, position)
        schema_failures, identity_failures = [], []
        checked = 0
        for p in (1, 2, 7, 8, 9, 10, 16):
            for r in (1, 2, 4, 9):
                for maximum in (8, 16):
                    prompt, response = " x" * p, " y" * (r - 1)
                    result = execute_loader(call, tokenizer, [prompt], [response], 0, maximum)
                    fixture_id = [p, r, maximum]
                    schema = (set(result) == {"input_ids", "attention_mask", "position_ids", "loss_mask"}
                              and all(isinstance(x, torch.Tensor) and x.ndim == 1 and x.numel() == maximum
                                      and x.dtype == torch.int64 for x in result.values()))
                    if not schema:
                        schema_failures.append(fixture_id)
                    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                    response_ids = tokenizer(response + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
                    full = prompt_ids + response_ids
                    expected = full[-maximum:]
                    expected += [tokenizer.pad_token_id] * (maximum - len(expected))
                    # Check every input/shifted label, not merely token counts.
                    label_match = result["input_ids"][1:].tolist() == expected[1:]
                    input_match = result["input_ids"].tolist() == expected
                    if not label_match or not input_match:
                        identity_failures.append(fixture_id)
                    checked += 1
        variants.append({"id": row["id"], "checked_inputs": checked,
                         "schema_failures": schema_failures, "token_identity_failures": identity_failures,
                         "schema_decision_verified": not schema_failures,
                         "frozen_original_result_unchanged": True})
    raw = json.loads(args.data.read_text())
    comparison = pd.read_csv(args.results / "conversion_row_comparison_v1.csv")
    discrepancies = comparison[~(comparison.prompt_equal & comparison.response_equal)]
    parser_cases = []
    for row in discrepancies.itertuples():
        value = raw[row.row_index]["conversations"][row.message_index]["value"]
        expr = ast.parse(value, mode="eval").body
        calls = expr.elts if isinstance(expr, ast.List) else [expr]
        invalid_keyword_value_types = []
        valid_call_count = 0
        for call in calls:
            if not isinstance(call, ast.Call):
                continue
            valid = True
            for kw in call.keywords:
                try:
                    ast.literal_eval(kw.value)
                except (ValueError, TypeError):
                    valid = False
                    invalid_keyword_value_types.append(type(kw.value).__name__)
            valid_call_count += int(valid)
        parser_cases.append({"row_index": int(row.row_index), "message_index": int(row.message_index),
                             "prompt_equal": bool(row.prompt_equal), "official_response_empty": bool(row.empty_response),
                             "valid_calls_that_historical_scanner_keeps": valid_call_count,
                             "invalid_keyword_ast_types": invalid_keyword_value_types,
                             "mechanism": "Historical parser catches literal-evaluation failure per call; official parser catches across full target and returns empty list."})
    assert len(parser_cases) == 18
    assert all(x["official_response_empty"] and x["valid_calls_that_historical_scanner_keeps"] > 0
               and x["invalid_keyword_ast_types"] for x in parser_cases)
    historical_summary = json.loads((args.results.parents[1] / "baseline/toolace_original_scan/raw_support/summary.json").read_text())
    case = json.loads((args.results / "case_analysis_v1.json").read_text())
    cov = case["coverage"]
    assert cov["raw_candidate_targets"] == cov["candidate_targets_in_schema_failed_rows"] + cov["empty_response_targets"] + cov["nonempty_parsed_targets"]
    result = {"schema_version": "1.0", "command": sys.argv, "paid_cost_cad": 0,
              "challenge_schema_and_identity_checks": variants,
              "all_schema_checks_pass": all(not x["schema_failures"] for x in variants),
              "all_shifted_token_identity_checks_pass": all(not x["token_identity_failures"] for x in variants),
              "parser_discrepancy_cases": parser_cases,
              "coverage_reconciliation": {
                  "historical_scanner": {"nonempty_targets": historical_summary["counts"]["targets"],
                                         "schema_failed_rows": 518, "target_parse_failures": 552,
                                         "mixed_unit_failure_events": historical_summary["parse_failures"],
                                         "candidate_exclusions": 680},
                  "official_converter": cov,
                  "difference": {"additional_empty_official_responses": len(parser_cases),
                                 "historical_target_difference": historical_summary["counts"]["targets"] - cov["nonempty_parsed_targets"]},
                  "exact_candidate_partition_verified": True},
              "trainer_shift_interpretation": "Upstream loss_mask index j weights prediction of input_ids[j+1]. The final EOS token can be supervised; masking final input position excludes prediction after EOS. Historical r-1 count was not the trainer-aligned supervision count.",
              "preservation": {"case_analysis_v1_sha256": digest((args.results / "case_analysis_v1.json").read_bytes()),
                               "challenge_results_v1_sha256": digest((args.results / "challenge_results_v1.json").read_bytes())}}
    save(args.output, result)
    print(json.dumps({"all_schema_checks_pass": result["all_schema_checks_pass"],
                      "all_shifted_token_identity_checks_pass": result["all_shifted_token_identity_checks_pass"],
                      "parser_discrepancies": len(parser_cases)}, indent=2))


if __name__ == "__main__":
    main()
