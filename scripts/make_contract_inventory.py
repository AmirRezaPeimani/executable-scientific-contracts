#!/usr/bin/env python3
"""Generate both contract-inventory tables from the saved counts and evidence paths."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path


CAPTION = "The 25 original checks, with source locations, roles, recorded classifications, and eligible data counts."
CURRENT = "study/camera_ready/results/"


def read_json(root, relative):
    return json.loads((root / relative).read_text())


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classification(value):
    return "convention sensitivity" if value in {"alternative convention", "alternative_convention"} else value


def esc(text):
    mapping = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(mapping.get(char, char) for char in str(text))


def breaks(text):
    return esc(text).replace(r"\_", r"\_\allowbreak{}").replace("/", r"/\allowbreak{}").replace(";", r";\allowbreak{}").replace("-", r"-\allowbreak{}")


def current_bindings(root):
    bindings = {}
    aprm_challenge_path = CURRENT + "agentprm/challenge_results_v2.json"
    aprm_challenge = read_json(root, aprm_challenge_path)
    original = next(row for row in aprm_challenge["rows"] if row["variant"] == "original" and row["mechanism"] == "terminal_discount" and row["eligible_examples"] > 0)
    bindings["APRM-TEMP-01"] = (aprm_challenge_path, original["eligible_examples"], "Original terminal-discount case: eligible_examples across both pinned caller paths.")
    states_path = CURRENT + "agentprm/state_targets_v1.csv"
    states = read_csv(root / states_path)
    bindings["APRM-TEMP-02"] = (states_path, sum(Decimal(row["gamma"]) == 1 for row in states), "Count saved state observations with gamma=1.")
    bindings["APRM-TEMP-03"] = (states_path, sum(Decimal(row["gamma"]) == Decimal("0.95") and int(row["terminal_score"]) == 1 and int(row["length"]) > 1 for row in states), "Count saved positive-terminal observations at gamma=0.95 with trajectory length greater than one.")
    aprm_case_path = CURRENT + "agentprm/case_analysis_v1.json"
    aprm_case = read_json(root, aprm_case_path)
    bindings["APRM-SCHEMA-01"] = (aprm_case_path, aprm_case["query_history"]["original"]["history_entries"], "Number of query-history entries in the original case; saved trajectory actions are correct.")
    scope_path = CURRENT + "agentprm/mean_scope_companion_v2.json"
    scope = read_json(root, scope_path)
    if "Complete trajectory visit mean is invariant" not in scope["scope"]:
        raise ValueError("AgentPRM mean-aggregation result is missing")
    bindings["APRM-METRIC-01"] = (scope_path, 0, "The original varying-terminal-outcome fixture has zero valid upstream instances. The replacement analysis compares means over trajectory visits and merged state-action pairs.")

    remaining_path = CURRENT + "remaining_evidence_v3.json"
    for row in read_json(root, remaining_path)["contracts"]:
        bindings[row["contract_id"]] = (remaining_path, row["coverage"]["eligible_examples"], "contracts[contract_id=" + row["contract_id"] + "].coverage.eligible_examples; " + row["coverage"]["unit"] + ".")

    tool_challenge_path = CURRENT + "toolace/challenge_results_verified_v2.json"
    tool_original = next(row for row in read_json(root, tool_challenge_path)["rows"] if row["id"] == "TACE-ORIGINAL")
    bindings["TACE-COORD-01"] = (tool_challenge_path, tool_original["eligible_examples"], "TACE-ORIGINAL eligible_examples in the saved controlled length grid.")
    tool_case_path = CURRENT + "toolace/case_analysis_v1.json"
    tool = read_json(root, tool_case_path)
    for cid in ("TACE-COORD-02", "TACE-SCHEMA-01"):
        bindings[cid] = (tool_case_path, tool["coverage"]["nonempty_parsed_targets"], "coverage.nonempty_parsed_targets from the official conversion.")
    for cid in ("TACE-MASK-01", "TACE-SUPPORT-01"):
        bindings[cid] = (tool_case_path, tool["official_8192_census"]["converted_rows"], "official_8192_census.converted_rows, including separately identified EOS-only examples.")

    aabs_case_path = CURRENT + "agentabstain/case_analysis_v1.json"
    aabs_case = read_json(root, aabs_case_path)
    for cid in ("AABS-PAIR-01", "AABS-MACRO-01"):
        bindings[cid] = (aabs_case_path, aabs_case["dataset"]["pairs"], "dataset.pairs in the released task-composition census.")
    aabs_challenge_path = CURRENT + "agentabstain/challenge_results_v2.json"
    aabs_rows = read_json(root, aabs_challenge_path)["rows"]
    for cid, mechanism in (("AABS-PAIR-02", "paired_conjunction"), ("AABS-CAR-01", "conditional_denominator")):
        row = next(row for row in aabs_rows if row["variant"] == "original_conformant" and row["mechanism"] == mechanism)
        bindings[cid] = (aabs_challenge_path, row["eligible_examples"], "Original conformant " + mechanism + " case: eligible_examples.")
    bindings["AABS-COMMIT-01"] = (aabs_case_path, len(aabs_case["primitive_checks"]["combined_operational"]), "Number of saved combined_operational Boolean cases in primitive_checks.")
    return bindings


def render_tex(rows):
    lines = [r"\begingroup\footnotesize", r"\setlength{\tabcolsep}{2pt}\renewcommand{\arraystretch}{1.12}",
             r"\begin{longtable}{@{}p{0.11\linewidth}p{0.25\linewidth}p{0.12\linewidth}p{0.12\linewidth}p{0.18\linewidth}p{0.14\linewidth}@{}}",
             r"\caption{" + CAPTION + r"}\label{tab:contract-inventory}\\",
             r"\toprule Check & Source & Role & Original classification & Eligible data & Evidence \\\midrule\endfirsthead",
             r"\toprule Check & Source & Role & Original classification & Eligible data & Evidence \\\midrule\endhead"]
    for row in rows:
        evidence = r"Original: \path{" + row["evidence"] + r"}\par Current: \path{" + row["current_evidence"] + "}"
        cells = [breaks(row["contract_id"]), breaks(row["source"]), esc(row["role"]),
                 esc(row["original_classification"]), esc(row["coverage"]), evidence]
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule\end{longtable}\endgroup")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = p.parse_args()
    root = args.repo_root.resolve()
    inventory_path = root / "tables/contract_inventory.csv"
    original_rows = read_csv(inventory_path)
    historical_path = root / "history/submitted/tables/contract_inventory.csv"
    history = {row["contract_id"]: row for row in read_csv(historical_path)}
    bindings = current_bindings(root)
    if {row["contract_id"] for row in original_rows} != set(history) or set(bindings) != set(history):
        raise ValueError("Historical, current and declared inventory identifiers differ")
    old_fields = ["contract_id", "repository", "source", "role", "original_classification", "eligible_examples", "coverage"]
    rows, evidence_report = [], []
    paths = {"history/submitted/tables/contract_inventory.csv"}
    for existing in original_rows:
        row = {name: existing[name] for name in old_fields}
        cid = row["contract_id"]
        historical = history[cid]
        if row["original_classification"] != classification(historical["outcome"]):
            raise ValueError(f"Historical classification changed: {cid}")
        current_path, count, basis = bindings[cid]
        if int(row["eligible_examples"]) != count:
            raise ValueError(f"Saved evidence count disagrees for {cid}: recorded {row['eligible_examples']} versus {count}")
        old_path = historical["evidence_location"]
        if not old_path.startswith("../results/"):
            raise ValueError(f"Unexpected historical evidence path for {cid}: {old_path}")
        historical_evidence = "history/submitted/" + old_path.removeprefix("../")
        row.update(evidence=historical_evidence, current_evidence=current_path, eligibility_basis=basis)
        for relative in (historical_evidence, current_path):
            resolved = (root / relative).resolve()
            if Path(relative).is_absolute() or not resolved.is_relative_to(root) or not resolved.is_file():
                raise ValueError(f"Evidence must name an existing repository-relative file: {relative}")
            paths.add(relative)
        rows.append(row)
        evidence_report.append({"contract_id": cid, "historical_classification": row["original_classification"],
                                "historical_evidence": historical_evidence, "current_evidence": current_path,
                                "eligible_examples": count, "eligibility_basis": basis})
    tex = render_tex(rows)
    for directory in (root / "tables", root / "manuscript/tables"):
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "contract_inventory.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        (directory / "contract_inventory.tex").write_text(tex)
    report = {"schema_version": "1.0", "row_count": len(rows),
              "original_classifications": dict(Counter(row["original_classification"] for row in rows)),
              "preserved_fields": old_fields, "evidence_paths_relative_to": "repository root",
              "reproduction_command": "python scripts/make_contract_inventory.py",
              "saved_evidence_only": True, "experiments_executed": False,
              "inputs": {relative: sha(root / relative) for relative in sorted(paths)},
              "rows": evidence_report,
              "outputs": {str(path.relative_to(root)): sha(path) for path in (root / "tables/contract_inventory.csv", root / "tables/contract_inventory.tex", root / "manuscript/tables/contract_inventory.csv", root / "manuscript/tables/contract_inventory.tex")}}
    (root / "tables/contract_inventory_evidence.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(rows), "cited_files_verified": len(paths), "original_classifications": report["original_classifications"], "all_recorded_counts_preserved": True}, indent=2))


if __name__ == "__main__":
    main()
