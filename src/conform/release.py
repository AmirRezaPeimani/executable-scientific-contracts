"""Validate included evidence and reconstruct the paper-facing summary."""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


AUDIT_TARGETS = ("agentprm", "contractbench", "toolace", "taubench")
REPOSITORY_LABELS = {
    "agentprm": "AgentPRM",
    "contractbench": "ContractBench",
    "toolace": "ToolACE",
    "taubench": "tau-bench",
    "agentabstain": "AgentAbstain",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_inventory(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _normalize_status(status: str) -> str:
    value = status.strip().lower().replace("_", " ")
    if value == "alternative convention":
        return "convention sensitivity"
    return value


def _contract_by_id(payload: dict[str, Any], contract_id: str) -> dict[str, Any]:
    matches = [
        item for item in payload["contracts"] if item["contract_id"] == contract_id
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one result for {contract_id}; found {len(matches)}")
    return matches[0]


def _status_counts(contracts: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(_normalize_status(item["status"]) for item in contracts)
    return dict(sorted(counts.items()))


def _validate_audit_counts(payload: dict[str, Any], path: Path) -> None:
    derived = Counter(item["status"] for item in payload["contracts"])
    recorded = {key: value for key, value in payload["counts"].items() if value}
    if dict(derived) != recorded:
        raise ValueError(f"recorded counts disagree with contract rows: {path}")


def build_summary(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / "contracts/contracts.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    inventory_path = root / "tables/contract_inventory.csv"
    inventory = _read_inventory(inventory_path)

    audits: dict[str, dict[str, Any]] = {}
    for target in AUDIT_TARGETS:
        path = root / f"results/reproduction/audits/{target}.json"
        payload = _read_json(path)
        _validate_audit_counts(payload, path)
        audits[target] = payload
    prospective = _read_json(root / "results/agentabstain_prospective_v2.json")
    prospective_counts = Counter(item["status"] for item in prospective["contracts"])
    if dict(prospective_counts) != prospective["counts"]:
        raise ValueError("AgentAbstain recorded counts disagree with contract rows")

    manifest_ids = {
        contract["id"]
        for target in manifest["targets"].values()
        for contract in target["contracts"]
    }
    evidence_contracts = [
        contract for payload in audits.values() for contract in payload["contracts"]
    ] + prospective["contracts"]
    evidence_ids = {contract["contract_id"] for contract in evidence_contracts}
    inventory_ids = {row["contract_id"] for row in inventory}
    if not (manifest_ids == evidence_ids == inventory_ids):
        raise ValueError("manifest, evidence, and inventory contract IDs disagree")
    if len(inventory_ids) != len(inventory):
        raise ValueError("contract inventory contains duplicate IDs")

    evidence_by_id = {item["contract_id"]: item for item in evidence_contracts}
    for row in inventory:
        result = evidence_by_id[row["contract_id"]]
        if _normalize_status(row["outcome"]) != _normalize_status(result["status"]):
            raise ValueError(f"inventory outcome disagrees for {row['contract_id']}")
        evidence_path = (inventory_path.parent / row["evidence_location"]).resolve()
        if not evidence_path.is_file():
            raise FileNotFoundError(evidence_path)

    original_contracts = [
        contract for target in AUDIT_TARGETS for contract in audits[target]["contracts"]
    ]
    prospective_contracts = prospective["contracts"]
    by_repository = {
        REPOSITORY_LABELS[target]: _status_counts(audits[target]["contracts"])
        for target in AUDIT_TARGETS
    }
    by_repository[REPOSITORY_LABELS["agentabstain"]] = _status_counts(
        prospective_contracts
    )

    mutations = _read_json(root / "results/reproduction/mutations.json")
    mutation_manifest = yaml.safe_load(
        (root / "contracts/mutations.yaml").read_text(encoding="utf-8")
    )
    mutation_ids = {item["id"] for item in mutation_manifest["mutations"]}
    result_mutation_ids = {item["id"] for item in mutations["mutations"]}
    if mutation_ids != result_mutation_ids:
        raise ValueError("mutation manifest and saved mutation results disagree")
    detected = sum(bool(item["detected"]) for item in mutations["mutations"])
    if mutations["overall"]["detected"] != detected:
        raise ValueError("saved study-mutation total disagrees with result rows")

    external_first = _read_json(root / "results/external_mutation_transfer_v1.json")
    external_corrected = _read_json(root / "results/external_mutation_transfer_v2.json")
    first_by_id = {item["id"]: item for item in external_first["mutations"]}
    corrected_by_id = {item["id"]: item for item in external_corrected["mutations"]}
    if set(first_by_id) != set(corrected_by_id):
        raise ValueError("external mutation executions contain different operators")
    for mutation_id in first_by_id:
        for key in ("external_family", "fixture", "transformation"):
            if first_by_id[mutation_id][key] != corrected_by_id[mutation_id][key]:
                raise ValueError(f"external mutation definition changed: {mutation_id}")
    for label, payload in (
        ("first", external_first),
        ("corrected", external_corrected),
    ):
        detected_external = sum(
            bool(item["detected"]) for item in payload["mutations"]
        )
        if payload["overall"]["detected"] != detected_external:
            raise ValueError(f"external {label} total disagrees with result rows")

    pair = _contract_by_id(prospective, "AABS-PAIR-01")["observed"]
    toolace_support = _read_json(root / "results/support/toolace_released_support.json")
    tool_counts = toolace_support["counts"]
    tool_audit_targets = _contract_by_id(
        audits["toolace"], "TACE-COORD-02"
    )["observed"]["targets"]
    tool_audit_overlength = _contract_by_id(
        audits["toolace"], "TACE-SUPPORT-01"
    )["observed"]["overlength_targets"]
    if tool_counts["targets"] != tool_audit_targets:
        raise ValueError("ToolACE parsed-target count disagrees with audit")
    if tool_counts["overlength_targets"] != tool_audit_overlength:
        raise ValueError("ToolACE overlength count disagrees with audit")

    mapping = _read_json(
        root / "results/support/contractbench_repository_mapping.json"
    )
    cb_metric = _contract_by_id(audits["contractbench"], "CBEN-METRIC-01")
    cb_log = _contract_by_id(audits["contractbench"], "CBEN-LOG-01")
    if mapping["reported_result"]["episodes"] != cb_metric["expected"]["episodes"]:
        raise ValueError("ContractBench reported denominator disagrees with audit")
    rounded_rate = round(
        100 * cb_metric["expected"]["successes"] / cb_metric["expected"]["episodes"],
        1,
    )
    if rounded_rate != mapping["reported_result"]["success_rate_percent"]:
        raise ValueError("ContractBench reported rate does not round from audit count")
    if mapping["released_artifact"]["successes"] != cb_metric["observed"]["successes"]:
        raise ValueError("ContractBench released successes disagree with audit")
    if (
        mapping["released_artifact"]["traces_without_terminal_assistant_response"]
        != cb_log["observed"]["output_without_assistant_message"]
    ):
        raise ValueError("ContractBench incomplete-trace count disagrees with audit")

    runtime = _read_json(root / "results/submission_evidence/execution_benchmark.json")
    runtime_summary: dict[str, Any] = {}
    for row in runtime["benchmarks"]:
        if row["repetitions"] != len(row["samples_seconds"]):
            raise ValueError(f"runtime repetition count disagrees for {row['target']}")
        if statistics.median(row["samples_seconds"]) != row["median_seconds"]:
            raise ValueError(f"runtime median disagrees for {row['target']}")
        runtime_summary[row["target"]] = {
            "repetitions": row["repetitions"],
            "median_seconds": row["median_seconds"],
            "p25_seconds": row["p25_seconds"],
            "p75_seconds": row["p75_seconds"],
        }

    revisions = {
        target: {
            key: entry[key]
            for key in ("revision", "data_revision")
            if key in entry
        }
        for target, entry in manifest["targets"].items()
    }
    if revisions["agentabstain"]["revision"] != prospective["source"][
        "repository_revision"
    ]:
        raise ValueError("AgentAbstain source revision disagrees with manifest")
    if revisions["agentabstain"]["data_revision"] != prospective["source"][
        "dataset_revision"
    ]:
        raise ValueError("AgentAbstain dataset revision disagrees with manifest")
    return {
        "status": "valid",
        "contracts": {
            "original": {
                "total": len(original_contracts),
                "outcomes": _status_counts(original_contracts),
            },
            "prospective_agentabstain": {
                "total": len(prospective_contracts),
                "outcomes": _status_counts(prospective_contracts),
            },
            "overall": {
                "total": len(evidence_contracts),
                "outcomes": _status_counts(evidence_contracts),
            },
            "by_repository": by_repository,
        },
        "mutations": {
            "study_specific": {
                "overall": {
                    "detected": detected,
                    "total": len(mutations["mutations"]),
                },
                "by_class": {
                    key: {
                        "detected": value["detected"],
                        "total": value["total"],
                    }
                    for key, value in sorted(mutations["by_class"].items())
                },
            },
            "external_first_execution": external_first["overall"],
            "external_after_fixture_correction": external_corrected["overall"],
        },
        "agentabstain": {
            "released_rows": pair["rows"],
            "complete_pairs": pair["pairs"],
            "scenarios": pair["categories"],
            "released_model_outputs_available": prospective[
                "released_model_outputs_available"
            ],
        },
        "toolace": {
            "candidate_targets": tool_counts["candidate_targets"],
            "parsed_targets": tool_counts["targets"],
            "parsing_fraction": tool_counts["targets"]
            / tool_counts["candidate_targets"],
            "overlength_targets": tool_counts["overlength_targets"],
            "configured_max_length": tool_counts["configured_max_length"],
            "maximum_sequence_tokens": tool_counts["maximum_sequence_tokens"],
        },
        "contractbench": {
            "reported_model": mapping["reported_result"]["model"],
            "reported_episodes": mapping["reported_result"]["episodes"],
            "reported_success_rate_percent": mapping["reported_result"][
                "success_rate_percent"
            ],
            "released_episodes": cb_metric["observed"]["episodes"],
            "released_successes": cb_metric["observed"]["successes"],
            "traces_without_terminal_assistant_response": cb_log["observed"][
                "output_without_assistant_message"
            ],
            "comparison_scope": "repository mapping",
            "exact_evaluation_slice_identity_verified": mapping[
                "exact_evaluation_slice_identity_verified"
            ],
            "corrected_model_performance_identifiable": mapping[
                "corrected_model_performance_identifiable"
            ],
        },
        "post_authoring_runtime": runtime_summary,
        "revisions": revisions,
    }


def write_summary(root: Path, output: Path) -> dict[str, Any]:
    payload = build_summary(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload
