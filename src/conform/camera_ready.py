"""Verify saved results and run upstream implementation checks."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

TARGETS = ("agentprm", "toolace", "agentabstain")
METHODS = ("schema", "native_tests", "source_pattern", "behavioral_contract")
DECISIONS = {"pass", "fail", "not_evaluated", "error"}
SOURCE_ROOT = Path(__file__).resolve().parents[2]
SOURCE_STUDY = SOURCE_ROOT / "study/camera_ready"
INSTALLED_STUDY = Path(sys.prefix) / "share/executable-scientific-contracts/study/camera_ready"


def study_root() -> Path:
    root = SOURCE_STUDY if SOURCE_STUDY.is_dir() else INSTALLED_STUDY
    if not root.is_dir():
        raise FileNotFoundError("Camera-ready study assets are missing; use the complete source release or reinstall the wheel.")
    return root


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_new(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def checked_child(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or Path(relative).is_absolute():
        raise ValueError(f"Unsafe evidence path: {relative}")
    return path


def score(rows):
    result = {}
    for method in METHODS:
        eligible = [r for r in rows if r["oracle_valid"] and r["eligible_examples"] > 0]
        faults = [r for r in eligible if r["expected_state"] == "fault"]
        repairs = [r for r in eligible if r["expected_state"] == "conformant" and "repair" in r["variant"].lower()]
        benign = [r for r in eligible if r["expected_state"] == "conformant" and "benign" in r["variant"].lower()]
        result[method] = {
            "decisions": dict(Counter(r["methods"][method]["decision"] for r in rows)),
            "fault_variants": len(faults),
            "faults_evaluated": sum(r["methods"][method]["decision"] in {"pass", "fail"} for r in faults),
            "faults_detected": sum(r["methods"][method]["decision"] == "fail" for r in faults),
            "repairs": len(repairs), "repairs_accepted": sum(r["methods"][method]["decision"] == "pass" for r in repairs),
            "benign_rewrites": len(benign), "benign_false_alarms": sum(r["methods"][method]["decision"] == "fail" for r in benign),
            "execution_errors": sum(r["methods"][method]["decision"] == "error" for r in rows),
        }
    return result


def validate_rows(rows, target):
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{target}: missing challenge rows")
    for row in rows:
        required = {"id", "target", "mechanism", "variant", "partition", "expected_state", "oracle_valid", "eligible_examples", "methods", "observed"}
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError(f"{target}: incomplete challenge row")
        label = str(row["id"])
        if not label or row["target"] != target:
            raise ValueError(f"{label}: target or identifier differs")
        if row["partition"] not in {"development", "locked_evaluation"}:
            raise ValueError(f"{label}: unknown partition")
        exposed = row.get("exposed_rerun") or "exposed" in str(row.get("evaluation_status", ""))
        if exposed and row["partition"] == "locked_evaluation":
            raise ValueError(f"{label}: previously evaluated cases must use the development partition")
        if row["expected_state"] not in {"fault", "conformant", "not_evaluated"}:
            raise ValueError(f"{label}: unknown expected state")
        if not isinstance(row["oracle_valid"], bool):
            raise ValueError(f"{label}: oracle_valid must be Boolean")
        count = row["eligible_examples"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"{label}: invalid eligibility count")
        if count and not isinstance(row.get("implementation"), dict):
            raise ValueError(f"{label}: source revision and file information are missing")
        if not isinstance(row["methods"], dict) or set(row["methods"]) != set(METHODS):
            raise ValueError(f"{label}: missing or unknown comparison method")
        for method in METHODS:
            item = row["methods"][method]
            if not isinstance(item, dict) or item.get("decision") not in DECISIONS:
                raise ValueError(f"{label}: unknown decision for {method}")
            if not isinstance(item.get("reason"), str) or not item["reason"].strip():
                raise ValueError(f"{label}: missing decision reason")
            if count == 0 and item["decision"] in {"pass", "fail"}:
                raise ValueError(f"{label}: zero eligible examples require not_evaluated")
        if row["expected_state"] == "not_evaluated" and (count or any(row["methods"][m]["decision"] in {"pass", "fail"} for m in METHODS)):
            raise ValueError(f"{label}: unavailable evidence incorrectly evaluated")


def verify_behavior_results(results_root: Path) -> dict:
    "Check saved result files against their manifests."
    all_rows, files = [], {}
    for target in TARGETS:
        directory = results_root / target
        manifest_path = directory / "camera_ready_target_manifest.json"
        public_fallback = not manifest_path.is_file()
        if not manifest_path.is_file():
            manifest_path = study_root() / "results" / target / "camera_ready_target_manifest.json"
        manifest = read(manifest_path)
        if manifest.get("target") != target or manifest.get("schema_version") != "2.0":
            raise ValueError(f"{target}: invalid evidence manifest")
        for relative, expected in manifest["files"].items():
            path = checked_child(directory, relative)
            if not path.is_file():
                raise FileNotFoundError(path)
            permitted = {expected}
            if public_fallback:
                permitted.add(manifest.get("source_artifact_raw_sha256", {}).get(relative))
            if sha(path) not in permitted:
                raise ValueError(f"{target}: evidence digest differs for {relative}")
        challenge_name = manifest["challenge_file"]
        if challenge_name not in manifest["files"]:
            raise ValueError(f"{target}: selected challenge is not sealed")
        path = checked_child(directory, challenge_name)
        payload = read(path)
        if payload.get("schema_version") != "2.0":
            raise ValueError(f"{target}: unsupported challenge schema")
        validate_rows(payload.get("rows"), target)
        files[target] = {"path": str(path), "sha256": sha(path), "manifest_sha256": sha(manifest_path)}
        all_rows.extend(payload["rows"])
    ids = [r["id"] for r in all_rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Challenge identifiers are not unique")
    excluded = [r["id"] for r in all_rows if not r["oracle_valid"]]
    return {"schema_version": "2.0", "status": "valid", "operation": "saved_result_verification",
            "upstream_implementations_executed": False, "inputs": files, "rows": len(all_rows),
            "excluded_unvalidated_rows": excluded,
            "by_target": {t: score([r for r in all_rows if r["target"] == t]) for t in TARGETS},
            "interpretation": "Results for the listed implementation variants. Unavailable coverage is recorded as not_evaluated."}


def legacy_adapter_path() -> Path:
    return Path(__file__).resolve().parent / "adapters/agentprm.py"


def execution_protocol(original: Path, destination: Path, target: str, *, confirmation=False, output=None):
    payload = read(original)
    changes = {}
    if "legacy_adapter" in payload:
        changes["legacy_adapter"] = {"before": payload["legacy_adapter"], "after": str(legacy_adapter_path())}
        payload["legacy_adapter"] = str(legacy_adapter_path())
    if confirmation and target == "toolace":
        for key, file in (("first_challenge_sha256", "challenge_results_v1.json"), ("case_analysis_v1_sha256", "case_analysis_v1.json")):
            changes[key] = {"before": payload[key], "after": sha(output / file)}
            payload[key] = sha(output / file)
    save_new(destination, payload)
    save_new(destination.with_suffix(".rebinding.json"), {"original_protocol": str(original), "original_sha256": sha(original),
             "execution_protocol_sha256": sha(destination), "changes": changes,
             "meaning": "Updated acquisition paths and references to the first-run result files."})
    return destination


def runner_command(target: str, source_root: Path, protocol: Path, output_dir: Path, *, dataset_jsonl=None, data=None, tokenizer=None, historical_tokenizer=None, assets=None) -> list[str]:
    if target not in TARGETS:
        raise ValueError(f"Unknown target: {target}")
    assets = assets or study_root()
    command = [sys.executable, str(assets / "analysis" / f"{target}_analysis.py"), "--source-root", str(source_root),
               "--protocol", str(protocol), "--output-dir", str(output_dir)]
    if target == "agentabstain":
        if dataset_jsonl is None:
            raise ValueError("agentabstain requires --dataset-jsonl")
        command += ["--dataset-jsonl", str(dataset_jsonl)]
    if target == "toolace":
        if data is None or tokenizer is None:
            raise ValueError("toolace requires --data and --tokenizer")
        command += ["--data", str(data), "--tokenizer", str(tokenizer)]
        if historical_tokenizer is not None:
            command += ["--historical-tokenizer", str(historical_tokenizer)]
    return command


def prepare_tokenizer(input_dir: Path, context: Path, assets: Path) -> Path:
    "Check and copy the required tokenizer files."
    recorded = read(assets / "metadata/toolace_tokenizer_acquisition.json")
    required = {"tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"}
    for name in required:
        path = input_dir / name
        if not path.is_file() or sha(path) != recorded["assets"][name]:
            raise ValueError(f"Pinned ToolACE tokenizer asset differs or is missing: {name}")
    destination = context / "tokenizer"
    destination.mkdir()
    copied = {}
    for name, expected in recorded["assets"].items():
        source = input_dir / name
        if source.is_file() and sha(source) == expected:
            shutil.copyfile(source, destination / name)
            copied[name] = expected
    recorded["runtime_reuse"] = {"source": str(input_dir), "validated_assets": copied,
                                  "validated_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "weights_copied": False}
    save_new(destination / "acquisition.json", recorded)
    return destination


def seal_target(output: Path, target: str):
    selection = "challenge_results_verified_v2.json" if target == "toolace" else "challenge_results_v2.json"
    names = ["case_analysis_v1.json", "challenge_results_v1.json", selection]
    for name in names:
        if not (output / name).is_file():
            raise FileNotFoundError(f"Runner did not produce {name}")
    save_new(output / "camera_ready_target_manifest.json", {"schema_version": "2.0", "target": target,
         "challenge_file": selection, "files": {name: sha(output / name) for name in names},
         "scope": "SHA-256 hashes of saved result files."})


def run_behavior(*, target: str, source_root: Path, protocol: Path, output_dir: Path, dataset_jsonl=None, data=None, tokenizer=None, historical_tokenizer=None):
    assets = study_root()
    source_root, protocol, output_dir = source_root.resolve(), protocol.resolve(), output_dir.resolve()
    # Validate required target arguments before creating execution artifacts.
    runner_command(target, source_root, protocol, output_dir, dataset_jsonl=dataset_jsonl, data=data, tokenizer=tokenizer,
                   historical_tokenizer=historical_tokenizer, assets=assets)
    if output_dir.exists():
        raise FileExistsError("Output directory already exists; choose a new --output-dir.")
    expected_pin = read(protocol)["source_revision"]
    pin = subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source_root), "status", "--porcelain"], text=True).strip()
    if pin != expected_pin or dirty:
        raise ValueError("Use a clean upstream checkout at the protocol's pinned revision.")
    context = output_dir.parent / f".{output_dir.name}.execution"
    context.mkdir(parents=True, exist_ok=False)
    execution = execution_protocol(protocol, context / "base_protocol.json", target)
    staged_tokenizer = prepare_tokenizer(Path(tokenizer).resolve(), context, assets) if target == "toolace" else None
    command = runner_command(target, source_root, execution, output_dir, dataset_jsonl=Path(dataset_jsonl).resolve() if dataset_jsonl else None,
                             data=Path(data).resolve() if data else None, tokenizer=staged_tokenizer,
                             historical_tokenizer=Path(historical_tokenizer).resolve() if historical_tokenizer else None, assets=assets)
    save_new(context / "base_command.json", {"command": command, "runner_sha256": sha(Path(command[1])),
             "source_revision": pin, "scope": "Upstream execution with source-availability and input-eligibility checks."})
    subprocess.run(command, check=True)
    confirmation_names = {"agentprm": ("agentprm_guarded_v2.py", "agentprm_guarded_v2.json"),
                          "toolace": ("toolace_evaluator_v2.py", "toolace_confirmation_v2.json"),
                          "agentabstain": ("agentabstain_guards_v2.py", "agentabstain_v2.json")}
    script, protocol_name = confirmation_names[target]
    confirmation = execution_protocol(assets / "protocols" / protocol_name, context / "confirmation_protocol.json", target,
                                      confirmation=True, output=output_dir)
    confirm = [sys.executable, str(assets / "analysis" / script), "--protocol", str(confirmation)]
    if target == "agentprm":
        confirm += ["--results-dir", str(output_dir)]
    elif target == "toolace":
        confirm += ["--source-root", str(source_root), "--tokenizer", str(staged_tokenizer), "--results", str(output_dir)]
    else:
        confirm += ["--source-root", str(source_root), "--output-dir", str(output_dir)]
    save_new(context / "confirmation_command.json", {"command": confirm, "runner_sha256": sha(Path(confirm[1])),
             "helper_sha256": sha(assets / "analysis" / f"{target}_analysis.py"), "reused_exposed_confirmation": True})
    subprocess.run(confirm, check=True)
    seal_target(output_dir, target)
    return {"status": "complete", "operation": "upstream_execution", "target": target, "output_dir": str(output_dir),
            "execution_context": str(context), "base_command": command, "confirmation_command": confirm,
            "confirmation_is_untouched_holdout": False}
