import copy
import json
import shutil
from pathlib import Path

import pytest

from conform import camera_ready
from conform.cli import main


def copied_results(tmp_path):
    destination = tmp_path / "results"
    shutil.copytree(camera_ready.study_root() / "results", destination)
    return destination


def mutate_challenge(root, edit, *, reseal=False):
    folder = root / "agentprm"
    manifest_path = folder / "camera_ready_target_manifest.json"
    manifest = camera_ready.read(manifest_path)
    path = folder / manifest["challenge_file"]
    payload = camera_ready.read(path)
    edit(payload)
    path.write_text(json.dumps(payload))
    if reseal:
        manifest["files"][path.name] = camera_ready.sha(path)
        manifest_path.write_text(json.dumps(manifest))


def test_saved_evidence_verifies_without_executing_upstream(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("saved-result verification must not dispatch a process")
    monkeypatch.setattr(camera_ready.subprocess, "run", forbidden)
    report = camera_ready.verify_behavior_results(camera_ready.study_root() / "results")
    assert report["status"] == "valid"
    assert report["operation"] == "saved_result_verification"
    assert not report["upstream_implementations_executed"]
    assert report["rows"] > 30
    for target in camera_ready.TARGETS:
        native = report["by_target"][target]["native_tests"]
        assert native["faults_evaluated"] == 0
        assert native["faults_detected"] == 0


def test_changed_decision_is_detected_by_integrity_seal(tmp_path):
    root = copied_results(tmp_path)
    mutate_challenge(root, lambda payload: payload["rows"][0]["methods"]["behavioral_contract"].update(decision="pass"))
    with pytest.raises(ValueError, match="digest differs"):
        camera_ready.verify_behavior_results(root)


def test_unknown_decision_rejected_even_when_hashes_are_consistent(tmp_path):
    root = copied_results(tmp_path)
    mutate_challenge(root, lambda payload: payload["rows"][0]["methods"]["schema"].update(decision="unknown"), reseal=True)
    with pytest.raises(ValueError, match="unknown decision"):
        camera_ready.verify_behavior_results(root)


def test_zero_eligibility_cannot_pass_even_when_hashes_are_consistent(tmp_path):
    root = copied_results(tmp_path)
    def change(payload):
        row = next(r for r in payload["rows"] if r["eligible_examples"] == 0)
        row["methods"]["behavioral_contract"]["decision"] = "pass"
    mutate_challenge(root, change, reseal=True)
    with pytest.raises(ValueError, match="zero eligible"):
        camera_ready.verify_behavior_results(root)


def test_missing_evidence_file_is_rejected(tmp_path):
    root = copied_results(tmp_path)
    (root / "toolace/case_analysis_v1.json").unlink()
    with pytest.raises(FileNotFoundError):
        camera_ready.verify_behavior_results(root)


def test_manifest_cannot_escape_results_directory(tmp_path):
    root = copied_results(tmp_path)
    path = root / "agentprm/camera_ready_target_manifest.json"
    manifest = camera_ready.read(path)
    manifest["files"]["../outside.json"] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unsafe evidence path"):
        camera_ready.verify_behavior_results(root)


def test_exposed_confirmation_cannot_be_reclassified_as_locked(tmp_path):
    root = copied_results(tmp_path)
    def change(payload):
        row = payload["rows"][0]
        row["exposed_rerun"] = True
        row["partition"] = "locked_evaluation"
    mutate_challenge(root, change, reseal=True)
    with pytest.raises(ValueError, match="exposed confirmation"):
        camera_ready.verify_behavior_results(root)


def test_error_and_unmanifested_fault_do_not_count_as_detection():
    path = camera_ready.study_root() / "results/agentprm/challenge_results_v2.json"
    row = copy.deepcopy(camera_ready.read(path)["rows"][0])
    row["oracle_valid"] = False
    row["methods"]["behavioral_contract"]["decision"] = "error"
    scored = camera_ready.score([row])["behavioral_contract"]
    assert scored["fault_variants"] == scored["faults_detected"] == 0
    assert scored["execution_errors"] == 1


def test_runner_command_propagates_explicit_inputs():
    cmd = camera_ready.runner_command("toolace", Path("/source"), Path("/protocol"), Path("/output"),
        data=Path("/data"), tokenizer=Path("/tokens"), historical_tokenizer=Path("/historical"), assets=Path("/assets"))
    assert cmd[1] == "/assets/analysis/toolace_analysis.py"
    assert cmd[cmd.index("--source-root") + 1] == "/source"
    assert cmd[cmd.index("--protocol") + 1] == "/protocol"
    assert cmd[cmd.index("--data") + 1] == "/data"
    assert cmd[cmd.index("--tokenizer") + 1] == "/tokens"
    assert cmd[cmd.index("--historical-tokenizer") + 1] == "/historical"
    with pytest.raises(ValueError, match="dataset-jsonl"):
        camera_ready.runner_command("agentabstain", Path("/s"), Path("/p"), Path("/o"))


def test_behavior_runner_dispatches_base_and_confirmation(monkeypatch, tmp_path):
    assets = camera_ready.study_root()
    monkeypatch.setattr(camera_ready, "study_root", lambda: assets)
    pin = camera_ready.read(assets / "protocols/agentprm_v1.json")["source_revision"]
    monkeypatch.setattr(camera_ready.subprocess, "check_output", lambda cmd, **kw: pin + "\n" if "rev-parse" in cmd else "")
    calls = []
    output = tmp_path / "agentprm"
    def dispatch(command, check):
        calls.append(command)
        output.mkdir(exist_ok=True)
        names = ["case_analysis_v1.json", "challenge_results_v1.json"] if len(calls) == 1 else ["challenge_results_v2.json"]
        for name in names:
            (output / name).write_text("{}")
    monkeypatch.setattr(camera_ready.subprocess, "run", dispatch)
    result = camera_ready.run_behavior(target="agentprm", source_root=tmp_path / "source",
        protocol=assets / "protocols/agentprm_v1.json", output_dir=output)
    assert len(calls) == 2
    assert calls[0][1].endswith("agentprm_analysis.py")
    assert calls[1][1].endswith("agentprm_guarded_v2.py")
    assert "--results-dir" in calls[1]
    assert not result["confirmation_is_untouched_holdout"]
    rebound = camera_ready.read(tmp_path / ".agentprm.execution/base_protocol.json")
    assert rebound["legacy_adapter"] == str(camera_ready.legacy_adapter_path())
    assert (output / "camera_ready_target_manifest.json").is_file()


def test_cli_help_distinguishes_verification_and_execution(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "behavior-check" in help_text and "behavior-run" in help_text and "scan-toolace" in help_text
    assert "does not execute upstream" in " ".join(help_text.split())
    assert main(["validate-manifest"]) == 0


def test_behavior_check_cli_saves_new_report_and_refuses_overwrite(tmp_path):
    output = tmp_path / "summary.json"
    args = ["behavior-check", "--results-root", str(camera_ready.study_root() / "results"), "--output", str(output)]
    assert main(args) == 0
    assert camera_ready.read(output)["status"] == "valid"
    with pytest.raises(FileExistsError):
        main(args)


def test_public_metadata_has_no_original_host_paths_or_raw_training_artifacts():
    study = camera_ready.study_root()
    for path in study.rglob("*"):
        if path.is_file():
            assert path.suffix not in {".parquet", ".safetensors", ".bin", ".log"}
            if path.suffix in {".json", ".csv", ".md"}:
                assert "/Users/" not in path.read_text()
