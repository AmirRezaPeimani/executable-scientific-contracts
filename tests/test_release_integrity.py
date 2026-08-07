from pathlib import Path

from conform.release import build_summary


ROOT = Path(__file__).resolve().parents[1]


def test_paper_facing_summary_is_derived_from_included_evidence() -> None:
    summary = build_summary(ROOT)

    assert summary["status"] == "valid"
    assert summary["contracts"]["original"] == {
        "total": 20,
        "outcomes": {"conformant": 12, "discrepant": 7, "inactive": 1},
    }
    assert summary["contracts"]["prospective_agentabstain"] == {
        "total": 5,
        "outcomes": {"conformant": 3, "convention sensitivity": 2},
    }
    assert summary["contracts"]["overall"] == {
        "total": 25,
        "outcomes": {
            "conformant": 15,
            "convention sensitivity": 2,
            "discrepant": 7,
            "inactive": 1,
        },
    }

    study = summary["mutations"]["study_specific"]
    assert study["overall"] == {"detected": 30, "total": 30}
    assert study["by_class"] == {
        "equation": {"detected": 8, "total": 8},
        "logging": {"detected": 8, "total": 8},
        "metric": {"detected": 6, "total": 6},
        "schema": {"detected": 8, "total": 8},
    }
    assert summary["mutations"]["external_first_execution"]["detected"] == 7
    assert summary["mutations"]["external_after_fixture_correction"]["detected"] == 8

    assert summary["agentabstain"]["released_rows"] == 526
    assert summary["agentabstain"]["complete_pairs"] == 263
    assert summary["agentabstain"]["scenarios"] == 8

    assert summary["toolace"]["candidate_targets"] == 10092
    assert summary["toolace"]["parsed_targets"] == 9412
    assert summary["toolace"]["overlength_targets"] == 0
    assert summary["toolace"]["configured_max_length"] == 8192
    assert summary["toolace"]["maximum_sequence_tokens"] == 3945

    contractbench = summary["contractbench"]
    assert contractbench["reported_model"] == "Qwen3.5-9B"
    assert contractbench["reported_episodes"] == 99
    assert contractbench["reported_success_rate_percent"] == 56.6
    assert contractbench["released_episodes"] == 99
    assert contractbench["released_successes"] == 0
    assert contractbench["traces_without_terminal_assistant_response"] == 3
    assert not contractbench["exact_evaluation_slice_identity_verified"]
    assert not contractbench["corrected_model_performance_identifiable"]


def test_fixed_revisions_and_runtime_medians() -> None:
    summary = build_summary(ROOT)
    revisions = summary["revisions"]
    assert revisions["agentprm"]["revision"].startswith("e4714717")
    assert revisions["contractbench"]["revision"].startswith("c50eefee")
    assert revisions["toolace"]["revision"].startswith("1156d649")
    assert revisions["taubench"]["revision"].startswith("59a200c6")
    assert revisions["agentabstain"]["revision"].startswith("f5812497")
    assert revisions["agentabstain"]["data_revision"].startswith("84222842")

    medians = {
        target: round(row["median_seconds"], 3)
        for target, row in summary["post_authoring_runtime"].items()
    }
    assert medians == {
        "agentprm": 0.098,
        "contractbench": 0.115,
        "toolace": 0.102,
        "taubench": 0.233,
        "agentabstain_prospective": 0.393,
    }
