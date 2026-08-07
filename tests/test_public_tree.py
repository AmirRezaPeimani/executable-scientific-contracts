from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_paper_visible_evidence_paths_and_figures_exist() -> None:
    evidence = [
        "results/reproduction/audits/agentprm.json",
        "results/reproduction/audits/contractbench.json",
        "results/reproduction/audits/toolace.json",
        "results/reproduction/audits/taubench.json",
        "results/agentabstain_prospective_v2.json",
    ]
    figures = [
        "figures/empirical_summary_v1.pdf",
        "figures/evidence_chain_v2.pdf",
        "figures/contract_outcome_landscape_full_v1.pdf",
        "figures/worked_contract_example_v1.pdf",
        "figures/execution_time_after_authoring_v1.pdf",
    ]
    for relative in evidence + figures:
        assert (ROOT / relative).is_file(), relative
