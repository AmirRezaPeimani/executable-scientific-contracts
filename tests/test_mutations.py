from pathlib import Path

from conform.mutations import run_mutations


ROOT = Path(__file__).resolve().parents[1]


def test_all_thirty_controlled_mutations_are_detected(tmp_path: Path) -> None:
    result = run_mutations(
        ROOT / "contracts/mutations.yaml", tmp_path / "mutations.json"
    )
    assert result["valid"]
    assert result["overall"]["total"] == 30
    assert result["overall"]["detected"] == 30
    assert result["clean_fixture_failures"] == []
