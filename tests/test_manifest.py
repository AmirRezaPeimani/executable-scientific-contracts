from pathlib import Path

from conform.core import load_manifest
from conform.specification import load_and_validate_manifest


ROOT = Path(__file__).resolve().parents[1]


def test_contract_manifest_has_five_targets_and_twenty_five_unique_ids() -> None:
    manifest = load_manifest(ROOT / "contracts/contracts.yaml")
    contracts = [
        contract
        for target in manifest["targets"].values()
        for contract in target["contracts"]
    ]
    assert len(manifest["targets"]) == 5
    assert len(contracts) == 25
    assert len({contract["id"] for contract in contracts}) == 25


def test_typed_manifest_validation() -> None:
    study = load_and_validate_manifest(ROOT / "contracts/contracts.yaml")
    assert len(study.targets) == 5
    assert sum(len(target.contracts) for target in study.targets) == 25
    assert len(study.manifest_sha256) == 64
