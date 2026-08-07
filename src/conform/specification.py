"""Validation and typed access for scientific-contract manifests."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REVISION = re.compile(r"^[0-9a-f]{40}$")
KINDS = {
    "equation",
    "schema",
    "logging",
    "metric",
    "support",
    "negative_control",
}


@dataclass(frozen=True)
class ContractSpec:
    contract_id: str
    kind: str
    severity_ceiling: int


@dataclass(frozen=True)
class TargetSpec:
    name: str
    revision: str
    contracts: tuple[ContractSpec, ...]
    data_revision: str | None = None


@dataclass(frozen=True)
class StudySpec:
    version: str
    targets: tuple[TargetSpec, ...]
    manifest_sha256: str


def _digest(value: dict[str, Any]) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_manifest(value: dict[str, Any]) -> StudySpec:
    if not isinstance(value, dict):
        raise ValueError("manifest must be a mapping")
    if not isinstance(value.get("version"), str):
        raise ValueError("manifest version must be a string")
    targets_value = value.get("targets")
    if not isinstance(targets_value, dict) or not targets_value:
        raise ValueError("manifest must contain at least one target")
    target_specs = []
    global_ids: set[str] = set()
    for target_name, target in targets_value.items():
        if not isinstance(target_name, str) or not target_name:
            raise ValueError("target names must be nonempty strings")
        if not isinstance(target, dict):
            raise ValueError(f"{target_name}: target specification must be a mapping")
        revision = target.get("revision")
        if not isinstance(revision, str) or not REVISION.fullmatch(revision):
            raise ValueError(f"{target_name}: revision must be a full Git SHA")
        data_revision = target.get("data_revision")
        if data_revision is not None and (
            not isinstance(data_revision, str)
            or not REVISION.fullmatch(data_revision)
        ):
            raise ValueError(f"{target_name}: data_revision must be a full Git SHA")
        contracts_value = target.get("contracts")
        if not isinstance(contracts_value, list) or not contracts_value:
            raise ValueError(f"{target_name}: at least one contract is required")
        contracts = []
        for index, contract in enumerate(contracts_value):
            if not isinstance(contract, dict):
                raise ValueError(f"{target_name}[{index}]: contract must be a mapping")
            identifier = contract.get("id")
            kind = contract.get("kind")
            ceiling = contract.get("severity_ceiling")
            if not isinstance(identifier, str) or not identifier:
                raise ValueError(f"{target_name}[{index}]: missing contract id")
            if identifier in global_ids:
                raise ValueError(f"duplicate contract id: {identifier}")
            global_ids.add(identifier)
            if kind not in KINDS:
                raise ValueError(f"{identifier}: unsupported kind {kind!r}")
            if not isinstance(ceiling, int) or not 0 <= ceiling <= 4:
                raise ValueError(f"{identifier}: severity_ceiling must be 0--4")
            contracts.append(
                ContractSpec(
                    contract_id=identifier,
                    kind=kind,
                    severity_ceiling=ceiling,
                )
            )
        target_specs.append(
            TargetSpec(
                name=target_name,
                revision=revision,
                data_revision=data_revision,
                contracts=tuple(contracts),
            )
        )
    return StudySpec(
        version=value["version"],
        targets=tuple(target_specs),
        manifest_sha256=_digest(value),
    )


def load_and_validate_manifest(path: Path) -> StudySpec:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return validate_manifest(value)
