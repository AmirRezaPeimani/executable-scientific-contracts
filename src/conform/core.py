from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ContractResult:
    contract_id: str
    target: str
    status: str
    observed: Any
    expected: Any
    severity: int
    consequence: str
    evidence: list[str]
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def git_revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def git_dirty(path: Path) -> bool:
    return bool(
        subprocess.check_output(
            ["git", "-C", str(path), "status", "--porcelain"], text=True
        ).strip()
    )


def load_manifest(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def timed_result(
    contract_id: str,
    target: str,
    status: str,
    observed: Any,
    expected: Any,
    severity: int,
    consequence: str,
    evidence: list[str],
    started: float,
) -> ContractResult:
    return ContractResult(
        contract_id=contract_id,
        target=target,
        status=status,
        observed=observed,
        expected=expected,
        severity=severity,
        consequence=consequence,
        evidence=evidence,
        elapsed_seconds=time.time() - started,
    )


def write_audit(
    output: Path,
    *,
    target: str,
    revision: str,
    dirty: bool,
    results: list[ContractResult],
    command: str,
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "target": target,
        "revision": revision,
        "dirty": dirty,
        "command": command,
        "contracts": [result.to_dict() for result in results],
        "counts": {
            status: sum(result.status == status for result in results)
            for status in ("conformant", "discrepant", "inactive", "error")
        },
        "maximum_demonstrated_severity": max(
            (result.severity for result in results), default=0
        ),
    }
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    payload["output_sha256"] = sha256(output)
    return payload
