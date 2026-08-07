from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .adapters import ADAPTERS
from .core import git_dirty, git_revision, load_manifest, write_audit
from .mutations import run_mutations
from .release import build_summary, write_summary
from .specification import load_and_validate_manifest


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CONTRACTS = PACKAGE_ROOT / "contracts"
INSTALLED_CONTRACTS = Path(sys.prefix) / "share/executable-scientific-contracts/contracts"
DEFAULT_CONTRACTS = SOURCE_CONTRACTS if SOURCE_CONTRACTS.exists() else INSTALLED_CONTRACTS
DEFAULT_ROOT = PACKAGE_ROOT if SOURCE_CONTRACTS.exists() else Path.cwd()


def _target_paths(workspace: Path) -> dict[str, Path]:
    return {
        "agentprm": workspace / "third_party/agent_prm",
        "contractbench": workspace / "third_party/contractbench_code",
        "toolace": workspace / "third_party/tool-rl-box",
        "taubench": workspace / "third_party/tau-bench",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="conform")
    parser.add_argument(
        "--manifest", type=Path, default=DEFAULT_CONTRACTS / "contracts.yaml"
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=DEFAULT_ROOT.parent,
        help="directory containing third_party/ after optional upstream acquisition",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list")
    subparsers.add_parser("validate-manifest")
    release = subparsers.add_parser(
        "release-check", help="validate included evidence and reconstruct the paper summary"
    )
    release.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    release.add_argument("--output", type=Path)
    audit = subparsers.add_parser("audit")
    audit.add_argument("target", choices=tuple(ADAPTERS))
    audit.add_argument("--output", type=Path, required=True)
    audit_all = subparsers.add_parser("audit-all")
    audit_all.add_argument("--output-dir", type=Path, required=True)
    mutate = subparsers.add_parser("mutate")
    mutate.add_argument(
        "--mutations", type=Path, default=DEFAULT_CONTRACTS / "mutations.yaml"
    )
    mutate.add_argument("--output", type=Path, required=True)
    return parser


def _audit(target: str, manifest: dict, output: Path, paths: dict[str, Path]) -> dict:
    root = paths[target]
    if not root.exists():
        raise FileNotFoundError(root)
    expected = manifest["targets"][target]["revision"]
    actual = git_revision(root)
    if actual != expected:
        raise RuntimeError(f"{target} revision {actual} != {expected}")
    results = ADAPTERS[target](root, manifest["targets"][target]["contracts"])
    return write_audit(
        output,
        target=target,
        revision=actual,
        dirty=git_dirty(root),
        results=results,
        command=f"conform audit {target}",
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = load_manifest(args.manifest)
    if args.command == "list":
        for target, entry in manifest["targets"].items():
            print(target)
            for contract in entry["contracts"]:
                print(f"  {contract['id']}  {contract['kind']}")
        return 0
    if args.command == "validate-manifest":
        study = load_and_validate_manifest(args.manifest)
        print(
            json.dumps(
                {
                    "status": "valid",
                    "version": study.version,
                    "targets": len(study.targets),
                    "contracts": sum(len(target.contracts) for target in study.targets),
                    "manifest_sha256": study.manifest_sha256,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "release-check":
        payload = (
            write_summary(args.root, args.output)
            if args.output
            else build_summary(args.root)
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    paths = _target_paths(args.workspace.resolve())
    if args.command == "audit":
        print(json.dumps(_audit(args.target, manifest, args.output, paths), indent=2))
        return 0
    if args.command == "audit-all":
        payload = {
            target: _audit(target, manifest, args.output_dir / f"{target}.json", paths)
            for target in ADAPTERS
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "mutate":
        payload = run_mutations(args.mutations, args.output)
        print(json.dumps(payload["overall"], indent=2, sort_keys=True))
        return 0 if payload["valid"] else 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
