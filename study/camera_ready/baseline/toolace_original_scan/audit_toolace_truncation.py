#!/usr/bin/env python3
"""Reproduce Tool-RL-Box ToolACE SFT conversion and audit truncation semantics.

This is a support audit, not an outcome experiment.  It reproduces the public
converter and the custom SFT dataset's 8,192-token left-truncation/mask logic
without training a model.  It records whether a retained supervised tool call
still has its named tool in the retained prompt.
"""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer


TOOLS_MARKER = "Here is a list of functions in JSON format that you can invoke:\n"
TOOLS_END = "\nShould you decide to return the function call(s)."
TOOLS_PREFIX = (
    "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
)
TOOLS_SUFFIX = (
    "\n</tools>\n\nFor each function call, return a json object with function name "
    "and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call>\n"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def parse_calls(text: str) -> list[dict[str, Any]]:
    """Match the official converter's Python-expression parser."""
    try:
        body = ast.parse(text, mode="eval").body
    except (SyntaxError, ValueError):
        return []
    nodes = body.elts if isinstance(body, ast.List) else [body]
    calls: list[dict[str, Any]] = []
    for call in nodes:
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        if isinstance(func, ast.Attribute):
            parts: list[str] = []
            while isinstance(func, ast.Attribute):
                parts.append(func.attr)
                func = func.value
            if isinstance(func, ast.Name):
                parts.append(func.id)
            name = ".".join(reversed(parts))
        elif isinstance(func, ast.Name):
            name = func.id
        else:
            continue
        try:
            arguments = {
                keyword.arg: ast.literal_eval(keyword.value)
                for keyword in call.keywords
                if keyword.arg is not None
            }
        except (ValueError, TypeError):
            continue
        calls.append({"name": name, "arguments": arguments})
    return calls


def convert_system(system: str) -> tuple[str, dict[str, str]]:
    """Reproduce data/convert_toolace.py's system-schema conversion."""
    raw = system.split(TOOLS_MARKER)[-1].split(TOOLS_END)[0].strip()
    # The released source removes the final period following the JSON list.
    if raw.endswith("."):
        raw = raw[:-1]
    tools = json.loads(raw.strip())
    translations: dict[str, str] = {}
    for tool in tools:
        name = tool["name"]
        if " " in name:
            translations[name] = name.replace(" ", "_")
            tool["name"] = translations[name]
    rendered = json.dumps(tools, indent=2)
    return TOOLS_PREFIX + rendered + TOOLS_SUFFIX, translations


def render_target(
    value: str, translations: dict[str, str]
) -> tuple[str, list[str]]:
    content = value
    for original, normalized in translations.items():
        if original in content:
            content = content.replace(original, normalized)
    calls = parse_calls(content)
    rendered = "".join(
        "<tool_call>\n" + json.dumps(call) + "\n</tool_call>\n" for call in calls
    )
    return rendered, [call["name"] for call in calls]


def coded_loss_count(
    prompt_length: int, response_length: int, max_length: int
) -> int:
    """Exactly reproduce the released loss-mask coordinate operations."""
    sequence_length = prompt_length + response_length
    kept = min(sequence_length, max_length)
    mask = torch.ones(kept, dtype=torch.int8)
    if prompt_length > 1:
        mask[: min(prompt_length, kept) - 1] = 0
    mask[min(prompt_length + response_length, kept) - 1] = 0
    return int(mask.sum().item())


def intended_retained_target_count(
    prompt_length: int, response_length: int, max_length: int
) -> int:
    """Response-token count after left crop, excluding the final EOS token."""
    sequence_length = prompt_length + response_length
    crop_start = max(0, sequence_length - max_length)
    response_start = prompt_length
    retained = max(0, sequence_length - max(crop_start, response_start))
    return max(0, retained - 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = json.loads(args.data.read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=False
    )

    counts: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    parse_failures: list[dict[str, Any]] = []

    for row_index, row in enumerate(rows):
        try:
            system_prompt, translations = convert_system(row["system"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            parse_failures.append(
                {"row_index": row_index, "stage": "system", "error": repr(exc)}
            )
            continue
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]
        target_index = 0
        for message_index, item in enumerate(row["conversations"]):
            role = item["from"]
            value = item["value"]
            if role == "assistant" and value.startswith("["):
                rendered, tool_names = render_target(value, translations)
                if not rendered:
                    parse_failures.append(
                        {
                            "row_index": row_index,
                            "message_index": message_index,
                            "stage": "target",
                            "value": value,
                        }
                    )
                    messages.append({"role": role, "content": rendered})
                    target_index += 1
                    continue

                prompt_string = tokenizer.apply_chat_template(
                    copy.deepcopy(messages),
                    add_generation_prompt=True,
                    tokenize=False,
                )
                prompt_ids = tokenizer(
                    prompt_string, add_special_tokens=False
                )["input_ids"]
                response_ids = tokenizer(
                    rendered + tokenizer.eos_token, add_special_tokens=False
                )["input_ids"]
                prompt_length = len(prompt_ids)
                response_length = len(response_ids)
                sequence_length = prompt_length + response_length
                crop_start = max(0, sequence_length - args.max_length)
                retained_prompt_ids = (
                    prompt_ids[crop_start:] if crop_start < prompt_length else []
                )
                retained_prompt = tokenizer.decode(
                    retained_prompt_ids, skip_special_tokens=False
                )
                names_present = [name in retained_prompt for name in tool_names]
                overlength = sequence_length > args.max_length
                code_loss = coded_loss_count(
                    prompt_length, response_length, args.max_length
                )
                intended_loss = intended_retained_target_count(
                    prompt_length, response_length, args.max_length
                )

                counts["targets"] += 1
                counts["tool_calls"] += len(tool_names)
                counts["overlength_targets"] += int(overlength)
                counts["overlength_tool_calls"] += int(overlength) * len(tool_names)
                counts["zero_coded_loss_targets"] += int(code_loss == 0)
                counts["positive_intended_zero_coded_targets"] += int(
                    intended_loss > 0 and code_loss == 0
                )
                counts["missing_named_schema_targets"] += int(not all(names_present))
                counts["missing_named_schema_tool_calls"] += sum(
                    not present for present in names_present
                )
                counts["overlength_missing_named_schema_targets"] += int(
                    overlength and not all(names_present)
                )

                records.append(
                    {
                        "row_index": row_index,
                        "message_index": message_index,
                        "target_index": target_index,
                        "tool_names": "|".join(tool_names),
                        "prompt_tokens": prompt_length,
                        "response_tokens": response_length,
                        "sequence_tokens": sequence_length,
                        "crop_start": crop_start,
                        "retained_prompt_tokens": len(retained_prompt_ids),
                        "overlength": overlength,
                        "coded_loss_tokens": code_loss,
                        "intended_retained_target_tokens": intended_loss,
                        "all_tool_names_in_retained_prompt": all(names_present),
                        "tool_name_presence": "|".join(
                            "1" if present else "0" for present in names_present
                        ),
                    }
                )
                messages.append({"role": role, "content": rendered})
                target_index += 1
            else:
                messages.append({"role": role, "content": value})

    def rate(numerator: str, denominator: str) -> float | None:
        den = counts[denominator]
        return counts[numerator] / den if den else None

    summary = {
        "audit_kind": "support_only_no_model_outcomes",
        "pipeline": {
            "repository": str(args.repo),
            "git_revision": git_revision(args.repo),
            "source_converter": "data/convert_toolace.py",
            "source_dataset_loader": (
                "verl/verl/utils/dataset/sft_dataset_wo_apply_template.py"
            ),
            "configured_max_length": args.max_length,
            "configured_truncation": "left",
        },
        "inputs": {
            "data": str(args.data),
            "data_sha256": sha256(args.data),
            "tokenizer": str(args.tokenizer),
            "tokenizer_json_sha256": sha256(args.tokenizer / "tokenizer.json"),
            "rows": len(rows),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
        },
        "counts": dict(sorted(counts.items())),
        "rates": {
            "overlength_target_rate": rate("overlength_targets", "targets"),
            "zero_coded_loss_target_rate": rate(
                "zero_coded_loss_targets", "targets"
            ),
            "positive_intended_zero_coded_rate": rate(
                "positive_intended_zero_coded_targets", "targets"
            ),
            "missing_named_schema_target_rate": rate(
                "missing_named_schema_targets", "targets"
            ),
            "overlength_missing_named_schema_rate_among_overlength": rate(
                "overlength_missing_named_schema_targets",
                "overlength_targets",
            ),
        },
        "parse_failures": len(parse_failures),
    }

    with (args.output_dir / "per_target.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    (args.output_dir / "parse_failures.json").write_text(
        json.dumps(parse_failures, indent=2, ensure_ascii=False) + "\n"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
