from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

from .core import load_manifest, sha256


def _equation_fixture() -> dict[str, Any]:
    gamma = 0.9
    horizon = 4
    return {
        "gamma": gamma,
        "horizon": horizon,
        "reward": 1.0,
        "values": [gamma ** (horizon - 1 - t) for t in range(horizon)],
    }


def _mutate_equation(fixture: dict[str, Any], operator: str) -> None:
    gamma = fixture["gamma"]
    horizon = fixture["horizon"]
    reward = fixture["reward"]
    if operator == "reverse_temporal_exponent":
        fixture["values"] = [gamma**t for t in range(horizon)]
    elif operator == "shift_temporal_index":
        fixture["values"] = [gamma ** (horizon - t) for t in range(horizon)]
    elif operator == "drop_terminal_offset":
        fixture["values"] = [gamma ** (horizon - t) for t in range(horizon)]
    elif operator == "invert_discount":
        fixture["values"] = [(1 / gamma) ** (horizon - 1 - t) for t in range(horizon)]
    elif operator == "ignore_discount":
        fixture["values"] = [reward] * horizon
    elif operator == "negate_reward":
        fixture["values"] = [-value for value in fixture["values"]]
    elif operator == "normalize_wrong_axis":
        total = sum(fixture["values"])
        fixture["values"] = [value / total for value in fixture["values"]]
    elif operator == "include_terminal_twice":
        fixture["values"][-1] *= 2
    else:
        raise KeyError(operator)


def _equation_conforms(fixture: dict[str, Any]) -> bool:
    expected = _equation_fixture()["values"]
    return len(fixture["values"]) == len(expected) and all(
        abs(left - right) < 1e-12
        for left, right in zip(fixture["values"], expected)
    )


def _schema_fixture() -> dict[str, Any]:
    return {
        "task_id": "fixture-1",
        "success": True,
        "action": "lookup",
        "observation": "record found",
        "tool": {"name": "lookup", "required_arguments": ["record_id"]},
        "arguments": {"record_id": "R-1"},
        "crop": {"prompt": 5, "response": 3, "maximum": 8, "response_start": 5},
        "loss_mask": [0, 0, 0, 0, 0, 1, 1, 0],
    }


def _mutate_schema(fixture: dict[str, Any], operator: str) -> None:
    if operator == "drop_required_field":
        del fixture["success"]
    elif operator == "rename_required_field":
        fixture["task"] = fixture.pop("task_id")
    elif operator == "wrong_field_type":
        fixture["success"] = "true"
    elif operator == "duplicate_action_from_observation":
        fixture["action"] = fixture["observation"]
    elif operator == "unknown_tool_name":
        fixture["action"] = "missing_tool"
    elif operator == "remove_tool_argument":
        fixture["arguments"].clear()
    elif operator == "shift_crop_coordinate":
        fixture["crop"]["response_start"] += 2
    elif operator == "supervise_prompt_token":
        fixture["loss_mask"][2] = 1
    else:
        raise KeyError(operator)


def _schema_conforms(fixture: dict[str, Any]) -> bool:
    required = {"task_id", "success", "action", "observation", "tool", "arguments"}
    if not required.issubset(fixture):
        return False
    if not isinstance(fixture["success"], bool):
        return False
    if fixture["action"] == fixture["observation"]:
        return False
    if fixture["action"] != fixture["tool"]["name"]:
        return False
    if not all(
        argument in fixture["arguments"]
        for argument in fixture["tool"]["required_arguments"]
    ):
        return False
    crop = fixture["crop"]
    if crop["response_start"] != crop["prompt"]:
        return False
    if any(fixture["loss_mask"][: crop["prompt"]]):
        return False
    if fixture["loss_mask"][-1] != 0:
        return False
    return True


def _logging_fixture() -> dict[str, Any]:
    return {
        "run_id": "run-1",
        "expected_run_id": "run-1",
        "output_tokens": 4,
        "messages": [
            {"role": "assistant", "tool_calls": [{"id": "c1", "name": "lookup"}]},
            {"role": "tool", "tool_call_id": "c1", "content": "found"},
            {"role": "assistant", "content": "DONE"},
        ],
        "terminal_events": [{"success": True, "payload": "complete"}],
    }


def _mutate_logging(fixture: dict[str, Any], operator: str) -> None:
    if operator == "drop_final_response":
        fixture["messages"].pop()
    elif operator == "count_unlogged_output":
        fixture["messages"] = []
    elif operator == "orphan_tool_result":
        fixture["messages"][1]["tool_call_id"] = "unknown"
    elif operator == "duplicate_terminal_event":
        fixture["terminal_events"].append(copy.deepcopy(fixture["terminal_events"][0]))
    elif operator == "reorder_call_and_result":
        fixture["messages"][0], fixture["messages"][1] = (
            fixture["messages"][1],
            fixture["messages"][0],
        )
    elif operator == "drop_success_event":
        fixture["terminal_events"] = []
    elif operator == "stale_run_identifier":
        fixture["run_id"] = "run-old"
    elif operator == "truncate_terminal_payload":
        fixture["terminal_events"][0]["payload"] = ""
    else:
        raise KeyError(operator)


def _logging_conforms(fixture: dict[str, Any]) -> bool:
    if fixture["run_id"] != fixture["expected_run_id"]:
        return False
    if fixture["output_tokens"] > 0 and not fixture["messages"]:
        return False
    pending: set[str] = set()
    for message in fixture["messages"]:
        for call in message.get("tool_calls", []):
            pending.add(call["id"])
        if message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            if call_id not in pending:
                return False
            pending.remove(call_id)
    if not fixture["messages"] or fixture["messages"][-1].get("role") != "assistant":
        return False
    if len(fixture["terminal_events"]) != 1:
        return False
    terminal = fixture["terminal_events"][0]
    return bool(terminal.get("success") and terminal.get("payload"))


def _metric_fixture() -> dict[str, Any]:
    rewards = [1.0, 0.0, 1.0, 0.0, 1.0]
    return {
        "rewards": rewards,
        "threshold": 1.0 - 1e-6,
        "eligible": len(rewards),
        "successes": 3,
        "reported": 3 / 5,
    }


def _mutate_metric(fixture: dict[str, Any], operator: str) -> None:
    if operator == "wrong_denominator":
        fixture["reported"] = fixture["successes"] / (fixture["eligible"] - 1)
    elif operator == "exclude_failures":
        fixture["reported"] = 1.0
    elif operator == "duplicate_success":
        fixture["successes"] += 1
        fixture["reported"] = fixture["successes"] / fixture["eligible"]
    elif operator == "threshold_shift":
        fixture["threshold"] = 0.5
        fixture["reported"] = sum(
            reward >= fixture["threshold"] for reward in fixture["rewards"]
        ) / fixture["eligible"]
    elif operator == "mean_instead_of_pass_rate":
        fixture["rewards"] = [0.8, 0.0, 1.0, 0.0, 1.0]
        fixture["reported"] = sum(fixture["rewards"]) / fixture["eligible"]
    elif operator == "integer_truncation":
        fixture["reported"] = fixture["successes"] // fixture["eligible"]
    else:
        raise KeyError(operator)


def _metric_conforms(fixture: dict[str, Any]) -> bool:
    expected_successes = sum(
        reward >= 1.0 - 1e-6 for reward in fixture["rewards"]
    )
    expected = expected_successes / len(fixture["rewards"])
    return (
        fixture["eligible"] == len(fixture["rewards"])
        and fixture["successes"] == expected_successes
        and abs(fixture["reported"] - expected) < 1e-12
        and abs(fixture["threshold"] - (1.0 - 1e-6)) < 1e-12
    )


RUNNERS = {
    "equation": (_equation_fixture, _mutate_equation, _equation_conforms),
    "schema": (_schema_fixture, _mutate_schema, _schema_conforms),
    "logging": (_logging_fixture, _mutate_logging, _logging_conforms),
    "metric": (_metric_fixture, _mutate_metric, _metric_conforms),
}


def _wilson(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = (
        z
        * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
        / denominator
    )
    return center - half, center + half


def run_mutations(manifest_path: Path, output: Path) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    results = []
    clean_failures = []
    for mutation in manifest["mutations"]:
        fixture_factory, mutator, conforms = RUNNERS[mutation["class"]]
        clean = fixture_factory()
        clean_passes = conforms(clean)
        if not clean_passes:
            clean_failures.append(mutation["id"])
        mutated = copy.deepcopy(clean)
        mutator(mutated, mutation["operator"])
        detected = clean_passes and not conforms(mutated)
        results.append({**mutation, "clean_passes": clean_passes, "detected": detected})

    by_class = {}
    for mutation_class in RUNNERS:
        subset = [result for result in results if result["class"] == mutation_class]
        successes = sum(result["detected"] for result in subset)
        low, high = _wilson(successes, len(subset))
        by_class[mutation_class] = {
            "detected": successes,
            "total": len(subset),
            "rate": successes / len(subset),
            "wilson_low_95": low,
            "wilson_high_95": high,
        }
    detected = sum(result["detected"] for result in results)
    low, high = _wilson(detected, len(results))
    payload = {
        "manifest_sha256": sha256(manifest_path),
        "mutations": results,
        "overall": {
            "detected": detected,
            "total": len(results),
            "rate": detected / len(results),
            "wilson_low_95": low,
            "wilson_high_95": high,
        },
        "by_class": by_class,
        "clean_fixture_failures": clean_failures,
        "valid": len(results) == 30 and not clean_failures,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload
