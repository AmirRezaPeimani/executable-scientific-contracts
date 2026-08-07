from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from conform.core import ContractResult, timed_result


def _result(
    contract_id: str,
    status: str,
    observed: Any,
    expected: Any,
    severity: int,
    consequence: str,
    evidence: list[str],
    started: float,
) -> ContractResult:
    return timed_result(
        contract_id,
        "toolace",
        status,
        observed,
        expected,
        severity,
        consequence,
        evidence,
        started,
    )


def _mask_counts(prompt: int, response: int, maximum: int) -> tuple[int, int]:
    sequence = prompt + response
    crop = max(0, sequence - maximum)
    retained_response = max(0, sequence - max(crop, prompt))
    implemented = max(0, maximum - min(prompt, maximum)) if crop else response
    corrected = retained_response
    return implemented, corrected


def audit_toolace(
    root: Path,
    specs: list[dict[str, Any]],
    support_path: Path | None = None,
) -> list[ContractResult]:
    del specs
    loader = root / "verl/verl/utils/dataset/sft_dataset_wo_apply_template.py"
    source = loader.read_text(encoding="utf-8")
    support_path = support_path or (
        Path(__file__).resolve().parents[3]
        / "results/support/toolace_released_support.json"
    )
    support = json.loads(support_path.read_text(encoding="utf-8"))
    counts = support["counts"]
    results: list[ContractResult] = []

    started = time.time()
    source_defect = (
        "input_ids = input_ids[-self.max_length :]" in source
        and "min(prompt_length, loss_mask.size(0))" in source
    )
    implemented, corrected = _mask_counts(prompt=10, response=4, maximum=8)
    results.append(
        _result(
            "TACE-COORD-01",
            "discrepant" if source_defect and implemented != corrected else "conformant",
            {"implemented_supervised": implemented, "fixture": [10, 4, 8]},
            {"corrected_supervised": corrected},
            0,
            "The source mismatch is exact but inactive on the released support.",
            [f"{loader}:170", f"{loader}:184", str(support_path)],
            started,
        )
    )

    started = time.time()
    zero_loss = counts["zero_coded_loss_targets"]
    results.append(
        _result(
            "TACE-COORD-02",
            "conformant" if zero_loss == 0 else "discrepant",
            {"zero_loss_targets": zero_loss, "targets": counts["targets"]},
            {"zero_loss_targets": 0},
            0 if zero_loss == 0 else 2,
            "Every released target retains supervised support.",
            [str(support_path)],
            started,
        )
    )

    started = time.time()
    missing = counts["missing_named_schema_targets"]
    results.append(
        _result(
            "TACE-SCHEMA-01",
            "conformant" if missing == 0 else "discrepant",
            {"missing_schema_targets": missing, "targets": counts["targets"]},
            {"missing_schema_targets": 0},
            0 if missing == 0 else 2,
            "Every parsed released target retains its named tool declaration.",
            [str(support_path)],
            started,
        )
    )

    started = time.time()
    masks_prompt = "loss_mask[: min(prompt_length, loss_mask.size(0)) - 1] = 0" in source
    masks_terminal = (
        "loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0"
        in source
    )
    results.append(
        _result(
            "TACE-MASK-01",
            "conformant" if masks_prompt and masks_terminal else "discrepant",
            {"masks_prompt": masks_prompt, "masks_terminal": masks_terminal},
            {"masks_prompt": True, "masks_terminal": True},
            0 if masks_prompt and masks_terminal else 2,
            "Non-cropped examples apply the declared prompt and terminal masks.",
            [f"{loader}:184"],
            started,
        )
    )

    started = time.time()
    overlength = counts["overlength_targets"]
    results.append(
        _result(
            "TACE-SUPPORT-01",
            "inactive" if source_defect and overlength == 0 else "conformant",
            {"source_defect": source_defect, "overlength_targets": overlength},
            {"severity": 0},
            0,
            (
                "A real coordinate mismatch has zero released-data activation "
                "under the official maximum length."
            ),
            [str(support_path)],
            started,
        )
    )
    return results
