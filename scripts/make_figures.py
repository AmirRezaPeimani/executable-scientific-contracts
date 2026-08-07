#!/usr/bin/env python3
"""Regenerate the five publication figures from included aggregate evidence."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conform.release import build_summary  # noqa: E402


INK = "#000000"
PAPER = "#FFFFFF"
GRID = "#D7DADF"
OUTLINE = "#777C82"
OUTCOMES = {
    "conformant": {"fill": "#DCEEF8", "edge": "#0072B2", "hatch": ""},
    "discrepant": {"fill": "#FBE6B8", "edge": "#D55E00", "hatch": "///"},
    "inactive": {"fill": "#DDF1E8", "edge": "#009E73", "hatch": "..."},
    "convention sensitivity": {
        "fill": "#F1E1EE",
        "edge": "#9C4F83",
        "hatch": "xx",
    },
}
CARD_FILLS = {
    "neutral": "#F8F9FA",
    "conformant": "#E7F3FA",
    "discrepant": "#FCECCB",
    "inactive": "#E7F5EF",
    "convention sensitivity": "#F4E8F1",
    "unavailable": "#ECEEF0",
}


mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.color": INK,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "savefig.facecolor": PAPER,
        "figure.facecolor": PAPER,
        "hatch.linewidth": 0.45,
    }
)


def read_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def read_inventory() -> list[dict[str, str]]:
    with (ROOT / "tables/contract_inventory.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        return list(csv.DictReader(handle))


def contract(payload: dict, contract_id: str) -> dict:
    return next(item for item in payload["contracts"] if item["contract_id"] == contract_id)


def save(fig: mpl.figure.Figure, output: Path, stem: str, pad: float = 0.035) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output / f"{stem}.pdf",
        bbox_inches="tight",
        pad_inches=pad,
        metadata={"Title": stem.replace("_", " ")},
    )
    plt.close(fig)


def card(ax: plt.Axes, x: float, y: float, w: float, h: float, text: str, status: str, size: float) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.003,rounding_size=0.008",
            facecolor=CARD_FILLS[status],
            edgecolor=OUTLINE,
            linewidth=0.62,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=size,
        color=INK,
        linespacing=1.12,
    )


def evidence_chain(output: Path) -> None:
    agent = read_json("results/reproduction/audits/agentprm.json")
    cb = read_json("results/reproduction/audits/contractbench.json")
    aabs = read_json("results/agentabstain_prospective_v2.json")
    summary = build_summary(ROOT)

    agent_metric = contract(agent, "APRM-METRIC-01")
    cb_metric = contract(cb, "CBEN-METRIC-01")
    cb_log = contract(cb, "CBEN-LOG-01")
    macro = contract(aabs, "AABS-MACRO-01")
    tool = summary["toolace"]
    mapping = summary["contractbench"]

    rows = [
        (
            "AgentPRM",
            "equation / value",
            [
                ("terminal-relative\ndiscount equation", "neutral"),
                ("implementation uses a\nforward exponent", "discrepant"),
                (
                    f"mean {agent_metric['observed']['mean']:.5f} →\n"
                    f"{agent_metric['expected']['mean']:.5f}",
                    "discrepant",
                ),
                ("released training\ntraces unavailable", "unavailable"),
                (
                    "controlled values change;\neffect on reported results\n"
                    "cannot be determined",
                    "discrepant",
                ),
            ],
        ),
        (
            "ContractBench",
            "logging / aggregate",
            [
                (
                    f"{mapping['reported_model']}\nn={mapping['reported_episodes']}; "
                    f"{mapping['reported_success_rate_percent']:.1f}%",
                    "neutral",
                ),
                (
                    f"released artifact\nn={cb_metric['observed']['episodes']}; "
                    f"{cb_metric['observed']['successes']}/"
                    f"{cb_metric['observed']['episodes']}",
                    "discrepant",
                ),
                (
                    "model name and n=99 match;\ncheckpoint, task set, and\n"
                    "configuration not verified",
                    "convention sensitivity",
                ),
                (
                    f"{cb_log['observed']['episodes']} released traces;\n"
                    f"{cb_log['observed']['output_without_assistant_message']} report "
                    "output tokens but\nhave no terminal response",
                    "unavailable",
                ),
                (
                    "0/99 released successes vs\n56.6% reported under the\n"
                    "repository mapping;\nhistorical performance\n"
                    "cannot be reconstructed",
                    "discrepant",
                ),
            ],
        ),
        (
            "ToolACE",
            "coordinate / support",
            [
                ("post-crop mask\ncoordinates", "neutral"),
                ("implementation retains\n0 supervised tokens", "discrepant"),
                ("corrected coordinates\nretain 4", "discrepant"),
                (
                    f"{tool['parsed_targets']:,}/{tool['candidate_targets']:,} parsed;\n"
                    f"0 exceed {tool['configured_max_length']:,};\n"
                    f"maximum observed: {tool['maximum_sequence_tokens']:,}",
                    "inactive",
                ),
                ("defect not activated among\nparsed released targets", "inactive"),
            ],
        ),
        (
            "AgentAbstain",
            "metric / aggregation",
            [
                ("macro over 8 scenarios", "convention sensitivity"),
                (
                    f"equal subtype weighting\nproduces "
                    f"{macro['observed']['headline_macro']:.3f}",
                    "convention sensitivity",
                ),
                (
                    f"pair-weighted alternative\nproduces "
                    f"{macro['expected']['headline_macro']:.3f}",
                    "convention sensitivity",
                ),
                ("released model\noutputs unavailable", "unavailable"),
                (
                    "within-scenario weighting\nis not specified;\nno discrepancy",
                    "convention sensitivity",
                ),
            ],
        ),
    ]

    fig, ax = plt.subplots(figsize=(7.20, 3.25))
    fig.subplots_adjust(left=0.008, right=0.992, top=0.995, bottom=0.005)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.10, 0.99)
    ax.axis("off")
    left = 0.148
    widths = [0.145, 0.148, 0.165, 0.168, 0.179]
    gap = 0.006
    headers = [
        "Scientific\nstatement",
        "Observed implementation\nor released artifact",
        "Comparison",
        "Available\nevidence",
        "What the evidence\nshows",
    ]
    x = left
    for width, header in zip(widths, headers):
        ax.text(
            x + width / 2,
            0.943,
            header,
            ha="center",
            va="center",
            fontsize=6.7,
            fontweight="bold",
            color=INK,
            linespacing=1.05,
        )
        x += width + gap
    row_h, row_gap, start_y = 0.181, 0.012, 0.697
    for index, (case, contract_type, cells) in enumerate(rows):
        y = start_y - index * (row_h + row_gap)
        ax.text(left - 0.012, y + row_h * 0.62, case, ha="right", va="center", fontsize=8.1, fontweight="bold")
        ax.text(left - 0.012, y + row_h * 0.30, contract_type, ha="right", va="center", fontsize=6.5)
        x = left
        for column, (text, status) in enumerate(cells):
            size = 5.55 if case == "ContractBench" else 6.05
            if column == 4 and case in {"AgentPRM", "ContractBench"}:
                size = 5.75
            card(ax, x, y, widths[column], row_h, text, status, size)
            x += widths[column] + gap
    save(fig, output, "evidence_chain_v2", 0.025)


def empirical_summary(output: Path) -> None:
    summary = build_summary(ROOT)
    repositories = ["AgentPRM", "ContractBench", "ToolACE", "tau-bench", "AgentAbstain"]
    counts = summary["contracts"]["by_repository"]
    outcome_order = ["conformant", "discrepant", "inactive", "convention sensitivity"]
    mutation = summary["mutations"]
    study = mutation["study_specific"]["by_class"]
    mutation_rows = [
        ("Equation", *study["equation"].values()),
        ("Schema / coordinate", *study["schema"].values()),
        ("Logging / serialization", *study["logging"].values()),
        ("Metric / aggregation", *study["metric"].values()),
        (
            "External — first run",
            mutation["external_first_execution"]["detected"],
            mutation["external_first_execution"]["total"],
        ),
        (
            "External — after fixture\ncorrection",
            mutation["external_after_fixture_correction"]["detected"],
            mutation["external_after_fixture_correction"]["total"],
        ),
    ]

    fig = plt.figure(figsize=(7.20, 3.15))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.24, 1.0], left=0.125, right=0.975, top=0.875, bottom=0.225, wspace=0.62)
    ax_a, ax_b = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    y_positions = [4, 3, 2, 1, 0]
    lefts = [0] * 5
    for outcome in outcome_order:
        values = [counts[repository].get(outcome, 0) for repository in repositories]
        style = OUTCOMES[outcome]
        bars = ax_a.barh(y_positions, values, left=lefts, height=0.62, color=style["fill"], edgecolor=style["edge"], linewidth=0.72, hatch=style["hatch"], zorder=3)
        for bar, value in zip(bars, values):
            if value:
                ax_a.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, str(value), ha="center", va="center", fontsize=7.2, fontweight="bold")
        lefts = [left + value for left, value in zip(lefts, values)]
    ax_a.set_title("(a) Outcomes by repository", loc="left", fontsize=8.8, fontweight="bold", pad=8)
    ax_a.set_xlim(0, 5)
    ax_a.set_ylim(-0.62, 4.62)
    ax_a.set_xticks(range(6))
    ax_a.set_xlabel("Number of checks", fontsize=7.5)
    ax_a.set_yticks(y_positions, repositories)
    ax_a.tick_params(axis="both", labelsize=7.3, length=2.8, width=0.65)
    ax_a.grid(axis="x", color=GRID, linewidth=0.55, zorder=0)
    ax_a.spines[["top", "right"]].set_visible(False)
    ax_a.text(-0.08, -0.31, "prospective", ha="right", va="top", fontsize=6.2, fontstyle="italic", clip_on=False)
    handles = [Patch(facecolor=OUTCOMES[name]["fill"], edgecolor=OUTCOMES[name]["edge"], hatch=OUTCOMES[name]["hatch"], label=name.title()) for name in outcome_order]
    ax_a.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.005, -0.27), ncol=2, frameon=False, fontsize=6.4, handlelength=1.25, columnspacing=1.15)

    mutation_y = [5.0, 4.0, 3.0, 2.0, 0.65, -0.35]
    for (label, detected, total), y in zip(mutation_rows, mutation_y):
        fraction = detected / total
        first = label == "External — first run"
        edge = OUTCOMES["discrepant"]["edge"] if first else OUTCOMES["conformant"]["edge"]
        ax_b.barh(y, fraction, height=0.52, color=PAPER if first else OUTCOMES["conformant"]["fill"], edgecolor=edge, linewidth=0.82, hatch="///" if first else "", zorder=3)
        ax_b.plot(fraction, y, marker="o", markersize=4.3, markerfacecolor=PAPER if first else edge, markeredgecolor=edge, markeredgewidth=0.9, zorder=4, clip_on=False)
        ax_b.text(1.035, y, f"{detected}/{total}", ha="left", va="center", fontsize=7.2, fontweight="bold", clip_on=False)
    ax_b.axhline(1.33, color=GRID, linewidth=0.65)
    ax_b.set_title("(b) Mutation sensitivity", loc="left", fontsize=8.8, fontweight="bold", pad=8)
    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(-0.85, 5.52)
    ax_b.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.50", "0.75", "1"])
    ax_b.set_xlabel("Proportion detected", fontsize=7.5)
    ax_b.set_yticks(mutation_y, [row[0] for row in mutation_rows])
    ax_b.tick_params(axis="both", labelsize=7.0, length=2.8, width=0.65)
    ax_b.grid(axis="x", color=GRID, linewidth=0.55, zorder=0)
    ax_b.spines[["top", "right"]].set_visible(False)
    save(fig, output, "empirical_summary_v1")


def _normal_outcome(value: str) -> str:
    value = value.replace("_", " ").lower()
    return "convention sensitivity" if value == "alternative convention" else value


def _contract_class(value: str) -> str:
    low = value.lower()
    if "equation" in low:
        return "equation"
    if "schema" in low or "coordinate" in low or "pair integrity" in low:
        return "schema"
    if "logging" in low or "serialization" in low:
        return "logging"
    if "metric" in low or "aggregation" in low or "conjunction" in low:
        return "metric"
    return "support / control"


def landscape(output: Path) -> None:
    rows = read_inventory()
    summary = build_summary(ROOT)
    totals = summary["contracts"]["overall"]["outcomes"]
    repositories = ["AgentPRM", "ContractBench", "ToolACE", "tau-bench", "AgentAbstain"]
    outcome_order = ["conformant", "discrepant", "inactive", "convention sensitivity"]
    markers = {"equation": "o", "schema": "s", "logging": "^", "metric": "D", "support / control": "P"}
    elements: list[tuple[str, object, float]] = []
    y = 0.0
    elements.append(("section", "Original cases · 20 checks", y))
    y += 0.72
    for index, repository in enumerate(repositories):
        if repository == "AgentAbstain":
            y += 0.20
            elements.append(("section", "AgentAbstain prospective case · 5 checks", y))
            y += 0.72
        elements.append(("repo", repository, y))
        y += 0.52
        for row in [item for item in rows if item["repository"] == repository]:
            elements.append(("row", row, y))
            y += 0.61
        if index < 4:
            y += 0.22

    fig, ax = plt.subplots(figsize=(7.35, 8.05))
    fig.subplots_adjust(left=0.04, right=0.985, top=0.97, bottom=0.12)
    ax.set_xlim(-3.38, 4.75)
    ax.set_ylim(-0.65, y + 0.10)
    ax.invert_yaxis()
    ax.axis("off")
    for index, outcome in enumerate(outcome_order):
        style = OUTCOMES[outcome]
        ax.add_patch(Rectangle((index - 0.41, -0.08), 0.82, y + 0.08, facecolor=style["edge"], edgecolor="none", alpha=0.055 if index % 2 == 0 else 0.032))
        label = outcome.title().replace("Convention Sensitivity", "Convention\nsensitivity")
        ax.text(index, -0.46, f"{label}\n({totals[outcome]})", ha="center", va="bottom", fontsize=7.4, fontweight="bold", color=style["edge"])
    ax.text(-2.98, -0.38, "Repository / check", fontsize=7.4, fontweight="bold", va="bottom")
    ax.text(4.18, -0.38, "Largest observed\nchange", fontsize=7.4, fontweight="bold", ha="center", va="bottom")
    for kind, payload, y_position in elements:
        if kind == "section":
            prospective = str(payload).startswith("AgentAbstain")
            ax.add_patch(Rectangle((-3.32, y_position - 0.31), 8.00, 0.49, facecolor="#E7F1F8" if prospective else "#ECEFF2", edgecolor="none"))
            ax.text(-3.18, y_position - 0.05, str(payload), fontsize=8.2, fontweight="bold", color=OUTCOMES["conformant"]["edge"] if prospective else INK, va="center")
        elif kind == "repo":
            ax.text(-3.16, y_position, str(payload), fontsize=8.4, fontweight="bold", va="center")
            ax.plot([-1.87, 4.62], [y_position, y_position], color=GRID, lw=0.75)
        else:
            row = payload
            assert isinstance(row, dict)
            outcome = _normal_outcome(row["outcome"])
            klass = _contract_class(row["invariant_class"])
            ax.text(-3.02, y_position, row["contract_id"], fontsize=7.2, va="center")
            ax.scatter([outcome_order.index(outcome)], [y_position], s=43, marker=markers[klass], facecolor=OUTCOMES[outcome]["edge"], edgecolor=INK, linewidth=0.6, zorder=3)
            match = re.search(r"\(([1-4])\)", row["demonstrated_consequence"])
            if match and match.group(1) in {"1", "2"}:
                ax.text(4.18, y_position, {"1": "value", "2": "metric"}[match.group(1)], ha="center", va="center", fontsize=7.2, fontweight="bold")
    handles = [Line2D([0], [0], marker=marker, linestyle="none", markerfacecolor="white", markeredgecolor=INK, markersize=6, label=label) for label, marker in markers.items()]
    ax.legend(handles=handles, title="Contract class (shape)", ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.52, -0.015), columnspacing=1.4, fontsize=7.1, title_fontsize=7.4)
    save(fig, output, "contract_outcome_landscape_full_v1")


def worked_toolace(output: Path) -> None:
    summary = build_summary(ROOT)
    tool = summary["toolace"]
    audit = read_json("results/reproduction/audits/toolace.json")
    controlled = contract(audit, "TACE-COORD-01")
    implemented = controlled["observed"]["implemented_supervised"]
    corrected = controlled["expected"]["corrected_supervised"]
    fig, ax = plt.subplots(figsize=(7.20, 2.752))
    fig.subplots_adjust(left=0.008, right=0.992, top=0.99, bottom=0.02)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.045, 0.905)
    ax.axis("off")
    panel_y, panel_h = 0.075, 0.785
    for x in (0.0, 0.515):
        ax.add_patch(FancyBboxPatch((x, panel_y), 0.485, panel_h, boxstyle="round,pad=0.003,rounding_size=0.009", facecolor="#F4F5F6", edgecolor=OUTLINE, linewidth=0.9))
    ax.text(0.022, 0.815, "Controlled check", ha="left", va="top", fontsize=8.4, fontweight="bold")
    ax.plot([0.018, 0.467], [0.775, 0.775], color=GRID, lw=0.7)
    ax.text(0.025, 0.720, "Scientific expectation", ha="left", va="top", fontsize=6.7, fontweight="bold")
    ax.text(0.025, 0.670, "Response tokens that remain after a left crop\nshould remain supervised.", ha="left", va="top", fontsize=6.8)
    ax.text(0.025, 0.588, "Code", ha="left", va="top", fontsize=6.7, fontweight="bold")
    ax.text(0.090, 0.588, "sft_dataset_wo_apply_template.py:170, 184", ha="left", va="top", fontsize=6.55)
    fixture = controlled["observed"]["fixture"]
    ax.text(0.025, 0.505, "Fixture", ha="left", va="top", fontsize=6.7, fontweight="bold")
    ax.text(0.135, 0.505, f"prompt = {fixture[0]}     response = {fixture[1]}     max_length = {fixture[2]}", ha="left", va="top", fontsize=6.55)
    ax.text(0.135, 0.365, "Implemented", ha="center", va="bottom", fontsize=7.0, fontweight="bold")
    ax.text(0.355, 0.365, "Corrected", ha="center", va="bottom", fontsize=7.0, fontweight="bold")
    ax.text(0.135, 0.245, str(implemented), ha="center", va="center", fontsize=22, fontweight="bold")
    ax.text(0.245, 0.245, "→", ha="center", va="center", fontsize=18, fontweight="bold")
    ax.text(0.355, 0.245, str(corrected), ha="center", va="center", fontsize=22, fontweight="bold")
    ax.text(0.245, 0.140, "supervised response tokens", ha="center", va="center", fontsize=7.0)
    ax.text(0.537, 0.815, "Released targets", ha="left", va="top", fontsize=8.4, fontweight="bold")
    ax.plot([0.533, 0.982], [0.775, 0.775], color=GRID, lw=0.7)
    ax.text(0.540, 0.710, "Parsed", ha="left", va="center", fontsize=6.8, fontweight="bold")
    ax.text(0.635, 0.710, f"{tool['parsed_targets']:,} / {tool['candidate_targets']:,} ({100*tool['parsing_fraction']:.1f}%)", ha="left", va="center", fontsize=8.2, fontweight="bold")
    start, end, y = 0.565, 0.950, 0.455
    observed_x = start + tool["maximum_sequence_tokens"] / tool["configured_max_length"] * (end - start)
    ax.plot([start, end], [y, y], color="#D1D4D8", lw=7.0, solid_capstyle="butt")
    ax.plot([start, observed_x], [y, y], color=OUTCOMES["conformant"]["edge"], lw=7.0, solid_capstyle="butt")
    ax.plot([observed_x, observed_x], [y - 0.035, y + 0.035], color=INK, lw=1.1)
    ax.plot([end, end], [y - 0.045, y + 0.045], color=INK, lw=1.1)
    ax.text(observed_x, 0.555, f"maximum observed sequence\n{tool['maximum_sequence_tokens']:,}", ha="center", va="center", fontsize=6.5, fontweight="bold")
    ax.text(end, 0.555, f"official limit\n{tool['configured_max_length']:,}", ha="center", va="center", fontsize=6.5, fontweight="bold")
    ax.text(0.757, 0.345, f"Overlength among parsed targets: {tool['overlength_targets']} / {tool['parsed_targets']:,}", ha="center", va="center", fontsize=7.0)
    ax.add_patch(FancyBboxPatch((0.545, 0.125), 0.425, 0.120, boxstyle="round,pad=0.003,rounding_size=0.007", facecolor=OUTCOMES["inactive"]["fill"], edgecolor=OUTCOMES["inactive"]["edge"], linewidth=0.8))
    ax.text(0.757, 0.185, "The defect was not activated among\nthe parsed released targets.", ha="center", va="center", fontsize=7.0, fontweight="bold")
    save(fig, output, "worked_contract_example_v1")


def runtime_figure(output: Path) -> None:
    summary = build_summary(ROOT)["post_authoring_runtime"]
    order = ["agentprm", "contractbench", "toolace", "taubench", "agentabstain_prospective"]
    labels = ["AgentPRM", "ContractBench", "ToolACE", "tau-bench", "AgentAbstain"]
    rows = [summary[target] for target in order]
    fig, ax = plt.subplots(figsize=(7.20, 2.35))
    fig.subplots_adjust(left=0.20, right=0.965, top=0.79, bottom=0.23)
    y = list(reversed(range(5)))
    medians = [row["median_seconds"] for row in rows]
    lower = [m - row["p25_seconds"] for m, row in zip(medians, rows)]
    upper = [row["p75_seconds"] - m for m, row in zip(medians, rows)]
    ax.errorbar(medians, y, xerr=[lower, upper], fmt="o", color=OUTCOMES["conformant"]["edge"], ecolor=OUTCOMES["conformant"]["edge"], capsize=3, markersize=4.5, linewidth=1.2)
    for x, yy in zip(medians, y):
        ax.text(x + 0.012, yy, f"{x:.3f} s", ha="left", va="center", fontsize=7.0)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 0.45)
    ax.set_xlabel("Seconds per complete invocation")
    ax.set_title("Execution time after contract authoring", loc="left", fontweight="bold", fontsize=10.2, pad=10)
    ax.grid(axis="x", color=GRID, linewidth=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7.3)
    save(fig, output, "execution_time_after_authoring_v1")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/figures")
    args = parser.parse_args()
    evidence_chain(args.output)
    empirical_summary(args.output)
    landscape(args.output)
    worked_toolace(args.output)
    runtime_figure(args.output)
    print(f"Wrote five evidence-derived PDF figures to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
