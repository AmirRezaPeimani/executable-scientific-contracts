# Recorded analyses and implementation checks

The three current implementation records are `results/agentprm/challenge_results_v2.json`, `results/toolace/challenge_results_verified_v2.json`, and `results/agentabstain/challenge_results_v2.json`. They contain 60 cases: 17 faulty variants, three repairs, 12 harmless rewrites, eight conformant original cases, and 20 missing-evidence or ineligible cases. The nine additional cases in `results/fresh_holdout_v1/` are separate.

The checks execute pinned upstream functions or unchanged function extractions whose hashes are recorded. The original adapters combine source inspection with locally reconstructed calculations. Relevant native test coverage was unavailable for the selected behaviors. The checks and outcome labels were assigned by one author. The corrected 60 cases were rerun after eligibility and prediction-mask accounting corrections; these reused cases do not provide independent held-out validation. The additional source families were constructed after the corrected checks were frozen, with their imports unchanged.

## Saved results and figures

```bash
conform behavior-check --results-root study/camera_ready/results --output /new/path/implementation_summary.json
python scripts/make_final_figures.py --results-root study/camera_ready/results --output-dir /new/path/figures
```

Saved-result verification does not execute upstream code. Each target manifest distinguishes normalized public-file hashes from the preserved raw artifact identities. `STUDY_ROOT` and `ACQUISITION_ROOT` are acquisition namespaces, not private host paths. Verification uses the supplied results directory. `results/behavioral_summary_v1.json` collects the current implementation records for figure reproduction; `results/remaining_evidence_v3.json` records roles and eligible data for the retained case checks.

## Complete executable example

The main-text listing is `scripts/agentabstain_contract_example.py`; its copy in `analysis/` is identical. Use Python 3.10 or newer, pandas 2.x, and AgentAbstain revision `f581249704b26804e28a39e37396f1be00b71a4d`:

```bash
python scripts/agentabstain_contract_example.py --source-root /path/to/pinned/agentabstain
```

The check prints ‘paired_accuracy: 0.25 PASS’ and the SHA-256 hash of the imported source file.

## Acquired implementation execution

Install the declared optional dependencies and acquire clean checkouts at the revisions recorded in `../../UPSTREAM.md` and the protocols:

```bash
python -m pip install -e '.[camera-ready,figures]'
conform behavior-run --target agentprm \
  --source-root /acquired/agent_prm \
  --protocol study/camera_ready/protocols/agentprm_v1.json \
  --output-dir /new/results/agentprm

conform behavior-run --target agentabstain \
  --source-root /acquired/agentabstain \
  --dataset-jsonl /acquired/agentabstain/tasks.jsonl \
  --protocol study/camera_ready/protocols/agentabstain_v1.json \
  --output-dir /new/results/agentabstain

conform scan-toolace \
  --source-root /acquired/tool-rl-box \
  --data /acquired/tool-rl-box/data/ToolACE/data.json \
  --tokenizer /acquired/Qwen2.5-1.5B-Instruct-tokenizer \
  --historical-tokenizer /acquired/Qwen2.5-0.5B-Instruct-tokenizer \
  --protocol study/camera_ready/protocols/toolace_v1.json \
  --output-dir /new/results/toolace
```

`behavior-run --target toolace` is equivalent to `scan-toolace`. The wrapper runs the base analysis and corrected availability/eligibility checks, records their exact commands, and preserves separate protocol path/first-artifact identity bindings. It stages only the small tokenizer assets matching `metadata/toolace_tokenizer_acquisition.json`; no model weights are required. The historical tokenizer is optional. Outputs must use a new directory. No acquisition or network request occurs implicitly.

AgentPRM executes target/preference processing and command-line exports. ToolACE executes the complete converter with storage redirection and the unchanged loader method with real tensors/tokenization; prediction masks are aligned with next-token labels. AgentAbstain executes the evaluator and the identified final aggregation computation. This establishes implementation behavior on the recorded inputs, not historical model-performance effects.

## Additional cases

Run the following commands from the repository root. Python 3.10 or newer and the `camera-ready` dependencies are required. The nine recorded cases use controlled inputs, three pinned source checkouts, two prior repaired-source directories, and a tokenizer. They do not require model weights, training, raw datasets, or another full ToolACE scan.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[camera-ready]'

FRESH_ACQUISITION_ROOT=/absolute/path/to/acquired
FRESH_PRIOR_ROOT=/absolute/path/to/prior-results
FRESH_OUTPUT_ROOT=/absolute/path/to/new-nine-case-reproduction
FRESH_TOKENIZER_ROOT="$FRESH_ACQUISITION_ROOT/qwen2.5-1.5b-tokenizer"
mkdir -p "$FRESH_ACQUISITION_ROOT" "$FRESH_PRIOR_ROOT"

git clone https://github.com/sanjibanc/agent_prm.git "$FRESH_ACQUISITION_ROOT/agent_prm"
git -C "$FRESH_ACQUISITION_ROOT/agent_prm" checkout e4714717f7f4bd4671848670c4ed54d0169f603a
git clone https://github.com/hypasd-art/Tool-RL-Box.git "$FRESH_ACQUISITION_ROOT/tool-rl-box"
git -C "$FRESH_ACQUISITION_ROOT/tool-rl-box" checkout 1156d649235235e686372956e99bfc50e4b1e3f6
git clone https://github.com/AntiQuality/agentabstain.git "$FRESH_ACQUISITION_ROOT/agentabstain"
git -C "$FRESH_ACQUISITION_ROOT/agentabstain" checkout f581249704b26804e28a39e37396f1be00b71a4d

python - "$FRESH_TOKENIZER_ROOT" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen2.5-1.5B-Instruct",
    revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    local_dir=sys.argv[1],
    allow_patterns=["tokenizer.json", "tokenizer_config.json", "vocab.json",
                    "merges.txt", "config.json", "generation_config.json",
                    "special_tokens_map.json", "added_tokens.json", "LICENSE"],
)
PY
```

Reuse existing clean checkouts and the recorded tokenizer rather than downloading them again. The runner checks acquired revisions and relevant file hashes. Set `FRESH_PRIOR_ROOT` to the parent of existing `agentprm` and `toolace` result directories. The required prior files are `agentprm/sources/repaired/` (the four recorded AgentPRM source files) and `toolace/implementation_variants/TACE-REPAIR/verl/verl/utils/dataset/sft_dataset_wo_apply_template.py`. Their contents must match the published repaired-source hashes.

If those prior runs are unavailable, the following optional source-only preparation reconstructs the identical repaired copies with the bundled helpers. It executes no cases or scans, creates no measured result, and refuses to overwrite existing copies:

```bash
python - "$FRESH_ACQUISITION_ROOT" "$FRESH_PRIOR_ROOT" <<'PY'
import json
import sys
from pathlib import Path
analysis = Path("study/camera_ready/analysis").resolve()
sys.path.insert(0, str(analysis))
from agentprm_analysis import FILES, variant
from toolace_analysis import LOADER, repaired_source
acquired, prior = map(Path, sys.argv[1:])
aprm = prior / "agentprm/sources/repaired"
toolace = prior / "toolace/implementation_variants/TACE-REPAIR" / LOADER
if aprm.exists() or toolace.exists():
    raise FileExistsError("Reuse existing repaired copies or choose a new prior directory")
spec = json.loads(Path("study/camera_ready/protocols/agentprm_v1.json").read_text())
repair = spec["variants"]["repaired"]
texts = {name: (acquired / "agent_prm" / name).read_text() for name in FILES}
variant(aprm.parent, texts, "repaired", repair["temporal_argument"],
        repair["history_expression"], repair.get("benign", ""))
toolace.parent.mkdir(parents=True)
toolace.write_text(repaired_source((acquired / "tool-rl-box" / LOADER).read_text()))
PY
```

Prepare the nine unchanged case definitions in a **new external output directory**, then execute them:

```bash
python study/camera_ready/analysis/fresh_holdout_v1.py --phase prepare \
  --workspace "$FRESH_OUTPUT_ROOT" \
  --agentprm-source-root "$FRESH_ACQUISITION_ROOT/agent_prm" \
  --toolace-source-root "$FRESH_ACQUISITION_ROOT/tool-rl-box" \
  --agentabstain-source-root "$FRESH_ACQUISITION_ROOT/agentabstain" \
  --agentprm-results "$FRESH_PRIOR_ROOT/agentprm" \
  --toolace-results "$FRESH_PRIOR_ROOT/toolace" \
  --tokenizer "$FRESH_TOKENIZER_ROOT"

python study/camera_ready/analysis/fresh_holdout_v1.py --phase run \
  --workspace "$FRESH_OUTPUT_ROOT"
```

The runner resolves evaluator scripts, scientific protocols, and the original AgentPRM adapter from this repository. `--workspace` controls outputs only; it must be outside the repository. `prepare` records all explicit acquired paths, source/fixture/tokenizer hashes, revisions, expected values, and evaluator/import hashes before execution. `run` reuses those bound paths and refuses a repeated execution. No private project layout or fixed tokenizer directory is assumed.

Outputs are `$FRESH_OUTPUT_ROOT/protocols/fresh_holdout_v1.json` and `$FRESH_OUTPUT_ROOT/results/fresh_holdout_v1/{challenge_results_v1.json,REPORT.md,raw/}`. The result records whether its summary matches the published summary; every fault has a separately saved manifestation witness. Reproduction reruns the published cases and does not create another untouched evaluation set. Bundled historical records are never overwritten.

`metadata/fresh_evidence_manifest.json` records the additional cases separately. Native coverage is unavailable for these cases. Original adapters cover two families and are unavailable for the AgentAbstain family; evaluated-case denominators are retained in `results/fresh_holdout_v1/INTERPRETATION.md`.

## Historical records and licenses

Historical fixture-validator results and original classifications are under `../../history/submitted/`. The exact historical conversion comparison is stated once in the repository README. First implementation executions remain alongside the corrected records to preserve their provenance. Original data, model weights, private acquisition files, and tokenizer assets are not distributed.

Upstream code retains its MIT or Apache-2.0 license; its source notices accompany separately acquired repositories. The AgentAbstain dataset is CC BY 4.0 and is not redistributed. The included original support scanner is MIT-licensed analysis code. Recorded source hashes and metadata identify the exact computations.
