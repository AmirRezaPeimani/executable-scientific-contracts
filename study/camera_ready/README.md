# Run the analyses

Commands and input requirements for the AgentPRM, ToolACE, and AgentAbstain analyses are listed below. Saved results are in `results/`; the nine additional cases are in `results/fresh_holdout_v1/`.

## Saved results and figures

```bash
conform behavior-check --results-root study/camera_ready/results --output /new/path/implementation_summary.json
python scripts/make_final_figures.py --results-root study/camera_ready/results --output-dir /new/path/figures
```

`behavior-check` validates saved files against their manifests. `behavior-run` executes the acquired implementations. `STUDY_ROOT` and `ACQUISITION_ROOT` identify the study and acquired inputs. `results/behavioral_summary_v1.json` supplies the figure data; `results/remaining_evidence_v3.json` records contract roles and eligible counts.

## Complete executable example

The supplement prints `scripts/agentabstain_contract_example.py`. Use Python 3.10 or newer, pandas 2.x, and AgentAbstain revision `f581249704b26804e28a39e37396f1be00b71a4d`:

```bash
python scripts/agentabstain_contract_example.py --source-root /path/to/pinned/agentabstain
```

The check prints ‘paired_accuracy: 0.25 PASS’ and the SHA-256 hash of the imported source file.

## Run upstream implementations

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

`behavior-run --target toolace` is equivalent to `scan-toolace`. The wrapper executes the analysis, checks source availability and input eligibility, and records the commands. It copies the tokenizer files listed in `metadata/toolace_tokenizer_acquisition.json`. The historical tokenizer is optional. Acquire the inputs first and choose a new output directory.

AgentPRM runs target and preference processing and command-line exports. ToolACE runs the converter with local storage paths and the loader with tensors and tokenization; prediction masks align with next-token labels. AgentAbstain runs the evaluator and ranking calculation.

## Additional cases

Run these commands from the repository root with Python 3.10 or newer and the `camera-ready` dependencies. Inputs are three source checkouts at the listed revisions, two directories of repaired source files, and the tokenizer.

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

Existing checkouts and tokenizer files can be reused. Set `FRESH_PRIOR_ROOT` to the parent of the `agentprm` and `toolace` result directories. Required files are `agentprm/sources/repaired/` (the four AgentPRM source files) and `toolace/implementation_variants/TACE-REPAIR/verl/verl/utils/dataset/sft_dataset_wo_apply_template.py`. The runner checks their revisions and hashes.

If the repaired files are unavailable, create them with the following commands. Use a new `FRESH_PRIOR_ROOT` directory:

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

Prepare the nine cases in a new directory outside the repository, then run them:

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

The runner loads evaluators, protocols, and the original AgentPRM adapter from this repository. `--workspace` sets the output directory. `prepare` records input paths, revisions, hashes, and expected values. `run` uses the recorded inputs and checks their hashes before execution.

Outputs are `$FRESH_OUTPUT_ROOT/protocols/fresh_holdout_v1.json` and `$FRESH_OUTPUT_ROOT/results/fresh_holdout_v1/{challenge_results_v1.json,REPORT.md,raw/}`. Results include the output demonstrating each fault and a comparison with the published summary.

`metadata/fresh_evidence_manifest.json` lists the files and hashes for the additional cases. Evaluated-case counts are in `results/fresh_holdout_v1/INTERPRETATION.md`.

## Historical records and licenses

The original fixture experiments and classifications are under `../../history/submitted/`. The supplement describes the custom and official ToolACE conversions. Original and repeated executions have separate result files. Acquire upstream data and tokenizer files using [UPSTREAM.md](../../UPSTREAM.md).

Upstream code retains its MIT or Apache-2.0 license; its source notices accompany separately acquired repositories. The AgentAbstain dataset is CC BY 4.0 and is not redistributed. The included original support scanner is MIT-licensed analysis code. Recorded source hashes and metadata identify the exact computations.
