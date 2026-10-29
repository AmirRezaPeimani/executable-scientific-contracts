# Executable Scientific Contracts for Verifiable LLM and Agent Research

Executable scientific contracts connect a paper claim, its implementation, a controlled input, activation in released data, and evidence about a reported result. This repository contains the analysis, results, figure sources, and executable example accompanying the final manuscript. Read the [camera-ready paper](manuscript/camera_ready.pdf), [supplement](manuscript/supplement.pdf), and [LaTeX source](manuscript/main.tex).

## Results

We examine 25 checks across AgentPRM, ContractBench, ToolACE, $\tau$-bench, and AgentAbstain. The inventory includes substantive checks, consistency checks, and summaries. The first four cases revisit findings already known when their 20 checks were constructed; the five AgentAbstain checks were specified before execution.

The experiment contains 60 cases: 17 faulty variants, comprising three original faults and 14 injected variants; three repairs; 12 harmless rewrites; eight conformant original cases; and 20 cases with missing evidence or ineligible inputs. A shared check returns ‘not evaluated’ when required evidence is missing or no eligible examples are available, before the compared methods run. We reran the same 60 cases after correcting the eligibility checks and prediction-mask accounting.

The contracts detected all 17 faulty variants, accepted all three repairs, and raised no flags on 12 harmless rewrites. The original adapters combine source inspection with locally reconstructed calculations; that comparison covers AgentPRM and ToolACE. Relevant native test coverage was unavailable for the selected behaviors. The eight conformant original cases are retained in the saved results and supplementary results table.

After freezing the corrected checks, we constructed nine additional cases covering terminal-reward remapping, response-token order, and replacement of the outer scenario mean with a median. Each family contains a fault, a reference implementation, and a harmless rewrite. The checks and their imports remained unchanged. These nine cases are reported separately.

### AgentPRM

For a successful four-action trajectory at the released $\gamma=0.95$, correcting the callers changes the signed targets from $(1,0.95,0.9025,0.857375)$ to $(0.857375,0.9025,0.95,1)$. The mean remains 0.92746875, but the target assigned to each state changes.

Reversal preserves the multiset over complete trajectory visits. Averaging deduplicated state-action targets can give different means before and after repair. The absolute-target exporter applies $0.5(Q+1)$; preference construction uses the signed values. For two successful continuations from a shared state, with lengths four and eight, the signed margin changes from zero to $0.95^3-0.95^7$, exceeding the released preference threshold $\delta=0.03$. A preference is added for each of the 28 chosen length pairs.

The rollout also copies observations into the action field of the query history. A test agent records six incorrect history entries before repair and none afterward. The separately saved trajectory contains the executed actions in both versions. Historical training traces were unavailable.

### ToolACE

The official conversion produces 9,964 examples. Running the loader with the recorded tokenizer finds no sequence above the 8,192-token limit; the maximum length is 3,945. The crop defect is therefore inactive on these converted examples.

The earlier custom conversion retained 9,412 of 10,092 candidate targets (93.3%). The official conversion produces 9,964 examples: 9,394 with nonempty responses and 570 with empty responses supervised on EOS only. Conversation-schema validation excludes 128 candidates. The official and custom converters disagree on 18 retained prompt/response rows; the conversion comparison records these differences.

### ContractBench and AgentAbstain

ContractBench reports 56.6% for Qwen3.5-9B on 99 tasks. The released artifact contains 0/99 success labels under the repository mapping. Three traces report 8,192 output tokens but contain no terminal assistant response. The model name and evaluation size match the report; the original checkpoint, task set, and evaluation configuration remain unverified. The successful-trace check has zero eligible examples and is marked ‘not evaluated’.

AgentAbstain’s released task file contains 526 rows forming 263 complete act/abstain pairs across eight scenarios. Pair integrity, paired-accuracy conjunction, and commit scoring match the source. Two choices remain unspecified: the treatment of an empty conditioned denominator and the weighting of subtypes within a scenario. The released subtype counts permit a maximum absolute difference of 2.412476 percentage points in the overall eight-scenario macro average. Individual scenario bounds range from zero to 5.882353 points. Constructed endpoint outcomes attain the overall bound through the unchanged ranking computation. Released model outputs were unavailable.

### Historical classifications and fixture mutations

Original classifications across the 25 checks were 15 conformant, seven discrepant, one inactive, and two convention sensitivities. These classifications are historical records, not estimates of defect prevalence.

The four fixture validators detected all 30 selected mutations: eight equation changes, eight schema changes, eight logging changes, and six aggregation changes. The set includes duplicate temporal transformations. The separate eight-operator experiment detected seven mutations on its first execution. Its relational mutation survived because the input contained no equality case. Adding that case increased detection to eight of eight with the check unchanged.

## Reproduction

Python 3.10 or newer is required. Install the declared dependencies:

```bash
python -m pip install -e '.[camera-ready,figures]'
```

Verify the saved implementation results without executing upstream code:

```bash
conform behavior-check --results-root study/camera_ready/results --output /new/path/implementation_summary.json
```

Regenerate the six manuscript figures from the recorded results:

```bash
python scripts/make_final_figures.py --results-root study/camera_ready/results --output-dir /new/path/figures
```

The paper prints `scripts/agentabstain_contract_example.py` directly from the supplied source. Acquire AgentAbstain at `f581249704b26804e28a39e37396f1be00b71a4d`; the example requires Python 3.10 or newer and pandas 2.x. It constructs all four Boolean pairs, imports the actual `compute_summary_metrics`, asserts paired accuracy 0.25 and four eligible pairs, and prints the source SHA-256:

```bash
python scripts/agentabstain_contract_example.py --source-root /path/to/pinned/agentabstain
```

Pinned source/data acquisition and full implementation execution commands are in [study/camera_ready/README.md](study/camera_ready/README.md) and [UPSTREAM.md](UPSTREAM.md). `behavior-run` and `scan-toolace` execute acquired implementations; `behavior-check` verifies existing result records. Source and dataset revisions, hashes, protocols, and commands accompany the results. No model training is required.

## Repository contents

| Path | Contents |
|---|---|
| `study/camera_ready/analysis/` | Versioned implementation analyses and executable checks |
| `study/camera_ready/results/` | Current results, quantitative CSVs, and separate additional-family records |
| `study/camera_ready/protocols/` | Recorded scientific protocols and source/data pins |
| `study/camera_ready/metadata/` | Public-file hashes and original raw-artifact identities |
| `tables/` | Final manuscript tables, including the role/eligibility inventory and all 60 implementation cases |
| `manuscript/` | Matching final PDFs and complete LaTeX sources |
| `scripts/` and `figures/` | Complete executable example and sources/assets for the six final figures |
| `history/submitted/` | Historical fixture experiments, original classifications, and adapters’ result records |
| `src/conform/` | Command interfaces, original comparison adapters, and result verification |

Historical records are preserved separately and do not replace the current implementation evidence. The historical summary can be reconstructed with `conform release-check --root history/submitted`; the selected fixture mutations can be reproduced with `conform mutate --output /new/path/fixture_mutations.json`. The external operators remain in `history/submitted/scripts/run_external_mutations.py` with separate `--execution first` and `--execution corrected` modes.

## Manuscript build

The LaTeX sources use the figures in `figures/` and the executable example in `scripts/`. From the repository root, compile with TeX Live and write temporary products outside the checkout:

```bash
paper_source="$PWD/manuscript"
paper_build=/path/to/paper-build
mkdir -p "$paper_build"
cd "$paper_source"
pdflatex -output-directory="$paper_build" -jobname=camera_ready main.tex
(cd "$paper_build" && BIBINPUTS="$paper_source:" bibtex camera_ready)
pdflatex -output-directory="$paper_build" -jobname=camera_ready main.tex
pdflatex -output-directory="$paper_build" -jobname=camera_ready main.tex
pdflatex -output-directory="$paper_build" supplement.tex
pdflatex -output-directory="$paper_build" supplement.tex
```

## Limitations

The study covers five repositories and manually selected faults. The checks and outcome labels were assigned by one author. Authoring time and agreement between independent authors were not measured. The proposed review workflow has not been evaluated in use. The recorded cases do not estimate population detection accuracy or establish historical model-performance changes.

## Citation and license

Amir Reza Peimani, University of Toronto — [amir.peimani@utoronto.ca](mailto:amir.peimani@utoronto.ca).

```bibtex
@software{peimani_executable_scientific_contracts_2026,
  author  = {Peimani, Amir Reza},
  title   = {Executable Scientific Contracts for Verifiable LLM and Agent Research},
  year    = {2026},
  version = {1.1.1},
  url     = {https://github.com/AmirRezaPeimani/executable-scientific-contracts}
}
```

The code is MIT licensed. Acquired upstream repositories and datasets retain their own licenses; raw upstream datasets, tokenizer assets, and model weights are not bundled. See [CITATION.cff](CITATION.cff) and the recorded upstream notices.
