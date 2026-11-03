# Executable Scientific Contracts for Verifiable LLM and Agent Research

This repository contains the code, results, figures, and executable example for the [paper](manuscript/camera_ready.pdf). The [supplement](manuscript/supplement.pdf) contains the complete results and contract inventory; [main.tex](manuscript/main.tex) contains the paper source.

## Results

The study examines 25 checks across AgentPRM, ContractBench, ToolACE, τ-bench, and AgentAbstain. Implementation tests detected all 17 faulty variants, accepted all three repairs, and raised no flags on 12 harmless rewrites.

- AgentPRM: correcting temporal discounting changes state targets and preferences while preserving the mean over complete trajectory visits.
- ToolACE: the crop defect is inactive across 9,964 converted examples. The longest sequence contains 3,945 tokens.
- AgentAbstain: the released task composition permits a maximum weighting difference of 2.412476 percentage points.

The [paper](manuscript/camera_ready.pdf) presents the cases. The [supplement](manuscript/supplement.pdf) contains the complete inventory, implementation results, and mutation experiments.

## Reproduction

Python 3.10 or newer is required. Install the declared dependencies:

```bash
python -m pip install -e '.[camera-ready,figures]'
```

Verify the saved implementation results:

```bash
conform behavior-check --results-root study/camera_ready/results --output /new/path/implementation_summary.json
```

Regenerate the six manuscript figures from the recorded results:

```bash
python scripts/make_final_figures.py --results-root study/camera_ready/results --output-dir /new/path/figures
```

The supplement prints `scripts/agentabstain_contract_example.py` directly from the supplied source. Acquire AgentAbstain at `f581249704b26804e28a39e37396f1be00b71a4d`; the example requires Python 3.10 or newer and pandas 2.x. It constructs all four Boolean pairs, imports the actual `compute_summary_metrics`, asserts paired accuracy 0.25 and four eligible pairs, and prints the source SHA-256:

```bash
python scripts/agentabstain_contract_example.py --source-root /path/to/pinned/agentabstain
```

Pinned source/data acquisition and full implementation execution commands are in [study/camera_ready/README.md](study/camera_ready/README.md) and [UPSTREAM.md](UPSTREAM.md). `behavior-run` and `scan-toolace` execute acquired implementations; `behavior-check` verifies existing result records. Source and dataset revisions, hashes, protocols, and commands accompany the results. No model training is required.

## Repository contents

| Path | Contents |
|---|---|
| `study/camera_ready/analysis/` | Versioned implementation analyses and executable checks |
| `study/camera_ready/results/` | Implementation results, quantitative CSVs, and additional cases |
| `study/camera_ready/protocols/` | Experimental protocols and source/data revisions |
| `study/camera_ready/metadata/` | Public-file hashes and original raw-artifact identities |
| `tables/` | Manuscript tables and the complete contract inventory |
| `manuscript/` | Matching final PDFs and complete LaTeX sources |
| `scripts/` and `figures/` | Complete executable example and sources/assets for the six final figures |
| `history/submitted/` | Historical fixture experiments, original classifications, and adapters’ result records |
| `src/conform/` | Command interfaces, original comparison adapters, and result verification |

The historical summary can be reconstructed with `conform release-check --root history/submitted`; the selected fixture mutations can be reproduced with `conform mutate --output /new/path/fixture_mutations.json`. The external operators remain in `history/submitted/scripts/run_external_mutations.py` with separate `--execution first` and `--execution corrected` modes.

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

## Citation and license

Amir Reza Peimani, University of Toronto, [amir.peimani@utoronto.ca](mailto:amir.peimani@utoronto.ca).

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
