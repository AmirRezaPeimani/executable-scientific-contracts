# From Claims to Consequences: Executable Scientific Contracts for Paper–Code Auditing

Research papers, implementations, and released artifacts can encode different
scientific objects even when each looks plausible in isolation. An executable
scientific contract turns one paper-facing expectation into a rerunnable,
evidence-bounded check.

Each contract preserves a five-link chain:

1. the scientific statement;
2. the implementation or released-artifact path;
3. a discriminating executable check;
4. activation evidence from the released support, when available; and
5. the strongest demonstrated consequence justified by that evidence.

A failed check is not automatically a paper-invalidating defect. The release
distinguishes four outcomes:

- **Conformant:** the observed behavior satisfies the contract.
- **Discrepant:** the implementation or released artifact differs from the
  stated expectation on the available check.
- **Inactive:** a source-level mismatch is present, but its triggering
  condition was not observed in the examined released support.
- **Convention sensitivity:** the source leaves a consequential choice
  unspecified, so different coherent conventions do not support a discrepancy
  claim.

## Study artifacts

The study contains five contracts for each of AgentPRM, ContractBench,
ToolACE, tau-bench, and AgentAbstain. The first four repositories form the
original 20-check set. AgentAbstain is a prospective fifth case: its five
checks were fixed before their outcomes were executed. The complete inventory
is in [`tables/contract_inventory.csv`](tables/contract_inventory.csv), and
the corresponding machine-readable evidence is under [`results/`](results/).

The included evidence reconstructs 15 conformant, 7 discrepant, 1 inactive,
and 2 convention-sensitivity outcomes across all 25 checks. These are
purposively selected check outcomes, not estimates of defect prevalence.

Contract adequacy was tested with 30 study-specific mutations: 8 equation,
8 schema/coordinate, 8 logging/serialization, and 6 metric/aggregation
mutations. All 30 were detected. A separate eight-operator set detected 7/8
on its frozen first execution and 8/8 after correcting only the fixture that
failed to instantiate its intended boundary. The two executions remain
separate in the released evidence.

## Quickstart: deterministic local validation

Python 3.10 or newer is required. After the declared PyPI dependencies have
been installed, the validation, mutation, summary, and figure commands use
only files included in this repository and do not make network calls.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test,figures]"
pytest
conform validate-manifest
conform release-check --output artifacts/paper_summary.json
conform mutate --output artifacts/study_mutations.json
python scripts/run_external_mutations.py --execution first --output artifacts/external_first.json
python scripts/run_external_mutations.py --execution corrected --output artifacts/external_corrected.json
python scripts/make_figures.py --output artifacts/figures
```

`conform release-check` is the principal validation entry point. It derives
the paper-facing summary from the audit JSON files, prospective result,
mutation outputs, support aggregates, runtime samples, and 25-check inventory;
it fails if those artifacts disagree or if an inventory evidence path is
missing.

The committed PDFs in [`figures/`](figures/) are the exact files used in the
final submission package. `scripts/make_figures.py` regenerates their five
paper-facing views from the included aggregate evidence. Rendering may vary
slightly with Matplotlib and font versions; the numerical content is checked
independently by `conform release-check` and the tests.

## Optional upstream audits

The source adapters can rerun the four original audits after the upstream
repositories and released artifacts have been acquired at the locked
revisions. The prospective AgentAbstain checks have a separate runner because
they require the pinned dataset and the upstream evaluation package.

Acquisition is intentionally separate from local validation; see
[`UPSTREAM.md`](UPSTREAM.md). No upstream source repository, dataset, model
output, API key, or paid service is bundled here.

After acquisition:

```bash
conform --workspace /path/to/acquisition-root audit agentprm --output artifacts/agentprm.json
conform --workspace /path/to/acquisition-root audit contractbench --output artifacts/contractbench.json
conform --workspace /path/to/acquisition-root audit toolace --output artifacts/toolace.json
conform --workspace /path/to/acquisition-root audit taubench --output artifacts/taubench.json

python scripts/run_agentabstain_checks.py \
  --source-root /path/to/acquisition-root/third_party/agentabstain \
  --dataset-jsonl /path/to/acquisition-root/third_party/agentabstain-data/tasks.jsonl \
  --output artifacts/agentabstain.json
```

The AgentAbstain runner requires the optional `prospective` dependency and the
dependencies declared by the pinned upstream repository.

```bash
python -m pip install -e ".[prospective]"
```

## Repository layout

| Path | Contents |
|---|---|
| `contracts/` | Five-target contract manifest and 30-mutation manifest |
| `src/conform/` | Contract result model, four source adapters, mutation fixtures, and release validator |
| `scripts/` | Prospective check, external-mutation, and figure entry points |
| `tests/` | Unit and release-integrity tests |
| `results/reproduction/audits/` | Sanitized exact audit records for the original four repositories |
| `results/support/` | Reader-facing aggregate support and mapping metadata |
| `tables/` | Complete 25-check inventory with paper-visible evidence paths |
| `figures/` | Final submission figures |
| `protocols/` | Frozen prospective and external-mutation protocols |

## Evidence boundaries

- ContractBench's released artifact has the same repository-internal model
  name and denominator as the reported Qwen3.5-9B result, but checkpoint,
  task-set identity, and evaluation configuration were not verified. The
  0/99-versus-56.6% comparison therefore holds only under the repository
  mapping. Corrected historical performance is not identifiable.
- The ToolACE source mismatch changes a controlled crop fixture, but none of
  the 9,412 parsed released targets exceeded the 8,192-token condition. The
  released aggregate includes 10,092 candidate targets, 9,412 parsed targets,
  and a maximum parsed sequence length of 3,945; third-party target text and
  per-target records are not redistributed.
- AgentAbstain released model outputs were unavailable. Three prospective
  checks conform; two expose source-under-specified conventions and are not
  defect claims.
- The repositories and checks were purposively selected. The release does not
  establish prevalence, completeness, automatic contract authorship, or a
  general proof of scientific correctness.

Legacy audit JSON fields named `severity` encode the original numeric
consequence rubric. Reader-facing summaries use **demonstrated consequence**
and do not assign a defect level to convention sensitivities.

## Fixed revisions

| Artifact | Revision |
|---|---|
| AgentPRM | `e4714717f7f4bd4671848670c4ed54d0169f603a` |
| ContractBench code | `c50eefee49b6925e2ccbf3c51a987ed705148725` |
| ContractBench released data | `457a8ad7d905cbb57ef6b892c5c087afee144171` |
| ToolACE / Tool-RL-Box | `1156d649235235e686372956e99bfc50e4b1e3f6` |
| tau-bench | `59a200c6d575d595120f1cb70fea53cef0632f6b` |
| AgentAbstain code | `f581249704b26804e28a39e37396f1be00b71a4d` |
| AgentAbstain dataset | `842228426c2a703347396501af61c7890972c7ee` |

## Citation and license

Author: Amir Reza Peimani, University of Toronto

Contact: [amir.peimani@utoronto.ca](mailto:amir.peimani@utoronto.ca)

To cite this software release:

```bibtex
@software{peimani_executable_scientific_contracts_2026,
  author  = {Peimani, Amir Reza},
  title   = {From Claims to Consequences: Executable Scientific Contracts for Paper–Code Auditing},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/AmirRezaPeimani/executable-scientific-contracts}
}
```

Citation metadata is also provided in [`CITATION.cff`](CITATION.cff).

The code in this repository is released under the MIT License. Upstream
repositories and datasets retain their own licenses; none are redistributed
here.
