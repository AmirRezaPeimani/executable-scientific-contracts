# AgentAbstain checks

Specified: 2026-07-29

Before specifying these checks, inspection covered repository metadata, the README, license, and file names. The evaluator, released task rows, and check results were examined afterward.

## Source and data

- Source: `AntiQuality/agentabstain`
- Source revision: `f581249704b26804e28a39e37396f1be00b71a4d`
- Source license: MIT
- Dataset: `antiquality/agentabstain`
- Dataset revision: `842228426c2a703347396501af61c7890972c7ee`
- Dataset license: CC BY 4.0
- Declared release: 263 task pairs, 42 environments, 8 scenarios

Acquire source and data separately. The package contains derived counts, hashes, and check results.

## Five checks

1. `AABS-PAIR-01`: each pair contains one `act` and one `abstain` variant under the same scenario and pair identifier.
2. `AABS-PAIR-02`: paired correctness is the conjunction of act and abstain correctness for each pair.
3. `AABS-CAR-01`: the conditioned abstention rate divides successful abstentions among successful act-side pairs by the number of successful act-side pairs. For an empty denominator, record the implementation's value and an undefined alternative.
4. `AABS-COMMIT-01`: successful operational abstention requires an uncommitted critical action and a passing response judgment.
5. `AABS-MACRO-01`: the headline paired score averages the eight scenario rates equally, including each scenario once. Within each scenario, compare equal subtype weighting with pair weighting, which the paper and README leave unspecified.

## Inputs

- `AABS-PAIR-01` uses every released task row.
- `AABS-PAIR-02`, `AABS-CAR-01`, and `AABS-MACRO-01` use constructed scores with unequal scenario sizes, zero successful acts, and discordant pair outcomes.
- `AABS-COMMIT-01` uses the four combinations of commit/no-commit and response-judge pass/fail.

Constructed inputs contain synthetic scores and traces. They execute the released implementation.

## Outcomes

- Record conformance when implementation outputs equal direct recomputation.
- Record `alternative_convention` for differences arising from unspecified edge cases or weighting rules; assign no defect severity to these cases.
- Severity 0: source mismatch without activation in released data.
- Severity 1: changed per-record value.
- Severity 2: changed aggregate metric.
- Severity 3: changed model ordering in released model results.
- Severity 4: changed stated conclusion supported by released model results.

## Comparisons

Compare repository tests at the recorded revision, required-field and schema checks, and source-pattern inspection. Record which checks each method covers and mark unavailable coverage explicitly.

## Execution and outputs

Run all five checks, require the original fixtures to pass, and save the complete results. Report the three checks of specified behavior and the two convention comparisons separately from the original 20 checks.

Save source and dataset revisions, file hashes, commands, elapsed time, expected and observed values, statuses, severity evidence, and unavailable fields. Use a new output directory for each run.
