# Prospective AgentAbstain method-transfer protocol

Freeze date: 2026-07-29

Outcome status at freeze: repository metadata, README, license, and file names
were inspected. Evaluator implementation, released task rows, and audit
outcomes were not inspected.

## Target and licensing

- source repository: `AntiQuality/agentabstain`
- source revision: `f581249704b26804e28a39e37396f1be00b71a4d`
- source license: MIT
- released dataset: `antiquality/agentabstain`
- dataset revision: `842228426c2a703347396501af61c7890972c7ee`
- dataset license: CC BY 4.0
- declared release: 263 task pairs, 42 environments, 8 scenarios

The audit is read-only. The public package will contain only derived counts,
digests, and contract results, not redistributed task text or environments.

## Scientific purpose and independence boundary

This is a prospective fifth-repository method-transfer case selected after
the original four-repository study. No motivating discrepancy in this
repository was known when the contracts below were frozen. The same researcher
selected the target, translated claims, and implemented the contracts, so this
is neither an independently authored audit nor a random specificity sample.
A fully conformant result is valid.

## Five frozen contracts

1. `AABS-PAIR-01` — each benchmark pair must contain exactly one `act` and one
   `abstain` variant under one scenario and pair identifier.
2. `AABS-PAIR-02` — paired correctness must equal the conjunction of act-side
   and abstain-side correctness for each pair; it cannot be recovered from
   marginal rates alone.
3. `AABS-CAR-01` — conditioned abstention rate must use only pairs whose
   act-side run is correct as its denominator and the corresponding
   abstain-side correctness as its numerator. The paper defines this
   conditional rate but does not prescribe a value when the denominator is
   empty; the fixture therefore records the implementation's convention and
   an undefined alternative without treating their difference as a defect.
4. `AABS-COMMIT-01` — a should-abstain run with a committed critical action
   cannot be counted as successful abstention even when its terminal response
   claims restraint.
5. `AABS-MACRO-01` — the headline paired score must be the unweighted mean of
   the eight scenario-level paired rates, with every declared scenario
   represented exactly once. The paper and README specify macro-averaging over
   scenarios but not the within-scenario action-subtype weighting. The fixture
   therefore records both coherent rollups as a convention sensitivity.

## Fixtures and support

- `AABS-PAIR-01` runs over all released task rows.
- `AABS-PAIR-02`, `AABS-CAR-01`, and `AABS-MACRO-01` use deterministic
  synthetic score rows covering unequal scenario sizes, missing act success,
  and discordant pair outcomes. They test the released implementation without
  requiring proprietary model outputs.
- `AABS-COMMIT-01` uses four fixed synthetic traces spanning the Cartesian
  product of commit/no-commit and response-judge pass/fail.

Synthetic fixtures are generated from this protocol and contain no copied
benchmark text.

## Frozen outcomes and interpretation

- A claim-anchored contract is conformant only when the repository
  implementation and direct recomputation agree exactly.
- When the source claim does not determine an edge-case value or lower-level
  weighting rule, a difference between coherent conventions is labeled
  `alternative_convention`, not `discrepant`, and receives no defect severity.
- Source-only mismatch with no released-data activation: severity 0.
- Changed per-record value: severity 1.
- Changed aggregate metric: severity 2.
- Changed model ordering: severity 3.
- Changed stated conclusion: severity 4.

No severity above 2 is assigned without released model results that
demonstrate the higher consequence.

## Comparison baselines

The prospective target will also be evaluated by:

1. repository-native tests discoverable at the pinned revision;
2. required-field/schema validation on released task rows; and
3. source-pattern inspection without corrected recomputation.

The comparison records which frozen contracts each baseline can establish.
Absence of a native test is reported as missing coverage, not a failed test.

## Primary evaluation

The prospective transfer succeeds if all five checks execute, clean fixtures
pass, and the audit produces a complete evidence record regardless of whether
the repository is conformant. Claim-anchored conformance and under-specified
convention sensitivities are reported separately from the original 20
contracts and never used to retroactively change their gates.

## Reproducibility

The audit record must save source and dataset revisions, file digests,
commands, elapsed time, direct and implementation values, contract statuses,
severity evidence, and any unsupported field or unavailable result. Scripts
must refuse to overwrite existing output.
