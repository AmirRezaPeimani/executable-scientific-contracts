# Prospective externally sourced mutation-transfer protocol

Freeze date: 2026-07-29

Outcome status at freeze: the original 30-mutant result was known. The eight
operators below were selected from mutation families documented independently
of this study. Their executions and detection outcomes were not inspected
before this protocol was saved.

## Purpose and claim boundary

This small evaluation asks whether the existing scientific-semantic oracles
also reject mutations selected from established software-testing operator
families rather than from the four study-specific discrepancy classes. It is
an operator-transfer check, not an independently authored test suite, a random
sample of software defects, or evidence about field-wide mutation adequacy.

The external operator source is:

- René Just, *The Major Mutation Framework: Efficient and Scalable Mutation
  Analysis for Java*, ISSTA 2014, Section 2.2, which enumerates binary and
  unary operator replacement, constant replacement, branch-condition
  manipulation, and statement deletion.
- PIT's public mutator catalog, consulted only for concrete names and
  boundary/return/deletion examples within those established families.

## Frozen operators and fixtures

Exactly eight single-site mutants will be evaluated:

| ID | External family | Frozen scientific fixture | Mutation |
|---|---|---|---|
| XM-01 | arithmetic-operator replacement | discounted terminal target | replace the decreasing exponent with an increasing exponent |
| XM-02 | relational-operator replacement | thresholded pass rate | replace `>=` by `>` at an attained boundary |
| XM-03 | constant-value replacement | terminal-relative index | replace the terminal offset 1 by 0 |
| XM-04 | unary-operator replacement | signed target values | negate every target |
| XM-05 | logical-operator replacement | paired success | replace conjunction by disjunction |
| XM-06 | statement deletion | terminal-response logging | delete the final response append |
| XM-07 | statement deletion | required tool arguments | delete the required-argument insertion |
| XM-08 | logical-operator replacement | commit-aware abstention | replace conjunction by disjunction |

Each clean fixture must pass its oracle before its mutant is evaluated. A
mutant is detected only if the same oracle rejects the mutated fixture.
Equivalent or invalid mutants are not silently replaced.

## Primary output

The primary output is the exact detected count out of eight, with a Wilson
95% interval. Results are also reported by external operator family. No
minimum detection threshold was used to select or revise the operators.

## Stopping and amendment rule

The script is run once. Any implementation error is recorded, corrected only
if it prevents the frozen transformation from being instantiated, and the
correction is documented without replacing the operator or fixture.

## Implementation amendment

The first execution (`external_mutation_transfer_v1.json`) instantiated
XM-02 with a threshold of `1 - 1e-6` and rewards equal to `1.0`. That did not
place any reward on the relational boundary, contrary to the frozen
"attained boundary" fixture, so replacing `>=` with `>` was equivalent.
Version 2 changes only that threshold to `1.0`. The operator, rewards, oracle,
other seven mutants, and reporting rule are unchanged; version 1 remains
preserved.
