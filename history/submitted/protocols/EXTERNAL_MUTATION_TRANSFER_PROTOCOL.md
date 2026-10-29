# Eight-operator mutation experiment

Specified: 2026-07-29

The original 30-mutation result was known when these eight operators were selected. The operators and fixtures below were recorded before their execution.

## Operators and fixtures

Operators follow René Just, *The Major Mutation Framework: Efficient and Scalable Mutation Analysis for Java*, ISSTA 2014, Section 2.2. PIT's mutator catalog supplied names and examples of boundary, return, and deletion mutations.

| ID | Operator family | Fixture | Mutation |
|---|---|---|---|
| XM-01 | arithmetic-operator replacement | discounted terminal target | replace the decreasing exponent with an increasing exponent |
| XM-02 | relational-operator replacement | thresholded pass rate | replace `>=` by `>` at an attained boundary |
| XM-03 | constant-value replacement | terminal-relative index | replace the terminal offset 1 by 0 |
| XM-04 | unary-operator replacement | signed target values | negate every target |
| XM-05 | logical-operator replacement | paired success | replace conjunction by disjunction |
| XM-06 | statement deletion | terminal-response logging | delete the final response append |
| XM-07 | statement deletion | required tool arguments | delete the required-argument insertion |
| XM-08 | logical-operator replacement | commit-aware abstention | replace conjunction by disjunction |

Each original fixture must pass before its mutant is evaluated. Detection means that the same check rejects the mutated fixture. Record equivalent or invalid mutants in the results.

## Execution and reporting

Run the eight operators once and report detection counts, a 95% Wilson interval, and counts by operator family. Selection is independent of any minimum detection threshold. Save execution errors; correct errors that prevent the specified transformation and record the correction.

## Equality input

The first run, `external_mutation_transfer_v1.json`, used a threshold of `1 - 1e-6` and rewards of `1.0` for XM-02. Both comparisons therefore returned the same result. The second run sets the threshold to `1.0`, placing the rewards at equality. The operator, rewards, check, other seven mutants, and reporting rule stay the same. Both runs are included.
