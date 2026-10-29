# Fresh source-family challenge

New cases were constructed after the v2 evaluators were frozen. The evaluators and imports were hash-verified and executed unchanged.

These are three researcher-constructed source families with paired reference and benign controls, not independent human validation or an accuracy estimate.

| Method | Manifested faults | Detected | Missed | Uncovered | Benign false alarms | Errors |
|---|---:|---:|---:|---:|---:|---:|
| schema | 3 | 0 | 3 | 0 | 0/3 | 0 |
| native_tests | 3 | 0 | 0 | 3 | 0/3 | 0 |
| source_pattern | 3 | 2 | 0 | 1 | 2/3 | 0 |
| behavioral_contract | 3 | 3 | 0 | 0 | 0/3 | 0 |

Every case has a separate actual-source manifestation witness and raw output. Missing native test coverage is unavailable, not a failed test. No existing result, protocol or evaluator was changed.
