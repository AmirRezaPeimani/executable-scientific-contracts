# Additional case results

The nine cases use the recorded inputs, expected outputs, and evaluator files.

Each family contains a fault, a reference implementation, and a harmless rewrite.

| Method | Faults with changed output | Detected | Missed | Uncovered | Benign false alarms | Errors |
|---|---:|---:|---:|---:|---:|---:|
| schema | 3 | 0 | 3 | 0 | 0/3 | 0 |
| native_tests | 3 | 0 | 0 | 3 | 0/3 | 0 |
| source_pattern | 3 | 2 | 0 | 1 | 2/3 | 0 |
| behavioral_contract | 3 | 3 | 0 | 0 | 0/3 | 0 |

Each case includes its raw output. Native test coverage is unavailable for these behaviors.
