# Coverage-aware interpretation

Nine new cases use three frozen source families, each with one source fault, one conformant reference and one semantically equivalent source rewrite. All v2 evaluator and import hashes match both before and after execution. Every source fault has a separate actual-implementation manifestation witness. All stated eligibility counts match observed coverage.

| Method | Faults detected / evaluated | Reference accepted / evaluated | Benign false alarms / evaluated | Unavailable cases |
|---|---:|---:|---:|---:|
| schema | 0/3 | 3/3 | 0/3 | 0 |
| native_tests | unavailable | unavailable | unavailable | 9 |
| source_pattern | 2/2 | 0/2 | 2/2 | 3 |
| behavioral_contract | 3/3 | 3/3 | 0/3 | 0 |

Schema validation misses all three manifested semantic faults. Historical source-pattern predicates flag their two covered faults but also both covered conformant references and both benign rewrites; the AgentAbstain source-pattern method is unavailable. Native test coverage is unavailable for all nine cases. Behavioral contracts accept the conformant references and benign rewrites while rejecting the three source faults.

This is a small check of new cases constructed after evaluator freeze by the same researcher. It provides neither independent authorship nor population detection accuracy, and uses no new repositories or observed model outcomes. Existing exposed confirmation results remain separate.
