# Historical submitted evidence

These preserved files support the earlier fixture-validator experiments, original 25-check classifications, and original-adapter result summaries. They are historical evidence; the current implementation results and official conversion are under `study/camera_ready/`.

The original classification totals are 15 conformant, seven discrepant, one inactive, and two convention sensitivities. A saved original classification is not changed by the current eligibility reporting. In particular, the ContractBench successful-trace check has zero eligible successful episodes and is not evaluated.

The fixture mutation set contains duplicate temporal transformations. Its 30 selected mutations exercise four fixture validators. The separate eight-operator experiment detected seven on the first execution and eight after adding an equality case with the check unchanged.

Run `conform release-check --root history/submitted` to reconstruct these historical summaries. `RELEASE_MANIFEST.json` records original artifact identities; the enclosing final release manifest records this presentation edition.
