# Original experiments

These files contain the original fixture experiments, 25-check classifications, and adapter results. Current implementation results and the official ToolACE conversion are under `study/camera_ready/`.

The original classifications are 15 conformant, seven discrepant, one inactive, and two convention sensitivities. The current inventory also reports eligible counts. The ContractBench successful-trace check has zero eligible episodes and returns `not_evaluated`.

The four fixture validators were tested with 30 mutations, including duplicate temporal transformations. The separate eight-operator experiment detected seven on the first run and eight after adding an equality case.

Run `conform release-check --root history/submitted` to reconstruct these summaries. `RELEASE_MANIFEST.json` lists the original file hashes; the root manifest lists hashes for the distributed files.
