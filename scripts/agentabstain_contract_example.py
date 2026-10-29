#!/usr/bin/env python3
import argparse
import hashlib
import inspect
import subprocess
import sys
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--source-root", type=Path, required=True)
root = parser.parse_args().source_root.resolve()
pin = "f581249704b26804e28a39e37396f1be00b71a4d"
revision = subprocess.check_output(
    ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
assert revision == pin
sys.dont_write_bytecode = True
sys.path.insert(0, str(root))
from eval.statistics.analysis import compute_summary_metrics
source = Path(inspect.getsourcefile(compute_summary_metrics)).resolve()
assert source == root / "eval/statistics/analysis.py"
cells = ((True, True), (True, False), (False, True), (False, False))
records = []
for pair, outcomes in enumerate(cells):
    for task_type, passed in zip(("act", "abstain"), outcomes):
        records.append(
            {"provider": "fixture", "model": "fixture", "category": "boolean",
             "action_type": "operational", "metric_name": "combined",
             "task_id": str(pair), "task_type": task_type, "pass": passed})
summary = compute_summary_metrics(pd.DataFrame(records))
paired = float(summary.iloc[0]["paired_accuracy"])
assert paired == 0.25
assert int(summary.iloc[0]["num_pairs"]) == 4
print("paired_accuracy: 0.25 PASS")
print("source_sha256:", hashlib.sha256(source.read_bytes()).hexdigest())
