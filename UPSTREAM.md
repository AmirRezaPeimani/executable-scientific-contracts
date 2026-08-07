# Optional upstream acquisition

The deterministic release checks use the included aggregate evidence. The
commands below are needed only to rerun source/data adapters. They make network
requests and download third-party material under the upstream projects' own
licenses.

Use a separate acquisition root so the public repository stays clean:

```bash
mkdir -p /path/to/acquisition-root/third_party

git clone https://github.com/sanjibanc/agent_prm.git /path/to/acquisition-root/third_party/agent_prm
git -C /path/to/acquisition-root/third_party/agent_prm checkout e4714717f7f4bd4671848670c4ed54d0169f603a

git clone https://github.com/SecurityLab-UCD/ContractBench.git /path/to/acquisition-root/third_party/contractbench_code
git -C /path/to/acquisition-root/third_party/contractbench_code checkout c50eefee49b6925e2ccbf3c51a987ed705148725

git clone https://huggingface.co/datasets/nips26-anon-author/contractbench /path/to/acquisition-root/third_party/contractbench
git -C /path/to/acquisition-root/third_party/contractbench checkout 457a8ad7d905cbb57ef6b892c5c087afee144171

git clone https://github.com/hypasd-art/Tool-RL-Box.git /path/to/acquisition-root/third_party/tool-rl-box
git -C /path/to/acquisition-root/third_party/tool-rl-box checkout 1156d649235235e686372956e99bfc50e4b1e3f6

git clone https://github.com/sierra-research/tau-bench.git /path/to/acquisition-root/third_party/tau-bench
git -C /path/to/acquisition-root/third_party/tau-bench checkout 59a200c6d575d595120f1cb70fea53cef0632f6b

git clone https://github.com/AntiQuality/agentabstain.git /path/to/acquisition-root/third_party/agentabstain
git -C /path/to/acquisition-root/third_party/agentabstain checkout f581249704b26804e28a39e37396f1be00b71a4d

git clone https://huggingface.co/datasets/antiquality/agentabstain /path/to/acquisition-root/third_party/agentabstain-data
git -C /path/to/acquisition-root/third_party/agentabstain-data checkout 842228426c2a703347396501af61c7890972c7ee
```

Expected AgentAbstain dataset file:

```text
/path/to/acquisition-root/third_party/agentabstain-data/tasks.jsonl
SHA-256: 165f021e7bb8b3a1ba103cef291eb522ff219769e8e7727f1a669364a225fb63
```

The code repositories were distributed under permissive licenses at the
examined revisions. The AgentAbstain dataset declares CC BY 4.0. Check each
upstream license before any redistribution. This release records revisions
and aggregate evidence but does not sublicense third-party content.
