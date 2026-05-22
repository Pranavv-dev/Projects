---
title: MacBook Air M4 Feasibility Analysis
tags: [setup, macos, apple-silicon, feasibility]
---

# Running BayesG on a MacBook Air M4 — Feasibility

Repo audited: `/tmp/BayesG` (commit @ 2026-05-22). Target: MacBook Air M4 (Apple Silicon, ARM64, unified memory, no NVIDIA GPU).

TL;DR: **Yes, it runs on an M4 with patches — but only CPU-bound.** The Grid scenario is comfortable (1–3 days for a full 1M-step run). Monaco is similar. NY51 is borderline (4–7 days). NY167 is impractical on an Air (likely 2+ weeks plus risk of thermal throttling). The dominant cost is SUMO + TraCI IPC, not torch — so MPS gives at best ~1.2–1.5× speedup, possibly less.

See also: [[Setup-Instructions]], [[../01-Architecture/Overview]], [[../04-Concepts/Training-Loop]].

---

## 1. Dependency Reality Check on macOS arm64

### Python 3.8.20
- Available via `pyenv`, `conda-forge`, or Homebrew (`python@3.8` was removed from core but is in `python-tk@3.8` casks/forks; safest path is `pyenv install 3.8.20` or `conda create -n bayesg python=3.8`).
- arm64-native CPython 3.8 builds **do** exist, but the ecosystem is thinner. **Recommendation: use 3.8.20 to match `requirements.txt` exactly** unless you are willing to do a port.

### torch 1.13.1
- **PyTorch 1.13.1 has arm64 macOS wheels** (`torch==1.13.1` from PyPI installs cleanly on M-series via `pip install torch==1.13.1`). The 1.12+ line introduced MPS.
- **No CUDA on Mac**, ever. So `torch.cuda.is_available()` returns `False` and the code's else-branch runs (CPU). See `models.py:164`.
- MPS works in 1.13.1 but has gaps; in particular `nn.LSTMCell` was unstable on MPS until ~PyTorch 2.1. With the LSTM-per-agent design here (one cell per node, called in a Python loop), **MPS in 1.13.1 will likely either crash or be slower than CPU due to per-op dispatch overhead**. Not worth porting unless you upgrade to torch >= 2.2 (which then conflicts with `numpy==1.23.5` pin — see Section 8).

### numpy 1.23.5
- arm64 wheels are available. **But** `numpy >= 1.20` deprecated `np.bool`; `numpy >= 1.24` removed it. 1.23.5 still emits a `DeprecationWarning` but works. Code uses `np.bool` in `agents/utils.py:119, 345` — patchable.

### tensorflow 2.13.1
- **Listed in `requirements.txt` but NEVER imported in any `.py` file.** `grep -rn "import tensorflow\|from tensorflow" /tmp/BayesG` returns nothing. The only "tensor" symbol used is `torch.utils.tensorboard.SummaryWriter`. Safe to drop.
- This matters because TF 2.13 has no native arm64 wheels for Python 3.8 on PyPI — you'd need `tensorflow-macos` from Apple's repo. **Skip it.**

### SUMO + traci 1.21.0 / sumolib 1.21.0
- macOS install: `brew install --cask sumo-gui` (or `brew install sumo` — Homebrew formula lives in the `dlr-ts/sumo` tap). Required env vars: `export SUMO_HOME="/opt/homebrew/share/sumo"` (Apple Silicon prefix) and `export PATH="$SUMO_HOME/bin:$PATH"`.
- The Python `traci` and `sumolib` packages are pip-installable as pure-Python wheels; version 1.21.0 should match a Homebrew SUMO 1.21.x. **SUMO version skew between the binary and the Python lib is the #1 cause of `traci.exceptions.FatalTraCIError` mismatches.** Pin both.
- TraCI port handling: `atsc_env.py:62-90` picks a random port via `socket.bind(('',0))`. Should work on macOS.

### sumo-rl 1.4.5
- Pure-Python, depends on `gymnasium`. Installs fine on arm64.

### wandb / sentry-sdk
- Listed in requirements but **not imported anywhere** (`grep` returns 0 hits). Safe to drop.

### mkl-service 2.4.0
- **Intel MKL — does not exist on arm64.** Will fail to install. **Must remove from requirements.** PyTorch on arm64 uses Apple Accelerate / OpenBLAS under the hood; mkl-service is unnecessary.

### numba 0.56.4 / llvmlite 0.39.1
- These have arm64 wheels for Py 3.8 (numba added arm64 wheels around 0.55). Should work, but **also not actually imported** in this codebase — safe to drop.

---

## 2. What the code will actually do on a Mac

Trace through `main.py:187`: `use_gpu = True`. Then `models.py:164`:

```python
if use_gpu and torch.cuda.is_available():  # False on Mac
    ...cuda path...
else:
    torch.manual_seed(seed)
    torch.set_num_threads(1)       # <-- IMPORTANT
    self.device = torch.device("cpu")
```

So **the code will silently fall back to CPU** without any crash from the device selector. But note `torch.set_num_threads(1)` — torch will use a single CPU thread for ops. On an M4 with 10 cores (4P+6E), this leaves nine cores idle for the torch side. SUMO is also single-threaded. So you have 8+ cores doing nothing during training. Consider removing the `set_num_threads(1)` for slight speedup (though per-op cost is so low it may not matter much — model is tiny).

`policies.py:109, 880, 1190` also use `torch.cuda.is_available()` inside the policy constructor — fine, all return CPU on Mac.

### No MPS code path exists
- Zero references to `torch.backends.mps` or `torch.device("mps")` in the codebase.
- Adding MPS is a one-line patch in `models.py:164`, but per Section 1 it is unlikely to help with torch 1.13.1.

### Linux-isms to patch
- `subprocess.run(['fuser', '-k', f'{self.port}/tcp'], ...)` at `atsc_env.py:257, 439`. macOS has no `fuser`. Wrap in try/except (already is — `try: ... except: pass`) so it won't crash, but the port-kill will silently no-op. Stale SUMO processes after a crash will leak ports. Replace with `lsof -ti tcp:{port} | xargs kill` on macOS, or just rely on the random-port logic.

### np.bool patches
- `agents/utils.py:119` and `:345`: change `dtype=np.bool` → `dtype=bool` (Python builtin, identical semantics).

---

## 3. Compute Load — Per-step FLOPs

Per-agent network (Grid config):
- `fc_x_layer`: Linear(n_s, 64) — n_s ~ 5–13 (wave dim depends on neighbors); ~13×64 ≈ 800 FLOPs.
- 3 GNN layers (obs / policy / hidden), GCN type: `Linear(in, 64)` plus `mm(adj, x)`. For one agent's ego-graph with at most 4 neighbors → 5 nodes × 64 features. ~5×64×64 ≈ 20k FLOPs per GNN, ×3 = 60k.
- `LSTMCell(192, 64)`: ~4 × (192×64 + 64×64) ≈ 65k FLOPs.
- `actor_head` Linear(64, 5) ≈ 320; `critic_head` Linear(~70, 1) ≈ 70.
- **Forward per agent per step ≈ ~130k FLOPs.**
- BayesG also runs a `GraphMaskGenerator` MLP per neighbor pair → ~4 × Linear(2d, 64) ≈ ~6k each, ~25k extra.

**Grid: 25 agents × ~155k FLOPs ≈ 4 MFLOPs per env step.** Negligible — M4 P-core does ~50 GFLOPS scalar, 200+ GFLOPS with NEON/AMX. Forward alone is microseconds.

The backward pass (over `batch_size=120` rollout) is ~120× the forward + grad bookkeeping. Still <1 GFLOP per gradient update. Roughly **per-update time on CPU ≈ 50–200 ms** based on similar PyTorch graph-NMARL benchmarks (most of it is Python overhead in the per-agent for-loops, not FLOPs).

---

## 4. SUMO Throughput — The Real Bottleneck

Per Grid episode (`config_BayesG_grid.ini`):
- `episode_length_sec = 3600`, `control_interval_sec = 5` → **720 control decisions per episode**.
- Each decision = 2s yellow + 3s green simulated → **3600 simulated seconds per episode**.
- One control step ≈ one Python→TraCI roundtrip per agent (25 agents × O(few) calls each for state/reward/phase set).

SUMO throughput on a single core for a 25-intersection grid is ~**100–300× realtime** (without TraCI overhead). TraCI per-step IPC adds ~0.5–2 ms × 25 agents = **15–50 ms wallclock per control step** dominated by IPC. So:

- **Grid: 720 × ~25 ms ≈ 18 s per episode of SUMO + ~25 × 50 ms = ~1.3 s of torch updates ≈ 20 s/episode.**
- Plus episode reset / SUMO subprocess spawn ≈ 1–3 s.

Total steps in config: `total_step = 1e6`. With `n_step=120` rollout, that's `1e6 / 120 ≈ 8333 model updates`, but each episode collects up to 720 control steps × n_step=120 (n_step is *rollout length per backward*, not steps per episode). Re-reading `utils.py:Trainer.explore`: the loop is `for _ in range(self.n_step)` and `backward` is called per rollout. So **each episode does ~720/120 ≈ 6 backward passes**, and the total number of episodes ≈ `1e6 / 720 ≈ 1390 episodes**.

**Wallclock Grid: 1390 episodes × ~22 s ≈ 8.5 hours. Realistic range with overhead, GC, periodic snapshots: 10–24 hours.**

This is faster than I initially feared because the network is tiny. The 50–200ms/update estimate was pessimistic — for batch_size=120 on a 25-agent toy graph on CPU, expect more like 20–80ms.

---

## 5. Memory

- Model: 25 agents × (~3 × 64×64 GNN + LSTMCell(192,64) + heads) ≈ ~50k params/agent × 25 ≈ **1.3M params total ≈ 5 MB fp32**.
- OnPolicyBuffer: 25 agents × n_step=120 × state_dim(~13 fp32) ≈ 150 KB.
- SUMO: ~200–500 MB resident for Grid; up to **3–5 GB for NY167**.
- Python + torch base: ~700 MB.

**Total Grid: <2 GB. Monaco: <3 GB. NY167: ~6–8 GB.** All fit comfortably in 16 GB unified RAM; 24 GB is overkill.

---

## 6. Scenario-by-Scenario Time Estimates

Assumptions: M4 Air, 10-core (4P+6E) — `__init` notes the unbinned 8c/10c chip; ranges below assume the **10-core** SKU (more common in 2025+). Numbers are **wall-clock for full `total_step` training**, single seed, no eval.

| Scenario | n_agents | total_step | episodes | CPU-only estimate | MPS-feasible? | Notes |
|---|---|---|---|---|---|---|
| **Grid** (`config_BayesG_grid.ini`) | 25 | 1e6 | ~1,390 | **10–24 h** | No real benefit | Smoke test in ~5 min |
| **Monaco** (`config_BayesG_monaco.ini`) | 28 | 1e6 | ~1,390 | **14–36 h** | No | Heterogeneous phases → more Python overhead |
| **NewYork33** (`config_BayesG_newyork33.ini`) | 33 | 5e5 | ~700 (control_interval 20s → 180 decisions/ep) | **8–20 h** | No | uses Large_city env path |
| **NewYork51** (`config_BayesG_newyork51.ini`) | 51 | 5e5 | ~700 | **24–72 h (1–3 days)** | Marginal | sumocfg complexity dominates |
| **NewYork167** (`config_BayesG_newyork167.ini`) | 167 | 1e6 | ~1,400 (40s control → 90 decisions/ep) | **2–4 weeks** | Marginal | Probably impractical on an Air — thermal throttling on sustained load |

**Why no MPS speedup**: The per-agent Python for-loops (`for i in range(self.n_agent): ...` in `_run_comm_layers`, `_run_critic_heads`) dominate. Each call dispatches dozens of micro-ops (Linear, mm, LSTMCell) on tensors of size ≤ 64. On MPS, kernel launch overhead per op (~50–100 µs) exceeds the op's CPU time. You would need to batch agents together to benefit from MPS — significant refactor.

If you upgrade to **PyTorch 2.2+ with MPS** *and* batch the agent loops, you might get **1.5–3× on Grid/Monaco**. But you'd then need numpy 1.24+, which requires the np.bool fix above plus rebuilding the locked dependency set.

---

## 7. Smoke Test Plan

To verify everything wires up in **~5 minutes** before committing to a long run:

1. **Patch `total_step`**: edit `config/config_BayesG_grid.ini` → `total_step = 1e4` (one log_interval).
2. **Patch `episode_length_sec`** (optional) → `300` to make each episode 60 control steps instead of 720.
3. **Confirm**: `python main.py --base-dir /tmp/bayesg-smoke train --config-dir ./config/config_BayesG_grid.ini` should produce ~15 episodes, write `train_reward.csv`, and dump a model.pt checkpoint. Expected wall time: 3–10 minutes.

See [[Setup-Instructions#Smoke-Test]] for exact commands.

---

## 8. Risks & Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| `np.bool` errors (numpy 1.23+) | Medium | Patch `agents/utils.py:119, :345` → `dtype=bool` |
| `mkl-service` install failure on arm64 | High | Remove from requirements.txt before `pip install -r` |
| `tensorflow==2.13.1` no arm64/Py3.8 wheel | High | Remove — never imported |
| `fuser` not found on macOS (port leak) | Low | Already in try/except. Optionally replace with `lsof` |
| SUMO version mismatch (Homebrew ships 1.22, traci pinned 1.21) | Medium | `brew install sumo@1.21` if available; else `pip install traci==1.22 sumolib==1.22` to match Homebrew |
| `torch.cuda.is_available()` path silently picks CPU | None (intended) | Verify in log: "Use cpu for pytorch..." |
| `wandb` / `sentry-sdk` heavy installs | Low | Remove from requirements — not imported |
| Thermal throttling on Air (no fan) under multi-day load | Medium for NY51+ | Use a cooling pad; or run on a Pro/Studio with active cooling |
| `numba==0.56.4` arm64 wheel for Py3.8 may be sparse | Low | Remove — not imported |
| SUMO 1.22 GUI on Apple Silicon needs Rosetta? | Low | Recent SUMO builds have native arm64; verify with `file $(which sumo)` |
| `torch.set_num_threads(1)` artificially limits CPU usage | Low | Comment it out if you want torch to use multiple cores (minor speedup) |
| `cloudpickle==1.2.2` pin — very old, may break | Low | Bump to compatible version if pip resolver complains |

---

## 9. Recommended Path

1. **Run the smoke test first** (Section 7). Confirm SUMO talks to traci on macOS and the Grid env reaches `done=True` once.
2. **Train Grid first** (10–24h overnight + 1 day). This is the cleanest synthetic benchmark.
3. **Then Monaco** for a real-net result.
4. **Skip NY51/NY167 on the Air.** Either rent a cloud CPU box (a Hetzner CPX41 or AWS c7g.4xlarge with Graviton arm64 will be 2–3× faster on the same code at $0.10–$0.40/hr) or borrow time on a CUDA Linux box.

See [[Setup-Instructions]] for concrete commands.
