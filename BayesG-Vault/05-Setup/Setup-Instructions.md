---
title: BayesG Setup on macOS (Apple Silicon)
tags: [setup, macos, install]
---

# BayesG Setup — macOS (Apple Silicon M1/M2/M3/M4)

Concrete, copy-pasteable. See [[MacM4-Feasibility]] for the why behind each step.

---

## 0. Prereqs

- macOS 13+ on Apple Silicon (M1/M2/M3/M4).
- Homebrew installed (`/opt/homebrew/bin/brew`).
- Xcode Command Line Tools: `xcode-select --install`.
- ~10 GB free disk (SUMO + conda env + checkpoints).

---

## 1. Install SUMO

```bash
brew tap dlr-ts/sumo
brew install --cask sumo-gui   # or: brew install sumo  for headless
```

Set env vars (add to `~/.zshrc`):

```bash
export SUMO_HOME="/opt/homebrew/share/sumo"
export PATH="$SUMO_HOME/bin:$PATH"
```

Verify:

```bash
sumo --version          # expect 1.21.x or 1.22.x
which sumo              # expect /opt/homebrew/bin/sumo
file $(which sumo)      # expect "Mach-O 64-bit executable arm64"
```

If the version is **not** 1.21, see Section 5 (pin traci to match).

---

## 2. Python environment

Use conda (Miniforge is the arm64-native distribution):

```bash
brew install --cask miniforge
conda init zsh   # restart shell after
conda create -n bayesg python=3.8.20 -y
conda activate bayesg
```

Alternative with pyenv:

```bash
brew install pyenv
pyenv install 3.8.20
pyenv shell 3.8.20
python -m venv ~/.venvs/bayesg
source ~/.venvs/bayesg/bin/activate
```

---

## 3. Patched requirements

Create `requirements-mac.txt` next to the upstream `requirements.txt`:

```bash
cd /tmp/BayesG
cat > requirements-mac.txt <<'EOF'
# core
numpy==1.23.5
scipy==1.10.1
pandas==2.0.3
matplotlib==3.7.5
seaborn==0.13.2
networkx==3.1
PyYAML==6.0.2
tqdm==4.62.3
psutil==7.0.0

# torch (arm64 wheels available from PyPI for 1.13.1)
torch==1.13.1
torchvision==0.14.1
torchaudio==0.13.1

# rl env
gym==0.14.0
gymnasium==1.1.1
pettingzoo==1.24.3
sumo-rl==1.4.5

# SUMO python bindings (pin to match brew-installed SUMO; adjust if needed)
traci==1.21.0
sumolib==1.21.0

# logging
tensorboardX==2.6.2.2

# misc that the code actually touches
ipython==8.12.3
ipdb==0.13.13
EOF

pip install --upgrade pip
pip install -r requirements-mac.txt
```

**Dropped from upstream `requirements.txt`:**

- `tensorflow==2.13.1`, `tensorflow-estimator`, `tensorflow-io-gcs-filesystem`, `tensorboard==2.13.0`, `keras` — never imported, no arm64/Py3.8 wheels.
- `mkl-service==2.4.0` — Intel-only, fails to install on arm64.
- `numba==0.56.4`, `llvmlite==0.39.1` — not imported.
- `wandb==0.19.8`, `sentry-sdk==2.24.1` — not imported.
- `cmake`, `code2flow`, `pylint`, `astroid`, etc. — dev tools, not needed at runtime.

Verify the install:

```bash
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
# expect: 1.13.1 True
python -c "import traci, sumolib; print(traci.__version__)"
# expect: 1.21.0 (or whatever you pinned)
```

---

## 4. Patch the codebase

Three small fixes (see [[MacM4-Feasibility#8-Risks--Mitigations]]).

### 4a. Fix `np.bool` (numpy 1.23 deprecation)

Edit `/tmp/BayesG/agents/utils.py`:

- Line 119: `dones = np.array(self.dones[:-1], dtype=np.bool)` → `dones = np.array(self.dones[:-1], dtype=bool)`
- Line 345: same change.

```bash
sed -i.bak 's/dtype=np\.bool\b/dtype=bool/g' /tmp/BayesG/agents/utils.py
```

### 4b. Keep CPU device (recommended; MPS in torch 1.13 is risky for LSTMCell)

`main.py:187` currently has `use_gpu = True`. **Leave it true** — the code in `models.py:164` checks `torch.cuda.is_available()` and gracefully falls back to CPU on Mac, so no change needed. Verify in the training log: it should print `Use cpu for pytorch...`.

Optionally remove the `torch.set_num_threads(1)` at `models.py:172` to let torch use multiple cores. Minor speedup (the model is tiny so it may not matter).

### 4c. `fuser` on macOS (optional)

`atsc_env.py:257, :439` use `fuser` to kill stale SUMO ports. It's already wrapped in `try/except`, so it will silently no-op on macOS. Stale ports after a crash get cleaned up by the random-port retry logic. No change needed unless you see port-bind errors.

---

## 5. SUMO version skew

If `brew install sumo` gave you **1.22** but you installed `traci==1.21.0`, you'll likely see:

```
traci.exceptions.FatalTraCIError: Mismatching TraCI API versions
```

Fix by re-pinning the Python side to match SUMO:

```bash
sumo --version  # note the version, e.g. 1.22.0
pip install --upgrade "traci==1.22.0" "sumolib==1.22.0"
```

The pure-Python TraCI wheel just needs to match the binary's wire protocol.

---

## 6. Smoke Test

Confirm everything wires up in ~5 minutes.

### 6a. Generate SUMO route files for Grid

```bash
cd /tmp/BayesG
python -c "
from envs.large_grid_data.build_file import gen_rou_file
gen_rou_file('./envs/large_grid_data/', 1100, 925, 0, seed=12, thread=0)
"
```

This writes `.rou.xml` and `.sumocfg` into `envs/large_grid_data/`.

### 6b. Patch the config for a short run

```bash
cp config/config_BayesG_grid.ini config/config_BayesG_grid_SMOKE.ini
sed -i.bak 's/^total_step = 1e6$/total_step = 1e4/' config/config_BayesG_grid_SMOKE.ini
sed -i.bak 's/^log_interval = 1e4$/log_interval = 1e3/' config/config_BayesG_grid_SMOKE.ini
sed -i.bak 's/^episode_length_sec = 3600$/episode_length_sec = 600/' config/config_BayesG_grid_SMOKE.ini
```

### 6c. Run

```bash
mkdir -p /tmp/bayesg-smoke
python main.py --base-dir /tmp/bayesg-smoke train \
    --config-dir ./config/config_BayesG_grid_SMOKE.ini
```

**Expected output (first 60s):**

```
... INFO ... Initializing SUMO environment with port XXXXX
... INFO ... Adjacency matrix Degree: [2. 3. 3. 3. 2. ...]
... INFO ... Training: a dim [5, 5, ...], agent dim: 25
... INFO ... Use cpu for pytorch...
... INFO ... Training: global step 1000, episode step ..., r: -..., ...
```

If you reach `global_step 10000` without crashing, **the install is good**.

Total smoke time: 3–8 minutes on M4.

---

## 7. Real training runs

### Grid (recommended first full run; ~10–24 h)

```bash
mkdir -p /tmp/bayesg-grid
python main.py --base-dir /tmp/bayesg-grid train \
    --config-dir ./config/config_BayesG_grid.ini
```

Monitor in another shell:

```bash
tensorboard --logdir /tmp/bayesg-grid/log --port 6006
# open http://localhost:6006
```

### Monaco (~14–36 h)

```bash
mkdir -p /tmp/bayesg-monaco
python main.py --base-dir /tmp/bayesg-monaco train \
    --config-dir ./config/config_BayesG_monaco.ini
```

### NewYork33 (~8–20 h)

```bash
mkdir -p /tmp/bayesg-ny33
python main.py --base-dir /tmp/bayesg-ny33 train \
    --config-dir ./config/config_BayesG_newyork33.ini
```

### NewYork51 / NewYork167

**Not recommended on a MacBook Air** (see [[MacM4-Feasibility#6-Scenario-by-Scenario-Time-Estimates]]). Run on a cloud Linux/arm64 host or a Mac with active cooling.

---

## 8. Evaluation / demo

After training writes a checkpoint to `<base-dir>/model/`:

```bash
python main.py --base-dir /tmp/bayesg-grid evaluate \
    --evaluation-seeds 2000,2010,2020 \
    --checkpoint /tmp/bayesg-grid/model/<timestamp>checkpoint.pt
```

Add `--demo` to launch SUMO-GUI.

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `OSError: libsumo... not found` | `SUMO_HOME` unset | Re-source `~/.zshrc`, confirm `echo $SUMO_HOME` |
| `traci.exceptions.FatalTraCIError: Mismatching TraCI API versions` | SUMO binary version ≠ python lib | See Section 5 |
| `AttributeError: module 'numpy' has no attribute 'bool'` | numpy>=1.24 | Apply patch in Section 4a, or pin `numpy==1.23.5` |
| `RuntimeError: ... mps backend out of memory` | Someone enabled MPS | Revert use_gpu logic, stay on CPU |
| Smoke test hangs at "Initializing SUMO" | Port collision | The code retries random ports; wait 30s. If persists, `lsof -ti tcp:8000-50000 \| xargs kill -9` |
| ` SUMO: cannot start GUI` in eval `--demo` | XQuartz/permission | Run without `--demo` first to confirm training works |
| Slow first episode | SUMO subprocess spawn + JIT | Normal; subsequent episodes are 5–10× faster |

---

## 10. Related

- [[MacM4-Feasibility]] — analysis behind these instructions
- [[../01-Architecture/Overview]] — codebase architecture
- [[../04-Concepts/Training-Loop]] — what's happening during training
- [[../06-LineByLine/main-py]] — `main.py` deep dive (if exists)
