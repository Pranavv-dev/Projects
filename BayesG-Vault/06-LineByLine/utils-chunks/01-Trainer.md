# utils.py — Lines 1-460 — Counter, Trainer, helpers (line-by-line)

This note dissects the top of [[utils.py]] (lines 1-460): the small filesystem/logging helpers, the [[Counter]] step-bookkeeper, and the workhorse [[Trainer]] class that orchestrates rollouts, NaN-guarded action sampling, and per-environment branching.

Related notes: [[02-Tester]], [[03-Evaluator]] (other halves of `utils.py`), [[Counter]], [[Trainer]], [[explore]], [[perform]], [[run]].

---

## Imports & module-level constants

```python
import itertools
import logging
import numpy as np
import time
import os
import pandas as pd
import subprocess
import shutil
import datetime
```

- `itertools` — used only for `itertools.count(1)` in [[Counter]].
- `logging` — used by `init_log` and pervasive `logging.info`/`logging.error` calls.
- `numpy as np` — heavy use inside `_get_policy`, `_get_value`, `explore`.
- `time` — used by `init_log` to time-stamp the log filename and (later in the file) for `time.sleep` between episodes.
- `os` — directory existence / creation.
- `pandas as pd` — imported but **not used in lines 1-460**. Likely used by `save` / dataframe dumps further down the file.
- `subprocess` — imported, but only referenced inside a commented-out version of `copy_file`. Dead import as of the current file.
- `shutil` — `shutil.copy` is the live implementation of `copy_file`.
- `datetime` — used to produce a `YYYYMMDDHHMM` timestamp folder name in `init_dir`.

No module-level constants are defined in this range.

---

## Helper functions

### `check_dir(cur_dir)` — lines 12-15

```python
def check_dir(cur_dir):
    if not os.path.exists(cur_dir):
        return False
    return True
```

- Trivial wrapper around `os.path.exists` that **inverts twice**. Could be written `return os.path.exists(cur_dir)`. It's a stylistic relic.
- Returns a bool — no side effects, no logging.

### `copy_file(src_dir, tar_dir)` — lines 18-28

```python
# def copy_file(src_dir, tar_dir):
#     cmd = ' cp %s %s' % (src_dir, tar_dir)
#     subprocess.check_call(cmd, shell=True)
def copy_file(src_dir, tar_dir):
    try:
        # Ensure the target directory exists
        os.makedirs(os.path.dirname(tar_dir), exist_ok=True)
        shutil.copy(src_dir, tar_dir)
        print(f"File copied successfully from {src_dir} to {tar_dir}")
    except Exception as e:
        print(f"Error while copying file: {e}")
```

- Lines 18-20 are the **commented-out** prior implementation using `subprocess` + `shell=True` (which is why the `subprocess` import is still present but dead — see [[Bugs / oddities]]).
- The live version:
  - **Line 24**: `os.makedirs(os.path.dirname(tar_dir), exist_ok=True)` — creates parent dirs of the *target file*. If `tar_dir` is itself a directory (no trailing filename), `os.path.dirname` will return the parent of that directory, which can silently mis-create things. The function assumes `tar_dir` is a full file path.
  - **Line 25**: `shutil.copy` — copies file contents *and* permission bits, but not full metadata (unlike `copy2`).
  - The `except Exception as e` block swallows **every** error and just prints — no re-raise, no `logging.error`. Calls relying on this for correctness will silently miss failures.

### `find_file(cur_dir, suffix='.ini')` — lines 30-35

```python
def find_file(cur_dir, suffix='.ini'):
    for file in os.listdir(cur_dir):
        if file.endswith(suffix):
            return cur_dir + '/' + file
    logging.error('Cannot find %s file' % suffix)
    return None
```

- Returns the **first** matching file (iteration order is `os.listdir`, which is OS-dependent and *not* sorted).
- Path concatenation uses `+ '/' +` instead of `os.path.join` — fine on POSIX, breaks portability on Windows.
- Failure mode: logs an error and returns `None`. Callers must check.

### `init_dir(base_dir, pathes=['log', 'data', 'model'], custom_log_dir=None, config=None)` — lines 38-83

```python
if not os.path.exists(base_dir):
    os.mkdir(base_dir)
dirs = {}

for path in pathes:
    if path == 'log' and custom_log_dir:
        cur_dir = custom_log_dir
        ...
    else:
        cur_dir = base_dir + '/%s/' % path

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
    cur_dir   = os.path.join(cur_dir, timestamp)
    os.makedirs(cur_dir, exist_ok=True)
    dirs[path] = cur_dir
return dirs
```

- **Line 47-48**: top-level `mkdir` (not `makedirs`). If `base_dir`'s parent doesn't exist, this raises.
- **Mutable default `pathes=['log', 'data', 'model']`** — classic Python footgun, though it's only read, so safe in practice.
- **The `'log'` branch (lines 52-72)** — only when both `path == 'log'` *and* `custom_log_dir` is truthy:
  - Pulls `is_graph_nn` and `gnn_type` from `config['MODEL_CONFIG']`.
  - For GAT, appends `{gnn_type}_heads_{n_heads}` (e.g. `gat_heads_4`).
  - For GCN or SAGE, appends just `{gnn_type}`.
  - Else raises `ValueError`.
  - If `is_mlp_gnn` is `True`, **mutates `base_dir`** (line 70) to the *basename* of the GNN dir and appends `_mlp_gnn`. Note that `base_dir` is the *function argument*, so this mutates the local variable — but it's only used inside this iteration, so the bug is harmless. Still confusing naming.
- **Lines 78-79**: every directory (log, data, model) gets a `YYYYMMDDHHMM` timestamp appended. Two runs starting in the same minute would *collide*, and `exist_ok=True` would silently merge them.
- Returns `dirs` — a dict keyed by the path-name (`'log'`, `'data'`, `'model'`).

### `init_log(log_dir)` — lines 86-92

```python
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler('%s/%d.log' % (log_dir, time.time())),
                        logging.StreamHandler()
                    ])
```

- `time.time()` is a float — `%d` truncates to int seconds, fine.
- `logging.basicConfig` is a **one-shot**: if the root logger already has handlers, this call is a no-op. Calling `init_log` twice in the same process won't add a second file handler.
- Both file + stream handlers attached, so logs go to disk and stderr.

### `init_test_flag(test_mode)` — lines 95-104

```python
if test_mode == 'no_test':         return False, False
if test_mode == 'in_train_test':   return True,  False
if test_mode == 'after_train_test':return False, True
if test_mode == 'all_test':        return True,  True
return False, False
```

- Returns `(in_train_test, after_train_test)`.
- Unrecognised modes silently fall through to `(False, False)` — no warning. Mistyped configs will degrade to no testing without any signal.

---

## class `Counter` — lines 107-134

A tiny step-bookkeeper consumed by [[Trainer]].

### `__init__` — lines 108-115

```python
def __init__(self, total_step, test_step, log_step):
    self.counter = itertools.count(1)
    self.cur_step = 0
    self.cur_test_step = 0
    self.total_step = total_step
    self.test_step = test_step
    self.log_step = log_step
    self.stop = False
```

- `itertools.count(1)` — infinite iterator starting at 1; `cur_step` becomes `0` until first `next()` call.
- `total_step` is the stopping budget, `test_step` the *interval* (in steps) between eval rollouts, `log_step` the modulus for `logging.info` lines.
- `self.stop` is a manual kill-switch; nothing in this file sets it, but external code may.

### `next()` — lines 117-119

```python
def next(self):
    self.cur_step = next(self.counter)
    return self.cur_step
```

- Advances the iterator and stores the new value. Called once per environment step inside `explore` (line 322).
- Note: shadows the built-in `next` only in scope; the inner `next(self.counter)` still refers to the builtin.

### `should_test()` — lines 121-126

```python
def should_test(self):
    test = False
    if (self.cur_step - self.cur_test_step) >= self.test_step:
        test = True
        self.cur_test_step = self.cur_step
    return test
```

- Trigger-and-reset pattern: returns `True` once every `test_step` global steps, then resets the baseline.
- Caller can rely on `should_test()` to be idempotent within a step (only the *first* `True` reading consumes the trigger).

### `should_log()` — lines 128-129

```python
def should_log(self):
    return (self.cur_step % self.log_step == 0)
```

- Pure mod check. Will fire on every multiple of `log_step` (including step 0 if `cur_step` is somehow 0, but `cur_step` jumps to 1 on first `next()`).

### `should_stop()` — lines 131-134

```python
def should_stop(self):
    if self.cur_step >= self.total_step:
        return True
    return self.stop
```

- Two-way stop: budget exhausted *or* `self.stop` flipped externally.

(There is **no** `log_step` *method* and **no** `test_step` *method* — those are just attributes. The methods are `should_log` / `should_test` / `should_stop`. The user's prompt listed `log_step` / `test_step` as methods; in the source they are integer attributes set in `__init__`.)

---

## class `Trainer` — lines 140-onwards

Docstring (lines 136-139):

> Trainer class is responsible for managing the training and evaluation processes of RL setup. It interacts with the environment, the model, and the global counters while logging and saving results.

### `__init__` — lines 141-187

Signature:

```python
def __init__(self, env, env_name, algo_name, model, global_counter,
             summary_writer, output_path=None, logger=None):
```

Parameters (docstring at lines 142-148 only documents a subset):

| param | purpose |
| --- | --- |
| `env` | environment object (CACC, ATSC, PowerGrid, Pandemic, Large_city, …). |
| `env_name` | string discriminator. Drives every per-env `if/elif` branch in the class. |
| `algo_name` | algorithm tag (`"ConseNet"`, `"IA2C_FP"`, `"IA2C"`, `"IA2C_LToS"`, else default). |
| `model` | RL policy/value network, exposing `forward`, `add_transition`, `backward`, `reset`, `n_agent`, etc. |
| `global_counter` | shared [[Counter]] instance. |
| `summary_writer` | TensorBoard `SummaryWriter`-style object with `add_scalar` / `flush`. |
| `output_path` | where logs/csvs get dumped (used by code further down the file). |
| `logger` | optional alt logger; stored as `self.logger` but the class otherwise uses `logging.*` globally. |

State assigned:

- `self.cur_step = 0` — *episode*-local step counter (distinct from `global_counter.cur_step`).
- `self.global_counter = global_counter`.
- `self.env`, `self.env_name`, `self.algo_name` stored verbatim.

**Agent-string selection (lines 154-166)** — this is the first env-branching block:

```python
if self.env_name == "Pandemic" or self.env_name == "Large_city":
    if   self.algo_name == "ConseNet":   self.agent = "ma2c_cu"
    elif self.algo_name == "IA2C_FP":    self.agent = "ia2c_fp"
    elif self.algo_name == "IA2C":       self.agent = "ia2c"
    elif self.algo_name == "IA2C_LToS":  self.agent = "ia2c_ltos"
    else:                                self.agent = "ma2c_cu"
else:
    self.agent = self.env.agent
```

- For Pandemic / Large_city, the algo name is mapped to an internal agent tag. Anything not recognised falls through to `"ma2c_cu"` (silent default — see [[Bugs / oddities]]).
- For every *other* env, `self.agent` is whatever the env exposes as `.agent` (CACC and ATSC envs set this themselves).

**`n_step` selection (lines 169-180)** — second env-branching block, governs how many env steps per rollout chunk:

```python
if   env_name in ("slowdown", "catchup"):       self.n_step = 600
elif env_name in ("Grid", "Monaco"):            self.n_step = 720
elif env_name == "PowerGrid":                   self.n_step = self.env.T
elif env_name == "Pandemic":                    self.n_step = self.env.T
elif env_name == "Large_city":                  self.n_step = self.env.T
else:                                           self.n_step = self.model.n_step
```

- CACC envs use a hard-coded 600.
- ATSC envs (Grid, Monaco) use a hard-coded 720 — with a parked-comment `# self.n_step = self.model.n_step` showing the original behaviour.
- PowerGrid / Pandemic / Large_city use the env's horizon `T` — so each rollout chunk is *a full episode*.
- Else: fall back to the model's `n_step`.

**Tail of init (lines 182-187):**

```python
self.summary_writer = summary_writer
assert self.env.T % self.n_step == 0
self.data = []
self.output_path = output_path
self.env.train_mode = True
self.logger = logger
```

- The `assert` enforces that the env horizon is an integer multiple of the rollout chunk — important so episode boundaries align with chunk boundaries.
- `self.data` — list of per-episode reward logs that `_log_episode` appends to.
- `self.env.train_mode = True` — flips the env into training mode (affects e.g. randomised demand for ATSC).

### `_add_summary(reward, global_step, is_train=True)` — lines 190-194

```python
if is_train: self.summary_writer.add_scalar('train_reward', reward, ...)
else:        self.summary_writer.add_scalar('test_reward', reward, ...)
```

- Single-purpose: stamps a scalar to TensorBoard. Called by `_log_episode` (always `is_train=True` in this slice).

### `_get_policy(ob, done, mode='train')` — lines 200-222

```python
if self.agent.startswith('ma2c'):
    self.ps = self.env.get_fingerprint()
    policy = self.model.forward(ob, done, self.ps)
else:
    policy = self.model.forward(ob, done)
action = []
for pi in policy:
    if np.any(np.isnan(pi)):
        pi = np.nan_to_num(pi, nan=-1000000)
        pi = np.exp(pi - np.max(pi))
        pi = pi / np.sum(pi)
    if mode == 'train':
        action.append(np.random.choice(np.arange(len(pi)), p=pi))
    else:
        action.append(np.argmax(pi))
return policy, np.array(action)
```

- **`ma2c` branch** (line 201): grabs *neighbour fingerprints* (`self.ps`) from the env, then forwards `(ob, done, ps)`. The fingerprint is the recent policy distribution of each agent's neighbours, used as extra conditioning.
- **Else branch** (line 205): a simple `forward(ob, done)`.
- The result `policy` is a per-agent list/array of distributions (or logits).

**The NaN guard (lines 209-216)** — this is the block worth dwelling on:

- `np.any(np.isnan(pi))` — checks whether any element of an agent's policy vector is NaN.
- If so, `np.nan_to_num(pi, nan=-1000000)` replaces NaNs with `-1e6` — a strong signal that those entries should be interpreted as *logits*, not probabilities.
- The next two lines (`pi = np.exp(pi - np.max(pi)); pi = pi / np.sum(pi)`) are a **softmax over the logits**, the standard numerically-stable variant.
- The trick: an entry of `-1e6` becomes `exp(-1e6 - max)` ≈ 0, so the original NaN slots get near-zero probability — without crashing `np.random.choice` (which would otherwise raise on NaNs/negatives/non-sum-1).
- **Subtle bug**: this block only runs *when there's a NaN*. If `pi` is already a valid probability vector (no NaNs), no softmax is applied and `np.random.choice(p=pi)` is given the raw `pi` — which assumes the model already outputs probabilities. If `pi` were already logits, sampling on the clean path would be wrong. So the function presumes `pi` is **probabilities on the happy path** and **logits on the NaN-recovery path**. That asymmetry is fragile.

**Sampling vs argmax (lines 218-221)**:
- `mode='train'` → stochastic sample (`np.random.choice`).
- `mode='test'` → deterministic `argmax`.

Returns `(policy, np.array(action))`.

### `_get_value(ob, done, action)` — lines 225-233

```python
if self.agent.startswith('ma2c'):
    value = self.model.forward(ob, done, self.ps, np.array(action), 'v')
else:
    self.naction = self.env.get_neighbor_action(action)
    if not self.naction:
        self.naction = np.nan
    value = self.model.forward(ob, done, self.naction, 'v')
return value
```

- For `ma2c*`, value head uses observation + done + policies-of-neighbours + this step's action.
- For other agents, fetches **neighbour actions** from the env. If empty / falsy, falls back to `np.nan` — note the `if not self.naction` check is a truthiness check, so an empty list, `None`, or `0`-length array would all trip it. A non-empty numpy array would error here on Python's truthiness rule, but in practice `get_neighbor_action` returns a list/dict.
- `mode='v'` is the value-head signal.

### `_log_episode(global_step, mean_reward, std_reward)` — lines 235-243

Appends a dict to `self.data` and writes the mean reward to TensorBoard, then flushes.

```python
log = {'agent': self.agent, 'step': global_step, 'test_id': -1,
       'avg_reward': mean_reward, 'std_reward': std_reward}
self.data.append(log)
self._add_summary(mean_reward, global_step)
self.summary_writer.flush()
```

- `test_id: -1` is the sentinel for "this is a training entry, not a test rollout".

---

### `explore(prev_ob, prev_done)` — lines 246-364 — **the per-rollout collector**

Top of function (lines 247-256):

```python
ob = prev_ob
if self.env_name == "Large_city":
    self.env.clear()
    self.env.reset()
    ob = self.env.get_state_()
done = prev_done
self.ep_reward = 0
```

- Carries forward the observation from the previous chunk.
- **Large_city special-case**: hard-resets the env on every rollout chunk and re-reads state. For Large_city `n_step == env.T`, so each chunk *is* an episode, hence the reset.

**The per-step loop (lines 260-345):**

For `_ in range(self.n_step):`

1. **Policy** (line 262): `policy, action = self._get_policy(ob, done)` — stochastic in train mode.
2. **Value** (line 265): `value = self._get_value(ob, done, action)`.
3. **Fingerprint update** (line 268): `self.env.update_fingerprint(policy)` — pushes the current policy into the env's fingerprint store so other agents can read it next step.
4. **Env step**, branched by `env_name`:

   - **CACC** (`slowdown` / `catchup`, lines 272-290):
     ```python
     next_ob, reward, done, _ = self.env.step(action)
     s = dp(np.array(next_ob))           # ← `dp` is undefined in this file!
     self.S.append(s.ravel())            # ← `self.S` is never initialised here!
     done = done[0]
     global_reward = np.sum(reward)
     if self.env.coop_gamma <= 0:
         reward = np.sum(reward)
     ```
     This branch references **`dp`** (probably meant `copy.deepcopy`) and **`self.S`**, neither of which is defined in lines 1-460. Either there's a missing import / attribute init below line 460, or this branch is broken at runtime. See [[Bugs / oddities]].

   - **Grid** (lines 292-304):
     ```python
     next_ob, reward, done, _ = self.env.step(action)
     done = done[0]
     global_reward = np.sum(reward)
     if self.env.coop_gamma <= 0:
         reward = np.sum(reward)
     ```
     Plain Grid step.

   - **Else** (line 307): `next_ob, reward, done, global_reward = self.env.step(action)`.
     Note **Monaco falls into this default branch** — not into the Grid branch. So for Monaco, `global_reward` comes directly from the env's 4th return value, and `done` is **not** indexed with `[0]`.

5. **Episode reward accumulator** (lines 310-315):
   ```python
   episode_r = reward
   if episode_r.ndim > 1:
       episode_r = episode_r.mean(axis=0)
   self.ep_reward += episode_r
   ```
   If reward is per-agent-per-something, average along axis 0 first. `self.ep_reward` is initialised to `0` at the start, so the first `+=` promotes it from int to array.

6. **Episode-reward log** (line 319): `self.episode_rewards.append(global_reward)`.
   - `self.episode_rewards` is **not initialised in `__init__`** within lines 1-460. It must be set elsewhere (probably in `run()` below line 460, or in the omitted portion). Calling `explore` on a fresh Trainer without first setting it would `AttributeError`. See [[Bugs / oddities]].

7. **Counter bump** (lines 322-324):
   ```python
   global_step = self.global_counter.next()
   self.cur_step += 1
   ```

8. **Add transition to buffer** (lines 326-330):
   ```python
   if self.agent.startswith('ma2c'):
       self.model.add_transition(ob, self.ps, action, reward, value, done)
   else:
       self.model.add_transition(ob, self.naction, action, reward, value, done)
   ```
   - `ma2c` agents store the neighbour *policy fingerprint* (`self.ps`).
   - Others store the neighbour *actions* (`self.naction` — set inside `_get_value` for non-ma2c agents).

9. **Logging cadence** (lines 332-338) — fires when `global_counter.should_log()` is True.

10. **Terminal handling** (lines 340-344):
    ```python
    if done:
        if self.env_name != "catchup":
            break
        else:
            continue
    ```
    - For most envs, a terminal `done` breaks the rollout loop early.
    - For `catchup`, `continue` instead — episodes can "wrap" inside a chunk. This is unusual and probably specific to the CACC catchup scenario.

11. **`ob = next_ob`** (line 345) — only reached if `done` was False (else the `continue`/`break` above skipped this). For `catchup` after `done=True`, the `continue` skips this line, meaning `ob` is *not* updated with `next_ob` — a subtle bug or intentional reset-to-prev-ob behaviour worth flagging. See [[Bugs / oddities]].

**Bootstrap R after the loop (lines 346-350):**

```python
if done:
    R = np.zeros(self.model.n_agent)
else:
    _, action = self._get_policy(ob, done)
    R = self._get_value(ob, done, action)
```

- Standard A2C truncated-return bootstrap: if episode ended, R = 0; otherwise bootstrap with V(s_{T+1}).

**Per-env episode-reward logging (lines 352-362):**

```python
if self.env_name == "Grid":
    episode_reward = np.array(self.ep_reward).sum()
    logging.info(f"Episode reward: {episode_reward}")
    self.summary_writer.add_scalar('episode_reward', episode_reward, ...)
elif self.env_name == "Monaco":
    episode_reward = global_reward      # ← uses last-step global_reward, NOT a sum
    logging.info(f"Episode reward: {episode_reward}")
elif self.env_name == "Large_city":
    episode_reward = np.array(self.ep_reward).sum()
    logging.info(f"Episode reward: {episode_reward}")
    self.summary_writer.add_scalar('episode_reward', episode_reward, ...)
```

- **Monaco's `episode_reward = global_reward` is suspicious** — `global_reward` is the *last step's* global reward, not the episode total. Either intentional (since Monaco uses cumulative-style rewards in the env) or a bug; flagged in [[Bugs / oddities]].
- Monaco also doesn't write to TensorBoard, while Grid and Large_city do.
- CACC envs (`slowdown` / `catchup`) and PowerGrid / Pandemic have **no** episode-reward log emitted from this block.

**Return value (line 364):** `return ob, done, R`.

---

### `perform(test_ind, gui=False)` — lines 369-460+ — test rollout

(Visible portion: 369-460. The function continues past line 460.)

**Reset (lines 370-378):**

```python
if self.env_name == "Pandemic":
    self.env.reset()
    ob = self.env.get_state_()
elif self.env_name == "Large_city":
    self.env.clear()
    self.env.reset()
    ob = self.env.get_state_()
else:
    ob = self.env.reset(gui=gui, test_ind=test_ind)
```

- Pandemic & Large_city use a 2-step reset (no `gui`/`test_ind` kwargs).
- All other envs use a `gui`/`test_ind`-aware reset (CACC uses `test_ind` to seed the trajectory; ATSC similarly).

**Local state init (lines 380-383):**
```python
rewards = []
done = True
self.model.reset()
step = 0
```

- `done = True` is a deliberate prime — many models use `done` to reset hidden state on the first forward pass.

**Control-interval discovery (lines 386-402):**

A three-tier fallback chain:

1. `self.env.sim.control_interval_sec` if available.
2. `self.env.config.getint('control_interval_sec')` wrapped in `try/except`.
3. Default `1`.

Each branch emits an `INFO` or `WARNING` log. The bare `except:` catches everything (line 395) — see [[Bugs / oddities]].

**Eval loop top (lines 404-431):**

```python
while True:
    print(f"step: {step}")
    print(f"control_interval: {control_interval}")
    ...
```

- Uses bare `print` for per-step traces — heavy noise in a test rollout. Should arguably be `logging.debug`.

Branch on agent type:

- `self.agent == 'greedy'` → `action = self.model.forward(ob)` (no policy, just action).
- Else: `policy, action = self._get_policy(ob, done, mode=...)`. The `mode` argument depends on whether `self.env.name.startswith('atsc')`:
  - ATSC envs: `mode='train'` (i.e. **stochastic sampling even at test time**) — this matches the docstring "test policy is typically stochastic for on-policy methods".
  - Others: `mode='test'` → `argmax`.

Then `self.env.update_fingerprint(policy)` (mirrors `explore`).

**Mask visualisation (lines 416-431):** every `500 * control_interval` simulation seconds, if `self.model.visualize_masks` exists, dump or display attention masks. Uses `self.mask_viz_dir` — another attribute **not initialised in `__init__`** within the visible slice. The two `try/except` blocks log errors but continue. See [[Bugs / oddities]].

**Env step branch (lines 433-459):**

- **CACC** (`slowdown`/`catchup`, 433-438): same shape as in `explore`.
- **Grid / Monaco** (439-444): note that here Monaco and Grid share the branch, **unlike in `explore`** where Monaco was in the `else` default. Inconsistency between train & test. See [[Bugs / oddities]].
- **Pandemic** (445-451):
  ```python
  done = done.astype(float64)   # ← `float64` is not imported; should be `np.float64`
  ```
  Bare `float64` — will `NameError` at runtime. See [[Bugs / oddities]].
- **Large_city** (452-457): same `float64` bare reference bug; **also** the `if not self.naction` analogue (no, this branch uses `coop_gamma`). And note **`done = done[0]` then `done = done.astype(...)`** — but `done[0]` may already be a scalar, on which `.astype` is fine. Still, the `float64` bug applies.
- **Else** (458-459): plain `next_ob, reward, done, global_reward = self.env.step(action)`.

The remainder of `perform` continues past line 460 (not in scope).

---

## Tensor shapes & types — what `explore()` returns

From the per-step contract:

- **`ob`** — per-agent observation; for ATSC/CACC it's typically a list/array of per-agent vectors, shape roughly `[n_agent, obs_dim]`. After a `break`, it's the **last successful `next_ob`** (or unchanged from entry if step 0 was terminal).
- **`done`** — `bool`. For CACC/Grid/Large_city/Pandemic the code unwraps `done = done[0]` so it ends up scalar. For Monaco (which goes through the `else` branch in `explore`), `done` is **whatever shape the env returns** — possibly an array — which is a latent inconsistency.
- **`R`** — bootstrap return, shape `(self.model.n_agent,)`. Either `np.zeros(n_agent)` or the value-head's output. dtype is whatever the model returns (typically `np.float32`).

Action inside `explore`:

- `action` from `_get_policy` is `np.array(action)` of length `n_agent`, dtype `int64` (since each element is `np.random.choice(np.arange(len(pi)), ...)` which returns Python int / np.int).

Policy:

- `policy` is a length-`n_agent` list of 1-D `np.ndarray`s, each of length = the discrete action set size for that agent (e.g. 5 actions per intersection in Grid). The docstring comment on line 267 says "list of 25 policies (each policy is a list of 5 actions)" — matches the 5×5 Grid ATSC scenario.

Value:

- Returned by `model.forward(..., 'v')`. Shape and dtype model-dependent; assumed `(n_agent,)`.

Reward:

- Per-step env reward. CACC/Grid may collapse to a scalar when `coop_gamma <= 0`. Otherwise per-agent vector. `episode_r.ndim > 1` branch implies it can occasionally be 2-D.

---

## Bugs / oddities

1. **`dp` is undefined** (line 277, CACC branch in `explore`).
   ```python
   s = dp(np.array(next_ob))
   ```
   Almost certainly meant `from copy import deepcopy as dp` — but no such import is in the file. CACC training will `NameError` at the first step.

2. **`self.S` and `self.episode_rewards` are never initialised in `__init__`** (within lines 1-460).
   - `self.S.append(...)` (line 278) and `self.episode_rewards.append(...)` (line 319) presume these attributes exist. Either they're set elsewhere (e.g. in `run()` below line 460) or in subclasses, or this is a latent `AttributeError`.

3. **Bare `float64` in `perform`** (lines 448 and 455).
   ```python
   done = done.astype(float64)
   ```
   `float64` is not imported as a bare name; the import is `import numpy as np`, so it should be `np.float64`. Pandemic and Large_city test rollouts will crash here.

4. **Monaco branching inconsistency.**
   - In `explore` (train), Monaco falls through to the `else` branch (line 307) which keeps `done` un-indexed and reads `global_reward` from the env's 4th return value.
   - In `perform` (test), Monaco is bundled with Grid (line 439) and gets `done = done[0]` plus `global_reward = np.sum(reward)`.
   - Either training or testing has Monaco's data flow wrong.

5. **Monaco's `episode_reward = global_reward`** (line 357) uses only the *last* step's global reward, not a sum — unlike Grid/Large_city. Probably wrong as an "episode total".

6. **Catchup `continue` skips `ob = next_ob`.**
   In the per-step loop, when `done` is True and env is `"catchup"`, the `continue` (line 344) jumps over the `ob = next_ob` assignment (line 345). So after a within-chunk episode end, the next step's policy is computed from the **stale** `ob`. This may be intentional ("wait for env to auto-reset") but is fragile.

7. **Silent fall-through in `init_test_flag`** (line 104): unrecognised `test_mode` strings degrade to `(False, False)` with no warning. Typos in configs silently disable testing.

8. **`copy_file` swallows all errors** (lines 27-28): any `Exception` becomes a `print`. No re-raise, no `logging.error`. Pipeline steps depending on the copy will fail later in confusing ways.

9. **Dead `subprocess` import** (line 7): only referenced inside the commented-out original `copy_file` (lines 18-20). The live implementation uses `shutil`, so `subprocess` is dead.

10. **Commented `np.random.seed`** — the user's prompt mentions this; in lines 1-460 I see **no `np.random.seed` line**, commented or otherwise. It may live in the omitted portion below line 460 or in a caller. Worth searching the rest of the file; flagging here as "not visible in this slice".

11. **`time.sleep(1)` between episodes** — the user's prompt references a 1s sleep before next episode. This is **not present in lines 1-460**. It must live in `run()` which begins after line 460. The visible code has `time` imported but only uses it for `time.time()` in `init_log` (line 90). The sleep magic number cannot be analysed from this slice.

12. **`run()` is not in the visible 1-460 range.** The user's prompt asks for a walkthrough of `run()` (reset env+model, while loop, `explore` + R bootstrap, `model.backward()`, episode termination handling, checkpoint saving). None of that is present in lines 1-460. The visible `Trainer` methods stop at the start of `perform()` and `perform()` itself doesn't return by line 460. The `run()` method must begin past line 460 and is **outside the requested range** — flagged here so a follow-up note can cover it.

13. **`init_dir`'s `os.mkdir`** (line 48) — uses `mkdir`, not `makedirs`, so a missing grandparent dir raises `FileNotFoundError`. Inconsistent with the later `os.makedirs(...)` at line 81.

14. **`init_dir` timestamp collisions** — two runs in the same minute collide on `YYYYMMDDHHMM`; `exist_ok=True` silently shares the directory.

15. **`find_file` returns *some* matching file**, not deterministically the first sorted one. Replaying the same directory on two machines may pick different `.ini` files.

16. **`_get_policy` NaN-guard asymmetry**: the softmax is only applied inside the NaN-recovery branch, implying `pi` is probabilities on the happy path and logits on the unhappy path. Type-confused contract.

17. **`perform`'s bare `except:`** (line 395): swallows everything from `config.getint`, including `KeyboardInterrupt` pre-3.8 semantics. Should be `except (configparser.Error, ValueError):` or similar.

18. **`self.mask_viz_dir`** (line 420) — referenced but not initialised in `__init__` within this slice. Either set externally by callers (probably the script that constructs the Trainer) or another latent `AttributeError`.

19. **Pandemic/Large_city agent fall-through** (line 164): unknown `algo_name` silently picks `"ma2c_cu"`. A misspelled algo flag would train the wrong agent type with no warning.

20. **`assert self.env.T % self.n_step == 0`** (line 183) — uses bare `assert`, which is stripped under `python -O`. Should be a real `raise ValueError` for a contract this load-bearing.

---

## Cross-references

- [[Counter]] — used by every method that calls `self.global_counter.next()` / `.should_log()` / `.should_test()`.
- [[explore]] — the rollout collector; consumed by `run()` (below line 460).
- [[perform]] — test-time analogue of `explore`; called by [[Tester]] / [[Evaluator]] in [[02-Tester]] / [[03-Evaluator]].
- [[Trainer.__init__ env-branching]] — the `env_name` `if/elif` ladder that drives every per-env split downstream.
- [[NaN logit guard]] — line 209-216 trick; only path through `_get_policy` that applies a softmax.
- [[init_dir]] — directory layout (`log/`, `data/`, `model/`) used by every script that imports from `utils.py`.
