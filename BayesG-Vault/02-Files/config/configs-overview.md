# BayesG Configs — Comparative Overview

A walkthrough of every `.ini` file under `/tmp/BayesG/config/` (40 files total). All configs follow the naming convention `config_<algo>_<scenario>.ini`, with two oddballs (`config_greedy.ini`, `config_ma2c_cnet_Large_city.ini` — note the capital `L`). They are consumed by the training entry-point and drive both the policy model (see [[policies-BayesG]], [[models]]) and the simulation environment (see [[envs]]).

---

## Config naming convention

There are 7 algorithm families crossed with up to 7 scenarios. The full matrix:

| algo \ scenario | grid | net (Monaco) | newyork33 | newyork51 | newyork167 | slowdown | other |
|-----------------|------|--------------|-----------|-----------|------------|----------|-------|
| **BayesG** (`BayesianGraph`)   | `config_BayesG_grid.ini`     | `config_BayesG_monaco.ini`    | `config_BayesG_newyork33.ini`    | `config_BayesG_newyork51.ini`    | `config_BayesG_newyork167.ini`    | —                            | —                                       |
| **LtoS** (`IA2C_LToS`)         | `config_LtoS_grid.ini`       | `config_LtoS_net.ini`         | `config_LtoS_newyork33.ini`      | —                                | —                                  | —                            | —                                       |
| **ia2c**                       | `config_ia2c_grid.ini`       | `config_ia2c_net.ini`         | `config_ia2c_newyork33.ini`      | `config_ia2c_newyork51.ini`      | `config_ia2c_newyork167.ini`      | `config_ia2c_slowdown.ini`   | —                                       |
| **ia2c_cu** (`Consensus`)      | `config_ia2c_cu_grid.ini`    | `config_ia2c_cu_net.ini`      | `config_ia2c_cu_newyork33.ini`   | `config_ia2c_cu_newyork51.ini`   | —                                  | —                            | —                                       |
| **ia2c_fp** (`IA2C_FP`)        | `config_ia2c_fp_grid.ini`    | `config_ia2c_fp_net.ini`      | `config_ia2c_fp_newyork33.ini`   | `config_ia2c_fp_newyork51.ini`   | —                                  | —                            | —                                       |
| **ma2c_cnet** (`CommNet`)      | `config_ma2c_cnet_grid.ini`  | `config_ma2c_cnet_net.ini`    | `config_ma2c_cnet_newyork33.ini` | `config_ma2c_cnet_newyork51.ini` | `config_ma2c_cnet_newyork167.ini` | `config_ma2c_cnet_slowdown.ini` | `config_ma2c_cnet_Large_city.ini` |
| **ma2c_dial**                  | `config_ma2c_dial_grid.ini`  | `config_ma2c_dial_net.ini`    | —                                | —                                | —                                  | —                            | —                                       |
| **ma2c_nc** (`NeurComm`)       | `config_ma2c_nc_grid.ini`    | `config_ma2c_nc_net.ini`      | `config_ma2c_nc_newyork33.ini`   | `config_ma2c_nc_newyork51.ini`   | `config_ma2c_nc_newyork167.ini`   | —                            | —                                       |
| **rand_graph_gnn** (`NeurComm` + GNN) | `config_rand_graph_gnn_grid.ini` | `config_rand_graph_gnn_net.ini` | —                          | `config_rand_graph_gnn_newyork51.ini` | —                            | —                            | —                                       |
| **greedy**                     | —                            | —                             | —                                | —                                | —                                  | —                            | `config_greedy.ini` (grid-only baseline) |

Total: 40 files. The five BayesG configs are the canonical training entry-points and are the focus of the comparisons below.

---

## Sections

Every config (except the baseline `config_greedy.ini` and `config_ia2c_slowdown.ini`) is divided into three INI sections.

### `[MODEL_CONFIG]`
Policy/optimizer hyperparameters consumed by [[models]] / [[policies-BayesG]]:

- `algo` — algorithm tag (e.g. `BayesianGraph`, `NeurComm`, `CommNet`, `Consensus`, `IA2C`, `IA2C_FP`, `IA2C_LToS`)
- RMSProp settings: `rmsp_alpha`, `rmsp_epsilon`, `max_grad_norm`
- A2C settings: `gamma`, `lr_init`, `lr_decay`, `entropy_coef`, `value_coef`
- Network dims: `num_lstm`, `num_fc`
- Rollout: `batch_size`, `reward_norm`, `reward_clip`
- BayesG-only: `is_graph_nn`, `n_attention_heads`, `gnn_type`, `unify_act_state_dim`, `learn_mask`
- NY-only on most algos: `torch_seed`

### `[TRAIN_CONFIG]`
Training loop / checkpoint cadence:

- `load_model` (bool)
- `total_step` (typically `1e6` for grid/Monaco, `5e5` for some NY runs)
- `test_interval`
- `log_interval`
- LtoS-only: `save_interval`

### `[ENV_CONFIG]`
Simulator wiring. Two distinct shapes exist:

**Grid / Monaco shape** (parametric SUMO setup):
```
clip_wave, clip_wait, control_interval_sec, agent, coop_gamma,
data_path, episode_length_sec, norm_wave, norm_wait, coef_wait,
peak_flow1, peak_flow2, init_density (grid only) OR flow_rate (Monaco only),
objective, scenario, env_name, seed, test_seeds, yellow_interval_sec
```

**NewYork shape** (pre-baked SUMO .net.xml / .sumocfg):
```
reward_scale, net_path, sim_path, agent, coop_gamma,
env_name = Large_city, seed, test_seeds, sampling_time
```

The NY configs **comment out** the entire Grid-shape block, suggesting they were forked from the Monaco configs.

---

## Hyperparameter comparison table — BayesG configs

| key                    | `BayesG_grid` | `BayesG_monaco` | `BayesG_newyork33` | `BayesG_newyork51` | `BayesG_newyork167` |
|------------------------|---------------|-----------------|--------------------|--------------------|----------------------|
| `lr_init`              | `5e-4`        | `5e-4`          | `5e-4`             | `5e-4`             | `5e-4`               |
| `gamma`                | `0.99`        | `0.99`          | `0.99`             | `0.99`             | `0.99`               |
| `batch_size`           | **120**       | **120**         | **20**             | **20**             | **20**               |
| `reward_norm`          | **2000.0**    | **2000.0**      | **2.5**            | **2.5**            | **2.5**              |
| `total_step`           | **1e6**       | **1e6**         | **5e5**            | **5e5**            | **1e6**              |
| `num_lstm`             | `64`          | `64`            | `64`               | `64`               | `64`                 |
| `num_fc`               | `64`          | `64`            | `64`               | `64`               | `64`                 |
| `gnn_type`             | `gcn`         | `gcn`           | `gcn`              | `gcn`              | `gcn`                |
| `n_attention_heads`    | `4`           | `4`             | `4`                | `4`                | `4`                  |
| `unify_act_state_dim`  | `true`        | `true`          | `true`             | `true`             | `true`               |
| `learn_mask`           | `true`        | `true`          | `true`             | `true`             | `true`               |
| `is_graph_nn`          | `true`        | `true`          | `true`             | `true`             | `true`               |
| `torch_seed`           | _absent_      | _absent_        | `85`               | `85`               | `85`                 |

The model-architecture knobs (LSTM/FC dims, GNN type, attention heads, mask learning, action/state unification) are **identical across all five BayesG configs** — only the rollout/sampling knobs and the seed differ.

---

## What CHANGES between scenarios for the same algo (BayesG)

### `reward_norm`: 2000 (grid/Monaco) vs 2.5 (NY)

The grid/Monaco configs include the inline note:

```ini
; same nodes may be updated by ~25 propagations,
; adjust norm accordingly
reward_norm = 2000.0
```

The NY configs say `~28 propagations` but use `reward_norm = 2.5`. The two scenarios use different **raw reward magnitudes** in the underlying environment — Grid/Monaco emit unnormalized queue counts (large integers summed over many lanes); the NY pipeline applies its own `reward_scale` first inside the env (see `[ENV_CONFIG]`: `reward_scale = 2`/`10`/`2000` per network) so the model-side `reward_norm` only needs a small final divisor. This is also why the NY configs ship an extra `reward_scale` env-side key.

### `batch_size`: 120 (grid/Monaco) vs 20 (NY)

```ini
# BayesG_grid
batch_size = 120

# BayesG_newyork33
batch_size = 20
```

The Monaco/Grid envs run 3600 s episodes at 5 s control intervals = 720 control steps per episode, so a rollout of 120 steps slices each episode into ~6 updates. The NY runs are much smaller (`sampling_time = 0.05`, no fixed `episode_length_sec`) and update more frequently with smaller rollouts.

### `total_step`: 1e6 vs 5e5

```ini
# BayesG_grid, BayesG_monaco, BayesG_newyork167
total_step = 1e6

# BayesG_newyork33, BayesG_newyork51
total_step = 5e5
```

NY33 and NY51 halve the training budget; NY167 bumps it back to 1e6 (presumably because the larger graph needs more samples).

### `episode_length_sec` vs `sampling_time`

These two keys are mutually exclusive — they encode the same concept (how long one episode runs) but for different env families.

```ini
# BayesG_grid / BayesG_monaco
control_interval_sec = 5
episode_length_sec   = 3600
```

```ini
# BayesG_newyork{33,51,167}
sampling_time = 0.05
; episode_length_sec = 3600     <-- commented out
; control_interval_sec = 5      <-- commented out
```

The NY family uses `sampling_time` as its primary stepping cadence (50 ms ticks); the grid/Monaco family uses second-resolution `control_interval_sec` and an explicit fixed-length episode.

### `test_interval` / `log_interval`

```ini
# Grid/Monaco
test_interval = 2e6
log_interval  = 1e4

# NY33/NY51/NY167
test_interval = 1e3   (or 1000)
log_interval  = 2000
```

Grid/Monaco runs effectively **never test during training** (test_interval > total_step), so testing is run as a separate post-hoc evaluation. NY runs test every 1000 steps, which suggests the NY pipeline checkpoints and evaluates inline.

### Env-name and reward-scale

| scenario   | `env_name`   | env-side scaling                                     |
|------------|--------------|------------------------------------------------------|
| grid       | `Grid_ATSC`  | `peak_flow1 = 1100`, `peak_flow2 = 925`              |
| monaco     | `Monaco`     | `flow_rate = 325`                                    |
| newyork33  | `Large_city` | `reward_scale = 2`                                   |
| newyork51  | `Large_city` | `reward_scale = 10`                                  |
| newyork167 | `Large_city` | `reward_scale = 2000`                                |

The fact that NY167 needs a 1000× larger `reward_scale` than NY33 is a strong hint about how raw reward magnitudes blow up with network size.

---

## What CHANGES between algos for the same scenario (grid)

Comparing `config_BayesG_grid.ini`, `config_ma2c_nc_grid.ini`, and `config_ia2c_grid.ini`:

### Identical
All three share the **exact same backbone**:

```ini
rmsp_alpha = 0.99
rmsp_epsilon = 1e-5
max_grad_norm = 40
gamma = 0.99
lr_init = 5e-4
lr_decay = constant
entropy_coef = 0.01
value_coef = 0.5
num_lstm = 64
num_fc = 64
batch_size = 120
reward_clip = -1
```

And the same env block (`peak_flow1 = 1100`, `peak_flow2 = 925`, `episode_length_sec = 3600`, `objective = queue`, `scenario = atsc_large_grid`, `env_name = Grid_ATSC`, etc.). The `agent` field is `ma2c_nc` for both BayesG and ma2c_nc, and `ia2c` for ia2c.

### Differences

| key           | `BayesG_grid`       | `ma2c_nc_grid` | `ia2c_grid` |
|---------------|---------------------|----------------|-------------|
| `algo`        | `BayesianGraph`     | `NeurComm`     | `IA2C`      |
| `reward_norm` | `2000.0`            | `2000.0`       | `100.0`     |
| `log_interval`| `1e4`               | `1e3`          | `1e4`       |
| `load_model`  | `false`             | `False`        | `False`     |

### BayesG-specific keys (only present in BayesG)
```ini
; whether to use graph neural network
is_graph_nn = true
; number of attention heads
n_attention_heads = 4
gnn_type = gcn
unify_act_state_dim = true
learn_mask = true
```

`ma2c_nc_net.ini` ships `unify_act_state_dim = true` as well, and the `rand_graph_gnn_*` family ships everything except `learn_mask` (using `use_random_mask = true` instead) plus a `gnn_version = v1` key. So `learn_mask` is the truly BayesG-unique knob — it toggles whether the per-edge attention mask is parametrized/learned (the BayesG core idea, see [[policies-BayesG]]) versus randomly sampled or fixed.

### ia2c vs ia2c_fp vs ia2c_cu (grid)

| key         | `ia2c_grid` | `ia2c_fp_grid` | `ia2c_cu_grid` |
|-------------|-------------|----------------|----------------|
| `algo`      | `IA2C`      | `IA2C_FP`      | `Consensus`    |
| `agent`     | `ia2c`      | `ia2c_fp`      | `ma2c_cu`      |
| `coop_gamma`| `0.9`       | `0.95`         | `0.9`          |

Otherwise byte-identical — coop_gamma is the only behavioral knob distinguishing fingerprint-IA2C from vanilla IA2C in the grid setting.

---

## The LToS configs — what unique keys they have

The three `config_LtoS_*.ini` files (grid, net, newyork33) carry a sizable extension to `[MODEL_CONFIG]`:

```ini
algo = IA2C_LToS
...
shared_dim = 32       # or 64 for net/ny33 — comment notes "num_fc and shared_dim should be the same"
use_lstm = True

# LToS specific parameters
update_frequency = 1
tau = 0.01
epsilon = 0.1
epsilon_decay = 0.995
epsilon_min = 0.01
buffer_size = 100000  # 50000 on net; commented out on ny33
gradient_steps = 1
```

And the Monaco version adds:

```ini
# LToS specific environment parameters
neighbor_radius = 2
reward_sharing = True
weighted_sharing = True
```

`[TRAIN_CONFIG]` also gains `save_interval = 5e4` — none of the other algos save mid-training.

Unique keys to LtoS:
- `shared_dim`, `use_lstm` (model)
- `update_frequency`, `tau`, `epsilon`, `epsilon_decay`, `epsilon_min`, `buffer_size`, `gradient_steps` (DQN-like target-network + replay setup, which is unusual for an A2C derivative)
- `neighbor_radius`, `reward_sharing`, `weighted_sharing` (env, Monaco only)
- `save_interval` (training)

The presence of `buffer_size` and `epsilon` suggests LtoS has an off-policy or epsilon-greedy mixing component on top of A2C — likely the weight-sharing meta-controller. The agent string is `ia2c` in the grid config but `ia2c_ltos` in the net/NY33 configs — likely a leftover inconsistency from when the grid config was first copied from `ia2c_grid.ini`.

---

## Unused / dead keys

Observations from a static read of the configs:

- **`load_model`** is set to `False`/`false` everywhere it appears (no config ships `True`), so the load-checkpoint codepath is never exercised from a fresh config. Two configs (`ma2c_dial_grid.ini`, `ma2c_dial_net.ini`) **don't define `load_model` at all** — relies on a code-side default.
- **`buffer_size = 100000`** in `config_LtoS_grid.ini` and `buffer_size = 50000` in `config_LtoS_net.ini`, but **commented out** in `config_LtoS_newyork33.ini`:
  ```ini
  ; buffer_size = 100000
  ```
  Either the LtoS-on-NY codepath doesn't need a buffer, or this is dead state — worth checking the LtoS model code.
- **`env_name = Grid_ATSC`** is missing from `config_ma2c_dial_grid.ini` and `config_ma2c_dial_net.ini` (those two omit `env_name` entirely), whereas every other config sets it. Likely safe via a code-side default but inconsistent.
- The Monaco/Grid configs include `clip_wave`, `clip_wait`, `norm_wave`, `norm_wait`, `coef_wait` — all of these are **commented out** in every NY config:
  ```ini
  ; clip_wave = -1
  ; clip_wait = -1
  ; norm_wave = 1.0
  ; norm_wait = -1
  ; coef_wait = 0
  ```
  These keys are still read by the env code in [[envs]] — so either the NY env class doesn't consult them (likely, since it's a different scenario class) or it relies on hard-coded defaults.
- `ma2c_cnet_grid.ini` **does not** set `unify_act_state_dim` while every other `ma2c_cnet_*` variant does. Looks like an oversight.
- `cat_obs_hidden = False` appears only in the `ma2c_cnet_*` family (and `Large_city`, NY33, NY51, NY167 variants). It is **not** documented anywhere else and not present in BayesG configs.
- `config_ia2c_slowdown.ini` has **no `algo` key** — same for `config_ma2c_cnet_slowdown.ini` and `config_ma2c_dial_*`. The algo must be inferred from the file name or the `agent` field.

---

## Bugs / oddities

1. **`torch_seed = 85` only in NY configs.** Every NY-flavored config (BayesG, ia2c, ia2c_cu, ia2c_fp, ma2c_cnet, ma2c_nc) ships:
   ```ini
   torch_seed = 85
   ```
   None of the grid/Monaco configs do. The grid/Monaco runs will produce non-deterministic PyTorch behavior; the NY runs will be (more) reproducible. Inconsistent — likely a partial migration.

2. **Whole `[ENV_CONFIG]` blocks commented out in NY configs.** Every NY config has the Monaco/grid env block pasted in but `; `-commented. Roughly 15 lines per file × 18 NY-shaped configs = ~270 lines of dead config. Example from `config_BayesG_newyork33.ini`:
   ```ini
   ; clip_wave = -1
   ; clip_wait = -1
   ; control_interval_sec = 5
   ; agent is greedy, ia2c, ia2c_fp, ma2c_som, ma2c_ic3, ma2c_nc.
   ...
   ; ; objective is chosen from queue, wait, hybrid    <-- note the doubled `;`
   ; objective = queue
   ; scenario = atsc_real_net
   env_name = Large_city
   ```
   Note the `; ; objective` line has a double comment marker, suggesting hand-editing across multiple passes.

3. **`reward_scale = 2` vs `reward_scale =2`.** `config_ia2c_newyork33.ini` line 27 reads `reward_scale =2` (no space). Every other NY config uses `reward_scale = 2`. ConfigParser strips whitespace so the value is fine, but it's a sloppiness signal.

4. **`config_BayesG_newyork33.ini` has typo "~28 propagations" but `BayesG_grid.ini` says "~25"** — the same template comment was forked but the propagation count was bumped only for one of NY33/NY51/NY167. The actual `reward_norm` value (`2.5`) is the same across all three so the comment vs value relationship is muddled.

5. **`total_step = 5e5` only for NY33 and NY51 BayesG; NY167 is `1e6`.** Possibly because NY167 is larger and needs more steps, or possibly because the author forgot to bump NY33/NY51 back up after experimenting with shorter runs.

6. **`config_ma2c_cnet_Large_city.ini`** is the only file with a capital letter in the scenario portion of the filename, breaking the `config_<algo>_<scenario>.ini` snake_case convention. It also has `test_interval = 1e3` whereas its sister `config_ma2c_cnet_newyork33.ini` uses `1e3` as well — likely a duplicated experiment file.

7. **`agent` mismatch in LtoS-grid.** `config_LtoS_grid.ini` says `agent = ia2c`, but the LtoS configs for net/NY33 say `agent = ia2c_ltos`. The grid config is presumably a stale copy of `config_ia2c_grid.ini`.

8. **`coop_gamma` inconsistency.** In Monaco/grid configs, `coop_gamma = 0.9` (or 0.75/0.95) is set meaningfully. In every NY config it is set to `-1.0` (i.e. disabled), but it's also written as a float (`-1.0`) rather than the int `-1` used elsewhere — minor inconsistency that ConfigParser is tolerant of.

9. **`load_model = False      `** (with trailing spaces) appears in many NY configs. Harmless under ConfigParser, but again a hand-editing artifact.

10. **`config_ma2c_dial_grid.ini` and `config_ma2c_dial_net.ini` have no `algo` field.** Compare to `config_ma2c_nc_grid.ini` which sets `algo = NeurComm`. The DIAL configs rely entirely on either filename inference or a default — and they also omit `env_name`.

---

## Cross-references

- For how these keys are wired into the policy network, see [[policies-BayesG]].
- For how `[ENV_CONFIG]` is consumed by SUMO/CACC scenarios, see [[envs]] and [[envs-data]].
- For the model factory that dispatches on `algo`, see [[models]].
- The corresponding agent classes are documented under [[agents]].
