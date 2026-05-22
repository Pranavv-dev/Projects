# BayesG Codebase Analysis

**Repository:** `Wei9711/BayesG` (https://github.com/Wei9711/BayesG)
**Paper:** Duan, Lu, Xuan. *"Bayesian Ego-graph Inference for Networked Multi-Agent Reinforcement Learning"*, NeurIPS 2025. ([OpenReview](https://openreview.net/forum?id=3qeTs05bRL))
**Built on:** [`cts198859/deeprl_network`](https://github.com/cts198859/deeprl_network) (Chu et al., ICLR 2020 — NeurComm).
**Size:** ~7,300 LOC of Python (plus a large SUMO/XML data tree).
**Analysis date:** 2026-05-22.

---

## 1. What the project is

BayesG is a research codebase for **networked multi-agent reinforcement learning (NMARL)** in which each agent only observes its local neighborhood. The core contribution is a decentralized actor–critic that, in addition to learning a policy, also **learns a sparse interaction graph** by treating each agent's "ego-graph" (the subset of neighbors it actually attends to) as a **latent variable inferred via variational Bayesian inference**.

Posterior over the per-agent binary mask `Z_i` ∈ {0,1}^{|N(i)|} is approximated by a variational `q(Z_i; φ_i)`, sampled via Gumbel-softmax, and optimized with an ELBO objective that balances policy log-likelihood against a Bernoulli sparsity prior and a mask entropy term. The sampled mask gates a GNN (GAT / GCN / GraphSAGE) used inside an actor–critic for **Adaptive Traffic Signal Control (ATSC)** on synthetic grids, the Monaco real network, and three Manhattan sub-networks (NY-33 / NY-51 / NY-167).

The repo is **published research code**, not a production library: many commits are bulk `Add files via upload`, baseline implementations sit alongside the new method in single large files, and several hyperparameters / thresholds are hardcoded.

---

## 2. Repository layout

```
BayesG/
├── main.py                      # CLI entrypoint (train | evaluate)
├── utils.py                     # Trainer / Tester / Evaluator + counters & dir helpers
├── requirements.txt             # Pinned (Python 3.8.20, torch 1.13.1, sumo / traci, gym 0.14)
├── README.md
├── BayesG.jpg                   # Paper figure
│
├── agents/
│   ├── models.py                # Algorithm classes: IA2C / IA2C_FP / IA2C_CU / IA2C_LToS /
│   │                            #   MA2C_NC / MA2C_CNET / MA2C_DIAL / BayesianGraph
│   ├── policies.py              # All neural policy modules (2,471 LOC — biggest file)
│   ├── gnn.py                   # GATLayer / GCNLayer / SAGELayer (from-scratch impls)
│   └── utils.py                 # OnPolicyBuffer / MultiAgentOnPolicyBuffer / LToSPolicyBuffer / Scheduler
│
├── envs/
│   ├── atsc_env.py              # Base SUMO ATSC env (TraCI wrapper)
│   ├── large_grid_env.py        # 5×5 synthetic grid (25 lights)
│   ├── real_net_env.py          # Monaco real network (~27 lights)
│   ├── Large_city.py            # NewYork 33 / 51 / 167 (gym.Wrapper)
│   ├── draw_net.py              # Visualization helpers
│   ├── large_grid_data/         # SUMO .nod/.edg/.con/.tll/.rou/.sumocfg + build_file.py
│   └── real_net_data/           # Monaco SUMO assets + build_file.py
│
└── config/                      # ~40 .ini configs, one per (algo × scenario)
    ├── config_BayesG_grid.ini
    ├── config_BayesG_monaco.ini
    ├── config_BayesG_newyork{33,51,167}.ini
    ├── config_ia2c_*.ini, config_ma2c_nc_*.ini, config_ma2c_cnet_*.ini, ...
    └── config_LtoS_*.ini, config_rand_graph_gnn_*.ini, config_greedy.ini
```

---

## 3. End-to-end data flow

```
main.py
  ├── parse_args()
  ├── train(args)
  │     ├── configparser reads config/*.ini   →  ENV_CONFIG, MODEL_CONFIG, TRAIN_CONFIG
  │     ├── init_env(env_name)                →  LargeGridEnv / RealNetEnv / Large_city_Env
  │     ├── init_agent(env, ...)              →  one of IA2C / IA2C_FP / IA2C_CU / IA2C_LToS /
  │     │                                          MA2C_NC / MA2C_CNET / MA2C_DIAL / BayesianGraph
  │     └── Trainer(env, model).run()         →  rollout (explore) + n-step backward
  └── evaluate(args)                          →  Evaluator(env, model).run()  → CSV / SUMO GUI
```

`Trainer.run()` in `utils.py:473–540` drives the standard on-policy actor–critic loop: reset env + policy LSTM, call `explore()` to collect `n_step` (default 120) transitions via `model.forward(... 'p' / 'v')` and `model.add_transition(...)`, then `model.backward(R, dt)` to update. Episode rewards are written to CSV; TensorBoard is fed via `SummaryWriter` (`main.py:206`).

---

## 4. Algorithms catalog (`agents/models.py`)

All algorithms share an actor–critic skeleton with per-agent LSTMs. They differ in **what neighbor information is shared and how it is aggregated**.

| Algo (class)          | Type        | Neighbor signal                                          | Notes                                                                                |
| --------------------- | ----------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `IA2C`                | IA2C        | neighbor states only                                     | Independent actor–critic per agent. Baseline ("PolicyInferring"-style).              |
| `IA2C_FP`             | IA2C        | + neighbor policy "fingerprints"                         | Foerster et al. 2017 stabilisation trick.                                            |
| `IA2C_CU`             | IA2C        | + consensus parameter averaging                          | Zhang et al. 2018. `consensus_update()` after each backward.                         |
| `IA2C_LToS`           | IA2C        | shared learned embeddings exchanged between neighbors    | Yi et al. NeurIPS 2022. Two-level high-/low-level policy.                            |
| `MA2C_NC`             | MA2C        | neighbor [state, policy, hidden] aggregated via MLP/GNN  | NeurComm (Chu et al. ICLR 2020) — the framework BayesG extends.                      |
| `MA2C_CNET`           | MA2C        | mean-pooled neighbor hidden states                       | CommNet (Sukhbaatar et al. 2016).                                                    |
| `MA2C_DIAL`           | MA2C        | mean-pooled neighbor hidden + own action                 | DIAL.                                                                                |
| **`BayesianGraph`**   | **MA2C**    | **GNN over a learned sparse mask of neighbors**          | **Paper's contribution. Inherits `MA2C_NC`. Uses `BayesianGraphCMultiAgentPolicy`.** |

`init_agent()` (`main.py:80–136`) is a registry-style if/elif on `MODEL_CONFIG.algo`. `Large_city` envs force `coop_gamma=-1` (no spatial reward discounting) and read `gnn_type` from config.

---

## 5. The core contribution: `BayesianGraphCMultiAgentPolicy`

Defined in `agents/policies.py:1214–1683`, inheriting `GraphCMultiAgentPolicy`. The companion `BayesianGraph` algorithm class is `agents/models.py:418–447`.

### 5.1 Variational posterior `q(Z_i; φ_i)`

`GraphMaskGenerator` (`policies.py:1184–1211`) parameterizes the per-edge Bernoulli posterior:

```python
# policies.py ~1188-1200
self.fc1 = nn.Linear(2 * d, hidden)
self.fc2 = nn.Linear(hidden, 1)
...
h = F.relu(self.fc1(torch.cat([s_i, s_j], dim=-1)))
logits = self.fc2(h)
prob = torch.sigmoid(logits)             # q(Z_ij = 1)
```

- One `GraphMaskGenerator` per agent (with at least one neighbor), held in `mask_generators` (`policies.py:1253`).
- Input features `(s_i, s_j)` are the per-agent embeddings produced by the shared encoder.

### 5.2 Reparameterized sampling (Gumbel-sigmoid)

```python
# policies.py ~1205
if self.training:
    g = -torch.log(-torch.log(torch.rand_like(logits) + eps) + eps)
    z = torch.sigmoid((logits + g) / tau)      # tau hard-coded to 0.5 (line 1190)
else:
    z = (prob > 0.08).float()                  # threshold also hard-coded
```

At inference the policy briefly enters `eval()` (`policies.py:1364–1371`) so the binary threshold is used.

### 5.3 Mask → GNN → policy/critic

For agent `i` with self-feature `x_i` and neighbor stack `x_{N(i)}`, three GNNs aggregate observation, policy, and hidden-state streams (`policies.py:1346–1356`):

```python
attn_mask[0, 1:] = z_mask                     # learned edges + self-loop (1343-1344)
x_combined = torch.cat([x_i, x_neighbors])    # 1347
s_x = self.gnn_obs[i](x_combined,    attn_mask)[0]
s_p = self.gnn_policy[i](p_combined, attn_mask)[0]
s_h = self.gnn_hidden[i](h_combined, attn_mask)[0]
s_i_new = torch.cat([s_x, s_p, s_h], dim=-1)  # 1352-1356
```

GNN type (`gat` / `gcn` / `sage`) is chosen by `_init_gnn_layer()` (`policies.py:953–962`); layers are home-grown in `agents/gnn.py:5–122`.

### 5.4 ELBO objective

Constructed in the backward path around `policies.py:1412–1452`:

```
likelihood    =  -policy_loss                                              # log p(a | s, Z)
prior         =   λ·log(prob) + (1-λ)·log(1-prob),  λ = 0.2                # sparsity prior
mask_entropy  = -(prob·log prob + (1-prob)·log(1-prob))                    # H[q(Z)]
ELBO          =  likelihood + prior - mask_entropy
loss          = -ELBO  +  v_coef·value_loss  -  e_coef·policy_entropy
```

`λ = 0.2` (the sparsity weight) is read via `getattr(self, 'lambda_', 0.2)` (`policies.py:1435`) but never actually set anywhere — i.e. **it is always 0.2**, contrary to what the `getattr` suggests.

### 5.5 Visualization

`visualize_masks(step, save_path, draw_whole)` (`policies.py:1476–1683`) renders learned masks at evaluation time. Two modes: a global heatmap + grid layout, or a per-agent panel of (local probs, local mask, global mask with highlights, network drawing). The threshold 0.08 (`policies.py:1504, 1635`) matches the inference threshold.

### 5.6 Architectural picture

```
       per-agent obs o_i
              │
              ▼
      ┌──────────────┐
      │  FC encoder  │  ─────────┐
      └──────────────┘            │
              │                   │
              ▼                   │
        embedding s_i             │
                                  │
   For each ordered pair (i, j ∈ N(i)):
        ┌─────────────────────────┐
        │ GraphMaskGenerator(s_i, s_j)│   →  prob ∈ (0,1), z ~ Gumbel-sigmoid
        └─────────────────────────┘
                    │
                    ▼
        Z_i  (sparse binary ego-graph)
                    │
                    ▼
     ┌────────────────────────────────┐
     │ GNN_obs / GNN_policy / GNN_hid │  ←  combined features over N(i) ∪ {i}, gated by Z_i
     └────────────────────────────────┘
                    │
                    ▼
              concat → LSTM → actor π_i(a|·) + critic V_i(·)
```

---

## 6. Environments (`envs/`)

### 6.1 Base — `TrafficSimulator` in `atsc_env.py:76–665`

- Wraps SUMO via `traci`. `connect(port=self.port)` (`atsc_env.py:452`), with ports drawn randomly per session (`main.py:65`) and explicitly killed on `terminate()` via `fuser`.
- **Per-agent state** = "wave" (incoming-lane vehicle density) optionally concatenated with "wait" (max waiting time), with neighbor states/fingerprints appended depending on `agent` mode (`atsc_env.py:308–341`).
- **Actions** = one phase index per traffic light. Heterogeneous: each light has its own number of phases (`n_a_ls`).
- **Reward** = `-queue` (default), `-max_wait`, or hybrid `-queue - coef_wait·wait`, clipped to QUEUE_MAX=10 (`atsc_env.py:490–548`).
- **Graph inputs**: `neighbor_mask` (binary adjacency) and `distance_mask` (hop distance) computed once in env init (`atsc_env.py:94–113`); these are exactly the inputs `BayesianGraph` learns to prune.
- `step()` (`atsc_env.py:203–249`) executes a yellow-phase interval then a green-phase interval, calling `traci.simulationStep()` inside; returns `(state, reward, done, global_reward)`.

### 6.2 Scenarios

| Scenario              | File                 | #agents | Net topology                  | Notes                                                              |
| --------------------- | -------------------- | ------- | ----------------------------- | ------------------------------------------------------------------ |
| ATSC Grid             | `large_grid_env.py`  | 25      | 5×5 synthetic                 | Hard-coded neighbor map; phases=5/intersection.                    |
| ATSC Monaco           | `real_net_env.py`    | ~27     | Real Monaco net               | Heterogeneous phases per TLS (`PHASES` dict, lines 47–66).         |
| NewYork 33 / 51 / 167 | `Large_city.py`      | 33/51/167 | Real Manhattan sub-nets    | `gym.Wrapper` style; 20s (40s for NY167) control interval; 500 steps/episode. |

`large_grid_data/build_file.py` (449 LOC) generates the full SUMO XML tree (`.nod / .typ / .edg / .con / .tll / .add`) for the 5×5 grid; `real_net_data/build_file.py` (167 LOC) generates the route file `most.rou.xml` (the Monaco `.net.xml` is shipped pre-built). NY net files live under `envs/NewYork{33,51,167}/`.

The repo ships a large number of pre-generated SUMO config files (`exp_NNNN.sumocfg`, `most_NNNN.sumocfg`), one per seed — these are committed as data, not generated at training time.

---

## 7. Training & evaluation loop (`utils.py`)

- **`Counter`** (~lines 1–60): total / test / log interval tracker shared by trainer & policy.
- **`Trainer.run()`** (`utils.py:473–540`): reset env + model LSTM; loop while `not counter.should_stop()`:
  1. `explore()` collects `n_step` transitions (`utils.py:246–364`).
  2. Compute bootstrap return `R = model.forward(obs, done, ..., out_type='v')`.
  3. `model.backward(R, dt, summary_writer, global_step)`.
  4. Periodically write checkpoints, log episode reward, run online tester.
- **`Tester`** (`utils.py:543–590`): offline evaluation across `test_seeds`; writes trip-level data.
- **`Evaluator`** (`utils.py:593–735`): used by `main.py evaluate`. Overrides `perform()` to call `model.visualize_masks(step, save_path, ...)` at checkpoints (`utils.py:666–679`) — this is how the BayesG mask figures from the paper are produced.
- Defends against `NaN`/`Inf` policy logits by re-softmaxing with `-1e6` substitution (`utils.py:210–222`) — clearly a band-aid for occasional policy network instability.

---

## 8. Configuration system

INI files in `config/`. Naming convention: `config_<algo>_<scenario>.ini`. Sections:

- **`[MODEL_CONFIG]`** — `algo`, learning hyperparams (`lr_init`, `gamma`, `rmsp_alpha/epsilon`, `max_grad_norm`, `value_coef`, `entropy_coef`), network sizes (`num_lstm`, `num_fc`), batch size, and BayesG-specific switches:
  - `is_graph_nn = true`
  - `gnn_type = gcn` (also `gat`, `sage`)
  - `n_attention_heads = 4`
  - `unify_act_state_dim = true` (heterogeneous-agent projection)
  - `learn_mask = true` (toggles BayesG mask vs fixed-graph GNN)
- **`[TRAIN_CONFIG]`** — `load_model`, `total_step` (1e6 for grid, 5e5 for NY33), `test_interval`, `log_interval`.
- **`[ENV_CONFIG]`** — scenario-specific: `env_name` (`Grid_ATSC` / `Monaco` / `Large_city`), `coop_gamma`, SUMO paths, normalization constants, `seed`, `test_seeds`.

Example reward-normalization mismatch: `config_BayesG_grid.ini` uses `reward_norm = 2000.0` while `config_BayesG_newyork33.ini` uses `reward_norm = 2.5` — the constants depend on env scale and are part of the per-scenario tuning.

CLI overrides exist for `--n-heads`, `--gnn-type`, `--is-mlp-gnn`, `--n-neighbor-hops` (`main.py:37–45`) but **none of them are actually wired into `train()`** — they are parsed and ignored. The ini file values are what take effect.

---

## 9. Dependencies & environment

- Python 3.8.20, PyTorch 1.13.1, gym 0.14 + gymnasium 1.1.1, SUMO ≥1.1.0, `traci 1.21.0`, `sumo-rl 1.4.5`.
- `tensorflow 2.13.1` and `keras 2.13.1` are pinned but **not used in code** (the repo imports only `torch`); they are dead dependencies inherited from upstream `deeprl_network`.
- Visualization stack: `matplotlib`, `seaborn`, `networkx`.
- Heavy/stale extras: `wandb 0.19.8`, `tensorboardX`, `gitpython`, `pylint`, `ipython`, `ipdb`, `tqdm` — many look like leftover dev tools rather than runtime requirements.

---

## 10. Code-quality observations

These don't affect the science but matter if anyone wants to build on the code.

**Design / structure**
1. **`policies.py` is 2,471 LOC and holds 9 classes** including baseline implementations (CommNet, DIAL, Consensus, NC, FP, LToS) plus the novel BayesG. Splitting baselines vs. proposed work would make the contribution easier to read and review.
2. **`main.py:init_agent`** is a long if/elif registry duplicating `coop_gamma=-1` and `gnn_type` overrides for the `Large_city` branch of every algo — a small algo→class table plus a per-env post-processor would deduplicate it.
3. **`utils.py:Trainer/Tester/Evaluator`** rely on env-name string branching (slowdown / catchup / Grid / Monaco / Large_city) — should be lifted into env-side polymorphism.

**Hard-coded constants in the novel path**
4. Gumbel-sigmoid temperature `τ = 0.5` (`policies.py:1190`) — not a config knob.
5. Mask binarization threshold `0.08` (`policies.py:1209, 1504, 1635`) — repeated, not a constant.
6. Bernoulli prior weight `λ = 0.2` (`policies.py:1435`) read via `getattr(self, 'lambda_', 0.2)` but **never actually set** — effectively always 0.2.
7. `value_coef`, `entropy_coef`, `max_grad_norm` are in config; the BayesG-specific terms are not. This makes ablation studies harder than they need to be.

**Bugs / smells**
8. **CLI flags are dropped on the floor** — `--n-heads`, `--gnn-type`, `--is-mlp-gnn`, `--n-neighbor-hops` in `main.py:37–45` are parsed and never read in `train()`.
9. `np.bool` is deprecated and used in `agents/utils.py:119, 345` — will raise on NumPy ≥ 1.20 unless the pin holds.
10. `IA2C_LToS` defines **two distinct `soft_update` methods** in `policies.py` (~2389 and ~2423) — the second silently overrides the first.
11. `IA2C_LToS.backward` (`models.py:607–820`) re-samples `trans_buffer[j]` inside the per-agent loop, which both **mutates the neighbor's buffer** and adds an O(n²) cost. This branch likely doesn't reproduce the LToS paper exactly.
12. `BayesianGraphCMultiAgentPolicy.mask_history` (`policies.py:1233, 1377–1381`) is appended every step with no cap — memory grows linearly with training length.
13. **`np.random.seed`** is commented out in `explore()` (`utils.py:478`); reproducibility relies on env-internal seed increments only.
14. SUMO port allocation uses `random.randint(0, 50000)` (`main.py:66`) — collision possible if multiple training runs share a host. `terminate()` falls back to `fuser` to free the port.
15. `tensorflow` / `keras` in `requirements.txt` aren't imported anywhere — should be dropped.
16. `is-mlp-gnn` CLI arg is typed `bool` (`main.py:42`) — argparse will treat any non-empty string (including `"False"`) as `True`. Should use `action='store_true'` or `lambda x: x.lower() == 'true'`.

**Stability patches that hint at deeper issues**
17. NaN-logit re-softmax fallback in `utils.py:210–222`.
18. Explicit NaN replacement in `ConsensusPolicy` (`policies.py:1787–1806`).
19. ELBO log-clamping with `+1e-8` (`policies.py:1436–1441`) but no equivalent guard on the main policy loss path.

---

## 11. Reproducing a BayesG run

From `README.md`:

```bash
# 1. Generate SUMO network files for the 5×5 grid
python envs/large_grid_data/build_file.py

# 2. Train BayesG on the grid
python3 main.py --base-dir /BayesG train \
    --config-dir ./config/config_BayesG_grid.ini

# 3. Watch TensorBoard
tensorboard --logdir=/BayesG/log

# 4. Evaluate with held-out seeds
python3 main.py --base-dir /BayesG evaluate \
    --evaluation-seeds 2000,2010,2020

# 5. SUMO-GUI demo of a saved checkpoint
python3 main.py --base-dir /BayesG evaluate \
    --checkpoint /BayesG/model/<timestamp>checkpoint.pt --demo
```

For Monaco / NewYork use `config_BayesG_monaco.ini` / `config_BayesG_newyork{33,51,167}.ini` respectively.

---

## 12. Suggested next steps for a fork

If you fork this to extend it:

1. **Promote BayesG hyperparameters to config** (`τ`, threshold, `λ`, learnable-mask toggle) — needed for any ablation.
2. **Fix or remove the dropped CLI flags** in `main.py` (or remove them entirely; config-only is cleaner).
3. **Split `policies.py`** into `policies/{base.py, ia2c.py, ma2c_nc.py, commnet.py, dial.py, ltos.py, bayesg.py}`.
4. **Trim `requirements.txt`** (drop `tensorflow`, `keras`, `wandb` if not used).
5. **Cap `mask_history`** or sample-and-discard rather than appending unboundedly.
6. **Use `torch_geometric`** instead of the from-scratch GAT/GCN/SAGE in `agents/gnn.py` — would also give you proper edge-index support and is what most reviewers expect.
7. **Switch `np.bool` → `bool`** in `agents/utils.py` to unpin NumPy.
8. **Add a `tests/` directory** with a tiny smoke test that runs ~1k env steps on the synthetic grid — this codebase has no test suite at all.

---

## 13. TL;DR

BayesG is the official implementation of a NeurIPS 2025 paper that learns sparse per-agent communication graphs in networked MARL by treating the ego-graph as a latent Bernoulli variable inferred by amortized variational inference. The implementation extends NeurComm/MA2C_NC: a `GraphMaskGenerator` produces a Gumbel-sigmoid-sampled mask per neighbor, the mask gates three GNN streams over obs / policy / hidden-state features, and an ELBO (policy log-likelihood − sparsity prior + mask entropy) is added to the standard A2C loss. Evaluated on SUMO ATSC across a synthetic 5×5 grid, Monaco, and three Manhattan sub-networks (33/51/167 lights), against a representative slate of NMARL baselines (IA2C, FP, ConsensusUpdate, LToS, CommNet, NeurComm). The code reproduces the paper but ships with the usual research-code smells: a 2.5k-LOC policy file, dropped CLI flags, a few hardcoded BayesG hyperparameters, dead TF dependencies, and no tests.
