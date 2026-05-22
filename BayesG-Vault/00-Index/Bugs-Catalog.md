---
title: Bug Catalog (every issue surfaced)
tags: [bugs, qa]
---

# BayesG Codebase — Consolidated Bug Catalog

Every issue surfaced by the line-by-line walkthroughs. Severities are this analyzer's read, not the author's; some "bugs" may be intentional ablations.

Legend: 🔴 likely incorrect / breaks training, 🟠 sloppy / latent, 🟡 cosmetic / dead code.

---

## Training correctness 🔴

| # | File / lines | Issue |
|---|---|---|
| 1 | `agents/utils.py:129` | **Operator-precedence drops bootstrap term.** `R = r.cpu().numpy() if isinstance(r, torch.Tensor) else r + self.gamma * R * (1.-done)` — when `r` is a tensor, `R` becomes just `r.cpu().numpy()` (no bootstrap). The multi-agent version on line 231 has the correct form, confirming this is unintended. Affects single-agent IA2C variants. |
| 2 | `agents/policies.py:543, 547` (NCMultiAgentPolicy_MLP) and analogous in `CommNetMultiAgentPolicy` | **`convert_state_linears` / `convert_action_linears` are plain Python lists, not `nn.ModuleList`.** Their parameters are invisible to `optimizer.parameters()` and `state_dict()` → never trained, never saved. Silent. |
| 3 | `agents/models.py:607–820` (`IA2C_LToS.backward`) | **Destructive neighbor-buffer resampling.** Inner loop calls `self.trans_buffer[j].sample_transition(...)` which `reset()`s buffer `j`. When the outer loop reaches `i = j`, buffer is empty → `continue` → agent `j` is never updated. Most agents are skipped each step. |
| 4 | `agents/policies.py:1214+` (BayesianGraphCMultiAgentPolicy) | **ELBO sign quirk.** Code computes `loss = -(likelihood + prior - mask_entropy)`. The canonical ELBO is `likelihood + prior + H[q]`; the minus on entropy means the optimizer minimizes `-H[q]`, **reducing** mask entropy. Worth checking against the paper's equation. |
| 5 | `agents/policies.py:1436-1437` | **`λ` vs `π` role swap in the prior term.** Code computes `λ·log(π) + (1-λ)·log(1-π)` which is cross-entropy *from* prior *to* posterior. The principled `E_q[log p(Z)]` is `π·log(λ) + (1-π)·log(1-λ)`. |
| 6 | `agents/policies.py:1435` | **`λ` is unsettable in practice.** Read via `getattr(self, 'lambda_', 0.2)` but never assigned, so effectively a constant. |
| 7 | `agents/policies.py` (FPPolicy `_encode_ob`) | **Zero-pads instead of using the fingerprint slice.** `ob[:, self.n_x:]` (the neighbor-policy fingerprint) is read into a slice that is then discarded; the LSTM sees `[fc(own_obs); zeros]`. The IA2C_FP fingerprint mechanism is silently disabled. |
| 8 | `agents/policies.py` (ConsensusPolicy) | **Double softmax.** NaN-recovery path replaces logits with `-1e6` then softmaxes, but the downstream `Categorical(logits=...)` softmaxes again. |
| 9 | `agents/policies.py` (CommNet `_init_net`) | **Dead `fc_p_layers`.** Allocated but never populated → CommNet drops neighbor policies entirely, despite the layer existing. |
| 10 | `agents/policies.py` (LToS) | **Duplicate `soft_update` method.** Second definition (lines 2423–2426) shadows the first (2389–2402); first version was the only one that updated Q-net. Driver calls the split helpers instead, so neither is actually live — but the shadowing is a footgun. |
| 11 | `agents/policies.py` (LToS `compute_target_critic`) | **Action argument ignored.** Always returns `max_a Q_target(s, a)`. |
| 12 | `agents/policies.py` (LToS `compute_actions`) | **Returns probabilities, caller casts to `.long()`** in `models.py:697-700`, flooring everything to 0. |
| 13 | `agents/policies.py` (LToS `compute_gradients`) | **Bare `except RuntimeError` returns zeros** when the autograd graph is freed — likely the common path, neutralizing the receiver-side message gradient that defines LToS. |
| 14 | `agents/models.py` (`IA2C_CU.backward`) | **Double `clip_grad_norm_`.** Once in `ConsensusPolicy.backward`, again in `MA2C_NC.backward`. |
| 15 | `agents/models.py` (`load`) | **Whitelists `GraphCMultiAgentPolicy` but not `BayesianGraphCMultiAgentPolicy`.** Resuming a BayesG run from a checkpoint may take the wrong branch. |
| 16 | `agents/policies.py` (GraphCMultiAgentPolicy `__init__`) | **Skips immediate parent**: `super(NCMultiAgentPolicy, self).__init__(...)` jumps to grandparent and rebuilds the network. Brittle if NC adds setup. |
| 17 | `agents/gnn.py` (GATLayer) | **`self.a` (attention vector) is dead.** Declared & xavier-init'd, never read. Attention is computed as `h_i @ h_j^T` (Luong dot-product), not the canonical GAT `a^T[Wh_i ‖ Wh_j]`. |
| 18 | `agents/gnn.py:40` | **`e.mean(dim=-1)` collapses heads.** After `matmul`, shape is `[N,N,H,H]` (the second H from h_j's transposed head dim). Averaging across that dim mixes heads — destroys multi-head independence. |
| 19 | `agents/gnn.py:60` | **GAT heads averaged, not concatenated** — halves expected output capacity vs canonical GAT. |
| 20 | `agents/gnn.py` (GCNLayer) | **No self-loops** (`A + I` not applied). Node's own state enters only via `linear.bias`. |
| 21 | `envs/Large_city.py:103` | **`np.fill_diagonal(adj_no_self, 1)`** contradicts the comment / name. |
| 22 | `envs/Large_city.py:151–156` | **`state_heterogeneous_space = ss` aliases instead of copying** and indexes `ss[j]` instead of `ss[indices[j]]`. Sizing nonsensical. |
| 23 | `envs/Large_city.py` (`reset`) | **Never closes prior traci connection.** Long runs leak SUMO processes. |
| 24 | `envs/Large_city.py:VEH_LEN_M` | `VEH_LEN_M = 200` — clearly wrong (vehicles are ~5–10m). |
| 25 | `envs/atsc_env.py` (init) | `terminate()` is called inside `__init__`, killing the just-spawned SUMO. |
| 26 | `envs/atsc_env.py` (`Node` default) | `neighbor=[]` is a mutable default arg — all `Node`s share the same list unless overwritten. |
| 27 | `envs/atsc_env.py` (`_set_phase` / `step`) | Real-net `QUEUE_MAX=10` cap applied only on `atsc_real_net`, not synthetic grids — asymmetric reward scale. |
| 28 | `envs/atsc_env.py` (`yellow_interval_sec`) | No config fallback; `getint` crashes if missing. If > `control_interval_sec`, green phase silently becomes a no-op. |
| 29 | `envs/atsc_env.py` real-net reward | `wait`/`queue` read `ild[0]` only; `wave` and `_measure_traffic_step` sum all segments. Observation vs reward use different aggregations. |
| 30 | `envs/atsc_env.py:617` | Calls `self.sim.lane.getLastStepHaltingNumber` on what should be a `lanearea` id — latent bug masked by id aliasing. |
| 31 | `envs/atsc_env.py` (`_measure_traffic_step`) | `init_data` runs once at construction; data lists accumulate across episodes with no flush. Memory leak in long runs. |
| 32 | `envs/atsc_env.py` (`_reset_state`) | Sets `prev_action=0`, not `-1` — defeats the `<0` no-previous-phase guard in `_get_node_phase`. |
| 33 | `envs/large_grid_env.py` (`_init_neighbor_map`) | Inconsistent neighbor-ordering: edge nodes `[N,E,W]` vs `[N,S,W]` vs interior `[N,E,S,W]`. Asymmetric. |
| 34 | `envs/large_grid_env.py` (file naming) | `exp_<sim_thread>.rou.xml` — keyed by thread, not seed. Shared `sim_thread=0` across workers races on `exp_0.rou.xml`. |
| 35 | `envs/real_net_env.py` (`EXTENDED_LANES`) | Duplicate key `('9431', '10099#3_1')` — second wins silently. |
| 36 | `envs/real_net_env.py` (`_init_neighbor_map`) | Asymmetric neighbor graph: `cluster_8751_9630` is listed as a neighbor of `cluster_9389_9689` but its own neighbor list is empty. Several isolated nodes. |
| 37 | `envs/real_net_env.py` (`_bfs`) | Off-by-one: returns `max_dist + 1`. |
| 38 | `agents/utils.py` (`OnPolicyBuffer.__init__`) | `alpha == 0` enters `_add_s_R_Adv` without `self.max_distance` set → crashes. |
| 39 | `agents/utils.py` (`Scheduler`) | Default `total_step=0` → division by zero, snaps to `val_min` immediately. |
| 40 | `utils.py` (`Trainer.run`) | After episode end, a 1-second `time.sleep` waits for SUMO cleanup. Adds ~25 minutes over 1390 episodes. |
| 41 | `utils.py` (`Tester.__init__`) | `super().__init__(env, model, global_counter, summary_writer)` — wrong arity (Trainer expects 6 positionals). Masked because nothing instantiates Tester. |
| 42 | `utils.py` (`Evaluator.__init__`) | Skips `super().__init__()`; `global_counter`, `summary_writer`, `cur_step`, `algo_name` never set. |
| 43 | `utils.py` (`Tester.run_offline`) | Treats `perform()`'s `(mean, std)` tuple as a scalar reward. Only `Evaluator.run` unpacks correctly. |
| 44 | `agents/policies.py` (BayesianGraph viz) | Visualization symmetrises an asymmetric model: `i→j` and `j→i` masks sampled independently but plot overwrites symmetrically. |
| 45 | `agents/policies.py` (BayesianGraph) | `mask_history` list appended every step with no cap — memory grows linearly. |

## Configuration / pipeline 🟠

| # | File | Issue |
|---|---|---|
| 46 | `main.py:37-45` | **4 CLI flags are unwired.** `--n-heads`, `--gnn-type`, `--is-mlp-gnn`, `--n-neighbor-hops` are parsed and never read in `train()`. |
| 47 | `main.py:42` | `--is-mlp-gnn` has `type=bool` — argparse pitfall (`bool("False") == True`). |
| 48 | `main.py:187` | `use_gpu = True` is hardcoded. |
| 49 | `main.py:285` | `evaluate_fn(..., port=1, ...)` — privileged port. |
| 50 | `main.py:24` | Default `--base-dir = '/BayesG'` — absolute root path will fail on most machines. |
| 51 | `config_LtoS_grid.ini` | `algo = ia2c` — should likely be `ia2c_ltos`. |
| 52 | `config_ma2c_dial_*.ini` | No `algo` key, no `env_name`. |
| 53 | `config_BayesG_newyork{33,51,167}.ini` | `torch_seed = 85` present only in NY configs — grid/Monaco runs are non-reproducible. |
| 54 | `config_ia2c_newyork33.ini` | `reward_scale =2` (typo, missing space). |
| 55 | NY configs | ~270 lines of `; `-commented dead config carried over from Grid template. |
| 56 | `requirements.txt` | `tensorflow==2.13.1`, `keras==2.13.1`, `wandb`, `sentry-sdk`, `numba`, `llvmlite`, `mkl-service` — listed but never imported. |
| 57 | `agents/utils.py:119, 345` | `np.bool` (removed in numpy ≥ 1.24). |

## Cosmetic / dead code 🟡

| # | File | Issue |
|---|---|---|
| 58 | `agents/utils.py` (`_add_st_R_Adv`) | Dead — defined but never dispatched; `dt` param accepted but unused. |
| 59 | `main.py` Large_city branch | `adj_order = 30` dead; `cal_n_order_matrix` call commented out (only CommNet uses it). |
| 60 | `main.py` BayesianGraph branch | Computes `gnn_type` (line 111), then doesn't forward it to the constructor. |
| 61 | `envs/draw_net.py` | Standalone script, **never imported anywhere**. `path = "NewYork33"` hardcoded. |
| 62 | `envs/large_grid_data/build_file.py` | Doesn't use `sumolib` — generates XML by string formatting + shells to `netconvert`. All 25 lights share identical 99s static program with offset=0. |
| 63 | `envs/real_net_data/build_file.py` | `__main__` is commented out; detector function `output_ild` is dead. Flow horizon 3300s in a 3600s sim window. |
| 64 | `agents/policies.py` (multiple) | Debug `print` statements left in `_init_policy`, `__init__` paths. |
| 65 | `agents/models.py` (LToS) | `current_epsilon` never decays. |
| 66 | `agents/policies.py` (LToS `__init__`) | `super().__init__(..., 'lstm', 'lstm', ...)` → every LToS policy has `self.name = 'lstm_lstm'` regardless of `agent_id`. |
| 67 | `agents/policies.py` (GraphMaskGenerator) | `UnboundLocalError` risk because `return mask, prob` is outside the `if/else`. |

## Severity tally

- 🔴 Likely incorrect: **45 items**
- 🟠 Sloppy / latent: **12 items**
- 🟡 Cosmetic / dead: **10 items**

**Total: 67 distinct issues.** Most are typical research-code rough edges; items #1, #2, #3, #4, #7, #18, and #44 are the ones I'd verify against the paper before trusting a reproduction.
