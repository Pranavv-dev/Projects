---
title: Architecture Overview
tags: [architecture, system]
---

# BayesG — System Architecture

## One-paragraph summary

BayesG extends the NeurComm (`MA2C_NC`) actor-critic framework with a **learned binary mask** over each agent's neighborhood. The mask is sampled from an amortized variational posterior `q(Z_i | s_i, s_N(i))`, parameterized by a small MLP per neighbor pair, with a Gumbel-sigmoid reparameterization for gradient flow. The masked subgraph is then passed through three GNNs (`gnn_obs` / `gnn_policy` / `gnn_hidden`) that aggregate observation / policy / hidden-state streams from neighbors, the outputs are concatenated, an LSTM ingests the result, and standard A2C actor + critic heads produce the policy distribution and the centralized value estimate. An ELBO loss term (likelihood + Bernoulli sparsity prior − mask entropy) is added to the usual A2C losses.

## Module dependency graph

```
main.py
 │
 ├── envs.large_grid_env       (Grid_ATSC)
 ├── envs.real_net_env         (Monaco)
 ├── envs.Large_city           (NewYork 33/51/167)
 │      │
 │      └── envs.atsc_env      (base TrafficSimulator — SUMO via TraCI)
 │
 ├── agents.models             (algorithm wrappers: IA2C / MA2C_NC / BayesianGraph / ...)
 │      │
 │      ├── agents.policies    (LSTM + GNN policy networks)
 │      │      │
 │      │      └── agents.gnn  (GAT / GCN / SAGE layers, from scratch)
 │      │
 │      └── agents.utils       (OnPolicyBuffer, MultiAgentOnPolicyBuffer, Scheduler)
 │
 └── utils                     (Trainer / Tester / Evaluator + dir helpers)
```

See [[../00-Index/README#File-inventory]] for the per-file walkthrough.

## End-to-end flow (training)

```
                              ┌─────────────────────────────────────┐
                              │ config/config_BayesG_grid.ini       │
                              └────────────────┬────────────────────┘
                                               │ configparser
                              ┌────────────────▼────────────────────┐
                              │ main.train():                       │
                              │  init_env  → LargeGridEnv (SUMO)    │
                              │  init_agent → BayesianGraph(...)    │
                              │  Trainer(env, model).run()          │
                              └────────────────┬────────────────────┘
                                               │
                ┌──────────────────────────────▼──────────────────────────────┐
                │ Trainer.run() loop                                          │
                │   while not counter.should_stop():                          │
                │     ob = env.reset(); model.reset()                         │
                │     for _ in range(n_step):                                 │
                │       p, v = model.forward(ob, done, ..., 'p'/'v')          │
                │       a    = sample(p);  next_ob, r, done = env.step(a)     │
                │       model.add_transition(ob, p, a, r, v, done)            │
                │       ob = next_ob                                          │
                │     R = model.forward(ob, done, 'v')        # bootstrap     │
                │     model.backward(R, dt)                                   │
                └────────────────────────────────────────────────────────────┘
                                               │
                ┌──────────────────────────────▼──────────────────────────────┐
                │ BayesianGraph.backward → BayesianGraphCMultiAgentPolicy.backward │
                │  obs, ps, acts, dones, Rs, Advs = buffer.sample_transition()│
                │  forward_pass (recompute) → policies + values + mask probs  │
                │  policy_loss   = -log_prob * Adv                            │
                │  value_loss    = (R - v)^2                                  │
                │  entropy_loss  = -H[π]                                      │
                │  elbo_loss     = -(policy_loss + prior(π_mask) - H[q])      │
                │  total         = elbo_loss + v_coef·value_loss              │
                │                              + e_coef·entropy_loss          │
                │  total.backward();  optimizer.step()                        │
                └────────────────────────────────────────────────────────────┘
```

See [[../06-LineByLine/policies-chunks/04-BayesianGraph-KEY]] for the exact code, tensor shapes, and ELBO math.

## Per-step forward pass through BayesianGraphCMultiAgentPolicy

For each agent `i`:

```
o_i ─── fc_x ─────────────────► x_i
                                │
                                ├─── mask_generator(s_i, s_j) → (z_ij, π_ij)
                                │       ∀j ∈ N(i)
                                │       z_ij = sigmoid((logit + Gumbel)/τ)
                                │       π_ij = sigmoid(logit)
                                │
                                ▼
attn_mask = [1, z_{i,1}, z_{i,2}, ..., z_{i,|N(i)|}]    # self-loop + learned edges
                                │
       x_combined = [x_i, x_{N(i)}]                     │
       p_combined = [p_i, p_{N(i)}]                     │
       h_combined = [h_i, h_{N(i)}]                     │
                                ▼
         s_x = gnn_obs   (x_combined, attn_mask)[0]    # gnn returns per-node; take row 0 (self)
         s_p = gnn_policy(p_combined, attn_mask)[0]
         s_h = gnn_hidden(h_combined, attn_mask)[0]
                                │
                  cat → [s_x, s_p, s_h]  (192-dim for n_fc=64)
                                │
                            LSTMCell
                                │
                 ┌──────────────┼──────────────┐
                 ▼                              ▼
         actor_head (n_a logits)         critic_head (scalar)
```

## Algorithm family

| Algorithm           | Wrapper class (`agents/models.py`) | Policy class (`agents/policies.py`)        | Neighbor signal aggregation                              |
| ------------------- | ---------------------------------- | ------------------------------------------ | -------------------------------------------------------- |
| Greedy              | (none, in env)                     | (none)                                     | local wave only                                          |
| IA2C                | `IA2C`                             | `LstmPolicy`                               | neighbor obs concatenated                                |
| IA2C_FP             | `IA2C_FP`                          | `FPPolicy`                                 | + neighbor policy fingerprints (broken — see bug #7)     |
| IA2C_CU (Consensus) | `IA2C_CU`                          | `ConsensusPolicy`                          | parameter averaging after each backward                  |
| IA2C_LToS           | `IA2C_LToS`                        | `LToSMultiAgentPolicy`                     | shared learned embeddings φ exchanged between neighbors  |
| MA2C_NC (NeurComm)  | `MA2C_NC`                          | `NCMultiAgentPolicy` / `_MLP`              | 3-stream MLP: [obs, policy, hidden] over neighbors       |
| CommNet             | `MA2C_CNET`                        | `CommNetMultiAgentPolicy`                  | mean-pooled neighbor hidden states                       |
| DIAL                | `MA2C_DIAL`                        | `DIALMultiAgentPolicy`                     | mean-pooled neighbor hidden + own action one-hot         |
| **BayesG**          | `BayesianGraph`                    | `BayesianGraphCMultiAgentPolicy`           | **learned mask × 3-stream GNN**                          |

## Environments

| Env class            | File                                     | #agents | Net source                              | Episode length                  |
| -------------------- | ---------------------------------------- | ------: | --------------------------------------- | ------------------------------- |
| `LargeGridEnv`       | `envs/large_grid_env.py`                 |      25 | Synthetic 5×5 (generated by build_file) | 3600s / 5s control = 720 steps  |
| `RealNetEnv`         | `envs/real_net_env.py`                   |    ~28  | Monaco real net (`most.net.xml`)        | 3600s / 5s control = 720 steps  |
| `Large_city_Env`     | `envs/Large_city.py` (gym.Wrapper)       | 33/51/167 | NewYork sub-nets                      | 500 steps × 20s (40s for NY167) |

## Key design choices (worth knowing before reading code)

1. **Per-agent for-loops everywhere.** No agent-batched tensor ops — every agent has its own LSTM/GNN/heads called in Python. Bottleneck for any GPU acceleration attempt.
2. **On-policy buffer with n-step rollouts.** No replay buffer; `n_step=120` for Grid, `n_step=20` for NY.
3. **Heterogeneous agents are handled via padding + masking,** not via per-agent shapes. `unify_act_state_dim=true` adds per-agent projections to a shared dim.
4. **Three independent GNN streams.** Not a single GNN producing one embedding — three separate GNNs over (obs, policy, hidden) whose outputs are concatenated. This is inherited from NeurComm and is the part that explodes the parameter count.
5. **The mask generator is per-agent, not global.** Each agent learns its own ego-graph posterior independently. No global graph distribution.
6. **Gumbel-sigmoid, not Gumbel-softmax.** Each edge has its own Bernoulli; no joint distribution over Z.

## Where to go next

- For a line-by-line of the variational method: [[../06-LineByLine/policies-chunks/04-BayesianGraph-KEY]]
- For the training loop: [[../06-LineByLine/utils-chunks/01-Trainer]]
- For SUMO/TraCI interaction: [[../06-LineByLine/atsc-chunks/01-init-step-reset]]
- For running on a Mac: [[../05-Setup/MacM4-Feasibility]]
- For every bug found: [[../00-Index/Bugs-Catalog]]
