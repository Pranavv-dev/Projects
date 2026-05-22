# `BayesianGraphCMultiAgentPolicy` — Line-by-Line Walkthrough

> [!IMPORTANT] PAPER'S MAIN CONTRIBUTION
> This class (lines **1214–1684** of [[policies.py]]) is the **core algorithmic contribution** of the BayesG NeurIPS 2025 paper. Everything else in the codebase — the GAT layers, the LSTM machinery, the heterogeneous-agent plumbing, the environment harness — exists to support this one class. If a reviewer reads only one file from this repository, it should be this one.
>
> The class operationalises the paper's "Bayesian Graph" idea: instead of fixing the communication adjacency a priori (as [[GraphCMultiAgentPolicy]] does) or learning a deterministic gate (as [[CommNetMultiAgentPolicy]] or [[DIALMultiAgentPolicy]] do), each pair $(i, j)$ along the original neighbor mask gets a **Bernoulli edge variable $Z^{ij}_t$** whose variational posterior $q_\phi(Z^{ij}_t \mid s_i, s_j)$ is parameterised by a 2-layer MLP — the [[GraphMaskGenerator]] at lines 1184–1211. Training maximises an **ELBO** combining the policy-gradient likelihood term, a Bernoulli prior, and the entropy of $q$. See section 8 below for the LaTeX.
>
> Lines covered here: **1214–1684**. Companion class [[GraphMaskGenerator]] at 1184–1211 is quoted where needed.

---

## 1. Class signature and inheritance

```python
class BayesianGraphCMultiAgentPolicy(GraphCMultiAgentPolicy):
    '''
    obs, policy, hidden state has seperate GNN layer
    s_i[GNN(obs_i, obs_neighbors, policy_neighbors, hidden_state_neighbors)]

    s_i[GNN(MLP(obs_i, obs_neighbors), MLP(policy_neighbors), MLP(hidden_state_neighbors))]
    '''
```
(lines 1214–1220)

### Inheritance chain

`BayesianGraphCMultiAgentPolicy` → [[GraphCMultiAgentPolicy]] (line 872) → [[NCMultiAgentPolicy]] (line 193) → `Policy` (base).

The class **does not redefine `__init__` heavily** — it delegates to `super().__init__` and only adds two attributes (`learn_mask`, `mask_history`). What it overrides explicitly:

| Method | Defined? | Purpose |
|---|---|---|
| `__init__` | Yes (1221–1233) | Adds `learn_mask`, `mask_history`, stores `n_heads` redundantly |
| `_init_comm_layer` | Yes (1235–1291) | Builds **three** GNNs (obs / policy / hidden) **plus** a `GraphMaskGenerator` per agent that has neighbors |
| `_get_comm_s` | Yes (1293–1356) | The new per-step routine that samples $Z_t$, builds the masked attention pattern, and passes through the three GNNs |
| `forward` | Yes (1358–1387) | Wraps the comm pass; **importantly toggles `self.eval()` regardless of training state** (see §6) |
| `backward` | Yes (1389–1457) | Replaces the standard A2C loss with the **ELBO** |
| `_update_tensorboard` | Yes (1459–1474) | Adds `prior_loss`, `mask_entropy_loss`, `elbo_loss` to the existing scalars |
| `visualize_masks` | Yes (1476–1683) | Two visualisation modes; threshold-based binarisation |

The class **does not** override `_run_comm_layers`, `_run_actor_heads`, `_run_critic_heads`, `_run_loss`, `_reset`, or `_init_net` — those come from [[NCMultiAgentPolicy]] / [[GraphCMultiAgentPolicy]] and call back into `_get_comm_s` and `_init_comm_layer` polymorphically.

---

## 2. `__init__` — what is new

```python
def __init__(self, n_s, n_a, n_agent, n_step, neighbor_mask, n_fc=64, n_h=64,
             n_s_ls=None, n_a_ls=None, identical=True, unify_act_state_dim=False, n_heads=4, gnn_type='gat', learn_mask=True):
    # Convert unify_act_state_dim to boolean explicitly
    self.unify_act_state_dim = unify_act_state_dim
    self.learn_mask = learn_mask

    # Call parent class constructor
    super().__init__(n_s, n_a, n_agent, n_step, neighbor_mask, n_fc, n_h,
                     n_s_ls, n_a_ls, identical, unify_act_state_dim=self.unify_act_state_dim,
                     n_heads=n_heads, gnn_type=gnn_type)

    self.n_heads = n_heads
    self.mask_history = []  # Store mask history for visualization
```
(lines 1221–1233)

### New parameters / attributes

- **`learn_mask` (default `True`, line 1222 / 1225)** — gates the entire Bayesian behaviour. If `False`, the class degenerates to a vanilla [[GraphCMultiAgentPolicy]]: in `_get_comm_s` the attention mask becomes the all-ones connection matrix (line 1341), and **no `latest_mask_probs` are ever populated**, so the ELBO branch in `backward` (line 1431 `if hasattr(self, 'latest_mask_probs') and i in self.latest_mask_probs:`) is skipped. That makes `learn_mask=False` a useful ablation knob but **silently** turns the loss into plain A2C.
- **`mask_history` (line 1233)** — a plain Python `list` accumulating dicts of the form `{'step': ..., 'masks': {i: np.ndarray, ...}}` (filled in `forward`, lines 1377–1381). Unbounded growth; never trimmed.
- **`n_heads` (line 1232)** — stored a second time even though `super().__init__` already saves it via [[GraphCMultiAgentPolicy]] line 889. Harmless duplication.
- **`mask_generators`** — *not* created in `__init__` itself. It is lazily constructed in `_init_comm_layer` the first time an agent with `n_n > 0` is encountered:
  ```python
  if not hasattr(self, 'mask_generators'):
      self.mask_generators = nn.ModuleList()
  ```
  (lines 1252–1253)
  This is fragile: if agent 0 happens to be the only agent with no neighbors, the `else` branch at line 1278 — `self.mask_generators.append(None)` — would raise `AttributeError` because `self.mask_generators` does not yet exist. In practice all grid layouts have agent 0 with neighbors, so the bug is latent.
- **`latest_mask_probs`** — also lazily created, inside `_get_comm_s` at line 1333:
  ```python
  if not hasattr(self, 'latest_mask_probs'):
      self.latest_mask_probs = {}
  self.latest_mask_probs[i] = prob_mask
  ```
  It is a `dict[int, Tensor]` indexed by agent id and holds the **probabilities** (post-sigmoid, pre-Gumbel) used for the ELBO. Each entry is overwritten on every `_get_comm_s` call, so only the most recent forward pass is in scope at the moment `backward` reads it.
- **`tau` (Gumbel temperature)** — *not* an attribute of this class. It is hardcoded as the default `temp=0.5` argument of [[GraphMaskGenerator.forward]] at line 1190 and never overridden by the calling code at line 1330. There is no schedule, no annealing, no constructor argument.
- **`lambda_` (sparsity prior weight)** — *not* set in `__init__`. It is read with a default in `backward` via `lambda_ = getattr(self, 'lambda_', 0.2)` (line 1435). So unless some external script mutates `policy.lambda_ = ...`, the prior is fixed at **0.2** for the entire run.
- **`threshold`** — also not stored. The eval-time binarisation threshold `0.08` is hardcoded inside [[GraphMaskGenerator.forward]] at line 1209 and again in `visualize_masks` at line 1504. A second threshold `0.1` appears twice more in `visualize_masks` (lines 1586, 1601, 1635) — three magic numbers for the same conceptual quantity. See §8.

The constructor does **not** call `_init_comm_layer` directly; that is done inside `super().__init__` → `_init_net` from [[NCMultiAgentPolicy]], which iterates over agents and calls the overridden `_init_comm_layer` polymorphically.

---

## `_init_comm_layer` — building the per-agent network stack

```python
def _init_comm_layer(self, n_n, n_ns, n_na, i_agent):
    '''Initialize communication layers with specified GNN type'''
    n_lstm_in = 3 * self.n_fc
```
(lines 1235–1237)

The `3 * self.n_fc` is structural: the LSTM consumes the concatenation of three GNN outputs (obs / policy / hidden), each of width `n_fc=64`, so `n_lstm_in=192`.

### Obs GNN — always built
```python
if self.identical:
    obs_input_dim = self.n_s
elif self.unify_act_state_dim:
    obs_input_dim = self.convert_state_shape
else:
    obs_input_dim = self.n_s_ls[i_agent]

gnn_obs = self._init_gnn_layer(obs_input_dim, self.n_fc)
self.gnn_obs.append(gnn_obs)
```
(lines 1240–1248)

`_init_gnn_layer` is inherited and dispatches on `self.gnn_type` (`'gat'`, `'gcn'`, etc.). The obs GNN exists for **every** agent, regardless of whether they have neighbors — important because an isolated agent still needs to encode its own observation through a GAT/GCN with `attn_mask = I`.

### `GraphMaskGenerator` — only for agents with neighbors
```python
if n_n:
    if not hasattr(self, 'mask_generators'):
        self.mask_generators = nn.ModuleList()

    # Each generator takes s_i and s_j as input (concatenated)
    mask_gen = GraphMaskGenerator(input_dim=2 * obs_input_dim, hidden_dim=self.n_fc)
    self.mask_generators.append(mask_gen)
```
(lines 1251–1257)

Key facts:
- `input_dim = 2 * obs_input_dim` because [[GraphMaskGenerator.forward]] concatenates `s_i` and `s_j` along the last dim (line 1198).
- `hidden_dim = self.n_fc = 64`, so the mask generator is a tiny `Linear(2d, 64) → ReLU → Linear(64, 1)` network.
- **One generator is appended per agent that has neighbors.** This means `mask_generators[k]` corresponds to the *k*-th agent (in insertion order) that has neighbors — i.e. **the same index as `gnn_obs`/`gnn_policy`/`gnn_hidden`** because they are all appended in the same `for i_agent in range(n_agent)` loop and all use the same branch. For isolated agents, all four lists get `None` via the `else` branch below.

### Policy / Hidden GNNs — also gated by `n_n > 0`
```python
    # === Policy GNN ===
    if self.identical:
        policy_input_dim = self.n_a
    elif self.unify_act_state_dim:
        policy_input_dim = self.convert_action_shape
    else:
        policy_input_dim = self.n_a
    gnn_policy = self._init_gnn_layer(policy_input_dim, self.n_fc)
    self.gnn_policy.append(gnn_policy)

    # === Hidden GNN ===
    gnn_hidden = self._init_gnn_layer(self.n_h, self.n_fc)
    self.gnn_hidden.append(gnn_hidden)

    # === LSTM ===
    self.fc_x_layers_for_no_neighbor.append(None)
    lstm_layer = nn.LSTMCell(n_lstm_in, self.n_h)
```
(lines 1259–1275)

> [!NOTE]
> The `elif self.unify_act_state_dim` branch on line 1262 sets `policy_input_dim = self.convert_action_shape`, but the `else` on line 1265 falls back to `self.n_a` — *not* `self.n_a_ls[i_agent]`. This means for **heterogeneous, non-unified** agents the policy GNN's input dim is wrong (uses the homogeneous default `n_a`) while `_get_comm_s` actually feeds `p_i` of dim `sum(self.na_ls_ls[i][j])`. This is a latent shape bug, but never hit in the BayesG paper's experiments because they always run with `identical=True` on the standard ATSC grids.

`lstm_layer = nn.LSTMCell(n_lstm_in, self.n_h)` confirms the `3 * n_fc → n_h` projection.

### Isolated-agent branch
```python
else:
    self.mask_generators.append(None)
    self.gnn_policy.append(None)
    self.gnn_hidden.append(None)

    if self.identical:
        no_neighbor_x_layer = nn.Linear(self.n_s, self.n_fc)
    else:
        no_neighbor_x_layer = nn.Linear(self.n_s_ls[i_agent], self.n_fc)
    self.fc_x_layers_for_no_neighbor.append(no_neighbor_x_layer)

    lstm_layer = nn.LSTMCell(self.n_fc, self.n_h)

init_layer(lstm_layer, 'lstm')
self.lstm_layers.append(lstm_layer)
```
(lines 1277–1291)

For an agent with no neighbors, the input to the LSTM is just `n_fc` (the single obs-only path), not `3 * n_fc`. The `mask_generators[i]` slot is `None`, which is precisely what `_get_comm_s` checks against at line 1324: `if self.learn_mask and self.mask_generators[i] is not None:`.

---

## `_get_comm_s` — the per-agent forward pipeline

This is the heart of the contribution. It is called from `_run_comm_layers` (inherited) once per agent per timestep.

```python
def _get_comm_s(self, i, n_n, x, h, p):
    # Get neighbor indices
    js = torch.nonzero(self.neighbor_mask[i]).squeeze(1)
    h_i = h[js]   # [n_n, hidden_dim]
```
(lines 1293–1296)

`x`, `h`, `p` are global stacks across all `n_agent` agents at the current timestep:
- `x`: `[n_agent, n_s]` observations
- `h`: `[n_agent, n_h]` LSTM hidden states from the previous step
- `p`: `[n_agent, n_a]` action distributions (the "fingerprint" `fp`)

`js` is the index vector of `i`'s neighbors per the **static** `neighbor_mask` — the Bayesian mask only prunes *within* this static set; it cannot create edges that weren't there to start with.

### Step 1 — gather neighbor features
```python
# === Feature Gathering ===
if self.identical:
    x_i = x[i].unsqueeze(0)       # shape: [1, obs_dim]
    nx_i = x[js]                  # [n_n, obs_dim]
    p_i = p[js]                   # [n_n, action_dim]
```
(lines 1298–1302)

In the identical-agent case (the one used in the paper), `x_i` is the self observation `[1, n_s]` and `nx_i` is the stack of neighbor observations `[n_n, n_s]`. `p_i` is the neighbor fingerprints (probabilities over actions from the previous policy).

The heterogeneous branch (lines 1303–1319) does the same thing with narrowing/unify-linears; I won't quote it because the paper does not use it.

### Step 2 — initialize attention mask
```python
# === Create attention mask ===
attn_mask = torch.zeros(n_n + 1, n_n + 1, device=self.device)
```
(line 1322)

The mask matrix has shape `[n_n+1, n_n+1]` because the GNN operates on the *augmented* neighborhood: self at index 0 plus `n_n` neighbors at indices `1..n_n`.

### Step 3 — sample $Z_t$ from the mask generator
```python
if self.learn_mask and self.mask_generators[i] is not None:
    # === Sample Z_t from GraphMaskGenerator ===
    mask_gen = self.mask_generators[i]
    s_i_state = x_i                        # [1, d]
    s_j_states = nx_i.view(n_n, -1)       # [n_n, d]

    z_mask, prob_mask = mask_gen(s_i_state, s_j_states, training=self.training)  # [n_n], [n_n]

    # Save for ELBO loss
    if not hasattr(self, 'latest_mask_probs'):
        self.latest_mask_probs = {}
    self.latest_mask_probs[i] = prob_mask  # detach if needed
```
(lines 1324–1335)

What [[GraphMaskGenerator.forward]] does internally (lines 1184–1211):

```python
x = torch.cat([s_i, s_j], dim=-1)             # [n_n, 2d]
logits = self.fc2(F.relu(self.fc1(x)))        # [n_n, 1]
prob   = torch.sigmoid(logits)

if training:
    u = torch.rand_like(prob)
    gumbel_noise = -torch.log(-torch.log(u + 1e-20) + 1e-20)
    z = torch.sigmoid((logits + gumbel_noise) / temp)
    return z.squeeze(-1), prob.squeeze(-1)
else:
    mask = (prob > 0.08).float()
    # NOTE: control reaches the unindented return below
return mask.squeeze(-1), prob.squeeze(-1)
```

So during training the returned `z` is the **Binary Concrete** (Gumbel-sigmoid) relaxation with `temp=0.5`. During eval it is a hard `0/1` mask thresholded at `0.08`. Both modes also return the soft `prob` (the sigmoid output, **without** noise), which is what the ELBO uses.

> [!WARNING] Indentation bug in `GraphMaskGenerator`
> The `mask = (prob > 0.08).float()` line at 1209 is inside the `else:` block, but the `return mask.squeeze(-1), prob.squeeze(-1)` at 1211 is **outside** the `if/else`. Therefore the train branch already returned at line 1206; the unindented return at 1211 is only reached from the eval branch, where `mask` was bound on line 1209. So the eval path works, but a reader would have to trace this carefully — and a future refactor that introduces another path would silently hit an `UnboundLocalError`.

The crucial side effect at line 1335: `self.latest_mask_probs[i] = prob_mask` — note this is the **soft probability tensor with grad**, *not* detached. The ELBO loss flows back through this exact tensor in `backward`.

### Step 4 — write the mask into the attention pattern
```python
    # Apply learned mask
    attn_mask[0, 1:] = attn_mask[1:, 0] = z_mask
else:
    # Use full mask (all connections enabled)
    attn_mask[0, 1:] = attn_mask[1:, 0] = 1

# Add self-loops to make sure nodes always attend to themselves
attn_mask += torch.eye(n_n + 1, device=self.device)
```
(lines 1337–1344)

Two things to notice:

1. **Star topology only.** The mask is written to row 0 / column 0 only. Edges between neighbors `(1..n_n)` are *never* unmasked — i.e. neighbor `j` cannot attend to neighbor `k` even if they would in the underlying global graph. So the GNN here is effectively a **per-agent star** with `i` at the centre. This is fine for a single GNN layer (you only aggregate one hop from `i`'s perspective), but it does mean the term "GNN" is generous: with one layer of GAT and a star mask it's basically an attention pooling over `{x_i, x_{j_1}, ..., x_{j_n}}`.
2. **Self-loop added unconditionally.** The `torch.eye` addition on line 1344 makes the diagonal `1` (or `1 + previous_diag` if it was nonzero — but it wasn't, because the writes on lines 1338/1341 only touch row/col 0 off-diagonals). So the diagonal becomes exactly `1`, the off-diagonals in row/col 0 carry the Gumbel-sampled `z_mask`, and everything else stays zero.

### Step 5 — concatenate and pass through the three GNNs
```python
# === Combine inputs for GNN ===
x_combined = torch.cat([x_i, nx_i.reshape(n_n, -1)], dim=0)
p_combined = torch.cat([p[i].unsqueeze(0), p_i.reshape(n_n, -1)], dim=0)
h_combined = torch.cat([h[i].unsqueeze(0), h_i.reshape(n_n, -1)], dim=0)

# === GNN feature aggregation ===
return torch.cat([
    self.gnn_obs[i](x_combined, attn_mask)[0].unsqueeze(0),
    self.gnn_policy[i](p_combined, attn_mask)[0].unsqueeze(0),
    self.gnn_hidden[i](h_combined, attn_mask)[0].unsqueeze(0)
], dim=1)
```
(lines 1346–1356)

Each `gnn_*[i](*, attn_mask)` returns a `[n_n+1, n_fc]` tensor of per-node features; `[0]` selects node 0 (the self node), and `.unsqueeze(0)` makes it `[1, n_fc]`. The three `[1, n_fc]` tensors are concatenated along `dim=1` into a `[1, 3 * n_fc] = [1, 192]` row, which is what the LSTMCell expects.

> [!NOTE] Same mask applied to all three GNNs
> The identical `attn_mask` is used for `gnn_obs`, `gnn_policy`, and `gnn_hidden`. The model does not learn a separate edge variable per GNN modality — communication, action sharing, and hidden-state sharing are all gated by one Bernoulli per edge per timestep. This is consistent with the paper's narrative of "should agent $i$ talk to agent $j$ at all this step?", but it is also a modelling choice you could critique.

---

## `forward` — single-step inference

```python
def forward(self, ob, done, fp, action=None, out_type='p'):
    ob = torch.from_numpy(np.expand_dims(ob, axis=0)).float()
    done = torch.from_numpy(np.expand_dims(done, axis=0)).float()
    fp = torch.from_numpy(np.expand_dims(fp, axis=0)).float()

    # Temporarily switch to eval mode for masking
    training_backup = self.training
    self.eval()

    h, new_states = self._run_comm_layers(ob, done, fp, self.states_fw)

    # Restore training flag
    if training_backup:
        self.train()

    self.states_fw = new_states.detach()

     # Store mask history for visualization
    if hasattr(self, 'latest_mask_probs'):
        self.mask_history.append({
            'step': getattr(self, 'current_step', 0),
            'masks': {i: p.detach().cpu().numpy()
                     for i, p in self.latest_mask_probs.items()}
        })

    if out_type.startswith('p'):
        return self._run_actor_heads(h, detach=True)
    else:
        action = torch.from_numpy(np.expand_dims(action, axis=1)).long()
        return self._run_critic_heads(h, action, detach=True)
```
(lines 1358–1387)

This is the inference path used by the trainer to roll out the policy. Critical observations:

- **Lines 1364–1365 force `self.eval()` for the rollout.** Then **line 1370–1371 restores `train()` only if `training_backup` was True**. This means during rollout the [[GraphMaskGenerator]] takes the `else` branch (hard threshold at 0.08), even if the model is otherwise in training mode. So at every collected transition the action is chosen using the *deterministic* mask, not the Gumbel-sampled one. That is a deliberate design decision: it gives a stable, low-variance behaviour policy.
- **`states_fw.detach()` (line 1373)** — the rollout LSTM state is detached so it doesn't carry gradient between optimiser steps. Standard A2C practice.
- **`mask_history.append(...)` (lines 1376–1381)** — every forward call appends one dict. `current_step` is read with `getattr(..., 0)` because the policy itself never sets `current_step`; the caller is expected to set it externally before calling `forward`. If they forget, all entries get `step=0`. Memory leak risk: the list grows forever and stores one `np.ndarray` per neighbored agent per environment step.
- **`out_type.startswith('p')` (line 1383)** — convention from the parent class: `'p'` → actor logits, anything else → critic value.

---

## `backward` — ELBO loss, line by line

```python
def backward(self, obs, fps, acts, dones, Rs, Advs,
         e_coef, v_coef, summary_writer=None, global_step=None):
    obs = torch.from_numpy(obs).float().transpose(0, 1).to(self.device)
    dones = torch.from_numpy(dones).float().to(self.device)
    fps = torch.from_numpy(fps).float().transpose(0, 1).to(self.device)
    acts = torch.from_numpy(acts).long().to(self.device)
```
(lines 1389–1394)

The `.transpose(0, 1)` swaps the batch axis from `[T, n_agent, dim]` (collected from the rollout) to `[n_agent, T, dim]` which is what `_run_comm_layers` expects.

```python
    # === Forward pass through comm layers ===
    hs, new_states = self._run_comm_layers(obs, dones, fps, self.states_bw)
    self.states_bw = new_states.detach()
```
(lines 1397–1398)

`_run_comm_layers` runs the inherited TBPTT through the LSTM. **It calls `_get_comm_s` once per (timestep, agent)** — and `_get_comm_s` overwrites `self.latest_mask_probs[i]` every time. So after the loop completes, `self.latest_mask_probs[i]` holds the **last timestep's** probabilities for agent `i`. The ELBO below is therefore computed only on the last step of the rollout, not summed across the rollout. (This is consistent with the SGD interpretation: one sample of $Z_t$ per minibatch.)

```python
    # Actor and critic outputs
    ps = self._run_actor_heads(hs)
    vs = self._run_critic_heads(hs, acts)

    # Rewards and advantages
    Rs = torch.from_numpy(Rs).float().to(self.device)
    Advs = torch.from_numpy(Advs).float().to(self.device)

    # Normalize advantages
    Advs = (Advs - Advs.mean()) / (Advs.std() + 1e-8)
```
(lines 1400–1409)

Per-batch advantage normalisation with the standard `1e-8` denominator guard. `1e-8` is the first of several numerical guards in this method.

```python
    # === Loss components ===
    likelihood_term = 0
    value_term = 0
    entropy_term = 0
    prior_term = 0
    mask_entropy_term = 0
```
(lines 1412–1417)

Five scalar accumulators — initialised to Python `0`, *not* `torch.tensor(0.)`. Because they are added to by `+=` with tensors, they get promoted to tensors on first addition. If an agent has no neighbors (so its block at line 1431 is skipped) and *no other agent* has neighbors either, `prior_term` and `mask_entropy_term` stay as Python ints — and later `-(likelihood_term + prior_term - mask_entropy_term)` will still work because the likelihood term is a tensor by then. So this is safe but ugly.

```python
    for i in range(self.n_agent):
        actor_dist = torch.distributions.categorical.Categorical(logits=ps[i])

        policy_loss_i, value_loss_i, entropy_loss_i = \
            self._run_loss(actor_dist, e_coef, v_coef, vs[i], acts[i], Rs[i], Advs[i])

        # Likelihood = - policy loss
        likelihood_term += -policy_loss_i
        value_term += value_loss_i
        entropy_term += entropy_loss_i
```
(lines 1419–1428)

`_run_loss` is inherited (defined on the base `Policy`). It returns a tuple `(policy_loss, value_loss, entropy_loss)` where `policy_loss = -E[log π(a|s) * A]` (positive when bad, by sign convention). The line `likelihood_term += -policy_loss_i` therefore makes `likelihood_term` the **expected log-likelihood weighted by advantage**, i.e. the surrogate REINFORCE objective — and this is the term that plays the role of $\mathbb{E}_q[\log p_\theta(a \mid s, Z)]$ in the ELBO. The value and entropy losses are accumulated separately and added at the *end* (line 1452), outside the ELBO proper.

### ELBO-specific terms
```python
        # === ELBO-specific terms ===
        if hasattr(self, 'latest_mask_probs') and i in self.latest_mask_probs:
            mask_probs = self.latest_mask_probs[i]

            # Bernoulli prior log-prob: log p(Z_t)
            lambda_ = getattr(self, 'lambda_', 0.2)
            prior_term += (lambda_ * torch.log(mask_probs + 1e-8) +
                        (1 - lambda_) * torch.log(1 - mask_probs + 1e-8)).sum()

            # Entropy of q(Z_t; φ): -log q
            mask_entropy = - (mask_probs * torch.log(mask_probs + 1e-8) +
                            (1 - mask_probs) * torch.log(1 - mask_probs + 1e-8))
            mask_entropy_term += mask_entropy.sum()
```
(lines 1430–1442)

Let $\pi_{ij} := q_\phi(Z^{ij} = 1 \mid s_i, s_j)$ be `mask_probs[idx]`.

- **`prior_term`** is $\sum_{ij} [\lambda \log \pi_{ij} + (1-\lambda) \log(1-\pi_{ij})]$. This is **not** the log of a Bernoulli prior in the usual sense (which would be $Z \log p + (1-Z) \log(1-p)$ where $p$ is the prior mean and $Z$ is the sample). Instead it is $\mathbb{E}_q[\log p(Z)]$ where the prior is Bernoulli($\lambda$) but the expectation has been taken **analytically** using $\mathbb{E}_q[Z]=\pi_{ij}$:
  $$\mathbb{E}_q[\log p(Z)] = \pi_{ij} \log \lambda + (1-\pi_{ij}) \log(1-\lambda).$$
  But the code uses $\lambda \log \pi_{ij} + (1-\lambda) \log(1-\pi_{ij})$ — i.e. **swapped roles**. This is the cross-entropy from the prior to the posterior, not $\mathbb{E}_q[\log p(Z)]$. The two only coincide when $\lambda = 0.5$. With the default `lambda_=0.2` it is a sparsity-encouraging cross-entropy term: large when $\pi_{ij}$ is far from `0.2`. The paper's text and the code disagree on which quantity is being computed; treat this as something to flag.
- **`mask_entropy_term`** is $H[q] = -\sum_{ij}[\pi_{ij}\log\pi_{ij} + (1-\pi_{ij})\log(1-\pi_{ij})]$. Standard Bernoulli entropy. The leading minus is on line 1440 ( `mask_entropy = -(... + ...)`).
- **`+1e-8`** appears four times in lines 1436–1441 — numerical guard so that `log(0)` and `log(1)` are well-behaved when $\pi$ saturates.

### Final ELBO and total loss
```python
    # === Final ELBO loss ===
    self.policy_loss = -likelihood_term
    self.value_loss = value_term
    self.entropy_loss = entropy_term
    self.prior_loss = prior_term
    self.mask_entropy_loss = mask_entropy_term
    self.elbo_loss = -(likelihood_term + prior_term - mask_entropy_term)

    self.loss = self.elbo_loss + self.value_loss + self.entropy_loss
    self.loss.backward()
```
(lines 1444–1453)

So the optimised loss is:
$$\mathcal{L} = -(\underbrace{L_{\text{lik}}}_{\text{REINFORCE}} + \underbrace{L_{\text{prior}}}_{\text{Bernoulli prior cross-entropy}} - \underbrace{H[q]}_{\text{mask entropy}}) + L_{\text{value}} + L_{\text{entropy}}$$

Mapping to the standard ELBO $\mathcal{F} = \mathbb{E}_q[\log p(a|s,Z)] - \mathrm{KL}(q\|p) = \mathbb{E}_q[\log p(a|s,Z)] + \mathbb{E}_q[\log p(Z)] + H[q]$:

| Code term | Sign in `elbo_loss` (which is negated) | ELBO sign | Role |
|---|---|---|---|
| `likelihood_term` | `+` inside `-( )` ⇒ minimised when negative | `+` $\mathbb{E}_q[\log p]$ | matches |
| `prior_term` | `+` inside `-( )` | `+` $\mathbb{E}_q[\log p(Z)]$ | matches **in sign** (modulo the $\lambda$ vs $\pi$ swap noted above) |
| `mask_entropy_term` | `-` inside `-( )` ⇒ becomes `+` after the outer negation | `+` $H[q]$ | sign is **wrong**: maximising the ELBO requires *adding* $H[q]$, but the code subtracts it inside the negated expression, so the optimiser is asked to **minimise** entropy. That makes the masks more peaked over time — possibly desirable as an annealing-by-overconfidence, but contradicts the paper's stated ELBO. Worth verifying against the paper's equation.

> [!CAUTION] Possible sign / formulation discrepancy
> Lines 1450 reads `self.elbo_loss = -(likelihood_term + prior_term - mask_entropy_term)`. After `self.loss.backward()` minimises this, the gradients act as if maximising `likelihood + prior - entropy`. The textbook ELBO maximises `likelihood + log p(Z) + entropy(q)` (because $-\mathrm{KL} = \mathbb{E}_q[\log p(Z)] - \mathbb{E}_q[\log q(Z)] = \mathbb{E}_q[\log p(Z)] + H[q]$). So unless the paper defines its ELBO differently, this is at minimum a sign quirk: the model is being trained with a **negative** entropy bonus on the mask distribution. Combine with the `lambda_/π` swap above and you have two non-obvious deviations from the canonical ELBO.

`self.loss.backward()` (line 1453) calls PyTorch's autograd; nothing fancy. The optimizer step is performed *outside* this method by the calling trainer (it lives in [[trainer.py]]).

```python
    # Logging (optional)
    if summary_writer is not None:
        self._update_tensorboard(summary_writer, global_step)
```
(lines 1456–1457)

---

## `_update_tensorboard` — observability

```python
def _update_tensorboard(self, summary_writer, global_step):
    # monitor training
    summary_writer.add_scalar('loss/{}_entropy_loss'.format(self.name), self.entropy_loss, global_step=global_step)
    summary_writer.add_scalar('loss/{}_policy_loss'.format(self.name), self.policy_loss, global_step=global_step)
    summary_writer.add_scalar('loss/{}_value_loss'.format(self.name), self.value_loss, global_step=global_step)
    summary_writer.add_scalar('loss/{}_total_loss'.format(self.name), self.loss, global_step=global_step)
    summary_writer.add_scalar('loss/{}_prior_loss'.format(self.name), self.prior_loss, global_step=global_step)
    summary_writer.add_scalar('loss/{}_mask_entropy_loss'.format(self.name), self.mask_entropy_loss, global_step=global_step)
    summary_writer.add_scalar('loss/{}_elbo_loss'.format(self.name), self.elbo_loss, global_step=global_step)
```
(lines 1459–1474)

Seven scalar streams. `self.name` is set to `'graph_nc'` by the parent constructor (line 900). The names are pure strings; no namespacing for the Bayesian sub-class — i.e. an ablation run vs. the Bayesian run will show up in TensorBoard under the same prefix unless the user provides a different `summary_writer` directory. The values are leaf tensors with grad attached, but `add_scalar` calls `.item()` internally, so this is fine.

---

## 6. Eval-mode behaviour — when does the threshold fire?

There are two distinct "eval-mode" pathways. They use different thresholds.

### 6a. Rollout `forward` (line 1364–1365)
```python
training_backup = self.training
self.eval()
```
This forces `self.training=False` for the duration of `_run_comm_layers` ⇒ `_get_comm_s` ⇒ `mask_gen(...)`. Inside [[GraphMaskGenerator.forward]] (line 1190) the `training` argument is `self.training` (line 1330), so it picks up `False`. That means:
```python
else:
    # During evaluation, use deterministic threshold
    mask = (prob > 0.08).float()
```
(line 1208–1209)

is taken. The threshold is **`0.08`**, hardcoded. After `_run_comm_layers` returns, line 1370–1371 restores `train()` if we were originally training. So thresholding only ever applies at *rollout* time; the gradient step on `backward` always sees the **Gumbel-sigmoid** `z_mask` because `backward` does not toggle eval.

### 6b. `backward` (no eval toggle)
At the top of `backward` there is no `self.eval()` call. The `_run_comm_layers` invocation at line 1397 therefore uses `self.training` as-is, which is `True` (since the trainer puts the model in train mode before calling `backward`). Inside the generator the `if training:` branch (line 1202) is taken, the Gumbel noise is sampled, and `z_mask` is the soft Concrete sample. The ELBO uses `prob_mask` (the noise-free sigmoid), which is stored at line 1335.

### 6c. `visualize_masks` (lines 1504, 1586, 1601, 1635)
Visualisation re-applies its own threshold of either `0.08` (line 1504, in the whole-graph mode) or `0.1` (lines 1586/1601/1635, in the per-agent mode). These thresholds do *not* affect training — they only control which edges are drawn in the matplotlib figure. The inconsistency is purely cosmetic but real: an edge with `prob = 0.09` will appear in the whole-graph plot but not in the per-agent plot.

---

## 7. `visualize_masks(step, save_path, draw_whole)`

The method has two completely independent code paths controlled by `draw_whole`. Both share the bookkeeping shell:

```python
def visualize_masks(self, step, save_path=None, draw_whole=False):
    if not hasattr(self, 'latest_mask_probs'):
        logging.warning("No mask probabilities available for visualization")
        return

    import networkx as nx
    import os
```
(lines 1476–1488)

`networkx` is **imported inside the function** — not at module top — presumably to avoid the import cost when visualisation is disabled. `os` is also re-imported here even though `os` is almost certainly already imported at module top. `plt` and `sns` are *not* imported here, so they must come from the module's top-level imports.

### 7a. `draw_whole=True` — whole-graph mode (lines 1490–1556)

```python
if draw_whole:
    # Create a single figure for the whole graph
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Store all selected edges
    selected_edges = []
    edge_probs = []

    # Collect all selected edges and their probabilities
    for i in range(self.n_agent):
        if i in self.latest_mask_probs:
            js = torch.nonzero(self.neighbor_mask[i]).squeeze(1)
            probs = self.latest_mask_probs[i].detach().cpu().numpy()
            for idx, j in enumerate(js):
                if probs[idx] > 0.08:  # Using the same threshold as in the model
                    selected_edges.append((i, j.item()))
                    edge_probs.append(float(probs[idx]))  # Convert to Python float
```
(lines 1490–1506)

Iterates over agents, pulls each agent's stored `prob_mask`, thresholds at `0.08`, and stores `(i, j)` pairs. The threshold matches the eval-mode threshold from [[GraphMaskGenerator]] — that's the only place the magic number `0.08` is consistent across files.

```python
    # Create the adjacency matrix
    adj_matrix = torch.zeros((self.n_agent, self.n_agent), dtype=torch.float32)
    for (i, j), prob in zip(selected_edges, edge_probs):
        adj_matrix[i, j] = float(prob)  # Ensure float type
        adj_matrix[j, i] = float(prob)  # Make it symmetric
```
(lines 1508–1512)

The adjacency is **made symmetric here** even though in the actual model agent `i`'s mask for edge `(i,j)` and agent `j`'s mask for edge `(j,i)` are sampled independently. So if `i`→`j` had `prob=0.7` and `j`→`i` had `prob=0.05` (below threshold), the plotted `adj_matrix[i,j]=0.7` *and* `adj_matrix[j,i]=0.7` — overwriting whatever `j` had decided. This is misleading: the visualisation suggests a symmetric graph but the underlying model is asymmetric.

```python
    # Plot adjacency matrix heatmap
    sns.heatmap(adj_matrix.cpu().numpy(), ax=ax1,
               cmap='YlOrRd',  # Yellow-Orange-Red colormap to match edge color
               vmin=0, vmax=0.5,
               cbar_kws={'label': 'Connection Probability'})
    # Adjust colorbar label font size after creation
    ax1.collections[0].colorbar.ax.set_ylabel('Connection Probability', fontsize=16)
    ax1.set_title('Whole Graph Adjacency Matrix', fontsize=18)
```
(lines 1514–1521)

`vmax=0.5` — probabilities above `0.5` get saturated to the same colour, so a `prob=0.9` edge is indistinguishable from a `prob=0.5` edge in the heatmap.

```python
    # Plot grid network
    G = nx.from_numpy_array(adj_matrix.cpu().numpy())

    # Calculate grid dimensions
    grid_size = int(np.sqrt(self.n_agent))
    if grid_size * grid_size != self.n_agent:
        grid_size = int(np.ceil(np.sqrt(self.n_agent)))

    # Create a proper grid layout
    pos = {}
    for j in range(self.n_agent):
        row = j // grid_size
        col = j % grid_size
        pos[j] = (col, -row)  # x=column, y=-row (negative to have 0,0 at top-left)
```
(lines 1523–1536)

Layout is a literal grid — agent `j` is placed at column `j % grid_size`, row `j // grid_size`. The negative `-row` flips the y-axis so agent 0 is at the top-left. This assumes the agent indexing matches the physical grid layout, which is only true for the [[grid env]] benchmarks; on Monaco or Manhattan it will produce a misleading rectangular layout.

```python
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax2,
                         node_color='#1F77B4',
                         node_size=800)

    # Draw edges with varying widths based on probability
    for (i, j), prob in zip(selected_edges, edge_probs):
        nx.draw_networkx_edges(G, pos, ax=ax2,
                             edgelist=[(i, j)],
                             edge_color='#E46D4C',
                             width=float(prob) * 10)  # Ensure float type
```
(lines 1538–1548)

Edge width is `prob * 10`. With `prob ∈ (0.08, 1.0]`, widths are in `(0.8, 10.0]`. This is a sensible visual scale.

```python
    # Add labels with font customization
    nx.draw_networkx_labels(G, pos, ax=ax2, font_size=18, font_color='white')
    ax2.set_title('Whole Grid Network\n(Edge width indicates probability)', fontsize=18)
    ax2.axis('off')

    plt.suptitle(f'Whole Graph Visualization at Step {step}', fontsize=18)
    plt.tight_layout()
```
(lines 1550–1556)

Standard matplotlib styling.

### 7b. `draw_whole=False` — per-agent mode (lines 1558–1655)

```python
else:
    # Original detailed per-agent visualization
    n_agents = self.n_agent
    n_cols = 4  # Local mask probs, Local adj, Global adj, Grid network
    fig, axes = plt.subplots(n_agents, n_cols, figsize=(6*n_cols, 4*n_agents))
    if n_agents == 1:
        axes = axes.reshape(1, -1)
```
(lines 1558–1564)

`figsize=(6*n_cols, 4*n_agents) = (24, 4*n_agents)`. For `n_agents=25` (the 5×5 grid), the figure is `24 × 100` inches — that's huge but matplotlib will scale on render.

For each agent `i` with stored probs:

```python
    if i in self.latest_mask_probs:
        # Get 1st hop neighbors
        js = torch.nonzero(self.neighbor_mask[i]).squeeze(1)
        n_neighbors = len(js)

        # 1. Plot local mask probabilities
        probs = self.latest_mask_probs[i].detach().cpu().numpy()
        logging.info(f"Agent {i} mask probabilities: {probs}")

        sns.barplot(x=range(n_neighbors), y=probs, ax=axes[i, 0])
        axes[i, 0].set_title(f'Agent {i} Local Mask Probabilities')
        axes[i, 0].set_xlabel('Neighbor Index')
        axes[i, 0].set_ylabel('Probability')
```
(lines 1566–1579)

Column 0: barplot of the `n_neighbors` Bernoulli probabilities for agent `i`. Also logs them (so per-step every-agent logging in INFO mode — noisy).

```python
        # 2. Plot local adjacency matrix (1st hop)
        local_mask = torch.zeros((n_neighbors + 1, n_neighbors + 1), device=self.device, dtype=torch.float32)
        # Add self-connection
        local_mask[0, 0] = 1.0
        # Add connections to neighbors based on mask probabilities
        local_mask[0, 1:] = torch.tensor((probs > 0.1).astype(np.float32), device=self.device)
        local_mask[1:, 0] = local_mask[0, 1:]  # Make it symmetric
```
(lines 1581–1587)

Column 1: a `[n_n+1, n_n+1]` binary local adjacency at threshold **`0.1`** (not 0.08 like the whole-graph mode).

```python
        # Create labels for the local adjacency matrix
        labels = ['self'] + [f'n{j.item()}' for j in js]

        sns.heatmap(local_mask.cpu().numpy(), ax=axes[i, 1],
                   cmap='YlOrRd', vmin=0, vmax=1,
                   xticklabels=labels, yticklabels=labels)
        axes[i, 1].set_title(f'Agent {i} Local Adjacency')
```
(lines 1589–1595)

```python
        # 3. Plot global adjacency matrix with highlighted local connections
        global_mask = self.neighbor_mask.clone().float()
        # Highlight local connections in a different color
        highlight_mask = torch.zeros_like(global_mask)
        highlight_mask[i, js] = torch.tensor((probs > 0.1).astype(np.float32), device=self.device)

        # Create a custom colormap that highlights local connections
        sns.heatmap(global_mask.cpu().numpy(), ax=axes[i, 2],
                   cmap='Blues', vmin=0, vmax=1, alpha=0.5)
        sns.heatmap(highlight_mask.cpu().numpy(), ax=axes[i, 2],
                   cmap='Reds', vmin=0, vmax=1, alpha=0.7)
        axes[i, 2].set_title(f'Agent {i} Global Adjacency\n(Red=Selected Local, Blue=Global)')
```
(lines 1597–1608)

Column 2: overlay of two heatmaps — blue for the static neighbor mask, red (with alpha) for agent `i`'s currently-selected edges, again at threshold `0.1`. The double `sns.heatmap` call into the same `ax` produces a layered colourmap; this looks visually busy in practice.

```python
        # 4. Plot grid network visualization
        G = nx.from_numpy_array(global_mask.cpu().numpy())

        # Calculate grid dimensions
        grid_size = int(np.sqrt(n_agents))
        if grid_size * grid_size != n_agents:
            grid_size = int(np.ceil(np.sqrt(n_agents)))

        # Create a proper grid layout
        pos = {}
        for j in range(n_agents):
            row = j // grid_size
            col = j % grid_size
            pos[j] = (col, -row)
```
(lines 1610–1623)

Column 3: per-agent grid network. Identical layout logic as `draw_whole=True`.

```python
        # Draw the base grid
        nx.draw_networkx_nodes(G, pos, ax=axes[i, 3],
                             node_color='lightblue',
                             node_size=500)
        nx.draw_networkx_edges(G, pos, ax=axes[i, 3],
                             edge_color='gray',
                             alpha=0.5)

        # Highlight selected local connections with edge weights
        for idx, j in enumerate(js):
            if probs[idx] > 0.1:  # Using the same threshold as in the model
                nx.draw_networkx_edges(
                    G, pos, ax=axes[i, 3],
                    edgelist=[(i, j.item())],
                    edge_color='red',
                    width=float(probs[idx]) * 5  # Ensure float type
                )

        # Highlight the current agent
        nx.draw_networkx_nodes(G, pos, ax=axes[i, 3],
                             nodelist=[i],
                             node_color='red',
                             node_size=500)
```
(lines 1625–1647)

Width scaling is `prob * 5` here (vs `prob * 10` in `draw_whole`). Yet another inconsistency.

The trailing block applies labels, axis-off, and titles, then `plt.suptitle` and `plt.tight_layout`.

### Save / show block (shared)
```python
if save_path:
    try:
        # Ensure the directory exists
        os.makedirs(save_path, exist_ok=True)

        # Create the full file path
        save_file = os.path.join(save_path, f'mask_step_{step}.png')

        # Save the figure
        plt.savefig(save_file, dpi=500, bbox_inches='tight')
        logging.info(f"Successfully saved visualization to {save_file}")

        # Verify the file was created
        if os.path.exists(save_file):
            file_size = os.path.getsize(save_file)
            logging.info(f"File created successfully. Size: {file_size} bytes")
        else:
            logging.error(f"File was not created at {save_file}")

    except Exception as e:
        logging.error(f"Error saving visualization: {str(e)}")
        logging.error(f"Save path: {save_path}")
        logging.error(f"Current working directory: {os.getcwd()}")
    finally:
        plt.close()
else:
    plt.show()
```
(lines 1657–1683)

`dpi=500` is very high — the per-agent mode at 5×5 grid produces a roughly 12000×50000 pixel PNG, dozens of MB on disk. Disk space risk if visualisation is enabled with `n_agents=25` in long runs.

---

## 8. Hardcoded constants — exhaustive inventory

| Constant | Value | Location | What it is |
|---|---|---|---|
| Gumbel temperature `temp` | `0.5` | [[policies.py]] line 1190 (default arg of `GraphMaskGenerator.forward`) | Concrete distribution temperature for differentiable Bernoulli sampling. **No annealing.** |
| Numerical guard inside Gumbel log | `1e-20` (twice) | [[policies.py]] line 1204 | Avoid `log(0)` in `-log(-log(u + 1e-20) + 1e-20)`. |
| Eval-time mask threshold | `0.08` | [[policies.py]] line 1209 | Binarises `prob` to `{0,1}` during rollout. |
| Visualisation threshold (`draw_whole`) | `0.08` | [[policies.py]] line 1504 | Matches the eval threshold. |
| Visualisation threshold (per-agent) | `0.1` (three times) | [[policies.py]] lines 1586, 1601, 1635 | Different from the eval threshold — inconsistent. |
| Edge width multiplier (`draw_whole`) | `10` | [[policies.py]] line 1548 | Visual only. |
| Edge width multiplier (per-agent) | `5` | [[policies.py]] line 1640 | Visual only. |
| Whole-graph heatmap vmax | `0.5` | [[policies.py]] line 1517 | Saturates probabilities ≥ 0.5. |
| Sparsity prior weight `lambda_` | `0.2` | [[policies.py]] line 1435 (`getattr(self, 'lambda_', 0.2)`) | Bernoulli prior mean. Not in `__init__`; settable only by external attribute injection. |
| ELBO log guards | `1e-8` (four times) | [[policies.py]] lines 1436, 1437, 1440, 1441 | Avoid `log(0)` when probabilities saturate. |
| Advantage normalisation guard | `1e-8` | [[policies.py]] line 1409 | Avoid divide-by-zero. |
| LSTM input width factor | `3` | [[policies.py]] line 1237 | Three GNN outputs concatenated. |
| Mask generator hidden width | `self.n_fc` (=`64` by default) | [[policies.py]] line 1256 | Default `n_fc` is set by the constructor of the parent class. |
| Default `current_step` | `0` | [[policies.py]] line 1378 | Used in `mask_history` if the caller forgets to set it. |
| `dpi` for visualisation save | `500` | [[policies.py]] line 1666 | Very high. |
| Node sizes | `800`, `500` | [[policies.py]] lines 1541, 1628, 1647 | Visual only. |
| Font sizes | `16`, `18` (multiple) | [[policies.py]] lines 1520, 1521, 1551, 1552, 1555 | Visual only. |

---

## 8. (re-numbered as §9 by request) Math vs code — the ELBO

The standard ELBO for a latent-edge stochastic policy is:
$$
\mathcal{F}(\theta, \phi) = \mathbb{E}_{q_\phi(Z \mid s)}\!\left[\log \pi_\theta(a \mid s, Z)\right] - \mathrm{KL}\!\big(q_\phi(Z \mid s)\,\|\,p(Z)\big)
$$
With Bernoulli $q_\phi$ and Bernoulli prior $p(Z)=\mathrm{Bern}(\lambda)$:
$$
-\mathrm{KL} = \mathbb{E}_q[\log p(Z)] - \mathbb{E}_q[\log q(Z)] = \mathbb{E}_q[\log p(Z)] + H[q]
$$
So:
$$
\mathcal{F} = \mathbb{E}_q[\log \pi(a|s,Z)] + \underbrace{\sum_{ij} \big[\pi_{ij}\log\lambda + (1-\pi_{ij})\log(1-\lambda)\big]}_{\mathbb{E}_q[\log p(Z)]} + \underbrace{H[q]}_{\text{entropy}}
$$

Mapping to the code (lines 1444–1452):

| Term | Math | Code |
|---|---|---|
| Likelihood | $\mathbb{E}_q[\log \pi(a|s,Z)]$ approximated by REINFORCE | `likelihood_term += -policy_loss_i` (line 1426) |
| Prior cross-entropy | $\sum [\pi_{ij}\log\lambda + (1-\pi_{ij})\log(1-\lambda)]$ | `prior_term += (lambda_ * log(π) + (1-lambda_) * log(1-π)).sum()` (lines 1436–1437) — **roles of $\pi$ and $\lambda$ are swapped relative to the math** |
| Entropy of $q$ | $H[q] = -\sum [\pi\log\pi + (1-\pi)\log(1-\pi)]$ | `mask_entropy = -(π log π + (1-π) log(1-π))` (lines 1440–1441) — **matches**, leading minus on line 1440 |
| Negation for loss | $\mathcal{L} = -\mathcal{F}$ | `self.elbo_loss = -(likelihood_term + prior_term - mask_entropy_term)` (line 1450) — **subtracts** entropy where it should add |
| Plus value/policy-entropy regularisers | A2C extras | `self.loss = self.elbo_loss + self.value_loss + self.entropy_loss` (line 1452) |

The sign of the mask-entropy term and the swap of `lambda_` and `mask_probs` in the prior term are the two non-obvious deviations between the mathematical ELBO and the implementation. They might be intentional (e.g. "we want to minimise mask entropy so the network commits to a sharp posterior" + "we want a sparsity cross-entropy term"), but they are not the canonical ELBO. Worth raising in a review.

---

## 9. (re-numbered as §10 by request) Tensor-shape walkthrough — 5×5 grid (`n_agent=25`)

Assume a standard ATSC 5×5 grid: `n_agent=25`, `n_s=12` (state features per agent — varies by env but illustrative), `n_a=5` actions, `n_fc=64`, `n_h=64`, `n_step=120`, batch size = `T_minibatch` rollout length.

Each *interior* agent has 4 neighbors (`n_n=4`); corner agents have 2; edge agents have 3. Below I trace agent 12 (the centre, `n_n=4`).

### Inside `_get_comm_s(i=12, n_n=4, x, h, p)` at a single timestep:
| Tensor | Shape | Line |
|---|---|---|
| `x` (global obs stack) | `[25, 12]` | input |
| `h` (LSTM hidden stack) | `[25, 64]` | input |
| `p` (fingerprint stack) | `[25, 5]` | input |
| `js` (neighbor indices) | `[4]` | 1295 |
| `h_i = h[js]` | `[4, 64]` | 1296 |
| `x_i = x[i].unsqueeze(0)` | `[1, 12]` | 1300 |
| `nx_i = x[js]` | `[4, 12]` | 1301 |
| `p_i = p[js]` | `[4, 5]` | 1302 |
| `attn_mask = zeros(5, 5)` | `[5, 5]` | 1322 |
| `s_i_state = x_i` | `[1, 12]` | 1327 |
| `s_j_states = nx_i.view(4, -1)` | `[4, 12]` | 1328 |
| `mask_gen` input after `cat` | `[4, 24]` | inside [[GraphMaskGenerator]] line 1198 |
| `fc1` output | `[4, 64]` | 1199 |
| `logits = fc2(...)` | `[4, 1]` | 1199 |
| `prob = sigmoid(logits)` | `[4, 1]` | 1200 |
| Gumbel noise `u` | `[4, 1]` | 1203 |
| `z = sigmoid((logits + noise)/0.5)` | `[4, 1]` | 1205 |
| Return `z.squeeze(-1)`, `prob.squeeze(-1)` | `[4]`, `[4]` | 1206 |
| `attn_mask` after writes (line 1338) | `[5, 5]` with row/col-0 off-diagonals = `z_mask` | 1338 |
| `attn_mask` after `+= eye` | `[5, 5]` diagonal=1, row/col-0 carry `z_mask` | 1344 |
| `x_combined` | `[5, 12]` | 1347 |
| `p_combined` | `[5, 5]` | 1348 |
| `h_combined` | `[5, 64]` | 1349 |
| `gnn_obs[i](x_combined, attn_mask)` | `[5, 64]` | 1353 |
| `[0].unsqueeze(0)` | `[1, 64]` | 1353 |
| Same for policy / hidden | each `[1, 64]` | 1354–1355 |
| Final `torch.cat([...], dim=1)` | **`[1, 192]`** | 1352–1356 |

### Aggregation over agents in `_run_comm_layers` (inherited)
For all 25 agents, the per-agent `[1, 192]` outputs are stacked into `[25, 192]`, then fed into 25 separate `LSTMCell`s (one per agent — the parent's `lstm_layers`), each producing `[1, 64]`. Stacked: `[25, 64]`. Repeated for each of `T_minibatch` timesteps inside `backward`, the per-timestep stack `[25, 64]` is collected into `hs` of shape `[T_minibatch, 25, 64]`.

### Inside `backward`
| Tensor | Shape |
|---|---|
| `obs` after `transpose(0,1)` | `[25, T_minibatch, 12]` |
| `dones` | `[T_minibatch]` (or `[T_minibatch, 25]` depending on env) |
| `fps` after transpose | `[25, T_minibatch, 5]` |
| `acts` | `[25, T_minibatch]` |
| `hs` | `[25, T_minibatch, 64]` (output of `_run_comm_layers`) |
| `ps[i]` (actor head per agent) | `[T_minibatch, 5]` |
| `vs[i]` (critic head per agent) | `[T_minibatch]` |
| `Rs[i]`, `Advs[i]` | `[T_minibatch]` each |
| `self.latest_mask_probs[i]` (interior agent) | `[4]` (last timestep only) |
| `prior_term` per interior agent | scalar (`.sum()`) |
| `mask_entropy_term` per interior agent | scalar |

For a corner agent, `latest_mask_probs[i]` has shape `[2]`; for an edge agent, `[3]`. The 4 corners (`n_n=2`) and the 12 edge agents (`n_n=3`) and the 9 interior agents (`n_n=4`) sum to `4·2 + 12·3 + 9·4 = 8 + 36 + 36 = 80` edges in the static `neighbor_mask`. The prior and entropy terms sum over all 80 of these per `backward` call.

### Total parameter count of the mask generators
Each `GraphMaskGenerator` for an identical-agent setup has:
- `fc1`: `2*12 → 64` = `24·64 + 64 = 1600`
- `fc2`: `64 → 1` = `64 + 1 = 65`
- Total: `1665` per agent.
- With 25 agents, all of which have neighbors: `25 · 1665 = 41,625` parameters added by the Bayesian mechanism — a small fraction of the total model.

---

## 11. Bugs / oddities / dead code

1. **`mask_generators` lazy creation is fragile (line 1252).** If the first agent in `_init_comm_layer` iteration has `n_n=0`, the `else` branch at line 1278 hits `self.mask_generators.append(None)` before the attribute is created. `AttributeError`. Not triggered in practice because all benchmark grids put agent 0 at a corner with neighbors, but a future env with an isolated agent at index 0 would crash.
2. **`UnboundLocalError` risk in [[GraphMaskGenerator.forward]] (lines 1207–1211).** The `return mask...` at line 1211 is outside the `if/else`; if a refactor ever adds a third branch without binding `mask`, it explodes. Currently safe because the train branch returns at line 1206.
3. **ELBO sign and parameter-swap quirks (lines 1436–1437, 1450).** See §9. The implementation departs from the textbook ELBO. Could be intentional but is undocumented.
4. **`self.training` toggle in `forward` (lines 1364–1371) does not preserve the state of submodules** that override `train()` non-trivially (none here, but worth noting). Also, if any other thread inspects `self.training` mid-`forward`, it will see `False`. Not relevant for synchronous PyTorch.
5. **`mask_history` is an unbounded list (line 1233 / 1377).** Memory leak in long training runs. Should be capped or use a `collections.deque(maxlen=...)`.
6. **`current_step` is read but never written by this class (line 1378).** Relies on the trainer to inject it; otherwise everything in `mask_history` is labelled step 0.
7. **`n_heads = n_heads` (line 1232) is redundant** — already set by the parent constructor (line 889).
8. **Asymmetric edge in symmetric plot (lines 1510–1512).** The `draw_whole` adjacency is forced symmetric by overwriting; agent `j`'s independent decision about edge `(j,i)` is silently discarded.
9. **Three different "probability thresholds":** `0.08` (real, lines 1209/1504), `0.1` (visual-only, lines 1586/1601/1635). Reader confusion.
10. **`lambda_` has no constructor / no schedule (line 1435).** `getattr(self, 'lambda_', 0.2)` means experiments using a different `λ` must monkey-patch the attribute after instantiation. This is a strange API choice for what the paper presents as a tunable hyperparameter.
11. **`elif self.unify_act_state_dim` / `else` mismatch in `_init_comm_layer` (lines 1262–1265).** Policy GNN input dim falls back to `self.n_a` for the heterogeneous-non-unified case, but `_get_comm_s` feeds it `sum(na_ls_ls[i])`. Latent shape bug, never hit by paper experiments.
12. **`visualize_masks` re-imports `networkx` and `os` on every call (lines 1487–1488).** Trivial cost but noisy.
13. **Comment on line 1335 says `# detach if needed`** — but the tensor is *not* detached, and the ELBO depends on it not being detached. The comment is misleading.
14. **Tensorboard tags use `self.name = 'graph_nc'`** (set by parent at line 900), not anything Bayesian-specific. Mixed-policy runs in the same logdir overlap.
15. **`fc_x_layers_for_no_neighbor.append(None)` at line 1274** is dead — the neighbor branch never reads from this list, but it must still be appended to keep indices aligned with `lstm_layers`. Confusing but necessary.
16. **`gnn_policy` / `gnn_hidden` are `None` for isolated agents (lines 1279–1280)** but `_get_comm_s` would crash if called with `n_n=0` (it indexes `gnn_policy[i]` etc.). In practice the inherited `_run_comm_layers` has a branch that uses `fc_x_layers_for_no_neighbor[i]` instead — verify in [[NCMultiAgentPolicy]] line ~193+ if you want certainty, but this class does not handle it explicitly.

---

## Where to go next

- [[GraphMaskGenerator]] (lines 1184–1211) — the variational posterior MLP. Tiny but central.
- [[GraphCMultiAgentPolicy]] (lines 872+) — the non-Bayesian parent. Diff against this class to see exactly what the Bayesian variant adds.
- [[NCMultiAgentPolicy]] (lines 193+) — grandparent. Owns `_run_comm_layers`, `_run_actor_heads`, `_run_critic_heads`, `_run_loss`.
- [[trainer.py]] — calls `forward` (rollout) and `backward` (gradient step). Look there for `current_step` setting and `visualize_masks` invocation.
- Paper sections to cross-check: ELBO definition (sign of $H[q]$, role of $\lambda$), temperature schedule, threshold value.
