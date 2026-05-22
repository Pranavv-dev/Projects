# 03 — `GraphCMultiAgentPolicy` and `GraphMaskGenerator`

Line-by-line walkthrough of `/tmp/BayesG/agents/policies.py` lines **872–1211**. Every claim below was verified against the actual source — nothing is assumed.

---

## `class GraphCMultiAgentPolicy(NCMultiAgentPolicy)` (lines 872–1182)

The docstring (lines 873–876) states the design:

> `obs, policy, hidden state has seperate GNN layer`
> `s_i[GNN(obs_i, obs_neighbors), GNN(policy_neighbors), GNN(hidden_state_neighbors)]`

So this class replaces the three MLP fanin layers of [[NCMultiAgentPolicy]] (`fc_x_layers`, `fc_p_layers`, `fc_m_layers`) with three **per-agent GNN modules**: `gnn_obs`, `gnn_policy`, `gnn_hidden`. The communication-feature output is then the concatenation of three GNN outputs (one per stream), fed to the LSTM exactly as before.

### `__init__` (lines 877–927)

Signature:

```python
def __init__(self, n_s, n_a, n_agent, n_step, neighbor_mask, n_fc=64, n_h=64,
             n_s_ls=None, n_a_ls=None, identical=True, unify_act_state_dim=False,
             n_heads=4, gnn_type='gat', use_random_mask=False):
```

New params **beyond** [[NCMultiAgentPolicy]]:

| Param                  | Default | Purpose                                                                 |
| ---------------------- | ------- | ----------------------------------------------------------------------- |
| `unify_act_state_dim`  | `False` | If `True` (heterogeneous case), project per-agent obs/action vectors to a common shape via `Linear` layers so a single GNN can consume them. |
| `n_heads`              | `4`     | Number of attention heads passed to `GATLayer` (only used when `gnn_type=='gat'`). |
| `gnn_type`             | `'gat'` | Selects between `GATLayer`, `GCNLayer`, `SAGELayer` (see [[gnn]]).      |
| `use_random_mask`      | `False` | If `True`, replaces the actual neighbor mask with a Bernoulli(0.5) random mask at every `_get_comm_s` call. |

Initialization order (verified):

1. **L880**: `self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` — set up-front because subsequent ops (e.g., `neighbor_mask.to(self.device)`) need it.
2. **L882**: `self.unify_act_state_dim = unify_act_state_dim`.
3. **L884–891**: stores `neighbor_mask` (moved to device), `n_agent`, `n_fc`, `n_h`, `n_heads`, `n_s`, `n_a`. There is a `print("neighbor_mask:", self.neighbor_mask.sum(axis=1))` on L885 (debug — see Open questions).
4. **L894**: `self.use_random_mask = use_random_mask`.
5. **L897**: `self.gnn_type = gnn_type.lower()`.
6. **L900**: `super(NCMultiAgentPolicy, self).__init__(...)` — **note**: this calls the grandparent (skipping `NCMultiAgentPolicy.__init__`). It passes `'graph_nc'` as the policy name and `None` as `n_n` (since here neighbor relationships are per-agent, not uniform). This is why `_init_net` and `_init_comm_layer` are fully overridden — the parent class wiring is bypassed.
7. **L902–915**: if `not identical`, requires `n_s_ls`/`n_a_ls`; if `unify_act_state_dim`, creates per-agent `Linear` layers:
   - `convert_state_linears`: `Linear(n_s_ls[i], 16)` — note the **hard-coded `convert_state_shape = 16`** on L909.
   - `convert_action_linears`: `Linear(n_a_ls[i], max(n_a_ls))` — i.e., pads action dims up to the max.
   The `print` statements list the created layers.
8. **L918**: `_precompute_neighbor_dims()`.
9. **L921**: `_init_net()`.
10. **L924**: `_reset()` (inherited).
11. **L927**: `self.to(self.device)`.

### `_create_linear` (lines 929–933)

Trivial helper: `nn.Linear(input_shape, output_shape).to(self.device)` and prints what was made.

### `_precompute_neighbor_dims` (lines 935–948)

Builds `self.neighbor_dims`: a list of length `n_agent`, each entry a 5-tuple `(n_n, n_ns, n_na, ns_ls, na_ls)`.

- `n_n` = neighbor count for agent i (= row sum of `neighbor_mask[i]`).
- **Homogeneous (`identical=True`, L941)**: `dims = (n_n, n_s * (n_n+1), n_a * n_n, [n_s]*n_n, [n_a]*n_n)`. Note `n_s * (n_n+1)` includes the agent itself (so the obs feature span is i + neighbors), but `n_a * n_n` does **not** include self (consistent with how policy fingerprints are gathered).
- **Heterogeneous (L943–947)**: looks up `n_s_ls[j]`/`n_a_ls[j]` for each neighbor `j` and sums. The neighbor obs dim is `n_s_ls[i] + sum(ns_ls)` (self + neighbors), action dim is `sum(na_ls)` (neighbors only).

### `_get_neighbor_dim` (lines 950–951)

Just returns `self.neighbor_dims[i_agent]`.

### `_init_gnn_layer` — the GNN factory (lines 953–962)

```python
def _init_gnn_layer(self, in_features, out_features, n_heads=None):
    if self.gnn_type == 'gat':
        return GATLayer(in_features, out_features, n_heads or self.n_heads)
    elif self.gnn_type == 'gcn':
        return GCNLayer(in_features, out_features)
    elif self.gnn_type == 'sage':
        return SAGELayer(in_features, out_features)
    else:
        raise ValueError(f"Unsupported GNN type: {self.gnn_type}")
```

Layer types come from [[gnn]]. Only `GATLayer` consumes `n_heads`; `GCNLayer` and `SAGELayer` ignore it.

### `_run_comm_layers` (lines 964–1019) — the forward pass

This is the time-stepped LSTM driver. It is structurally **identical** to the parent `NCMultiAgentPolicy._run_comm_layers`: convert obs/dones/fps to tensors → `batch_to_seq` → split LSTM state into `h, c` → loop over time → for each agent compute `s_i` and step the per-agent LSTM cell. The only material difference is that the per-agent communication features come from `_get_comm_s` (which uses GNNs) rather than the MLP-based comm in the parent.

Key fragments:

- **L972–976**: `batch_to_seq` reshapes `[B, n_agent, *]` to `[T, n_agent, *]`.
- **L979**: `h, c = torch.chunk(states, 2, dim=1)` — recurrent state is stored as concatenated `[h; c]`.
- **L990–1006**: branches on `n_n`:
  - `if n_n:` call `self._get_comm_s(i, n_n, x, h, p)` — GNN path.
  - `else:` fall back to `F.relu(self.fc_x_layers_for_no_neighbor[i](x_i))`. Heterogeneous case narrows `x[i]` to `n_s_ls[i]` first.
- **L1009–1010**: `done`-gated reset of `h_i`, `c_i` (multiply by `1-done`).
- **L1012**: `next_h_i, next_c_i = self.lstm_layers[i](s_i, (h_i, c_i))`.
- **L1019**: returns `(outputs.transpose(0, 1), torch.cat([h, c], dim=1))` — outputs in `[B, T, n_h]`-style, state in `[n_agent, 2*n_h]`-style (matches parent contract).

### `_get_comm_s` (lines 1021–1075) — where the GNN is actually invoked

Builds the per-agent communication feature `s_i`.

1. **L1023**: `js = torch.nonzero(self.neighbor_mask[i]).squeeze(1)` — neighbor indices.
2. **L1024**: `h_i = h[js]` (note: this shadows the typical meaning of `h_i` — it's the **neighbor** hidden states, not the agent's own).
3. **Identical path (L1026–1030)**: `x_i = x[i].unsqueeze(0)`, `nx_i = x[js]`, `p_i = p[js]`.
4. **Heterogeneous path (L1032–1050)**: narrows `x[i]` to `n_s_ls[i]`; optionally projects through `convert_state_linears[i]`; for each neighbor builds `p_i_ls` and `nx_i_ls` (optionally via `convert_action_linears`/`convert_state_linears`); concatenates and `unsqueeze(0)`.

**Attention mask construction (L1052–1063)** — this is the heart of the **random-mask** code path:

```python
attn_mask = torch.zeros(n_n + 1, n_n + 1, device=self.device)

if self.use_random_mask:
    random_mask = torch.bernoulli(torch.ones(n_n, device=self.device) * 0.5)
    attn_mask[0, 1:] = attn_mask[1:, 0] = random_mask
else:
    attn_mask[0, 1:] = attn_mask[1:, 0] = 1

attn_mask += torch.eye(attn_mask.size(0), device=attn_mask.device)
```

- The mask has shape `(n_n+1, n_n+1)`: index 0 is the agent itself, indices `1..n_n` are its neighbors.
- **`use_random_mask=True`**: each edge (self↔neighbor) is independently kept with **p=0.5** via `torch.bernoulli`. Self-loops to neighbor-only edges between neighbors stay 0.
- **`use_random_mask=False`**: full star — agent connects to all of its (already-filtered-by-`neighbor_mask`) neighbors with weight 1.
- The final `+= I` adds the diagonal (self-loops everywhere), so even non-edges have self-attention. The mask is **symmetric** (line 1058/1061 set `[0,1:]` and `[1:,0]` together).

**Feature stacking (L1066–1068)**:

```python
x_combined = torch.cat([x_i, nx_i.reshape(n_n, -1)], dim=0)        # (n_n+1, d_x)
p_combined = torch.cat([p[i].unsqueeze(0), p_i.reshape(n_n, -1)], dim=0)  # (n_n+1, d_p)
h_combined = torch.cat([h[i].unsqueeze(0), h_i.reshape(n_n, -1)], dim=0)  # (n_n+1, d_h)
```

So each stream's input to its GNN is a `(n_n+1) × d` matrix with row 0 = self, rows 1.. = neighbors.

**Three GNNs in parallel (L1071–1075)**:

```python
return torch.cat([
        self.gnn_obs[i](x_combined, attn_mask)[0].unsqueeze(0),
        self.gnn_policy[i](p_combined, attn_mask)[0].unsqueeze(0),
        self.gnn_hidden[i](h_combined, attn_mask)[0].unsqueeze(0)
    ], dim=1)
```

- Each GNN returns a `(n_n+1, n_fc)` matrix; `[0]` grabs **agent i's row** (the readout for the self node).
- `unsqueeze(0)` reshapes to `(1, n_fc)`; `dim=1` cat produces `(1, 3*n_fc)` — exactly the `3*self.n_fc` expected by the LSTM in `_init_comm_layer` (L1111 `n_lstm_in = 3 * self.n_fc`).

### `_init_net` (lines 1077–1107)

Replaces the parent's MLP-based init. Builds:

- `self.lstm_layers`, `self.actor_heads`, `self.critic_heads` (the standard heads).
- **`self.gnn_obs`, `self.gnn_policy`, `self.gnn_hidden`** — three parallel `ModuleList`s, one entry per agent (or `None` for agents without neighbors, see below).
- Caches `ns_ls_ls`, `na_ls_ls`, `n_n_ls`.
- `self.fc_x_layers_for_no_neighbor` — fallback `Linear` for the (`n_n == 0`) case.

Loop body (L1097–1107): per agent, unpacks `neighbor_dims[i]`, calls `_init_comm_layer`, `_init_actor_head`, `_init_critic_head`. The critic head is sized off `n_na` (neighbor action concat width) so the critic conditions on neighbor actions — consistent with `_run_critic_heads` below.

### `_init_comm_layer` (lines 1109–1153)

`n_lstm_in = 3 * self.n_fc` (L1111).

- **`gnn_obs`** (always created, L1113–1122): input dim is one of
  - `self.n_s` (identical),
  - `self.convert_state_shape` (== 16, heterogeneous + unify),
  - `self.n_s_ls[i_agent]` (heterogeneous + no-unify).
  Output: `self.n_fc`.

- **If `n_n > 0` (L1124–1140)**:
  - `gnn_policy`: input `self.n_a` / `self.convert_action_shape` / `self.n_a` (note: heterogeneous-no-unify uses `self.n_a` — possible bug, see Open questions); output `self.n_fc`.
  - `gnn_hidden`: input `self.n_h`, output `self.n_fc`.
  - `fc_x_layers_for_no_neighbor.append(None)` (placeholder).
  - LSTM cell: `nn.LSTMCell(3*n_fc, n_h)`.

- **If `n_n == 0` (L1141–1150)**:
  - `gnn_policy.append(None)`, `gnn_hidden.append(None)`.
  - Plain `Linear(n_s, n_fc)` (or `Linear(n_s_ls[i_agent], n_fc)`).
  - LSTM cell: `nn.LSTMCell(n_fc, n_h)` — narrower input because only `gnn_obs`/fallback feeds it.

- **L1152**: `init_layer(lstm_layer, 'lstm')` — custom weight init (defined elsewhere).

### `_run_critic_heads` (lines 1155–1182)

Same semantics as parent: for each agent gather neighbor actions, one-hot-encode each using `na_ls_ls[i][j]`, concat onto `hs[i]`, run through `self.critic_heads[i]`. Returns a list of value tensors (or numpy arrays if `detach=True`). No GNN involvement in the critic — the critic only uses the LSTM hidden state plus raw neighbor actions.

### Visualization helpers

**There are none in the inspected range.** Lines 872–1182 contain no `plot_*`, `visualize_*`, `matplotlib` calls, or attention-weight extraction methods. Any visualization for this class lives elsewhere (likely in a subclass such as [[BayesianGraphCMultiAgentPolicy]] or in a separate plotting utility).

---

## `class GraphMaskGenerator(nn.Module)` (lines 1184–1211)

A small two-layer MLP that emits a **Gumbel-sigmoid soft mask** over candidate edges. Used (by callers outside this snippet) to learn which neighbor links to keep.

### `__init__` (lines 1185–1188)

```python
def __init__(self, input_dim, hidden_dim):
    super(GraphMaskGenerator, self).__init__()
    self.fc1 = nn.Linear(input_dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, 1)
```

- `input_dim` is the **per-pair** input width (caller must pass `2*d` because `forward` concatenates `s_i` and `s_j` along dim -1 — verified at L1198).
- `hidden_dim` is the hidden layer width — entirely caller-controlled, **not** hard-coded in this class.
- **`tau` is NOT stored on `self`**. It is a per-call argument `temp` to `forward` (default `0.5`). So strictly speaking the Gumbel temperature lives on the method signature, not as an `__init__` field.

### `forward` (lines 1190–1211)

Signature: `forward(self, s_i, s_j, temp=0.5, training=True)`.

```python
if s_i.shape[0] == 1:
    s_i = s_i.repeat(s_j.shape[0], 1)  # broadcast self to all neighbors

x = torch.cat([s_i, s_j], dim=-1)            # [n_n, 2d]
logits = self.fc2(F.relu(self.fc1(x)))       # [n_n, 1]
prob = torch.sigmoid(logits)                 # deterministic edge prob
```

Then it splits on `training`:

**Training branch (L1202–1206)** — Gumbel-sigmoid:

```python
u = torch.rand_like(prob)
gumbel_noise = -torch.log(-torch.log(u + 1e-20) + 1e-20)
z = torch.sigmoid((logits + gumbel_noise) / temp)
return z.squeeze(-1), prob.squeeze(-1)
```

Verification of the math:

- `u ~ Uniform(0, 1)` (`torch.rand_like(prob)`).
- **Gumbel noise**: `g = -log(-log(u + 1e-20) + 1e-20)`. The `1e-20` inside the **inner** `log` guards against `log(0)` when `u==0`; the `1e-20` outside guards against `log(0)` when `u==1` (which would make `-log(u)==0`). This is the standard Gumbel(0,1) sample.
- **Gumbel-sigmoid**: `z = sigmoid((logits + g) / temp)`. **Confirmed exact form**: `z = sigmoid((logits + g) / tau)`.

Note: a textbook **binary** Gumbel-softmax draws **two** Gumbel noises (one for the "on" logit, one for the implicit 0 logit) and takes their difference. This code uses only **one** Gumbel noise added to the logit, which is a known simplified Gumbel-sigmoid trick. Both formulations are commonly seen; this is a slightly biased estimator vs. the symmetric two-Gumbel form. (Flagging this in Open questions.)

**Eval branch (L1207–1211)**:

```python
else:
    # During evaluation, use deterministic threshold
    mask = (prob > 0.08).float()

return mask.squeeze(-1), prob.squeeze(-1)
```

- The threshold is **0.08** (very low — keeps most edges).
- **Indentation bug**: the final `return mask.squeeze(-1), prob.squeeze(-1)` (L1211) is at the outer `def`-level, not inside the `else`. So in **training mode** the function returns from L1206 (`return z.squeeze(-1), prob.squeeze(-1)`), and the L1211 return is reachable only via the `else` branch (where `mask` is defined). Functionally correct, but stylistically the L1211 return *looks* like it could be reached without `mask` being defined — only by inspection does it become clear that the training branch already returned. Flagged below.

---

## Important constants — quoted exactly

| Constant                   | Where                                  | Value     |
| -------------------------- | -------------------------------------- | --------- |
| Gumbel temperature `temp`  | `GraphMaskGenerator.forward` arg (L1190) | `0.5` (default) — passed per-call, not stored. |
| Eval mask threshold        | `GraphMaskGenerator.forward` (L1209)   | `0.08`    |
| Gumbel numerical floor     | `GraphMaskGenerator.forward` (L1204)   | `1e-20` (used twice — inside inner log and inside outer log) |
| `GraphMaskGenerator` hidden dim | `__init__` arg (L1185)            | caller-provided (`hidden_dim`); no default. |
| `convert_state_shape`      | `GraphCMultiAgentPolicy.__init__` (L909) | `16`    |
| `convert_action_shape`     | `GraphCMultiAgentPolicy.__init__` (L913) | `max(n_a_ls)` |
| Random-mask Bernoulli p    | `_get_comm_s` (L1057)                  | `0.5`     |
| LSTM input width           | `_init_comm_layer` (L1111)             | `3 * self.n_fc` (== 192 for `n_fc=64`) |
| Default `n_heads`          | `__init__` (L878)                      | `4`       |
| Default `n_fc`, `n_h`      | `__init__` (L877)                      | `64`, `64` |

---

## How `GraphCMultiAgentPolicy` differs from `NCMultiAgentPolicy`

| Aspect                          | `NCMultiAgentPolicy`                          | `GraphCMultiAgentPolicy`                                              |
| ------------------------------- | --------------------------------------------- | --------------------------------------------------------------------- |
| Communication backbone          | MLPs (`fc_x_layers`, `fc_p_layers`, `fc_m_layers`) | Three GNNs per agent: `gnn_obs`, `gnn_policy`, `gnn_hidden`           |
| GNN type                        | n/a                                           | Selectable: `'gat'` / `'gcn'` / `'sage'` via `gnn_type` (L897, L953–962) |
| Attention heads                 | n/a                                           | `n_heads` param (default 4) used by `GATLayer` only                   |
| Edge mask                       | Deterministic from `neighbor_mask`            | Same — **unless** `use_random_mask=True`, which substitutes a fresh `Bernoulli(0.5)` mask each call (L1055–1058) |
| Heterogeneous dim unification   | Not present                                   | `unify_act_state_dim` flag (L882, L907–915) creates per-agent `Linear` projections to `convert_state_shape=16` and `convert_action_shape=max(n_a_ls)` |
| `super().__init__` call         | Calls parent normally                         | Calls **grandparent** (`super(NCMultiAgentPolicy, self).__init__`) at L900 — skips `NCMultiAgentPolicy.__init__` and rebuilds the network from scratch |
| `_init_net`                     | Inherited                                     | Fully overridden (L1077–1107)                                         |
| `_init_comm_layer`              | MLP-based                                     | GNN-based; agents with `n_n==0` fall back to a plain `Linear` (L1141–1150) |
| `_run_comm_layers` / `_get_comm_s` | MLP fan-in                                 | Three parallel GNN calls; outputs concat into `3*n_fc` LSTM input    |
| Critic head                     | Same — uses neighbor one-hot actions          | **Same** (`_run_critic_heads` at L1155–1182 is essentially identical) |
| Device handling                 | Implicit                                      | Explicit `self.device` set first (L880), `.to(self.device)` at end of init (L927) |

---

## Cross-references

- [[BayesianGraphCMultiAgentPolicy]] — inherits from this class; presumably adds Bayesian / learned-mask logic on top.
- [[gnn]] — provides `GATLayer`, `GCNLayer`, `SAGELayer` used by `_init_gnn_layer`.
- [[NCMultiAgentPolicy]] — parent class; `GraphCMultiAgentPolicy` skips its `__init__` (via `super(NCMultiAgentPolicy, ...)` on L900) and replaces its MLP communication backbone.
- [[GraphMaskGenerator]] — the learned-mask module; the snippet does **not** show it being used inside `GraphCMultiAgentPolicy`. It is presumably wired in by the Bayesian subclass or an external trainer.
- [[batch_to_seq]], [[one_hot]], [[init_layer]] — utility helpers imported at module top.

---

## Open questions / bugs / suspicious patterns

1. **`super(NCMultiAgentPolicy, self).__init__(...)` on L900** is intentional (it forwards directly to the grandparent), but it is fragile: any future change to `NCMultiAgentPolicy.__init__` will silently *not* run for this subclass. A comment explaining the skip would be helpful.

2. **`print` calls in production code** (L885 `print("neighbor_mask:", ...)`, L911, L915, L932). These will spam stdout every time the model is instantiated.

3. **Heterogeneous + `use_random_mask=False` + no-unify policy GNN input dim**: at L1132 the heterogeneous-no-unify branch sets `gnn_policy = self._init_gnn_layer(self.n_a, self.n_fc)` — using the **scalar** `self.n_a` (which in heterogeneous mode is just whatever was passed positionally to `__init__`, typically `max(n_a_ls)` or similar). The actual `p_combined` tensor passed at runtime has per-row width `na_ls[j]` (from `narrow`) without padding. **This is likely a shape-mismatch bug** in the heterogeneous-no-unify code path. The `unify_act_state_dim=True` branch sidesteps it by padding everything to `convert_action_shape=max(n_a_ls)`.

4. **`gnn_obs` is always created (L1115–1122)**, even for agents with no neighbors — but if `n_n == 0`, `_get_comm_s` is never called and the `fc_x_layers_for_no_neighbor[i]` Linear is used instead. So `gnn_obs[i]` is dead weight for isolated agents. Minor memory waste, not a bug.

5. **`use_random_mask` resamples on every call** (L1057) — this means the mask is non-stationary across rollouts and gradient steps. If this flag is intended for ablation/robustness, fine; if it's intended to learn under a fixed random topology, it's wrong (the topology should be fixed once per episode or epoch).

6. **`GraphMaskGenerator` Gumbel formulation uses one Gumbel noise** (L1204) rather than the symmetric two-Gumbel form for binary Gumbel-softmax. This is a common shortcut but yields a slightly different (biased toward sigmoid) sampler. Worth verifying against whatever paper this is replicating.

7. **`GraphMaskGenerator` eval threshold of `0.08`** (L1209) is suspiciously low — it keeps any edge with probability > 8%. If the generator learns to output `prob ≈ 0.5` everywhere, the eval mask becomes all-ones, defeating the purpose. Why not 0.5?

8. **Return-statement layout in `GraphMaskGenerator.forward`** (L1211): the final `return mask.squeeze(-1), prob.squeeze(-1)` sits at outer-function indentation, but `mask` is only defined inside the `else`. The function works because the `if training:` branch returns at L1206. This relies entirely on control flow rather than static structure — a refactor where someone removes the early `return` would crash on `mask` being undefined. Tidier form: put the return inside the `else`, or initialize `mask` outside.

9. **`GraphMaskGenerator` is defined but not invoked anywhere in the inspected range (872–1211).** Its integration point lives elsewhere (likely [[BayesianGraphCMultiAgentPolicy]]). Worth tracing.

10. **`tau` is not stored on `self`** in `GraphMaskGenerator` — it must be threaded through every `forward` call. If a caller forgets `temp=`, they get `0.5` silently. Storing it as `self.tau` would make experiments more reproducible/loggable.
