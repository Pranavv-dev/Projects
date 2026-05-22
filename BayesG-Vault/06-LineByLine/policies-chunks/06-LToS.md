# LToSMultiAgentPolicy — Line-by-Line Walkthrough

> Source: `agents/policies.py` lines **2047–2471**.
> Companion driver: `agents/models.py::IA2C_LToS` (lines 481–805).
> Buffer: `agents/utils.py::LToSPolicyBuffer` (lines 310–354).
>
> Reference: Yi et al., "Learning to Share in Networked Multi-Agent Reinforcement Learning," NeurIPS 2022.
> Related notes: [[05-MA2C]], [[04-IA2C]], [[03-Policy-base]], [[IA2C_LToS-models]], [[LToSPolicyBuffer]].

---

## Class signature, base class, what it overrides

```python
class LToSMultiAgentPolicy(Policy):
    """Multi-agent policy that implements Learning to Share (LToS) approach
       with high and low level policies"""
```

- **Base:** `Policy` (defined at `policies.py:15`), which is itself `nn.Module`. The base just stores `n_a`, `n_s`, `n_step`, `name`, `identical`, and exposes the abstract `forward` (raises `NotImplementedError`).
- **Override behavior:**
  - Does **NOT** override `forward` — the inherited `forward` from `Policy` would raise `NotImplementedError`. Instead the driver `IA2C_LToS.forward` (models.py:559) calls bespoke methods (`compute_w_out`, `compute_actions`, `compute_critic`) directly on the policy.
  - Does **NOT** use the inherited helpers `_init_actor_head`, `_init_critic_head`, `_run_critic_head` from `Policy`. It builds its own networks in `_init_net`.
  - Adds many new public methods: `compute_w_out`, `compute_critic`, `compute_target_critic`, `compute_reward_sharing`, `compute_gradients`, `compute_actions`, `soft_update`, `soft_update_target_phi`, `soft_update_target_policy`, `_update_tensorboard`, `_reset`, `_init_net`.

Compare to neighbors in this file: e.g. [[05-MA2C]] policies override `forward` and use `_run_critic_head`. LToS does neither — it is more of a Q-actor-critic hybrid than the on-policy AC the base class was built for.

---

## `__init__` — every parameter

```python
def __init__(self, n_s_ls, n_a_ls, neighbor_mask, n_step,
             shared_dim=64, n_fc=64, use_lstm=True, identical=True, agent_id=0):
```

Line-by-line (lines 2049–2091):

| Line | Code | Meaning |
|------|------|---------|
| 2051 | `self.n_s_ls = n_s_ls if isinstance(n_s_ls, list) else [n_s_ls]` | Coerce to list. Stores **full** per-agent obs-dim list. |
| 2052 | `self.n_a_ls = n_a_ls if isinstance(n_a_ls, list) else [n_a_ls]` | Same for action dims. |
| 2053 | `self.n_neighbors = int(torch.sum(neighbor_mask[agent_id]))` | Number of neighbors **for this agent only**. Read from row `agent_id` of the mask. |
| 2056–2057 | `n_s = self.n_s_ls[agent_id]; n_a = self.n_a_ls[agent_id]` | Pull this agent's dims. |
| 2059 | `super().__init__(n_a, n_s, n_step, 'lstm', 'lstm', identical)` | Calls `Policy.__init__`. The two `'lstm'` strings become `policy_name='lstm'`, `agent_name='lstm'` → final `self.name = 'lstm_lstm'`. **This is a copy-paste oddity** — the second arg is supposed to be the agent identifier; passing `'lstm'` means every LToS policy gets the same name (no `agent_id` suffix). |
| 2060 | `self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` | Per-instance device decision. Note: the base `Policy` does not handle device; this is local. |
| 2062–2063 | `self.n_s = n_s; self.n_a = n_a` | Re-store (base class also stored them — harmless dup). |
| 2064 | `self.neighbor_mask = neighbor_mask` | Keeps the **full** adjacency, not just this agent's row. |
| 2067–2072 | Store `n_step, shared_dim, n_fc, use_lstm, identical, agent_id` | Vanilla. |
| 2075–2083 | Init tensorboard tracker scalars/tensors | `q_loss, policy_loss, total_loss, actor_loss = 0.0`; `q_values, w_out, w_in, w_in_grads = None`; `self.epsilon = 1.0`. The `self.epsilon = 1.0` here is **dead** — the driver (`IA2C_LToS`) keeps its own `current_epsilon` and passes it explicitly to `compute_w_out` / `compute_actions`. |
| 2086 | `self._init_net()` | Build all the nets (next section). |
| 2089–2091 | `self.states_fw = None; self.target_states_fw = None; self._reset()` | LSTM hidden state placeholders, then `_reset` actually allocates the zero tensors. |

**Missing parameter you might expect:**
- No explicit `tau` — that lives on the driver `IA2C_LToS.tau` (models.py:489) and is passed as default `tau=0.01` to `soft_update*` methods. The policy never reads `self.tau`.
- No `gamma` — also on the driver.
- No `lr` — optimizer is built outside.

---

## Two-level policy structure (built in `_init_net`, lines 2093–2154)

### High-level policy φ (the "phi" embedding network)

```python
self.phi = nn.Sequential(
    nn.Linear(self.n_s, self.n_fc),
    nn.ReLU(),
    nn.Linear(self.n_fc, self.shared_dim)
)
```
- Maps **own observation** `o_i ∈ ℝ^{n_s}` → embedding `w_out ∈ ℝ^{shared_dim}`.
- This is the message agent i broadcasts to its neighbors.
- In Yi et al.'s notation this is `φ_{θ_i}(o_i)` producing `w_i^out`.
- A clone `target_phi` (lines 2103–2107) with identical architecture is built.

> **Quirk:** the driver `IA2C_LToS.backward` reaches inside via `self.policy[i].phi[0](obs)` (models.py:644, 736) — it only uses the first `nn.Linear` layer, skipping the ReLU + second Linear! That is **not** φ as defined here. See [[Bugs section]] below.

### Low-level policy μ (action policy conditioned on neighbor embeddings)

Two branches based on `use_lstm`:

LSTM branch (lines 2110–2122):
```python
self.lstm_cell = nn.LSTMCell(
    self.n_s + self.shared_dim * self.n_neighbors,  # obs + neighbor embeds
    self.shared_dim
)
self.actor_head = nn.Linear(self.shared_dim, self.n_a)
# + target_lstm_cell, target_actor_head (identical arch)
```

Non-LSTM branch (lines 2123–2133):
```python
self.actor_layer = nn.Sequential(
    nn.Linear(self.n_s + self.shared_dim * self.n_neighbors, self.n_fc),
    nn.ReLU(),
    nn.Linear(self.n_fc, self.n_a)
)
# + target_actor_layer (identical arch)
```

- **Input:** `[o_i ; w_in^1 ; w_in^2 ; … ; w_in^{n_neighbors}]`, the agent's own obs concatenated with each neighbor's `w_out` embedding.
- **Output:** logits over `n_a` discrete actions.
- LSTM cell output dim is `shared_dim`, not `n_fc`. The actor head sits on top.
- The LSTM "state" is concatenated `[h ; c]` of width `2 * shared_dim`, stored in `self.states_fw` — see `_reset`.

### Q-network (critic)

Lines 2136–2145:
```python
self.q_net = nn.Sequential(
    nn.Linear(self.shared_dim + self.n_a, self.n_fc),
    nn.ReLU(),
    nn.Linear(self.n_fc, 1)
)
self.target_q_net = nn.Sequential(...)  # identical arch
```
- **Input:** `[w_vec ; one_hot(a)]` of width `shared_dim + n_a`.
- **Output:** scalar Q-value.
- Critically, `w_vec` here is **either** `w_out_i` (agent i's own embedding) **or** `w_in[n]` (a neighbor's embedding) depending on which method invokes it — the same Q-net is reused for both. See `compute_gradients`.

### Target networks (initialization)

Lines 2148–2154:
```python
self.target_phi.load_state_dict(self.phi.state_dict())
if self.use_lstm:
    self.target_lstm_cell.load_state_dict(self.lstm_cell.state_dict())
    self.target_actor_head.load_state_dict(self.actor_head.state_dict())
else:
    self.target_actor_layer.load_state_dict(self.actor_layer.state_dict())
self.target_q_net.load_state_dict(self.q_net.state_dict())
```
Hard copy at construction. Subsequent updates are polyak via `soft_update_*`.

---

## Methods — walkthrough

### `compute_w_out(self, obs, agent_id, epsilon=0.1)` — lines 2156–2173

```python
obs_i = torch.from_numpy(np.array(obs)).float().to(self.device)
if obs_i.dim() == 1:
    obs_i = obs_i.unsqueeze(0)
w = self.phi(obs_i)
if random.random() < epsilon:
    w = w + torch.randn_like(w) * 0.1
self.w_out = w.detach()
return w
```
- Wraps `obs` as a tensor; adds batch dim if 1-D.
- Runs the **full** `phi` (Linear → ReLU → Linear) → returns `w ∈ ℝ^{1 × shared_dim}`.
- **Exploration noise:** with prob ε, adds Gaussian `N(0, 0.1²)` to the embedding. Not the standard ε-greedy — this is "ε-noisy continuous" on the message. Magnitude 0.1 is hardcoded.
- `agent_id` parameter is **unused** inside this function. It's a no-op param. The driver still passes it (`models.py:574`).
- Returns the **non-detached** `w` (the call to `.detach()` only affects what's saved for tensorboard). So the autograd graph through `phi` is retained when this is called during `forward`.

### `compute_critic(self, w_vec, a)` — lines 2175–2207

Two paths:

(a) `a is None` — for value estimation over **all** actions:
```python
a = torch.arange(self.n_a, device=self.device)            # [n_a]
w_vec = w_vec.expand(self.n_a, -1)                        # [n_a, shared_dim]
a = F.one_hot(a, self.n_a).float()                        # [n_a, n_a]
```
Concat to `[n_a, shared_dim + n_a]`, run through `q_net` → `[n_a, 1]`.

(b) `a` provided:
```python
a = F.one_hot(a.long(), self.n_a).float()
if a.dim() == 1: a = a.unsqueeze(0)
if w_vec.dim() == 1: w_vec = w_vec.unsqueeze(0)
w_vec = w_vec.expand(a.size(0), -1)
```
- After one-hot, `a` is `[batch, n_a]`. `w_vec.expand(batch, -1)` **assumes `w_vec` was broadcastable along dim 0**, i.e. that it has size 1 in batch dim or already matches. If `w_vec` has a true batch size ≠ 1 and ≠ `a.size(0)`, `expand` will silently/explicitly fail. In practice the driver calls it with shapes that line up (batch already matches).
- Concat: `[batch, shared_dim + n_a]` → `q_net` → `[batch, 1]`.
- `self.q_values = q_values.detach()` for tensorboard.

### `compute_target_critic(self, w_vec, a)` — lines 2209–2236

Mirrors `compute_critic` but ends with:
```python
w_vec = w_vec.unsqueeze(1).expand(a.size(0), a.size(1), -1)  # Expand to [120,6,32]
...
return torch.max(q_values, dim=1)[0].squeeze(-1)
```

Crucial differences:
- The else-branch (when `a` is provided) builds a **3-D** input `[batch, a.size(1), shared_dim + n_a]` — but `a` after one-hot is `[batch, n_a]`, so `a.size(1) == n_a`. The result is a Q-value tensor of shape `[batch, n_a, 1]`.
- Returns `max(q, dim=1)[0].squeeze(-1)` → shape `[batch]`. This is `max_{a'} Q_target(w, a')`.
- So **even though the caller passes an action `a`, the method ignores it semantically** — it always returns the max over actions. The action one-hot just sets up the broadcast shape. This is a **bug-like surprise**: in the driver (models.py:703) the code reads:
  ```python
  target_q_vals = self.policy[i].compute_target_critic(next_w_out, next_actions)
  y = Rs + self.gamma * target_q_vals * (1 - dones)
  ```
  `next_actions` is computed via `compute_actions` and converted via `torch.from_numpy(next_actions).long()` — but those are **softmax probabilities** (see `compute_actions` below), not action indices, so casting to `.long()` floors them all to 0. Either way, target_q ignores them and takes the max — so the bug is masked. The Bellman target is effectively `r + γ max_{a'} Q_target(w_target_neighbor_only, a')`. See [[Bugs section]].

### `compute_reward_sharing(self, rewards, weights=None)` — lines 2238–2260

Not called from the driver's `backward` (lines 481–795 of models.py); appears to be dead/legacy. Implements:

```
r_shared_i = Σ_{j ∈ N(i)} w_{ij} * r_j
```

with either uniform `1/|N(i)|` weights or provided `weights[i][neighbors]`. Returns `torch.stack(r_shared)`.

### `compute_gradients(self, obs, w_in, actions)` — lines 2262–2321

Computes `g_i^in = ∇_{w_i^in} Q(o_i, argmax_a Q(·) ; w_i^in)` per the docstring (Yi et al. eq.).

Walkthrough:

```python
n_neighbors = w_in.size(1)   # e.g. 4
w_in_grads = []
for n in range(n_neighbors):
    w_in_n = w_in[:, n, :]                   # [B, shared_dim]
    a_all = torch.arange(self.n_a, ...)      # [n_a]
    a_all_1hot = F.one_hot(a_all, self.n_a).float()   # [n_a, n_a]
    w_in_n_expanded     = w_in_n.unsqueeze(1).expand(-1, self.n_a, -1)  # [B, n_a, shared_dim]
    a_all_1hot_expanded = a_all_1hot.unsqueeze(0).expand(w_in_n.size(0), -1, -1)  # [B, n_a, n_a]
    w_in_n_flat       = w_in_n_expanded.reshape(-1, w_in_n.size(-1))   # [B*n_a, shared_dim]
    a_all_1hot_flat   = a_all_1hot_expanded.reshape(-1, self.n_a)      # [B*n_a, n_a]
    q_input  = torch.cat([w_in_n_flat, a_all_1hot_flat], dim=-1)
    q_values = self.q_net(q_input).reshape(B, self.n_a)
    max_actions      = q_values.argmax(dim=1)                  # [B]
    max_actions_1hot = F.one_hot(max_actions, self.n_a).float()# [B, n_a]
    q_input_max = torch.cat([w_in_n, max_actions_1hot], dim=-1)# [B, shared_dim + n_a]
    q_values_max = self.q_net(q_input_max)                     # [B, 1]
    try:
        w_in_grad = torch.autograd.grad(
            q_values_max.sum(),
            w_in_n,
            create_graph=True, retain_graph=True
        )[0]
    except RuntimeError:
        w_in_grad = torch.zeros_like(w_in_n)
    w_in_grads.append(w_in_grad)
w_in_grads = torch.stack(w_in_grads, dim=1)                    # [B, n_neighbors, shared_dim]
self.w_in_grads = w_in_grads.detach()
return w_in_grads
```

Notes:
- The argmax is non-differentiable, but here it's only used to **select** which one-hot to pass to `q_net`; the gradient is taken w.r.t. `w_in_n` of `q_net(cat(w_in_n, argmax_onehot))`. This matches Yi et al.'s "envelope" trick.
- The function **must** receive `w_in` with `requires_grad=True` for `autograd.grad` to succeed. In the driver (models.py:728), `stored_w_in` comes from the buffer (a list of detached `.detach()` tensors stored by `add_transition` — see [[LToSPolicyBuffer]]), so the bare except clause hides this: `RuntimeError` falls back to zero grads.

  Actually — checking `LToSPolicyBuffer.add_transition` (utils.py:328–333): it stores `w_in` **without** explicit `.detach()` — the cloned-detached line is commented out. The driver in `IA2C_LToS.forward` builds `self.w_in[i]` from `self.w_out[j]` which is the un-detached return of `compute_w_out` (the `.detach()` there only assigns to `self.w_out` tracker, not the returned tensor). So `stored_w_in` retains the graph until the buffer reset clears tensors. **However**, `sample_transition` stacks them later, and the original autograd graph may already be freed (`retain_graph` not used at compute time). So in practice the `RuntimeError` fallback to zeros is likely the common path. → see [[Bugs section]].

### `compute_actions(self, obs, w_in, done=None, epsilon=0.1)` — lines 2323–2381

```python
if isinstance(obs, np.ndarray):
    obs_i = torch.from_numpy(obs).float().to(self.device)
else:
    obs_i = obs

if not isinstance(w_in, list):
    w_in = [w_in]
self.w_in = w_in                      # tensorboard
n_n = len(w_in)
```

If `n_n > 0`:
```python
w_in_cat = torch.cat(w_in, dim=-1)
if obs_i.dim() == 1: obs_i = obs_i.unsqueeze(0)
if w_in_cat.dim() == 1: w_in_cat = w_in_cat.unsqueeze(0)
input_vec = torch.cat([obs_i, w_in_cat], dim=-1)   # [B, n_s + shared_dim * n_n]
```
If `n_n == 0`:
```python
input_vec = obs_i                     # [B, n_s] — but the LSTM expects n_s + shared_dim*n_neighbors!
```
**No-neighbor branch is broken** when `n_neighbors > 0` was used to size the LSTM input. If `n_neighbors == 0` at construction, the LSTM was built with input_size = `n_s + 0` and it works. But if some agents have `n_neighbors > 0` and at runtime `w_in` arrives empty, the LSTM will fail with a shape mismatch. The `else` branch is essentially safe only when `n_neighbors == 0` from the start.

LSTM step (lines 2360–2375):
```python
h, c = torch.chunk(self.states_fw, 2, dim=1)
if done is not None:
    done = torch.tensor(done, dtype=torch.float32).to(self.device)
    h = h * (1 - done)
    c = c * (1 - done)
batch_size = input_vec.size(0)
h = h.expand(batch_size, -1)
c = c.expand(batch_size, -1)
h_new, c_new = self.lstm_cell(input_vec, (h, c))
self.states_fw = torch.cat([h_new, c_new], dim=1).detach()
logits = self.actor_head(h_new)
```

- **LSTM state semantics:** `states_fw` is `[1, 2*shared_dim]` (from `_reset`). The `.expand(batch_size, -1)` shares the same state across all batch items — this is fine at action time (batch=1) but **at training time the driver passes a batch of `B` transitions through the same hidden state**, which is incorrect for proper LSTM truncation. Each transition in the batch should have its own hidden state.
- After the LSTM step, `states_fw = cat(h_new, c_new).detach()` — `detach` cuts backprop through time. Effectively the LSTM is used like a feedforward network with carried state.
- `done` zero-resets `(h, c)` element-wise — but only the latest state, not per-transition.

Non-LSTM (line 2377): `logits = self.actor_layer(input_vec)`.

Final two lines:
```python
logits = F.softmax(logits, dim=1).squeeze()
return logits.detach().cpu().numpy()
```
- **Returns softmax probabilities, not action samples or argmax.** The signature comment says "return logits for each action," but they're actually softmax probs (after softmax) and called "actions" by callers. The driver `forward` returns this list as `actions` (models.py:594), where each "action" is a probability vector of length `n_a`. Whatever consumes this had better do sampling/argmax itself.
- `.squeeze()` removes singleton dims, which is fine for batch=1 but lossy for multi-batch.

### `_reset(self)` — lines 2383–2387

```python
if self.use_lstm:
    self.states_fw = torch.zeros(1, self.shared_dim * 2, device=self.device)
    self.target_states_fw = torch.zeros(1, self.shared_dim * 2, device=self.device)
```
Only allocates LSTM states; no-op if `use_lstm` is False.

### `soft_update(self, tau=0.01)` — lines 2389–2402 (**FIRST DEFINITION**)

```python
def soft_update(self, tau=0.01):
    """Soft update target networks using polyak averaging"""
    for target_param, param in zip(self.target_phi.parameters(), self.phi.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    if self.use_lstm:
        for target_param, param in zip(self.target_lstm_cell.parameters(), self.lstm_cell.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.target_actor_head.parameters(), self.actor_head.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    else:
        for target_param, param in zip(self.target_actor_layer.parameters(), self.actor_layer.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
```
- Updates target φ + target μ (LSTM or non-LSTM).
- **Does NOT update `target_q_net`** in this version.

### `soft_update_target_phi(self, tau=0.01)` — lines 2404–2407

```python
def soft_update_target_phi(self, tau=0.01):
    """Soft update target high-level policy (θ_i')"""
    for target_param, param in zip(self.target_phi.parameters(), self.phi.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
```
Only φ.

### `soft_update_target_policy(self, tau=0.01)` — lines 2409–2421

```python
def soft_update_target_policy(self, tau=0.01):
    """Soft update target low-level policy (μ_i')"""
    if self.use_lstm:
        for target_param, param in zip(self.target_lstm_cell.parameters(), self.lstm_cell.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.target_actor_head.parameters(), self.actor_head.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    else:
        for target_param, param in zip(self.target_actor_layer.parameters(), self.actor_layer.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    # Also update target Q-network
    for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
```
- Updates μ + the **target Q-net**. (Note the docstring says "low-level policy" but it bundles the Q-net update here.)

### `soft_update(self, tau=0.01)` — lines 2423–2426 (**SECOND DEFINITION — overrides the first**)

```python
def soft_update(self, tau=0.01):
    """Legacy method that updates all target networks at once"""
    self.soft_update_target_phi(tau)
    self.soft_update_target_policy(tau)
```

### The duplicate `soft_update` method

There are **two** definitions of `soft_update` in this class. Python class-body evaluation is top-to-bottom: the second binding **shadows** the first. So:

- **First `soft_update` (lines 2389–2402)** — does φ + μ, **omits Q-net**. **DEAD CODE.**
- **Second `soft_update` (lines 2423–2426)** — calls `soft_update_target_phi` + `soft_update_target_policy`, which **does update Q-net** (because `soft_update_target_policy` includes it).

Practical impact: if anything calls `policy.soft_update(tau)`, it gets the second (full) version. The driver `IA2C_LToS.backward` (models.py:786–788) calls the two split methods explicitly anyway, so neither version of `soft_update` is invoked from the main training loop. The first is doubly dead. The second exists only as a legacy convenience.

### `_update_tensorboard(self, summary_writer, global_step)` — lines 2428–2471

Writes to tensorboard:
- `agent_{i}/high_level/q_loss`, `policy_loss`, `total_loss` (lines 2440–2444)
- `agent_{i}/low_level/actor_loss` (line 2448)
- `agent_{i}/q_values/{mean,max,min}` (2452–2454)
- `agent_{i}/weights/w_out_norm` (2458)
- `agent_{i}/weights/w_in_norm` — mean of per-neighbor norm (2461–2462)
- `agent_{i}/gradients/w_in_grad_norm` (2466)
- `agent_{i}/exploration/epsilon` (2470)

Quirks:
- The `if hasattr(self, 'q_loss')` etc. always pass because these are set to `0.0` in `__init__`. Effectively unconditional.
- `q_loss`, `policy_loss`, `total_loss`, `actor_loss` are **never updated by the policy itself**. The driver only sets `self.q_losses[i]` / `self.policy_losses[i]` on the *driver* (models.py:711, 715), not on the policy. So these tensorboard scalars stay at `0.0` forever.
- Conversely, the driver does write to its own tensorboard via `_update_tensorboard` (models.py:806), but the **per-agent** loss bars are commented out (models.py:815–817). So the actual loss curves are not logged from either side. → [[Bugs section]].

---

## How LToS does message passing — neighbors exchange w_out/w_in embeddings, not raw observations

The flow (lines 559–605 of `models.py::IA2C_LToS.forward`, calling the policy's methods):

1. **Each agent encodes its own obs** to a `shared_dim` embedding:
   ```python
   w = self.policy[i].compute_w_out(obs[i], agent_id=i, epsilon=self.current_epsilon)
   self.w_out.append(w)         # w_out[i] = φ_i(o_i)
   ```

2. **Each agent collects neighbors' embeddings** (lines 578–582):
   ```python
   self.w_in = [[] for _ in range(self.n_agent)]
   for i in range(self.n_agent):
       neighbors = torch.where(self.neighbor_mask[i] == 1)[0]
       for j in neighbors:
           self.w_in[i].append(self.w_out[j])
   ```
   So `self.w_in[i]` is a list of `n_neighbors_i` tensors, each `[1, shared_dim]`, **from neighbors' encodings** — never the raw observation.

3. **Each agent acts** conditioned on own obs + neighbor embeddings:
   ```python
   action = self.policy[i].compute_actions(obs[i], self.w_in[i], ...)
   ```
   In `compute_actions` (line 2344): `w_in_cat = torch.cat(w_in, dim=-1)` then `input_vec = torch.cat([obs_i, w_in_cat], dim=-1)`.

Compared with [[05-MA2C]] / NCMultiAgent which exchange raw `fp` (action distributions) or `obs_i` neighbor-stacks: LToS exchanges **learned low-dim embeddings**. The neighbor sees `φ_j(o_j)`, not `o_j` itself. Communication width: `n_neighbors * shared_dim` (e.g. 4 × 32 = 128) instead of `n_neighbors * n_s` (e.g. 4 × 48 = 192). The embeddings are also **learned end-to-end via the policy-gradient on the Q-net**.

`compute_gradients` then back-propagates `∂Q/∂w_in_n` so each agent can tell its neighbors **how to adjust the message they send** to maximize the receiver's Q. The driver gathers these gradients and applies them to the sender's φ via `theta_grad` (models.py:756–779).

---

## Training-loop interaction — the resampling-of-neighbor-buffer issue

In `IA2C_LToS.backward` (models.py:607–795), the outer loop is over agents `i`:

```python
for i in range(self.n_agent):
    obs, nactions, actions, dones, Rs, Advs, stored_w_in = \
        self.trans_buffer[i].sample_transition(Rends[i], dt)   # agent i's buffer
    ...
    for j in neighbors:
        neighbor_obs, _, _, _, _, _, _ = \
            self.trans_buffer[j].sample_transition(Rends[j], dt)   # *** RESAMPLES neighbor's buffer ***
        if len(neighbor_obs) == 0:
            continue
        neighbor_obs = torch.from_numpy(neighbor_obs).float().to(self.device)
        next_w_out = self.policy[j].target_phi[0](neighbor_obs)
        valid_next_w_in.append(next_w_out)
```

Why this is suspicious — looking at `LToSPolicyBuffer.sample_transition` (utils.py:335–354):

```python
def sample_transition(self, R, dt=0):
    ...
    obs = np.array(self.obs, dtype=np.float32)
    ...
    stored_w_in = self.w_in[:len(obs)]
    self.w_in = self.w_in[len(obs):]   # consume w_in
    self.reset(self.dones[-1])         # *** clears obs, adds, acts, rs, Rs, Advs, vs, dones ***
    return obs, nas, acts, dones, Rs, Advs, stored_w_in
```

`reset` (utils.py:316–326) explicitly drops all the transition arrays. **`sample_transition` is destructive — it drains the buffer.**

Consequences:

1. **When agent `i`'s loop hits neighbor `j` (line 663), it calls `self.trans_buffer[j].sample_transition(Rends[j], dt)`**, which **drains agent j's buffer**.
2. When the outer loop reaches `i = j`, agent j's `sample_transition` returns empty arrays (`len(obs) == 0`), so `continue` on line 623 fires and **agent j's actual update is skipped this round**.
3. Worse, if multiple agents share a common neighbor `k`, the **first** agent to reach `k` in the inner loop drains it; every later agent gets empty `neighbor_obs` and hits the `continue` on line 667, then later pads with zeros (line 681). So most agents see **zeroed neighbor embeddings** in `next_w_in` for target computation.
4. The `Rends[j]` passed to `sample_transition` is the **bootstrap return for the inner index `j` at the current time** — but the function then calls `self.reset(self.dones[-1])` which wipes everything, so when `j`'s own outer iteration comes around, there's nothing to sample. Reset preserves the last `done` flag, but all returns/values are gone.

This breaks the algorithm in at least three ways:
- **No symmetric updates across agents** in one `backward` call — only the first ~few agents (or even just agent 0) effectively train each round.
- **Bellman target uses near-empty/zero-padded neighbor embeddings**, so `target_q_vals` is mostly junk for late agents.
- **`stored_w_in` (the recorded message history) is consumed independently** of the obs arrays at the inner-loop calls — but at the inner call, only `obs` is unpacked (line 663: `neighbor_obs, _, _, _, _, _, _ = …`). So the discarded `stored_w_in` for agent j is also lost when j's outer iteration arrives.

The intended Yi et al. design uses **independent buffers** and the neighbor-target lookup should be done by **peeking** at the recorded `obs` for the same timesteps, not by calling `sample_transition` (which is consumption). A correct implementation would either share one buffer with timestamp-aligned slices, or store neighbor obs directly in this agent's buffer when adding the transition.

→ Documented as the **primary correctness bug** in [[Bugs section]] below.

---

## Tensor-shape walkthrough for a typical forward

Assume: `n_s=48`, `n_a=6`, `shared_dim=32`, `n_fc=32`, `use_lstm=True`, `n_neighbors=4`, batch=1 at action time.

| Step | Tensor | Shape | Source |
|---|---|---|---|
| obs in | `obs_i` | `[1, 48]` | `compute_w_out` line 2163 |
| phi → w_out | `w` | `[1, 32]` | `phi(obs_i)` line 2165 |
| collect w_in | `self.w_in[i]` (list) | `4 × [1, 32]` | `IA2C_LToS.forward` 580–582 |
| cat w_in | `w_in_cat` | `[1, 128]` | `compute_actions` 2344 |
| input_vec | `[obs ; w_in]` | `[1, 176]` | line 2353 |
| states_fw | `[h ; c]` | `[1, 64]` | `_reset` |
| h, c after chunk | each | `[1, 32]` | line 2361 |
| h_new, c_new | each | `[1, 32]` | line 2373 |
| states_fw new | `[h_new ; c_new]` | `[1, 64]` | line 2374 |
| logits | | `[1, 6]` | `actor_head(h_new)` 2375 |
| return | softmax-probs after .squeeze() | `[6]` | line 2379, 2381 |

For backward with batch B = 120 (per the comments in the driver):

| Step | Tensor | Shape |
|---|---|---|
| obs | | `[120, 48]` |
| `w_out = self.phi[0](obs)` | only first Linear! | `[120, 32]` |
| `stored_w_in` (after stack/squeeze) | | `[120, 4, 32]` |
| `compute_gradients(obs, stored_w_in, actions)` per-n: |  |  |
| `w_in_n` | | `[120, 32]` |
| `q_input` | | `[120 * 6, 32 + 6] = [720, 38]` |
| `q_values` | reshaped | `[120, 6]` |
| `q_input_max` | | `[120, 38]` |
| `q_values_max` | | `[120, 1]` |
| `w_in_grad` per neighbor | | `[120, 32]` |
| stacked `w_in_grads` | | `[120, 4, 32]` |
| target step: `next_w_out` per neighbor | | `[B_j, 32]` (B_j typically 0 because of drain bug) |
| `next_w_in` after stack+reshape | | `[B_i, 128]` (if padded) |

---

## Bugs / oddities (consolidated)

1. **Duplicate `soft_update` definition** — lines 2389 and 2423. The second silently overrides the first; the first (which omits Q-net update) is dead. The shadowed first is bugged-by-omission (no Q-net polyak).

2. **Driver bypasses φ's MLP**: `IA2C_LToS.backward` (models.py:644, 736) calls `self.policy[i].phi[0](obs)` — indexing into the `nn.Sequential` to grab just the **first `Linear`**, skipping `ReLU` + second `Linear`. So during training, the embedding is the linear projection only; during action selection (`compute_w_out`) it's the full MLP. **The two are inconsistent.** Same applies to `self.policy[j].target_phi[0]` on line 670.

3. **Destructive neighbor-buffer resample** in `IA2C_LToS.backward` lines 663–676. `sample_transition` drains the neighbor's buffer (utils.py:351–353), so subsequent agents in the outer loop have nothing left. See [[Training loop interaction]] section above. This effectively makes most agents' updates degenerate to "skip" (line 623) or "zero-padded targets" (line 681).

4. **`compute_target_critic` ignores the action argument** (lines 2228–2236). The else-branch builds a `[B, n_a, n_a]` action one-hot block, then computes `max` over actions. The caller's `next_actions` is irrelevant to the result. Compounding this, the caller (models.py:697–700) passes the **softmax-probability vector** as `next_actions` and casts via `.long()` — which floors all positive-<1 entries to 0. Both bugs cancel: target ends up `max_a' Q_target(w_next, a')`, the right Bellman target by accident.

5. **`compute_actions` returns softmax probabilities, not actions**, despite the name and signature comment "return logits for each action". The downstream caller `IA2C_LToS.forward` just stores them in `actions` and returns; in `backward` line 697 they're cast to `.long()` and used as Bellman-target action indices, which collapses everything to 0 (see #4).

6. **`compute_actions` no-neighbor branch is shape-broken** when the LSTM was sized for `n_neighbors > 0` (line 2358 builds `input_vec = obs_i` of width `n_s`, but `lstm_cell` expects `n_s + shared_dim * n_neighbors`). Only safe if `n_neighbors == 0` at construction.

7. **LSTM hidden state shared across batch transitions** (lines 2370–2371): `.expand(batch_size, -1)` means every transition in a B=120 batch uses the same `(h, c)`. There is no per-timestep BPTT, no per-sequence state — effectively, the LSTM is a stateless feedforward at training time, with whatever state happened to be in `self.states_fw` from the last action call.

8. **`states_fw.detach()` on line 2374**: cuts the recurrent autograd chain after every step. No BPTT possible.

9. **`agent_id` parameter to `compute_w_out`** (line 2156) is **unused** in the function body.

10. **Tensorboard loss scalars are never updated**: `_update_tensorboard` reads `self.q_loss`, `self.policy_loss`, `self.total_loss`, `self.actor_loss` — but the driver writes to `self.policy[i].q_losses[i]` and `policy_losses[i]` on the **driver**, not on the policy. So the policy-level scalars stay at the initial `0.0`. (Lines 2440–2448 always log `0.0`.)

11. **`compute_gradients` swallows RuntimeError silently** (lines 2309–2311) and substitutes `zeros_like(w_in_n)`. If `stored_w_in` lost its autograd graph (which it usually does because the original `forward` graph is freed before `backward` runs), this except branch fires every time and **all message-direction gradients are zero**. → φ never learns from the receiver-side signal that LToS is supposed to provide. Combined with bug #3, the "two-level" learning is effectively non-functional.

12. **`compute_critic.expand` assumption** (line 2196): `w_vec.expand(a.size(0), -1)` requires `w_vec.shape[0] ∈ {1, a.size(0)}`. If batch dims mismatch, expand raises. Not robust.

13. **`compute_target_critic` `unsqueeze(0)` path** (line 2219, the `a is None` case): `w_vec.unsqueeze(0).expand(self.n_a, -1)` — assumes `w_vec` was 1-D. If `w_vec` has a batch dim, this adds another, producing `[1, B, shared_dim]` then expands the first dim to `n_a`. The else-branch handles 2-D differently. Inconsistent handling of input rank.

14. **`super().__init__(n_a, n_s, n_step, 'lstm', 'lstm', identical)`** (line 2059) passes `'lstm'` as the `agent_name`, so every LToS policy across all agents gets the same `self.name = 'lstm_lstm'`. The `agent_id` is **not** included in the name. Cosmetic.

15. **`self.epsilon = 1.0` on the policy** (line 2083) is dead — the driver maintains `current_epsilon` and passes it as an argument to `compute_w_out` / `compute_actions`. The policy attribute is only ever read by `_update_tensorboard` (line 2470).

---

## Summary

`LToSMultiAgentPolicy` provides the per-agent building blocks for LToS:
- φ (high-level / message encoder, mapping `o_i` → `w_out_i ∈ ℝ^{shared_dim}`)
- μ (low-level / action policy, taking `[o_i ; w_in]` → action distribution, LSTM or MLP)
- a Q-net used both for the actor's policy-gradient term and for receiver-side message gradients (`compute_gradients`)
- target φ, target μ, target Q
- soft-update helpers (with a duplicate definition; the kinder one wins)

It is structurally faithful to Yi et al. (NeurIPS 2022), but the *driver* `IA2C_LToS.backward` has multiple implementation bugs (#2, #3, #4, #11) that together neutralize most of the learning signal. Notably: φ is updated using only its first linear layer during training; neighbor target embeddings are mostly zero-padded due to destructive buffer resampling; receiver-side message gradients (the whole point of LToS) silently degrade to zero whenever the autograd graph is freed before `compute_gradients` runs. Use this file as a reference but **do not treat it as a working LToS baseline without fixing #2, #3, and #11 at minimum**.

See also: [[IA2C_LToS-models]], [[LToSPolicyBuffer]], [[05-MA2C]], [[03-Policy-base]].
