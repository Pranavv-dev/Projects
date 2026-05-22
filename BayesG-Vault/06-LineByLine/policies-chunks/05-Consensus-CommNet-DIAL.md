# 05 — Consensus, CommNet, DIAL Multi-Agent Policies

Line-by-line walkthrough of three communication-flavoured subclasses of [[NCMultiAgentPolicy]] in `/tmp/BayesG/agents/policies.py`:

- `ConsensusPolicy` (lines 1686–1837) — used by [[IA2C_CU]]
- `CommNetMultiAgentPolicy` (lines 1840–2002) — used by [[MA2C_CNET]]
- `DIALMultiAgentPolicy` (lines 2004–2045) — used by [[MA2C_DIAL]]

All three inherit from `NCMultiAgentPolicy` but plug in their own communication aggregation. Some also override `_init_net`, `_run_comm_layers`, and `backward`.

---

## class ConsensusPolicy(NCMultiAgentPolicy) — lines 1686–1837

### `__init__` (lines 1687–1703)

```python
def __init__(self, n_s, n_a, n_agent, n_step, neighbor_mask, n_fc=64, n_h=64,
             n_s_ls=None, n_a_ls=None, identical=True):
    Policy.__init__(self, n_a, n_s, n_step, 'cu', None, identical)
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

- Line 1689: Notice the call is `Policy.__init__`, **not** `super().__init__()`. The agent_name is `'cu'` (consensus update).
- Line 1690: device picked from `torch.cuda.is_available()`.
- Lines 1692–1694: Only stores `n_s_ls` / `n_a_ls` when agents are heterogeneous.
- Lines 1695–1700: Stores network config and calls `self._init_net()` then `self._reset()` (inherited from `NCMultiAgentPolicy`).
- Line 1703: `self.max_grad_norm = 0.5` — explicit gradient clipping is set up at the policy level (the parent class `MA2C_NC.backward` in `models.py` also clips via `nn.utils.clip_grad_norm_`, so this is a second clip — see Bugs section).

### `_init_net` (lines 1705–1734)

Builds **per-agent** networks. The key thing: no `fc_p_layers` / `fc_m_layers` exist. The consensus policy doesn't read neighbor states/policies/messages at runtime — each agent is independent locally, then weights are averaged across the neighborhood after backward:

```python
self.fc_x_layers = nn.ModuleList()
self.lstm_layers = nn.ModuleList()
self.actor_heads = nn.ModuleList()
self.critic_heads = nn.ModuleList()
```

For each agent `i`:
- Line 1713: `_get_neighbor_dim(i)` returns `(n_n, n_ns, n_na, _, na_ls)`. Note `ns_ls` is discarded with `_`.
- Lines 1719–1722: `fc_x_layer = nn.Linear(n_s, self.n_fc)` — input dim is **own** `n_s` (not `n_ns`!). Each agent's MLP only sees its own obs.
- Lines 1720–1721: orthogonal init with gain `sqrt(2)`, zero bias.
- Lines 1724–1730: `LSTMCell(self.n_fc, self.n_h)` with custom orthogonal init for weight params, zero init for biases.
- Line 1733: `_init_actor_head(n_a)` — actor reads only own hidden state.
- Line 1734: `_init_critic_head(n_na)` — critic takes `n_na` extra dims (neighbor actions). This is inherited; the actual critic forward in `NCMultiAgentPolicy._run_critic_heads` concatenates neighbor actions onto `h_i` before the linear head. So although there's no fc_p / fc_m, the critic still consumes neighbor actions as features.

### `consensus_update` (lines 1736–1742)

```python
def consensus_update(self):
    consensus_update = []
    with torch.no_grad():
        for i in range(self.n_agent):
            mean_wts = self._get_critic_wts(i)
            for param, wt in zip(self.lstm_layers[i].parameters(), mean_wts):
                param.copy_(wt)
```

What it does:
- Iterates over every agent.
- Calls `self._get_critic_wts(i)` which returns the **mean** of the LSTM weights of agent `i` and its neighbors (despite the name "critic_wts", it averages **LSTM** weights, not critic head weights).
- Overwrites each agent's LSTM parameters in-place with the mean.

The local variable `consensus_update = []` on line 1737 is allocated but never used — dead code (shadows the method name in scope but doesn't break anything).

### `_get_critic_wts` (lines 1744–1755)

```python
def _get_critic_wts(self, i_agent):
    wts = []
    for wt in self.lstm_layers[i_agent].parameters():
        wts.append(wt.detach())
    neighbors = list(torch.where(self.neighbor_mask[i_agent] == 1)[0])
    for j in neighbors:
        for k, wt in enumerate(self.lstm_layers[j].parameters()):
            wts[k] += wt.detach()
    n = 1 + len(neighbors)
    for k in range(len(wts)):
        wts[k] /= n
    return wts
```

- Line 1747: detach so accumulation doesn't track gradients.
- Line 1748: `torch.where(self.neighbor_mask[i_agent] == 1)[0]` finds neighbor indices.
- Lines 1749–1751: sums in-place into `wts[k]` (this mutates the detached tensor of agent `i`'s own LSTM... see Bugs).
- Line 1752: `n = 1 + len(neighbors)` accounts for self + neighbors.
- Lines 1753–1754: in-place divide.

**Important subtlety**: at line 1747 `wt.detach()` returns a view sharing storage with the original parameter. Then line 1751 does `wts[k] += wt.detach()`, which is in-place addition on that view — but since detach shares storage, this would corrupt agent `i`'s LSTM parameters. **However**, line 1754 then does `wts[k] /= n`, which is also in-place. So agent `i`'s LSTM weights are temporarily corrupted, then averaged, then in `consensus_update` line 1742 `param.copy_(wt)` copies the averaged tensor back. The net effect is roughly correct **for agent i** because the value at end is the mean. But for **agent j** processed later, when we do `wts[0] = self.lstm_layers[j].param.detach()`, agent j's params are still pristine (assuming j hasn't been "consensus-updated" yet in the same outer pass). Order matters: by the time we process agent `j`, agent `i`'s LSTM has been overwritten with the average centered at `i`. So agent `j`'s neighbor (which is `i`) contributes the already-averaged params to `j`'s average. This is a known consensus-iteration ordering issue but the algorithm is technically "Jacobi-vs-Gauss-Seidel"; here it's Gauss-Seidel style with in-place destructive updates.

### `_run_comm_layers` (lines 1757–1776)

```python
def _run_comm_layers(self, obs, dones, fps, states):
    obs = obs.transpose(0, 1).to(self.device) # [28,1,22]
    dones = torch.as_tensor(dones, device=self.device)
    hs = []
    new_states = []
    if self.identical:
        for i in range(self.n_agent):
            xs_i = F.relu(self.fc_x_layers[i](obs[i]))
            hs_i, new_states_i = run_rnn(self.lstm_layers[i], xs_i, dones, states[i])
            hs.append(hs_i.unsqueeze(0))
            new_states.append(new_states_i.unsqueeze(0))
    else:
        obs_dim = self.n_s_ls
        for i in range(self.n_agent):
            xs_i = F.relu(self.fc_x_layers[i](obs[i][:, :obs_dim[i]]))
            ...
    return torch.cat(hs), torch.cat(new_states)
```

- Line 1759: transposes obs from `[T, N, m]` to `[N, T, m]` (per inline comment shape `[28,1,22]` suggests 28 agents, 1 timestep, 22-dim obs).
- The `fps` argument is **completely ignored** — Consensus does not read neighbor fingerprints inline. Compare to NC which slurps them into `_get_comm_s`.
- Lines 1763–1768 (identical case): each agent processes own `obs[i]` through `fc_x_layers[i]` → ReLU → LSTM.
- Lines 1769–1775 (heterogeneous): same but narrows obs to per-agent `n_s_ls[i]` dims.
- Line 1776: concatenates per-agent results.

So the **inputs are local-only**; only the actor/critic heads use neighbor info indirectly through `_init_critic_head(n_na)`, and the LSTM parameters get periodically synced via `consensus_update()`.

### `backward` (lines 1778–1837)

The interesting parts beyond the NC parent:

```python
obs = torch.nan_to_num(obs, nan=0.0)
fps = torch.nan_to_num(fps, nan=0.0)
```

Lines 1787–1788: NaN scrubbing on inputs. NaNs become 0. This is defensive.

```python
hs, new_states = self._run_comm_layers(obs, dones, fps, self.states_bw)
self.states_bw = new_states.detach()
```

Lines 1790–1791: standard — run comm layers, detach new states.

Lines 1794–1795: commented-out check `if torch.isnan(hs).any(): raise ValueError(...)`. Dead.

```python
ps = self._run_actor_heads(hs)
vs = self._run_critic_heads(hs, acts)

for i, p in enumerate(ps):
    if torch.isnan(p).any():
        ps[i] = torch.nan_to_num(p, nan=-1000000.0)
        ps[i] = F.softmax(ps[i], dim=-1)
```

Lines 1801–1806: per-agent NaN check on policy logits:
- If any NaN in `p`, replace with `-1e6` (near-zero probability after softmax).
- Then apply softmax — so `ps[i]` becomes **probabilities**, not logits.

This is a **logic bug**: the very next block on line 1819 does `torch.distributions.categorical.Categorical(logits=ps[i])`. If the NaN-recovery path ran, `ps[i]` is already softmax probabilities but is being passed as `logits`, which the Categorical constructor will internally treat as unnormalized log-probs and softmax **again**. The agent that suffered NaNs ends up with a doubly-softmaxed distribution (extremely flat). For agents without NaNs, `ps[i]` stays as raw logits and is correct.

### NaN-handling guard — what is it guarding against?

Lines 1787–1788 (`nan_to_num` on `obs`/`fps`) and 1801–1806 (NaN check on logits) guard against:
- NaNs leaking in from `np.float32` overflow / divide-by-zero in the env's reward/obs normalization.
- Exploding LSTM gradients producing NaNs in `hs` which then propagate to logits.
- Hetero-padding zeros causing the actor to underflow.

### When is it triggered? — called from `MA2C_NC.backward` in `models.py`?

Indirectly. Trace:
1. [[IA2C_CU]] inherits from `MA2C_NC` (`models.py` line 398).
2. [[IA2C_CU]] overrides `backward`:
   ```python
   def backward(self, Rends, dt, summary_writer=None, global_step=None):
       super(IA2C_CU, self).backward(Rends, dt, summary_writer, global_step)
       self.policy.consensus_update()
   ```
   (`models.py` lines 414–416)
3. `super().backward(...)` is `MA2C_NC.backward` (line 313 of `models.py`), which calls `self.policy.backward(obs, ps, acts, dones, Rs, Advs, ...)` — and `self.policy` is a `ConsensusPolicy` instance, so the overridden `ConsensusPolicy.backward` (line 1778) runs.
4. After that returns, `self.policy.consensus_update()` runs — the LSTM parameter averaging happens **every** backward pass.

So NaN handling runs every gradient step; consensus averaging runs every gradient step too. Both are on the hot path.

---

## class CommNetMultiAgentPolicy(NCMultiAgentPolicy) — lines 1840–2002

Class docstring (lines 1841–1844):

```python
"""Reference code: https://github.com/IC3Net/IC3Net/blob/master/comm.py.
   Note in CommNet, the message is generated from hidden state only, so current state
   and neigbor policies are not included in the inputs.
   s_i=[MLP(obs_i, obs_neighbors)+MLP(hidden_state_neighbors)]"""
```

The docstring is slightly misleading — `obs_neighbors` (current observation of neighbors) **is** included; what's excluded relative to NC is **neighbor policies** (`fps`/fingerprints), and the message from neighbors is just **mean-pooled hidden states** rather than concatenated then projected.

### `__init__` (lines 1845–1872)

```python
Policy.__init__(self, n_a, n_s, n_step, 'cnet', None, identical)
```

- Line 1847: agent name `'cnet'`.
- Lines 1851 and 1857–1865: heterogeneous-agent dimension unification. If `unify_act_state_dim=True`, builds `convert_state_linears` (per-agent `Linear(n_s_ls[i], 16)`) and `convert_action_linears` (per-agent `Linear(n_a_ls[i], max(n_a_ls))`).

Note: `convert_state_linears` is a **plain Python list** (line 1860), not an `nn.ModuleList`. That means PyTorch will **not** register those parameters automatically — they will **not** show up in `self.parameters()` and therefore **will not be optimized**. See Bugs. The `.to(self.device)` is called on each layer in `_create_linear` so they at least live on the right device.

### `_create_linear` (lines 1874–1878)

Just a `nn.Linear(in, out).to(device)`. Identical to inline.

### `_init_net` (lines 1880–1901)

```python
self.fc_x_layers = nn.ModuleList()
self.fc_p_layers = nn.ModuleList()    # <-- created but never used!
self.fc_m_layers = nn.ModuleList()
self.lstm_layers = nn.ModuleList()
...
```

- Line 1882: `fc_p_layers` is created (likely inherited convention from NC) but **never appended to or used** in CommNet because `_init_comm_layer` only handles `fc_x` and `fc_m`. So it remains an empty `ModuleList`. Dead.
- Lines 1895–1897: if `unify_act_state_dim`, override `n_ns = self.convert_state_shape * (n_n + 1)` so the `fc_x` input matches `16 * (1 + neighbors)`.
- Line 1898: calls `self._init_comm_layer(n_n, n_ns)` (note: **2 args**, not 3 like NC's `_init_comm_layer(n_n, n_ns, n_na)`).

### `_init_comm_layer` (lines 1903–1919)

```python
fc_x_layer = nn.Linear(n_ns, self.n_fc)
...
if n_n:
    fc_m_layer = nn.Linear(self.n_h, self.n_fc)
    ...
    self.fc_m_layers.append(fc_m_layer)
else:
    self.fc_m_layers.append(None)
lstm_layer = nn.LSTMCell(self.n_fc, self.n_h)
```

Key contrast vs NC: `fc_m_layer = nn.Linear(self.n_h, self.n_fc)` — input is `n_h` (not `n_h * n_n`). This is because messages are **mean-pooled** to a single `n_h`-dim vector before being projected, not concatenated.

Also: LSTM input is `self.n_fc` (not `3 * self.n_fc` like NC). The architecture is "additive": `s_i = relu(fc_x([x_i, nx_i])) + fc_m(mean(neighbor_h))`, giving a single `n_fc` vector.

### `_run_comm_layers` (lines 1925–1977)

```python
obs = torch.as_tensor(obs, device=self.device)
dones = torch.as_tensor(dones, device=self.device)
fps = torch.as_tensor(fps, device=self.device)
obs = batch_to_seq(obs)
dones = batch_to_seq(dones)
fps = batch_to_seq(fps)
h, c = torch.chunk(states, 2, dim=1)
```

- Lines 1927–1929: convert numpy → tensor (handles both forward and backward callers).
- Lines 1933–1937: `batch_to_seq` converts `[N, T, *]` (or similar) to a Python list of `[1, *]` per timestep.
- Line 1940: split LSTM cell state into `h, c`.

For each timestep:
```python
for t, (x, p, done) in enumerate(zip(obs, fps, dones)):
    next_h = []
    next_c = []
    x = x.squeeze(0)
    p = p.squeeze(0)
    for i in range(self.n_agent):
        n_n = self.n_n_ls[i]
        if n_n:
            s_i = self._get_comm_s(i, n_n, x, h, p)
        else:
            if self.identical:
                x_i = x[i].unsqueeze(0)
            else:
                x_i_temp = x[i].narrow(0, 0, self.n_s_ls[i]).unsqueeze(0)
                if self.unify_act_state_dim:
                    x_i = self.convert_state_linears[i](x_i_temp)
                else:
                    x_i = x_i_temp
            s_i = F.relu(self.fc_x_layers[i](x_i))
        h_i = h[i].unsqueeze(0) * (1-done)
        c_i = c[i].unsqueeze(0) * (1-done)
        next_h_i, next_c_i = self.lstm_layers[i](s_i, (h_i, c_i))
        next_h.append(next_h_i)
        next_c.append(next_c_i)
    h = torch.cat(next_h)
    c = torch.cat(next_c)
    outputs.append(h.unsqueeze(0))
outputs = torch.cat(outputs)
return outputs.transpose(0, 1), torch.cat([h, c], dim=1)
```

- Lines 1944–1948: per-timestep, squeeze the batch dim.
- Lines 1950–1964: per-agent — if has neighbors, compute `s_i` via `_get_comm_s`; otherwise just process own obs (with optional dim conversion).
- Lines 1967–1968: reset hidden state on done.
- Line 1970: LSTM step.
- Lines 1973–1976: stack new states.
- Line 1977: transpose back to `[N, T, n_h]` and stack hidden+cell into `[N, 2*n_h]`.

The `p` (fingerprints) is unpacked and squeezed but **never used** in this function or downstream `_get_comm_s` — see Bugs.

### `_get_comm_s` (lines 1979–2002)

```python
js = torch.nonzero(self.neighbor_mask[i]).squeeze(1)
m_i = torch.index_select(h, 0, js).mean(dim=0, keepdim=True)
nx_i = torch.index_select(x, 0, js)
if self.identical:
    nx_i = nx_i.view(1, self.n_s * n_n)
    x_i = x[i].unsqueeze(0)
else:
    x_i = x[i].narrow(0, 0, self.n_s_ls[i]).unsqueeze(0)
    if self.unify_act_state_dim:
        x_i = self.convert_state_linears[i](x_i)
    nx_i_ls = []
    for j in range(n_n):
        nx_i_temp = x[js[j]].narrow(0, 0, self.ns_ls_ls[i][j])
        if self.unify_act_state_dim:
            nx_i_ls.append(self.convert_state_linears[js[j]](nx_i_temp))
        else:
            nx_i_ls.append(nx_i_temp)
    nx_i = torch.cat(nx_i_ls).unsqueeze(0)
return F.relu(self.fc_x_layers[i](torch.cat([x_i, nx_i], dim=1))) + \
       self.fc_m_layers[i](m_i)
```

- Line 1981: indices of neighbors.
- Line 1982: **mean-pool neighbors' hidden states** → `m_i` shape `[1, n_h]`. This is the CommNet hallmark.
- Lines 1983–1999: gather neighbor observations and concat with own (identical case) or per-agent unified (hetero case).
- Lines 2001–2002: `s_i = relu(fc_x([own_obs, neighbor_obs])) + fc_m(mean_neighbor_hidden)`. Note `fc_m` output is **not** wrapped in ReLU here, unlike NC which applies ReLU to all three streams before concat. The result is the sum, not concatenation.

### Message aggregation: NC vs CommNet

| Aspect | NC | CommNet |
|---|---|---|
| Neighbor hidden state | `view(1, n_h * n_n)` concatenated, projected by `fc_m: n_h*n_n → n_fc` | `.mean(dim=0)` over neighbors → `[1, n_h]`, projected by `fc_m: n_h → n_fc` |
| Neighbor policies (fps) | Concatenated + projected via `fc_p` | Not used at all |
| Neighbor obs | Concatenated with own, projected by `fc_x` | Same |
| How streams combine | `cat([relu(fc_x), relu(fc_p), relu(fc_m)], dim=1)` → LSTM input `3*n_fc` | `relu(fc_x(...)) + fc_m(mean_h)` → LSTM input `n_fc` |
| LSTM input size | `3 * n_fc` | `n_fc` |

CommNet is much cheaper per step but ignores neighbor policy fingerprints entirely.

### `backward`

CommNet does **not** override `backward`. It inherits `NCMultiAgentPolicy.backward` (line 228) wholesale. That backward calls `self._run_comm_layers(obs, dones, fps, ...)` which dispatches to the CommNet override.

---

## class DIALMultiAgentPolicy(NCMultiAgentPolicy) — lines 2004–2045

DIAL = Differentiable Inter-Agent Learning.

### `__init__` (lines 2005–2016)

```python
Policy.__init__(self, n_a, n_s, n_step, 'dial', None, identical)
```

- Line 2007: agent name `'dial'`.
- Identical structure to ConsensusPolicy `__init__` minus the `device` and `max_grad_norm` attributes — DIAL **doesn't** set `self.device`. This means anywhere DIAL code calls `.to(self.device)` it will inherit from a parent or fail. Actually scanning the class: `_init_comm_layer` and `_get_comm_s` don't reference `self.device`. Inherited methods like `_run_comm_layers` (from NC) use the same approach.
- Lines 2015–2016: `_init_net()` and `_reset()`. Note DIAL doesn't override `_init_net` — it uses the NC parent's `_init_net` which calls `self._init_comm_layer(n_n, n_ns, n_na, ...)` per agent. So DIAL's overridden `_init_comm_layer` is used.

### `_init_comm_layer` (lines 2018–2030)

```python
def _init_comm_layer(self, n_n, n_ns, n_na):
    fc_x_layer = nn.Linear(n_ns, self.n_fc)
    init_layer(fc_x_layer, 'fc')
    self.fc_x_layers.append(fc_x_layer)
    if n_n:
        fc_m_layer = nn.Linear(self.n_h*n_n, self.n_fc)
        init_layer(fc_m_layer, 'fc')
        self.fc_m_layers.append(fc_m_layer)
    else:
        self.fc_m_layers.append(None)
    lstm_layer = nn.LSTMCell(self.n_fc, self.n_h)
    init_layer(lstm_layer, 'lstm')
    self.lstm_layers.append(lstm_layer)
```

- Same as NC's `_init_comm_layer` but **drops the `fc_p` layer entirely** (no neighbor-policy MLP).
- `fc_m_layer` input is `n_h * n_n` (concatenated, like NC, not mean-pooled like CommNet).
- LSTM input is `self.n_fc` (same as CommNet), **not** `3 * self.n_fc` like NC. So DIAL's combining strategy must be additive too.

Note: NC's `_init_net` (parent, lines 392-onward) initializes `self.fc_p_layers = nn.ModuleList()`, so it exists as an empty/None-filled list for DIAL. The DIAL `_init_comm_layer` doesn't append to it, so it stays empty.

### `_get_comm_s` (lines 2032–2045)

```python
def _get_comm_s(self, i, n_n, x, h, p):
    js = torch.nonzero(self.neighbor_mask[i]).squeeze(1)
    m_i = torch.index_select(h, 0, js).view(1, self.n_h * n_n)
    nx_i = torch.index_select(x, 0, js)
    if self.identical:
        nx_i = nx_i.view(1, self.n_s * n_n)
    else:
        nx_i_ls = []
        for j in range(n_n):
            nx_i_ls.append(nx_i[j].narrow(0, 0, self.ns_ls_ls[i][j]))
        nx_i = torch.cat(nx_i_ls).unsqueeze(0)
    a_i = one_hot(p[i].argmax().unsqueeze(0), self.n_fc)
    return F.relu(self.fc_x_layers[i](torch.cat([x[i].unsqueeze(0), nx_i], dim=1))) + \
           F.relu(self.fc_m_layers[i](m_i)) + a_i
```

- Line 2033: neighbor indices.
- Line 2034: `m_i = ... .view(1, self.n_h * n_n)` — concatenated neighbor hidden states (DIAL-style, like NC).
- Lines 2036–2042: neighbor obs concat (identical case `view`; hetero case loop with `narrow`).
- **Line 2043**: `a_i = one_hot(p[i].argmax().unsqueeze(0), self.n_fc)` — **own** fingerprint's argmax (i.e., the action with highest probability under own policy) is one-hot encoded into a vector of size `self.n_fc`.
- **Line 2044–2045**: `s_i = relu(fc_x([own_obs, neighbor_obs])) + relu(fc_m(neighbor_h_concat)) + a_i`. The one-hot action vector is **added directly** to the LSTM input.

### How DIAL appends action info to messages

DIAL doesn't actually "append" — it **adds** a one-hot encoding of the agent's own current most-likely action (`p[i].argmax()`, where `p` here is the fingerprint passed in) directly to the combined state vector. The one-hot is dimensioned to `self.n_fc`, the LSTM input width. So action information becomes a sparse additive bias to the LSTM input.

Quirks:
1. The "action info" being added is **own** action (`p[i]`), not neighbors' actions. So the message DIAL receives from neighbors is still just hidden states (`m_i`); the own-action term is more of a self-conditioning signal.
2. `one_hot(p[i].argmax(), self.n_fc)` requires `self.n_fc >= n_a` (else the scatter index would error). Configurable but a potential trap.
3. `argmax` is **non-differentiable**, breaking the "differentiable" claim of DIAL in the original paper. The proper DIAL would pass continuous logits or use Gumbel-softmax.

### Compare to CommNet

| Aspect | CommNet | DIAL |
|---|---|---|
| Neighbor hidden state aggregation | mean → `[1, n_h]` → `fc_m: n_h→n_fc` | concat → `[1, n_h*n_n]` → `fc_m: n_h*n_n→n_fc` |
| Action info | None | Own action argmax, one-hot to `n_fc`, added |
| Neighbor obs | Concatenated with own | Concatenated with own (same) |
| ReLU on `fc_m` output | No | Yes |
| LSTM input width | `n_fc` | `n_fc` |
| Combine | `relu(fc_x) + fc_m(mean_h)` | `relu(fc_x) + relu(fc_m(cat_h)) + a_i` |

DIAL is closer to NC architecturally (concat hidden states) but drops the per-neighbor policy stream and replaces it with a single own-action one-hot. CommNet is the minimalist: no action info, mean-pool hidden.

---

## Comparison Table — NC vs Consensus vs CommNet vs DIAL

| Feature | NCMultiAgentPolicy | ConsensusPolicy | CommNetMultiAgentPolicy | DIALMultiAgentPolicy |
|---|---|---|---|---|
| `agent_name` | (parent) `'nc'` | `'cu'` | `'cnet'` | `'dial'` |
| Neighbor obs shared? | Yes, concat into `fc_x` | **No** (only own obs) | Yes, concat into `fc_x` | Yes, concat into `fc_x` |
| Neighbor policies (fps) shared? | Yes, concat → `fc_p` | No | No | Own-action one-hot only |
| Neighbor hidden states shared? | Yes, concat → `fc_m: n_h*n_n→n_fc` | No (only synced via weight averaging) | Yes, **mean-pool** → `fc_m: n_h→n_fc` | Yes, concat → `fc_m: n_h*n_n→n_fc` |
| Stream combine | `cat(3 relu streams)`, LSTM in `3*n_fc` | None — only own | Additive `relu(fc_x) + fc_m(mean_h)`, LSTM in `n_fc` | Additive `relu(fc_x) + relu(fc_m) + a_i`, LSTM in `n_fc` |
| Critic input | own h + neighbor actions (`n_na`) | own h + neighbor actions (`n_na`) | own h + neighbor actions (`n_na`) | own h + neighbor actions (`n_na`) |
| Inter-agent coupling | Per-step, through inputs | Per-step (critic actions) + periodic LSTM weight averaging | Per-step, through inputs | Per-step, through inputs |
| Overrides backward? | (it is the parent) | Yes — adds NaN guards + extra grad clip | No (uses NC's) | No (uses NC's) |
| Overrides `_run_comm_layers`? | No | Yes (own-only obs) | Yes (additive combine, ignores `p`) | No (uses NC's, which calls overridden `_get_comm_s`) |
| Overrides `_get_comm_s`? | (it is the parent) | N/A (no comm) | Yes (mean-pool) | Yes (one-hot action + concat hidden) |

---

## Cross-references — `models.py`

### [[IA2C_CU]] (`models.py` lines 398–416)

```python
class IA2C_CU(MA2C_NC):
    def __init__(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                 total_step, model_config, seed=0, use_gpu=False):
        self.name = 'ma2c_cu'
        self._init_algo(...)

    def _init_policy(self):
        if self.identical_agent:
            return ConsensusPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                   self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm)
        else:
            return ConsensusPolicy(... identical=False)

    def backward(self, Rends, dt, summary_writer=None, global_step=None):
        super(IA2C_CU, self).backward(Rends, dt, summary_writer, global_step)
        self.policy.consensus_update()
```

The only thing IA2C_CU does beyond MA2C_NC is call `consensus_update()` after each backward.

### [[MA2C_CNET]] (`models.py` lines 465–479)

```python
class MA2C_CNET(MA2C_NC):
    def __init__(self, ...):
        self.name = 'ma2c_ic3'  # note: name is 'ma2c_ic3' not 'ma2c_cnet'
        self._init_algo(...)

    def _init_policy(self):
        if self.identical_agent:
            return CommNetMultiAgentPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                           self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm)
        else:
            return CommNetMultiAgentPolicy(... identical=False)
```

The class is `MA2C_CNET` (suggesting CommNet) but `self.name = 'ma2c_ic3'` (referring to IC3Net). Reflects the docstring on line 1841 pointing at the IC3Net repo. Whatever logging/checkpoint code keys off `self.name` will use `'ma2c_ic3'`. Note also `_init_policy` never passes `unify_act_state_dim`, so the hetero unify path in `CommNetMultiAgentPolicy.__init__` (lines 1857–1865) cannot be activated from MA2C_CNET — it'd require a different caller or hand-construction.

### [[MA2C_DIAL]] (`models.py` lines 449–463)

```python
class MA2C_DIAL(MA2C_NC):
    def __init__(self, ...):
        self.name = 'ma2c_dial'
        self._init_algo(...)

    def _init_policy(self):
        if self.identical_agent:
            return DIALMultiAgentPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                        self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm)
        else:
            return DIALMultiAgentPolicy(... identical=False)
```

Trivial subclass — just swaps the policy class.

---

## Bugs / Oddities

### ConsensusPolicy

1. **Dead local `consensus_update = []`** (line 1737). Allocated, never used. Shadows the method name in local scope but harmless.
2. **In-place mutation of detached views** in `_get_critic_wts` (line 1747 `wts.append(wt.detach())`, line 1751 `wts[k] += wt.detach()`, line 1754 `wts[k] /= n`). `wt.detach()` shares storage with the original `Parameter`. Each `+=` and `/=` mutates the underlying tensor. This means by the time `consensus_update` (line 1742) calls `param.copy_(wt)`, the source `wt` has already been modified in-place. The result is mathematically correct for agent `i` (its own params end up equal to the mean), but it causes destructive ordering effects on subsequent agents (Gauss-Seidel-style consensus). May or may not be intentional.
3. **NaN-recovery path produces wrong inputs to Categorical**: at lines 1801–1806, the NaN-tainted logits are first `nan_to_num`'d, then `F.softmax` is applied. The resulting tensor is **probabilities**. Then on line 1819, `Categorical(logits=ps[i])` is called, which **re-softmaxes** them. For affected agents, the action distribution becomes essentially uniform (double-softmax → flat). Non-NaN agents are unaffected.
4. **Commented-out NaN guard** at lines 1794–1795 — would have raised on NaN in hidden states. Removed but not deleted.
5. **Double gradient clipping**: `ConsensusPolicy.backward` clips with `max_grad_norm=0.5` on `self.parameters()` (line 1834), then `MA2C_NC.backward` (parent in models.py line 319–320) clips again with `self.max_grad_norm` (config-driven). Two consecutive `clip_grad_norm_` calls don't break anything but the second one is on already-clipped grads with a different threshold. Likely unintended.
6. **`fps` arg ignored** in `_run_comm_layers` (line 1757). The signature accepts it for API compatibility but Consensus genuinely doesn't use it.
7. **`Policy.__init__` direct call** (line 1689) instead of `super().__init__()`. Breaks MRO if the inheritance ever gets diamond-shaped, but currently fine.

### CommNetMultiAgentPolicy

8. **`convert_state_linears` and `convert_action_linears` are plain Python lists**, not `nn.ModuleList` (lines 1860, 1864). PyTorch's `.parameters()` and `.to()` won't recurse into a plain list. So those Linear layers' parameters **won't be returned by `self.parameters()`** and won't be touched by the optimizer set up in `MA2C_NC._init_algo`. They will not learn. Workaround happens because `_create_linear` calls `.to(self.device)`, so they at least live on GPU.
9. **`convert_action_linears` is initialized but never used** in the visible code. Created on line 1864 but `_get_comm_s` and `_run_comm_layers` never reference it.
10. **`fps` parameter is unpacked, squeezed, but never consumed** (lines 1937, 1944, 1948). `p` flows through to `_get_comm_s` as an argument but CommNet's `_get_comm_s` (line 1979) doesn't reference `p` either. Pure dead data flow.
11. **`fc_p_layers = nn.ModuleList()` allocated but never populated** (line 1882). Dead.
12. **`unify_act_state_dim` is plumbed but never used from MA2C_CNET**. The `_init_policy` in `MA2C_CNET` (models.py 472–479) doesn't pass `unify_act_state_dim`, so the convert paths are dead from the normal entry point.
13. **Asymmetric ReLU**: in `_get_comm_s` line 2001–2002, `fc_x` is ReLU'd but `fc_m_layers[i](m_i)` is not. Compare to DIAL which ReLUs both. Minor architectural inconsistency.

### DIALMultiAgentPolicy

14. **`p[i].argmax()` is non-differentiable** (line 2043). For an algorithm literally named "Differentiable Inter-Agent Learning", using argmax to extract the action signal defeats the differentiability claim. The original DIAL passes continuous communication channels.
15. **One-hot dimension is `n_fc`, not `n_a`** (line 2043). This requires `n_fc >= n_a` and the resulting one-hot vector has mostly-zeros padding. Could be intentional (to add to LSTM input of width `n_fc`) but the action information is sparse in a high-dim space.
16. **The "action" in `p[i]` is the agent's **own** action**, not a message from neighbors. In NC, the analogous `p` was neighbor fingerprints; DIAL only adds own-action self-conditioning.
17. **No `self.device`** is set in `__init__`. If any inherited or future method accesses `self.device`, AttributeError. Currently safe because NC's `_run_comm_layers` (parent) doesn't reference `self.device` either.
18. **Heterogeneous path doesn't narrow `x[i]`** for own obs in `_get_comm_s` line 2044 — uses `x[i].unsqueeze(0)` unconditionally. If `x[i]` is zero-padded (from `_convert_hetero_states` in `MA2C_NC.add_transition`), the unused dims become zeros fed into `fc_x_layers[i]`. The hetero branch (lines 2039–2042) narrows neighbor obs correctly but `x[i]` (own) is not narrowed. The expected input dim of `fc_x_layers[i]` is `n_ns = n_s_ls[i] + sum(neighbor n_s_ls)`, so passing the **full padded** `x[i]` (which has length `self.n_s`, the max) will cause a shape mismatch in the hetero non-identical case. This looks like a real bug for the hetero branch.

---

## File paths

- Source: `/tmp/BayesG/agents/policies.py` lines 1686–2045
- Models: `/tmp/BayesG/agents/models.py` lines 398–479
- Helpers: `/tmp/BayesG/agents/utils.py` (`one_hot`, `batch_to_seq`, `run_rnn`, `init_layer`)
- Parent class context: `/tmp/BayesG/agents/policies.py` lines 193–515 (`NCMultiAgentPolicy`)
