# `agents/policies.py` — Lines 1–192 (Base classes: `Policy`, `LstmPolicy`, `FPPolicy`)

**Source:** `/tmp/BayesG/agents/policies.py`, lines 1–192
**Scope of this chunk:** the abstract `Policy` base class plus the two simplest concrete subclasses, `LstmPolicy` and `FPPolicy`. Everything multi-agent / graph-based (`NCMultiAgentPolicy`, `GraphCMultiAgentPolicy`, `BayesianGraphCMultiAgentPolicy`, etc.) lives below line 192 and is covered in later chunks.

---

## Imports (lines 1–14)

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.utils import batch_to_seq, init_layer, one_hot, run_rnn
from agents.gnn import GATLayer, GCNLayer, SAGELayer
import random
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import networkx as nx
import os
```

Line-by-line:

- **Line 1 — `numpy as np`.** Used to wrap observations / actions before they are converted to torch tensors. In this 192-line window it appears at line 139 (`np.expand_dims(ob, axis=0)`) and line 147 (`np.array([naction])`).
- **Line 2 — `torch`.** Core tensor library. Used for `torch.from_numpy`, `torch.device`, `torch.zeros`, `torch.cat`, `torch.chunk`, `torch.squeeze`, `torch.distributions.categorical.Categorical`.
- **Line 3 — `torch.nn as nn`.** Provides `nn.Module` (line 15 — base class of `Policy`), `nn.Linear` (lines 35, 53, 153, 178), `nn.LSTMCell` (lines 155, 181).
- **Line 4 — `torch.nn.functional as F`.** Provides `F.relu` (lines 150, 187), `F.log_softmax` (line 126), `F.softmax` (line 145), `F.pad` (line 190).
- **Line 5 — `from agents.utils import batch_to_seq, init_layer, one_hot, run_rnn`.** Four helpers from [[agents-utils]]:
  - `batch_to_seq(x)` — splits a `[T, …]` tensor into a length-T tuple via `torch.chunk(x, n_step)`. Used **indirectly** here through `run_rnn`.
  - `init_layer(layer, layer_type)` — orthogonal init for weights, zero init for biases; supports `'fc'` and `'lstm'`. Called on every `nn.Linear` / `nn.LSTMCell` constructed in this file (lines 36, 54, 154, 156, 179, 182).
  - `one_hot(x, oh_dim, dim=-1)` — converts an integer index tensor into a one-hot tensor on the same device as input. Used in `_run_critic_head` (lines 68, 74) to encode neighbor actions.
  - `run_rnn(layer, xs, dones, s)` — iterates an `LSTMCell` step by step, **zeroing `h` and `c` on `done==1`**, returns `(outputs, final_state)` where `final_state = cat([h, c])`. Used at lines 123 (backward) and 142 (forward).
- **Line 6 — `from agents.gnn import GATLayer, GCNLayer, SAGELayer`.** Graph layers from [[agents-gnn]]. **Not referenced in lines 15–192** — these are used by `NCMultiAgentPolicy` and friends below the window. Imported here at module scope.
- **Line 7 — `random`.** Not used in lines 15–192.
- **Line 8 — `copy`.** Not used in lines 15–192.
- **Line 9 — `matplotlib.pyplot as plt`.** Not used in lines 15–192 (used later by graph-visualization code in `BayesianGraphCMultiAgentPolicy`).
- **Line 10 — `seaborn as sns`.** Not used in lines 15–192.
- **Line 11 — `logging`.** Not used in lines 15–192.
- **Line 12 — `networkx as nx`.** Not used in lines 15–192.
- **Line 13 — `os`.** Not used in lines 15–192.

> Verified: of the 13 imports, only `numpy`, `torch`, `torch.nn`, `torch.nn.functional`, and the four functions from `agents.utils` are referenced inside the 15–192 range. The remaining seven (`GATLayer`/`GCNLayer`/`SAGELayer`, `random`, `copy`, `plt`, `sns`, `logging`, `nx`, `os`) are dead with respect to this chunk and exist for downstream classes.

---

## `class Policy(nn.Module)` (lines 15–103)

This is the abstract base. It does not implement `forward` itself (line 28 raises `NotImplementedError`) but provides shared init/run helpers for the actor head, critic head, and the [[A2C]]-style loss.

### `__init__` (lines 16–25)

```python
def __init__(self, n_a, n_s, n_step, policy_name, agent_name, identical):
    super(Policy, self).__init__()
    self.name = policy_name
    if agent_name is not None:
        # for multi-agent system
        self.name += '_' + str(agent_name)
    self.n_a = n_a
    self.n_s = n_s
    self.n_step = n_step
    self.identical = identical
```

Parameters:

- **`n_a`** — number of *discrete* actions. Note line 34's comment: "only discrete control is supported for now". This is the size of the actor head's output logit vector.
- **`n_s`** — number of state features (observation dimensionality). Used downstream as the input dim of `fc_layer` (line 153) and to derive `n_x` in `FPPolicy._init_net` (lines 175, 177).
- **`n_step`** — number of rollout steps the policy is expected to process in one `backward`. Stored but not directly consumed by `Policy` itself; consumers (e.g. the trans buffer in `models.py`) rely on it.
- **`policy_name`** — string tag used to construct `self.name` (e.g. `'lstm'` in line 108). The `LstmPolicy` constructor hard-codes `'lstm'`; `FPPolicy` inherits that without changing it.
- **`agent_name`** — optional suffix. If non-None it becomes `'_<agent_name>'`, so a `LstmPolicy` for agent 3 ends up `self.name = 'lstm_3'`. This is exactly the pattern `models.py` uses: `name='{:d}'.format(i)` at line 208.
- **`identical`** — bool. `True` ⇒ all agents have the same action dimension; `False` ⇒ heterogeneous, and `self.na_dim_ls` (a per-neighbor list of action dims) must be supplied.

The line `super(Policy, self).__init__()` runs `nn.Module.__init__()` so that `self._parameters`, `self._modules`, etc. are properly initialized — required before assigning any `nn.Linear` / `nn.LSTMCell` to `self`.

### `forward` (lines 27–28)

```python
def forward(self, ob, *_args, **_kwargs):
    raise NotImplementedError()
```

Abstract. Concrete subclasses (`LstmPolicy.forward` at line 138, `FPPolicy` inheriting it) override this. `*_args, **_kwargs` are named with a leading underscore to mark them unused — pure signature padding so the base method does not constrain the subclass signatures.

### `_init_actor_head` (lines 30–36)

```python
def _init_actor_head(self, n_h, n_a=None):
    if n_a is None:
        n_a = self.n_a
    self.actor_head = nn.Linear(n_h, n_a)
    init_layer(self.actor_head, 'fc')
```

- **`n_h`** — hidden state size feeding the head (typically `n_lstm`; see lines 157, 183).
- **`n_a`** — output action-logit count; defaults to `self.n_a`. The override path is used by classes (further down the file) that have per-agent action dims, but in this 192-line window only the default path is taken.
- The comment on line 34 is correct: this is *discrete* policy output — a linear layer producing logits that are later passed through `F.log_softmax` (training, line 126) or `F.softmax` (rollout, line 145).
- `init_layer(..., 'fc')` performs orthogonal init on `weight`, zero init on `bias` (see [[agents-utils]]).

### `_init_critic_head` (lines 43–54)

```python
def _init_critic_head(self, n_h, n_n=None):
    if n_n is None:
        n_n = int(self.n_n)
    if n_n:
        if self.identical:
            n_na_sparse = self.n_a*n_n
        else:
            n_na_sparse = sum(self.na_dim_ls)
        n_h += n_na_sparse
    self.critic_head = nn.Linear(n_h, 1)
    init_layer(self.critic_head, 'fc')
```

- **`n_h`** — base hidden-state size (e.g. `n_lstm`).
- **`n_n`** — neighbor count; falls back to `int(self.n_n)`. Note: `self.n_n` is set in `LstmPolicy.__init__` (line 114), not in `Policy.__init__`. So calling `_init_critic_head` on a bare `Policy` (no `self.n_n`) would `AttributeError`. The base class is genuinely abstract here.
- **`if n_n:`** — when there is at least one neighbor, the critic is `Q(s, a_neighbors)` rather than `V(s)`. The neighbor actions are concatenated as a sparse one-hot block.
- **Identical case (line 48):** every neighbor has `self.n_a` actions, so the sparse block has `self.n_a * n_n` columns.
- **Heterogeneous case (line 50):** each neighbor j contributes `na_dim_ls[j]` columns, totalling `sum(self.na_dim_ls)`.
- **Line 51 — `n_h += n_na_sparse`** — the critic's input width grows accordingly.
- **Line 53 — `nn.Linear(n_h, 1)`** — single scalar output. Comment "V(s) / V(a,s)" notes the value-vs-action-value duality depending on whether the sparse block is present. (Strictly: when `n_n>0` it's `V(s, a_{-i})`, a centralized critic conditioning on neighbor actions; see [[Centralized-Critic]].)

### `_run_critic_head` (lines 62–79)

```python
def _run_critic_head(self, h, na, n_n=None):
    if n_n is None:
        n_n = int(self.n_n)
    if n_n:
        na = torch.from_numpy(na).long()
        if self.identical:
            na_sparse = one_hot(na, self.n_a)
            na_sparse = na_sparse.view(-1, self.n_a*n_n)
        else:
            na_sparse = []
            na_ls = torch.chunk(na, n_n, dim=1)
            for na_val, na_dim in zip(na_ls, self.na_dim_ls):
                na_sparse.append(torch.squeeze(one_hot(na_val, na_dim), dim=1))
            na_sparse = torch.cat(na_sparse, dim=1)
        na_sparse = na_sparse.to(h.device)
        h = torch.cat([h, na_sparse], dim=1)
    return self.critic_head(h).squeeze()
```

- **`h`** — hidden state tensor of shape `[T, n_lstm]` (backward) or `[1, n_lstm]` (forward), produced upstream by `run_rnn`.
- **`na`** — numpy array of neighbor actions. Shape expectations:
  - Backward path (line 127): `nactions` has shape `[T, n_n]` (one column per neighbor, one row per rollout step).
  - Forward path (line 147): wrapped as `np.array([naction])`, so shape `[1, n_n]`.
- **Line 66 — `torch.from_numpy(na).long()`** — convert to int64. `.to(device)` is **deliberately deferred** until after one-hot encoding (line 77), because `one_hot` (in `agents/utils.py`) reads `device = x.device` from the input — putting the integer tensor on CPU first means `one_hot` allocates the zero tensor on CPU too, then we move the result. Functionally OK but mildly wasteful when GPU is in use.
- **Identical path (lines 68–69):**
  - `one_hot(na, self.n_a)` returns shape `[T, n_n, n_a]`.
  - `.view(-1, self.n_a*n_n)` reshapes to `[T, n_n*n_a]`.
- **Heterogeneous path (lines 71–75):**
  - `torch.chunk(na, n_n, dim=1)` splits the `[T, n_n]` action matrix into `n_n` tensors of shape `[T, 1]`.
  - For each neighbor column `na_val` and its dim `na_dim`, `one_hot(na_val, na_dim)` returns `[T, 1, na_dim]`, then `squeeze(dim=1)` → `[T, na_dim]`.
  - Concatenating across dim=1 gives `[T, sum(na_dim_ls)]`.
- **Line 77 — `na_sparse = na_sparse.to(h.device)`** — explicit cross-device safety. Comment at line 76 confirms intent.
- **Line 78 — `torch.cat([h, na_sparse], dim=1)`** — produces `[T, n_lstm + sparse_width]`, matching the critic head's input.
- **Line 79 — `.squeeze()`** — drops the trailing dim-1 from the `nn.Linear(_, 1)` output. With `T==1` (forward path), this collapses to a 0-dim tensor; otherwise to `[T]`. (Possible footgun if `T==1` and the caller expects shape `[1]` — but the only forward caller calls `.detach().cpu().numpy()` so it ends up as a 0-d numpy scalar, which is fine.)

### `_run_loss` (lines 82–91)

```python
def _run_loss(self, actor_dist, e_coef, v_coef, vs, As, Rs, Advs):
    log_probs = actor_dist.log_prob(As)
    policy_loss = -(log_probs * Advs).mean()
    entropy_loss = -(actor_dist.entropy()).mean() * e_coef
    value_loss = (Rs - vs).pow(2).mean() * v_coef
    return policy_loss, value_loss, entropy_loss
```

Standard [[A2C]] loss. Parameters:

- **`actor_dist`** — a `torch.distributions.Categorical` built from softmax-normalized logits (constructed at line 126 in the backward).
- **`e_coef`** — entropy-regularization coefficient (passed from `models.py` config, see `entropy_coef` at line 239 of `models.py`).
- **`v_coef`** — value-loss coefficient (`value_coef` in config).
- **`vs`** — predicted values, `[T]` (from `_run_critic_head`).
- **`As`** — actions actually taken, `[T]` long-tensor.
- **`Rs`** — bootstrapped returns, `[T]` float.
- **`Advs`** — advantages, `[T]` float.

Components:

1. **`log_probs = actor_dist.log_prob(As)`** — `[T]` log-probability of each executed action.
2. **`policy_loss = -(log_probs * Advs).mean()`** — **standard policy gradient** with advantage weighting. The `.mean()` averages across the rollout. Note `Advs` is treated as a constant (no `.detach()` here, but advantages come from numpy in `LstmPolicy.backward` at line 132 so they're already detached from any compute graph).
3. **`entropy_loss = -(actor_dist.entropy()).mean() * e_coef`** — **negative** entropy. Because the total loss is summed (line 133), minimizing total loss ⇒ minimizing `-entropy` ⇒ maximizing entropy ⇒ exploration bonus. Note the sign convention: a more entropic distribution makes `entropy_loss` more negative, lowering total loss. Confirmed by tensorboard logging it directly under `loss/..._entropy_loss` (line 95).
4. **`value_loss = (Rs - vs).pow(2).mean() * v_coef`** — MSE between Monte-Carlo returns and the critic's predicted value, weighted by `v_coef`.

Return order: **`(policy_loss, value_loss, entropy_loss)`** — caller at line 128 unpacks into `(self.policy_loss, self.value_loss, self.entropy_loss)` in matching order.

> Sign sanity: total = policy + value + entropy = `-E[log_pi * A]` + `v_coef * MSE` + `-e_coef * E[H]`. All three are minimized; the entropy term encourages spread. Standard.

### `_update_tensorboard` (lines 93–102)

```python
def _update_tensorboard(self, summary_writer, global_step):
    summary_writer.add_scalar('loss/{}_entropy_loss'.format(self.name), self.entropy_loss, global_step=global_step)
    summary_writer.add_scalar('loss/{}_policy_loss'.format(self.name), self.policy_loss, global_step=global_step)
    summary_writer.add_scalar('loss/{}_value_loss'.format(self.name), self.value_loss, global_step=global_step)
    summary_writer.add_scalar('loss/{}_total_loss'.format(self.name), self.loss, global_step=global_step)
```

Logs the four scalar losses under `loss/<policy_name>_<which>_loss`. Relies on `self.name` (set in `__init__`) and on `self.policy_loss / self.value_loss / self.entropy_loss / self.loss` having been assigned (this happens inside `LstmPolicy.backward` at lines 128–133). Called conditionally from `backward` at line 135–136 when a `summary_writer` is provided.

---

## `class LstmPolicy(Policy)` (lines 105–164)

A single-agent (or independent-multi-agent) policy with: `obs → FC → LSTM → (actor head, critic head)`.

### `__init__` (lines 106–116)

```python
def __init__(self, n_s, n_a, n_n, n_step, n_fc=64, n_lstm=64, name=None,
             na_dim_ls=None, identical=True):
    super(LstmPolicy, self).__init__(n_a, n_s, n_step, 'lstm', name, identical)
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not self.identical:
        self.na_dim_ls = na_dim_ls
    self.n_lstm = n_lstm
    self.n_fc = n_fc
    self.n_n = n_n
    self._init_net()
    self._reset()
```

Parameters & subtle points:

- **Argument order difference vs. `Policy`.** `LstmPolicy.__init__` takes `(n_s, n_a, …)` but `Policy.__init__` takes `(n_a, n_s, …)`. Line 108 swaps them on the way up: `super().__init__(n_a, n_s, n_step, 'lstm', name, identical)`. Easy to mis-read — verified the swap is correct.
- **Hard-coded `policy_name='lstm'`** (line 108). So `self.name` becomes `'lstm'` or `'lstm_<i>'`. `FPPolicy` inherits this — see "Open questions" below.
- **`self.device`** (line 109) — set per-policy, not globally. Each policy independently picks CUDA if available. Note `Policy.__init__` does not set `self.device`, so the parent class cannot be used standalone — `_run_critic_head` references `h.device` but does not rely on `self.device`, so that's fine.
- **Line 110–111** — `na_dim_ls` is only stored when `identical=False`. If `identical=True` and a caller passes `na_dim_ls`, it is silently dropped. (Conversely, if `identical=False` and the caller forgets `na_dim_ls`, `self.na_dim_ls` becomes `None` and downstream `sum(self.na_dim_ls)` will crash.)
- **`n_fc=64`, `n_lstm=64`** — default widths. `models.py` reads these from config and passes them explicitly (lines 208, 218 of `models.py`).
- **`n_n`** — number of neighbors. Stored as `self.n_n` (line 114), consumed by `_init_critic_head` and `_run_critic_head` via the `int(self.n_n)` fallback. In `IA2C._init_policy` (`models.py` line 207) it's passed as `n_n = torch.sum(self.neighbor_mask[i])` — a 0-d torch tensor. Hence `int(self.n_n)` is needed downstream.
- **`self._init_net()`** (line 115) constructs the FC + LSTM + heads.
- **`self._reset()`** (line 116) zeros the LSTM state buffers.

### `forward` (lines 138–147)

```python
def forward(self, ob, done, naction=None, out_type='p'):
    ob = torch.from_numpy(np.expand_dims(ob, axis=0)).float().to(self.device)
    done = torch.from_numpy(np.expand_dims(done, axis=0)).float().to(self.device)
    x = self._encode_ob(ob)
    h, new_states = run_rnn(self.lstm_layer, x, done, self.states_fw)
    if out_type.startswith('p'):
        self.states_fw = new_states.detach()
        return F.softmax(self.actor_head(h), dim=1).squeeze().detach().cpu().numpy()
    else:
        return self._run_critic_head(h, np.array([naction])).detach().cpu().numpy()
```

Used for **single-step rollout** (acting in the env), not training. Detailed walk-through:

- **`ob`** — single observation, originally numpy `[n_s]`. `np.expand_dims(ob, axis=0)` → `[1, n_s]`. `.float().to(self.device)` → CUDA/CPU float32 tensor.
- **`done`** — scalar 0/1. Expanded to `[1]` then converted to float on device. (Note: `run_rnn` will call `batch_to_seq(dones)` which expects shape `[T]` or `[T, 1]`; `[1]` is fine — `batch_to_seq` will unsqueeze to `[1,1]` and chunk into one piece.)
- **`x = self._encode_ob(ob)`** — `LstmPolicy._encode_ob` is `F.relu(self.fc_layer(ob))` → `[1, n_fc]`. (`FPPolicy` overrides this, see below.)
- **`run_rnn(self.lstm_layer, x, done, self.states_fw)`** — single-step LSTM update. Returns `(h, new_states)`:
  - `h` has shape `[1, n_lstm]` (one row per timestep; here T=1).
  - `new_states` has shape `[n_lstm * 2]` (the concatenated `[h; c]` after squeezing).
- **`out_type.startswith('p')` (policy branch):**
  - **Important:** `self.states_fw = new_states.detach()` is updated **only when `out_type == 'p'`** (or any string starting with `p`). If a caller does a critic query (else branch) the forward LSTM state is *not* advanced. This is intentional — the actor is called once per env step; value queries are separate diagnostic / bootstrap calls that should not advance the rolling state.
  - `F.softmax(self.actor_head(h), dim=1)` → `[1, n_a]` probability distribution.
  - `.squeeze().detach().cpu().numpy()` → 1-D numpy `[n_a]`.
- **Else branch (value):**
  - `np.array([naction])` wraps the neighbor-action vector with an extra batch dim. If `naction` is a list/array of length `n_n`, this yields shape `[1, n_n]`, which is what `_run_critic_head` expects.
  - Output is a 0-d numpy scalar (since `T=1` and `_run_critic_head` calls `.squeeze()`).

> Edge case: `out_type.startswith('p')` means strings like `'policy'`, `'pi'`, `'probs'` all hit the actor branch. Anything else (commonly `'v'`) hits the critic branch. No validation.

### `backward` (lines 118–136)

```python
def backward(self, obs, nactions, acts, dones, Rs, Advs,
             e_coef, v_coef, summary_writer=None, global_step=None):
    obs = torch.from_numpy(obs).float().to(self.device)
    dones = torch.from_numpy(dones).float().to(self.device)
    xs = self._encode_ob(obs)
    hs, new_states = run_rnn(self.lstm_layer, xs, dones, self.states_bw)
    self.states_bw = new_states.detach()
    actor_dist = torch.distributions.categorical.Categorical(logits=F.log_softmax(self.actor_head(hs), dim=1))
    vs = self._run_critic_head(hs, nactions)
    self.policy_loss, self.value_loss, self.entropy_loss = \
        self._run_loss(actor_dist, e_coef, v_coef, vs,
                       torch.from_numpy(acts).long().to(self.device),
                       torch.from_numpy(Rs).float().to(self.device),
                       torch.from_numpy(Advs).float().to(self.device))
    self.loss = self.policy_loss + self.value_loss + self.entropy_loss
    self.loss.backward()
    if summary_writer is not None:
        self._update_tensorboard(summary_writer, global_step)
```

Argument shapes (all numpy on entry):

- **`obs`** — `[T, n_s]` (full rollout).
- **`nactions`** — `[T, n_n]` int (neighbor actions per step).
- **`acts`** — `[T]` int (this agent's actions).
- **`dones`** — `[T]` float (episode-termination flags).
- **`Rs`** — `[T]` float (bootstrapped returns).
- **`Advs`** — `[T]` float (advantages).
- **`e_coef`, `v_coef`** — scalars.
- **`summary_writer`, `global_step`** — optional tensorboard hooks.

Step-by-step:

1. **Line 120–121** — move `obs`, `dones` to device as float32.
2. **Line 122 — `xs = self._encode_ob(obs)`** — `[T, n_fc]` after FC+ReLU.
3. **Line 123 — `run_rnn(...)`** — unrolls the LSTM **with done-resets** along the rollout. Crucially, `run_rnn` uses `c = c * (1-done); h = h * (1-done)` *before* the step (line 38–39 of `agents/utils.py`), so an episode boundary zeros the state going into the new episode. Returns `hs: [T, n_lstm]` and `new_states: [n_lstm*2]`.
4. **Line 125 — `self.states_bw = new_states.detach()`.** Carries LSTM state across **mini-batches** but breaks the gradient graph at the boundary (truncated BPTT, window = `n_step`). The comment on line 124 — "backward grad is limited to the minibatch" — confirms this is intentional truncated backprop.
5. **Line 126 — `actor_dist = Categorical(logits=F.log_softmax(self.actor_head(hs), dim=1))`.**
   - `self.actor_head(hs)` → `[T, n_a]` raw logits.
   - `F.log_softmax(..., dim=1)` → log-probs (already normalized).
   - `Categorical(logits=log_probs)` — PyTorch treats the `logits` kwarg as unnormalized log-probs and internally renormalizes via log-sum-exp. **Passing already-log-softmaxed values is redundant but harmless** (log-softmax is idempotent under further log-softmax up to a constant, which Categorical normalizes away). Mildly wasteful but correct.
6. **Line 127 — `vs = self._run_critic_head(hs, nactions)`.** `[T]` value estimates.
7. **Lines 128–132 — `_run_loss(...)`** with numpy→tensor conversion inline for `acts`, `Rs`, `Advs`. Note `acts` is `.long()` so `Categorical.log_prob(acts)` indexes correctly.
8. **Line 133 — `self.loss = policy + value + entropy`** — total scalar loss.
9. **Line 134 — `self.loss.backward()`** — populates `.grad` for every parameter in this `nn.Module` instance. No optimizer step here; the optimizer lives on the parent model (`models.py` line 244) and steps after `backward` is called on each agent's policy.
10. **Lines 135–136** — optional tensorboard logging.

### `_encode_ob` (lines 149–150)

```python
def _encode_ob(self, ob):
    return F.relu(self.fc_layer(ob))
```

Simple FC + ReLU. Input `[*, n_s]`, output `[*, n_fc]`. Overridden by `FPPolicy`.

### `_init_net` (lines 152–158)

```python
def _init_net(self):
    self.fc_layer = nn.Linear(self.n_s, self.n_fc)
    init_layer(self.fc_layer, 'fc')
    self.lstm_layer = nn.LSTMCell(self.n_fc, self.n_lstm)
    init_layer(self.lstm_layer, 'lstm')
    self._init_actor_head(self.n_lstm)
    self._init_critic_head(self.n_lstm)
```

Architecture:

```
ob [n_s] ──Linear─→ [n_fc] ──ReLU─→ ──LSTMCell─→ h [n_lstm] ─┬→ actor_head → [n_a] logits
                                                              └→ critic_head(h ⊕ na_sparse) → scalar
```

- `nn.LSTMCell(input_size=n_fc, hidden_size=n_lstm)` — note: `LSTMCell`, not `LSTM`. This means the unrolling is done manually in `run_rnn` (one step at a time), which is precisely what lets `run_rnn` apply per-step `done` masks.
- Both heads sit on top of `n_lstm`-dimensional hidden state.

### `_reset` (lines 160–163)

```python
def _reset(self):
    self.states_fw = torch.zeros(self.n_lstm * 2, device=self.device)
    self.states_bw = torch.zeros(self.n_lstm * 2, device=self.device)
```

Two separate state buffers:

- **`states_fw`** — used by `forward` (env-rollout / acting). Advanced one step at a time.
- **`states_bw`** — used by `backward` (training). Advanced one minibatch (`n_step` steps) at a time.

Both are 1-D length `n_lstm*2` (concatenated `h` and `c`). `run_rnn` slices them via `torch.chunk(s, 2, dim=1)` after `unsqueeze(0)` (see `agents/utils.py` lines 34–35).

> Comment at line 161 says "forget the cumulative states every cum_step" — the actual forget mechanism is *not* in `_reset` (which only initializes), it's in `run_rnn`'s per-step `done` masking. `_reset` is called once at construction (line 116). See [[LSTM-State-Management]].

---

## `class FPPolicy(LstmPolicy)` (lines 167–192)

"FP" stands for **fingerprint policy** — see `IA2C_FP` docstring in `models.py` line 264: "In fingerprint IA2C, neighborhood policies (fingerprints) are also included." This is from the *Fingerprint MARL* idea (concatenate neighbor policy outputs into the observation to stabilize independent learners — [[Fingerprints-MARL]]).

### `__init__` (lines 168–171)

```python
def __init__(self, n_s, n_a, n_n, n_step, n_fc=64, n_lstm=64, name=None,
             na_dim_ls=None, identical=True):
    super(FPPolicy, self).__init__(n_s, n_a, n_n, n_step, n_fc, n_lstm, name,
                     na_dim_ls, identical)
```

Same signature as `LstmPolicy.__init__`, just passes through. Because the `super().__init__` calls `LstmPolicy.__init__`, which in turn calls `self._init_net()` and `self._reset()`, and Python resolves `self._init_net` via MRO to **`FPPolicy._init_net`** (line 173), the FPPolicy-specific net is built. Same for `_encode_ob`.

### `_init_net` (lines 173–184)

```python
def _init_net(self):
    if self.identical:
        self.n_x = self.n_s - self.n_n * self.n_a
    else:
        self.n_x = int(self.n_s - sum(self.na_dim_ls))
    self.fc_x_layer = nn.Linear(self.n_x, self.n_fc)
    init_layer(self.fc_x_layer, 'fc')
    # Always use 128 as input size for LSTM
    self.lstm_layer = nn.LSTMCell(self.n_fc * 2, self.n_lstm)
    init_layer(self.lstm_layer, 'lstm')
    self._init_actor_head(self.n_lstm)
    self._init_critic_head(self.n_lstm)
```

What differs from `LstmPolicy._init_net`:

1. **The observation is decomposed.** The full observation `n_s` is treated as the *own observation* (`n_x`) **concatenated with neighbor policy fingerprints** (`n_n * n_a` slots in the identical case, or `sum(na_dim_ls)` in the heterogeneous case). The FC layer is sized for the *own* part only:
   - **Identical:** `n_x = n_s - n_n * n_a` — each neighbor contributes `n_a` policy-prob entries.
   - **Heterogeneous:** `n_x = n_s - sum(na_dim_ls)` — each neighbor contributes its own `n_a`.
2. **FC layer name change.** `LstmPolicy` calls it `fc_layer`; `FPPolicy` calls it `fc_x_layer`. **Important:** `FPPolicy` does **not** create `self.fc_layer`. If any code path went through `LstmPolicy._encode_ob` it would crash with `AttributeError: 'FPPolicy' object has no attribute 'fc_layer'`. This is prevented only because `FPPolicy._encode_ob` (line 186) overrides the base.
3. **LSTM input size = `n_fc * 2 = 128`** (assuming default `n_fc=64`). The comment at line 180 hard-codes the "128" magic number, but the code itself parameterizes it as `n_fc*2`. The doubling is to make room for the fingerprint padding (see `_encode_ob` below).
4. Heads are identical in shape to `LstmPolicy`.

Cross-check against `IA2C_FP._init_policy` (`models.py` lines 277–288):

- `n_s1 = self.n_s_ls[i] + self.n_a*n_n` (identical case) — so the *combined* `n_s1` is passed in. Then `FPPolicy._init_net` subtracts `n_n*n_a` to recover the bare-obs size. Consistent.

### `_encode_ob` (lines 186–191)

```python
def _encode_ob(self, ob):
    x = F.relu(self.fc_x_layer(ob[:, :self.n_x]))
    # Always pad to 128 features
    if x.size(1) < self.n_fc * 2:
        x = F.pad(x, (0, self.n_fc * 2 - x.size(1)))
    return x
```

Line-by-line:

- **`ob[:, :self.n_x]`** — slice the *first `n_x` columns* of the observation (the agent's own state, **discarding** the neighbor-fingerprint columns `ob[:, n_x:]`). Shape `[T, n_x]`.
- **`F.relu(self.fc_x_layer(...))`** — `[T, n_fc]`.
- **Pad to `n_fc*2`:** `F.pad(x, (0, n_fc*2 - n_fc))` appends `n_fc` zeros on the right. Final `x` has shape `[T, n_fc*2]`.

> **This looks wrong, or at least odd.** The whole point of the FP architecture (per the paper) is to embed neighbor policies (the *fingerprints*) into the LSTM input. But here the neighbor-policy part `ob[:, self.n_x:]` is *sliced away*, the FC is applied to the own-obs only, and the second half of the LSTM input is **zero-padded**. So the LSTM sees `[own_obs_features; zeros]`. The fingerprint information is **never used**. Either:
> (a) this is intentionally a baseline/ablation that defeats FP-ness,
> (b) the code is a regression from a version that concatenated something like `F.relu(self.fc_p_layer(ob[:, self.n_x:]))` instead of zero padding, or
> (c) the comment "Always pad to 128 features" suggests the dev wanted shape compatibility with another architecture (maybe `NCMultiAgentPolicy`) and lost track of the fingerprint semantics.
>
> See "Open questions" below.

---

## Cross-references

Who instantiates these classes? From `grep -rn "LstmPolicy\|FPPolicy"`:

- **`/tmp/BayesG/agents/policies.py`** — class definitions only (lines 105, 108, 167, 170).
- **`/tmp/BayesG/agents/models.py:8`** — imports both: `from agents.policies import (LstmPolicy, FPPolicy, ConsensusPolicy, …)`.
- **`/tmp/BayesG/agents/models.py:207`** — `IA2C._init_policy`, identical-agent branch: `LstmPolicy(self.n_s_ls[i], self.n_a_ls[i], n_n, self.n_step, n_fc=…, n_lstm=…, name='{:d}'.format(i))`.
- **`/tmp/BayesG/agents/models.py:217`** — `IA2C._init_policy`, heterogeneous branch: same call with `na_dim_ls=…, identical=False`.
- **`/tmp/BayesG/agents/models.py:279`** — `IA2C_FP._init_policy`, identical: `FPPolicy(n_s1, self.n_a, int(n_n), self.n_step, …)` where `n_s1 = self.n_s_ls[i] + self.n_a*n_n`.
- **`/tmp/BayesG/agents/models.py:286`** — `IA2C_FP._init_policy`, heterogeneous: `FPPolicy(n_s1, self.n_a_ls[i], …, na_dim_ls=…, identical=False)`.

So `LstmPolicy` powers the [[IA2C]] baseline ("independent A2C"), and `FPPolicy` powers [[IA2C-FP]] ("independent A2C with fingerprints"). Both are constructed once per agent inside `nn.ModuleList`.

Related classes downstream in the same file (covered in later chunks):

- [[NCMultiAgentPolicy]] (line 193) — networked-comms multi-agent A2C, the architecture this codebase actually centers on.
- [[GraphCMultiAgentPolicy]] (line 872) — subclass that swaps in GNN layers.
- [[BayesianGraphCMultiAgentPolicy]] (line 1214) — the project's headline contribution; uses a Bayesian graph posterior. See [[BayesG-Overview]].
- [[ConsensusPolicy]] (line 1686), [[CommNetMultiAgentPolicy]] (line 1840), [[DIALMultiAgentPolicy]] (line 2004), [[LToSMultiAgentPolicy]] (line 2047) — other baselines.

---

## Open questions / suspicious code

1. **`FPPolicy._encode_ob` zero-pads instead of encoding fingerprints.** As discussed above, the neighbor-fingerprint slice `ob[:, self.n_x:]` is computed (it sits in `ob`, sized by the caller in `models.py:278`) but **never read**. The LSTM input is `[fc(own_obs); zeros]`. This either silently disables the FP mechanism or is a stale refactor. Worth diffing against an upstream / earlier commit. Concrete fix would be something like
   ```python
   p = F.relu(self.fc_p_layer(ob[:, self.n_x:]))
   return torch.cat([x, p], dim=1)
   ```
   with `self.fc_p_layer = nn.Linear(n_s - n_x, n_fc)` created in `_init_net`.

2. **`FPPolicy` keeps `policy_name='lstm'`.** `LstmPolicy.__init__` hard-codes `'lstm'` (line 108) and `FPPolicy.__init__` does not override it. So a FPPolicy logs its losses under `loss/lstm_<i>_*` in tensorboard, indistinguishable from a plain `LstmPolicy`. Minor but confusing when comparing IA2C vs IA2C_FP runs in the same tensorboard dir.

3. **`Categorical(logits=F.log_softmax(...))` (line 126) is redundant.** PyTorch's `Categorical` with `logits=` performs log-softmax internally. Double log-softmaxing is mathematically benign (still a valid log-prob up to a constant that Categorical strips) but wastes a softmax pass and obscures intent. Plain `Categorical(logits=self.actor_head(hs))` is the idiomatic form.

4. **`self.n_n` not initialized in `Policy`.** `_init_critic_head` and `_run_critic_head` read `self.n_n` as a fallback. If a subclass forgets to set it (the base `Policy` does not), these methods will `AttributeError`. The base class is effectively abstract w.r.t. `n_n` even though that isn't enforced.

5. **`na_dim_ls` not initialized when `identical=True`.** `LstmPolicy.__init__` only sets `self.na_dim_ls` on the heterogeneous path. Any future code that reads `self.na_dim_ls` unconditionally will break for identical-agent setups. Currently safe only because `_init_critic_head` and `_run_critic_head` gate on `self.identical`.

6. **Device-of-na transfer is two-step.** In `_run_critic_head`, `na` is created on CPU (line 66) regardless of `h.device`. `one_hot` allocates the result on CPU, then line 77 moves it to GPU. Cheaper would be `torch.from_numpy(na).long().to(h.device)` before the `one_hot` call. Not a correctness issue.

7. **Critic-head shape on T=1.** `self.critic_head(h).squeeze()` (line 79) returns a **0-d** tensor when T=1, but `[T]` when T>1. The single caller in the forward path immediately runs `.detach().cpu().numpy()`, so the consumer gets a 0-d numpy scalar — silent but slightly inconsistent.

8. **`run_rnn` consumes a numpy `dones` of shape `[1]` in `forward`** (line 140 produces `[1]` via `np.expand_dims(done, axis=0)`). Tracing into `run_rnn`/`batch_to_seq`: `batch_to_seq` checks `len(x.shape) == 1` and unsqueezes to `[1,1]`, then `torch.chunk(x, 1)` returns `(x,)`. Fine, just non-obvious.

9. **`n_step` is stored but unused in this 192-line range.** It's read by the buffer / trainer in `models.py`. Listed for transparency — not a bug.

10. **`_init_net` rebuilds `self.lstm_layer` in `FPPolicy` after the parent had not built it yet.** The MRO arrangement is OK because `LstmPolicy.__init__` calls `self._init_net()` which dispatches to `FPPolicy._init_net` directly — `LstmPolicy._init_net` is never executed for an `FPPolicy` instance. Verified.

---

**End of chunk.** Next chunks in `policies-chunks/` should cover lines 193+ ( [[NCMultiAgentPolicy]] and onward).
