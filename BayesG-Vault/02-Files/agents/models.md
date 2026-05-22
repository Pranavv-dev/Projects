# `agents/models.py` — Line-by-Line Walkthrough

> File: `/tmp/BayesG/agents/models.py` (821 lines on disk; prompt rounded to 820)
> Companion notes: [[policies.py]] · [[utils.py]] · [[main.py]]

This module defines the **algorithm-level wrappers** that sit one rung above the policy networks. Each class owns:

- a `self.policy` (an `nn.Module` or `nn.ModuleList`) coming from [[policies.py]],
- a `self.trans_buffer` coming from [[utils.py]] (`OnPolicyBuffer` / `MultiAgentOnPolicyBuffer` / `LToSPolicyBuffer`),
- an `optim.RMSprop` optimizer, an `lr_scheduler` (`Scheduler` from [[utils.py]]),
- a `name` string that the training loop in [[main.py]] uses to dispatch.

Eight classes are defined:

```
IA2C  ←  IA2C_FP
IA2C  ←  MA2C_NC  ←  IA2C_CU
                  ←  BayesianGraph
                  ←  MA2C_DIAL
                  ←  MA2C_CNET
IA2C  ←  IA2C_LToS
```

---

## Imports

```python
import os
import torch
import torch.nn as nn
import torch.optim as optim
from agents.utils import OnPolicyBuffer, MultiAgentOnPolicyBuffer, Scheduler, LToSPolicyBuffer
from agents.policies import (LstmPolicy, FPPolicy, ConsensusPolicy,
                             NCMultiAgentPolicy,
                             CommNetMultiAgentPolicy,
                             DIALMultiAgentPolicy,
                             GraphCMultiAgentPolicy,
                             BayesianGraphCMultiAgentPolicy,
                             NCMultiAgentPolicy_MLP,
                             LToSMultiAgentPolicy)
import logging
import numpy as np
import torch.nn.functional as F
```

Observations:

- Pulls **four** buffer classes from [[utils.py]] (only three are actually imported here — `Scheduler` is a 4th). `LToSPolicyBuffer` exists alongside the on-policy buffers, telling us LToS is off-policy-flavoured.
- Pulls **nine** policy classes from [[policies.py]]. Every concrete algorithm class below picks exactly one.
- `torch.nn.functional as F` is only used inside `IA2C_LToS.backward` (for `F.mse_loss`), foreshadowing that LToS does Q-learning instead of A2C.

---

## class IA2C (base)

```python
class IA2C:
    def __init__(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                 total_step, model_config, seed=0, use_gpu=False):
        self.name = 'ia2c'
        self._init_algo(...)
```

### Signature & inheritance

- Plain Python class (no parent). Acts as the **abstract base** for every other algorithm in the file.
- Constructor positional args (used identically by every subclass except `MA2C_NC` which adds `gnn_type`):
  `(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma, total_step, model_config, seed=0, use_gpu=False)`.

### `__init__`

- Sets `self.name = 'ia2c'`.
- Forwards everything into `self._init_algo(...)`. The `__init__` for **every** subclass follows the same pattern: set `self.name`, then call `_init_algo`. So `_init_algo` is the real constructor.

### `_init_algo` (shared)

Lines 116–188. Does, in order:

1. **Persists config**: `self.model_config = model_config` (line 131). Important — `MA2C_NC._init_policy` and `BayesianGraph._init_policy` both reach back into this.
2. **State/action vectors**: `self.n_s_ls`, `self.n_a_ls`.
3. **Agent identification** (lines 139–146):
   ```python
   self.identical_agent = False
   if (max(self.n_a_ls) == min(self.n_a_ls)):
       self.identical_agent = True
       self.n_s = n_s_ls[0]
       self.n_a = n_a_ls[0]
   else:
       self.n_s = max(self.n_s_ls)
       self.n_a = max(self.n_a_ls)
   ```
   Note: heterogeneity is detected from **action dims only**, then `n_s` defaults to `n_s_ls[0]` when identical (even if `n_s_ls` aren't all equal). This is an oddity (see Bugs).
4. **`self.n_agent = len(neighbor_mask)`**.
5. **Reward shaping**: `self.reward_clip`, `self.reward_norm`.
6. **`self.n_step = model_config.getint('batch_size')`** (rollout length, default `120` per comment).
7. **Network sizes**: `self.n_fc`, `self.n_lstm` (both `64` per comment).
8. **Device** (lines 164–174): GPU branch sets `cudnn.deterministic = True` and seeds `cuda`; CPU branch sets `torch.set_num_threads(1)`.
9. **Masks promoted to tensors on `self.device`**: `neighbor_mask` → `long`, `distance_mask` → `float`.
10. **Policy**: `self.policy = self._init_policy()` then `.to(self.device)`. This is the polymorphic seam — every subclass overrides `_init_policy`.
11. **Training resources** (only if `total_step`): `self._init_train(model_config, distance_mask, coop_gamma)`.

### `_init_policy` (base)

Lines 194–222. Returns `nn.ModuleList` of `LstmPolicy` from [[policies.py#LstmPolicy]] — one per agent.

```python
for i in range(self.n_agent):
    n_n = int(torch.sum(self.neighbor_mask[i]))
    if self.identical_agent:
        local_policy = LstmPolicy(self.n_s_ls[i], self.n_a_ls[i], n_n, self.n_step,
                                  n_fc=self.n_fc, n_lstm=self.n_lstm, name='{:d}'.format(i))
    else:
        na_dim_ls = []
        for j in torch.where(self.neighbor_mask[i] == 1)[0]:
                na_dim_ls.append(self.n_a_ls[j])
        local_policy = LstmPolicy(..., na_dim_ls=na_dim_ls, identical=False)
```

Heterogeneous branch collects neighbour action dims so the LSTM can size its neighbour-action embedding correctly.

### `_init_scheduler`

Lines 224–232. Reads `lr_init` / `lr_decay`; if not `constant`, also reads `lr_min` and instantiates `Scheduler(lr_init, lr_min, total_step, decay=…)`. Otherwise constant.

### `_init_train`

Lines 234–248. Pure orchestration:

```python
self._init_scheduler(model_config)
self.v_coef = model_config.getfloat('value_coef')
self.e_coef = model_config.getfloat('entropy_coef')
self.max_grad_norm = model_config.getfloat('max_grad_norm')
alpha   = model_config.getfloat('rmsp_alpha')
epsilon = model_config.getfloat('rmsp_epsilon')
self.optimizer = optim.RMSprop(self.policy.parameters(), self.lr_init,
                               eps=epsilon, alpha=alpha)
gamma = model_config.getfloat('gamma')
self._init_trans_buffer(gamma, distance_mask, coop_gamma)
```

So **all eight algorithms share RMSprop**, all share the same `value_coef`/`entropy_coef`/`max_grad_norm` interface, and they only differ in `_init_trans_buffer` and `_init_policy`.

### `_init_trans_buffer` (base)

Lines 250–254. One `OnPolicyBuffer(gamma, coop_gamma, distance_mask[i])` per agent — `self.trans_buffer` is a **list**.

### `_update_lr`

Lines 256–260. Pulls `cur_lr = self.lr_scheduler.get(self.n_step)` and broadcasts into every `optimizer.param_groups[*]['lr']`. The `# TODO: refactor this using optim.lr_scheduler` comment is honest — they reimplemented LR scheduling instead of using PyTorch's.

### `add_transition` (base)

Lines 37–46.

```python
def add_transition(self, ob, naction, action, reward, value, done):
    reward = torch.as_tensor(reward, device=self.device)
    if self.reward_norm > 0:
        reward = reward / self.reward_norm
    if self.reward_clip > 0:
        reward = torch.clamp(reward, -self.reward_clip, self.reward_clip)
    for i in range(self.n_agent):
        self.trans_buffer[i].add_transition(ob[i], naction[i], action[i],
                                            reward, value[i], done)
```

Per-agent fan-out. Signature: `(ob, naction, action, reward, value, done)` — `naction` = **neighbour actions** (used by `LstmPolicy` for the FC input). Reward is a **scalar** shared across agents (this is the implicit cooperation; per-agent rewards are reconstructed by `OnPolicyBuffer` using `distance_mask` and `coop_gamma` — see [[utils.py]]).

### `forward` (base)

Lines 66–73. Identical loop, with `out_type` switching policy vs value via [[policies.py#LstmPolicy.forward]]:

```python
def forward(self, obs, done, nactions=None, out_type='p'):
    out = []
    if nactions is None:
        nactions = [None] * self.n_agent
    for i in range(self.n_agent):
        cur_out = self.policy[i](obs[i], done, nactions[i], out_type)
        out.append(cur_out)
    return out
```

- `out_type='p'` → action distribution / sampled action.
- `out_type='v'` → value scalar.

The branching is delegated entirely to the policy — `IA2C` itself just iterates agents.

### `backward` (base)

Lines 48–63.

```python
self.optimizer.zero_grad()
for i in range(self.n_agent):
    obs, nas, acts, dones, Rs, Advs = self.trans_buffer[i].sample_transition(Rends[i], dt)
    if i == 0:
        self.policy[i].backward(obs, nas, acts, dones, Rs, Advs,
                                self.e_coef, self.v_coef,
                                summary_writer=summary_writer, global_step=global_step)
    else:
        self.policy[i].backward(...)        # no summary writer
if self.max_grad_norm > 0:
    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
self.optimizer.step()
if self.lr_decay != 'constant':
    self._update_lr()
```

Each agent's `OnPolicyBuffer.sample_transition` returns `(obs, neighbour_actions, actions, dones, R-bootstraps, advantages)`. Only agent 0 logs to TensorBoard — `summary_writer` is suppressed for others to avoid duplicate scalars. Gradients accumulate from each `policy[i].backward()` call (each does `.backward()` internally) and are clipped jointly across all agents before a single `optimizer.step()`.

### `load`

Lines 75–102. Three branches:

1. `os.path.isfile(model_path)` → `torch.load(model_path, map_location='cpu')`.
2. `os.path.isdir(model_path)`:
   - If `isinstance(self.policy, GraphCMultiAgentPolicy)` → look for `graph_model.pt` inside; if missing **and** `train_mode`, print and return `False` (i.e. continue training from scratch).
   - Else → load `model.pt`.
3. Otherwise raise.

After loading, `self.policy.load_state_dict(checkpoint['model_state_dict'])` and `.train()`/`.eval()`.

**Bug-adjacent**: the `isinstance` check is only against `GraphCMultiAgentPolicy`, not `BayesianGraphCMultiAgentPolicy` (the paper's wrapper). So checkpoint loading for `BayesianGraph` falls into the generic `model.pt` branch.

### `save`

Lines 108–114. Always writes `checkpoint-{global_step}.pt` containing `global_step`, `model_state_dict`, `optimizer_state_dict`. Nothing special.

### `reset`

Lines 104–106. Calls `policy[i]._reset()` per agent (resets LSTM hidden states). Overridden by `MA2C_NC` (single policy, single `_reset`).

---

## class IA2C_FP

```python
class IA2C_FP(IA2C):
    """In fingerprint IA2C, neighborhood policies (fingerprints) are also included."""
```

### Inheritance & `__init__`

- Inherits `IA2C`.
- `__init__`: sets `self.name = 'ia2c_fp'` then `_init_algo(...)` — re-implements the constructor verbatim (this is a pattern across the file; subclasses don't `super().__init__`, they re-call `_init_algo` because they need their own `name` set **before** policy init).

### Only override: `_init_policy` (lines 272–289)

Same per-agent loop as the base, but uses `FPPolicy` and concatenates **neighbour action probabilities** into the state:

```python
if self.identical_agent:
    n_s1 = int(self.n_s_ls[i] + self.n_a * n_n)
    policy.append(FPPolicy(n_s1, self.n_a, int(n_n), ...))
else:
    na_dim_ls = [self.n_a_ls[j] for j in torch.where(self.neighbor_mask[i] == 1)[0]]
    n_s1 = int(self.n_s_ls[i] + sum(na_dim_ls))
    policy.append(FPPolicy(n_s1, self.n_a_ls[i], int(n_n), ...,
                           na_dim_ls=na_dim_ls, identical=False))
```

- Identical: input dim = own state + `n_a × n_neighbours`.
- Heterogeneous: input dim = own state + `Σ neighbour action dims`.

See [[policies.py#FPPolicy]] for how that augmented vector is consumed. **Everything else is inherited** — `add_transition`, `forward`, `backward`, `load`, `save`, `reset`, `_init_trans_buffer`.

---

## class MA2C_NC

This is the **trunk** for every multi-agent algorithm in the file. Everything below it (`IA2C_CU`, `BayesianGraph`, `MA2C_DIAL`, `MA2C_CNET`) inherits from `MA2C_NC`, not `IA2C`.

### Inheritance & `__init__`

```python
class MA2C_NC(IA2C):
    def __init__(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                 total_step, model_config, seed=0, use_gpu=False, gnn_type='gat'):
        self.name = 'ma2c_nc'
        self._init_algo(...)
```

Notice the `gnn_type='gat'` kwarg **which is never used** inside the body — `self.gnn_type` is read off `model_config` in `_init_policy` instead. The kwarg is left over from [[main.py]] line 100 where the dispatcher passes `gnn_type=gnn_type` for `Large_city`.

### `add_transition` (override, lines 299–310)

```python
def add_transition(self, ob, p, action, reward, value, done):
    if self.reward_norm > 0:
        reward = reward / self.reward_norm
    if self.reward_clip > 0:
        reward = np.clip(reward, -self.reward_clip, self.reward_clip)
    if self.identical_agent:
        self.trans_buffer.add_transition(np.array(ob), np.array(p), action,
                                         reward, value, done)
    else:
        pad_ob, pad_p = self._convert_hetero_states(ob, p)
        self.trans_buffer.add_transition(pad_ob, pad_p, action,
                                         reward, value, done)
```

Differences from base:

- Second arg renamed `naction → p` — `p` is the **softmax policy output** of neighbours, not their sampled action. This is the "NeurComm" message: continuous policy fingerprints, not discrete actions.
- Reward shaping uses `np.clip` (not `torch.clamp`); reward is **not** wrapped to a tensor here.
- `self.trans_buffer` is **single**, not a list — see `_init_trans_buffer` below.
- Heterogeneous path pads to `(n_agent, n_s)` and `(n_agent, n_a)` rectangles via `_convert_hetero_states`.

### `_convert_hetero_states` (lines 338–344)

```python
pad_ob = np.zeros((self.n_agent, self.n_s))
pad_p  = np.zeros((self.n_agent, self.n_a))
for i in range(self.n_agent):
    pad_ob[i, :len(ob[i])] = ob[i]
    pad_p[i, :len(p[i])]   = p[i]
return pad_ob, pad_p
```

Right-pads each agent's vector to `max(n_s_ls)` / `max(n_a_ls)` so a single batched tensor can be built.

### `forward` (override, lines 326–333)

```python
def forward(self, obs, done, ps, actions=None, out_type='p'):
    if self.identical_agent:
        return self.policy.forward(np.array(obs), done, np.array(ps),
                                   actions, out_type)
    else:
        pad_ob, pad_p = self._convert_hetero_states(obs, ps)
        return self.policy.forward(pad_ob, done, pad_p, actions, out_type)
```

Signature gains a `ps` argument (neighbour policy fingerprints). Calls the **single** policy once (not per-agent loop) — `NCMultiAgentPolicy` handles all agents internally (see [[policies.py#NCMultiAgentPolicy]]).

### `backward` (override, lines 313–323)

```python
self.optimizer.zero_grad()
obs, ps, acts, dones, Rs, Advs = self.trans_buffer.sample_transition(Rends, dt)
self.policy.backward(obs, ps, acts, dones, Rs, Advs, self.e_coef, self.v_coef,
                     summary_writer=summary_writer, global_step=global_step)
if self.max_grad_norm > 0:
    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
self.optimizer.step()
if self.lr_decay != 'constant':
    self._update_lr()
```

Single `sample_transition`, single `policy.backward`. Much simpler than `IA2C.backward`.

### `reset` (override, line 335)

Just `self.policy._reset()` (single policy, no agent loop).

### `_init_policy` (lines 354–393)

The big one. There is a **commented-out simple version** above it (lines 346–353) — the live version reads three flags from config:

```python
self.is_graph_nn  = self.model_config.getboolean('is_graph_nn', False)
self.gnn_version  = self.model_config.get('gnn_version', 'v2')
```

Builds `policy_params` dict with the shared 7 keys (`n_s`, `n_a`, `n_agent`, `n_step`, `neighbor_mask`, `n_fc`, `n_h`). Adds heterogeneous extras when needed.

Then branches:

- `is_graph_nn = True` → adds `gnn_type` (default `'gcn'`), `n_heads` (default `4`), `unify_act_state_dim`, `use_random_mask`, returns `GraphCMultiAgentPolicy(**policy_params)`.
- `is_graph_nn = False`:
  - if `not identical_agent and unify_act_state_dim` → `NCMultiAgentPolicy_MLP(**policy_params)`,
  - else → `NCMultiAgentPolicy(**policy_params)`.

So the same class name `MA2C_NC` covers **three** concrete policies. See [[policies.py#NCMultiAgentPolicy]], [[policies.py#NCMultiAgentPolicy_MLP]], [[policies.py#GraphCMultiAgentPolicy]].

The `print(f"Config is_graph_nn value: …")` on line 356 is a debug print left in production.

### `_init_trans_buffer` (line 395–396)

```python
def _init_trans_buffer(self, gamma, distance_mask, coop_gamma):
    self.trans_buffer = MultiAgentOnPolicyBuffer(gamma, coop_gamma, distance_mask)
```

**Single** `MultiAgentOnPolicyBuffer` (not a list). Takes the full `distance_mask` matrix. See [[utils.py#MultiAgentOnPolicyBuffer]].

---

## class IA2C_CU

```python
class IA2C_CU(MA2C_NC):
    def __init__(...):
        self.name = 'ma2c_cu'
        self._init_algo(...)
```

> Note the inconsistency: class is `IA2C_CU` but `self.name = 'ma2c_cu'`.

### `_init_policy` (lines 405–412)

Returns `ConsensusPolicy` (identical) or `ConsensusPolicy(..., n_s_ls=..., n_a_ls=..., identical=False)` heterogeneous. See [[policies.py#ConsensusPolicy]].

Crucially: **does not** go through the `MA2C_NC._init_policy` graph/MLP branching. Plain consensus.

### `backward` (lines 414–416)

```python
def backward(self, Rends, dt, summary_writer=None, global_step=None):
    super(IA2C_CU, self).backward(Rends, dt, summary_writer, global_step)
    self.policy.consensus_update()
```

Wraps `MA2C_NC.backward` and tacks on `policy.consensus_update()` — the consensus averaging step that defines this algorithm. Inherits `add_transition`, `forward`, `_init_trans_buffer` from `MA2C_NC`.

---

## class BayesianGraph ⭐ (paper's wrapper)

```python
class BayesianGraph(MA2C_NC):
    def __init__(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                 total_step, model_config, seed=0, use_gpu=False):
        self.name = 'bayesian_graph'
        self._init_algo(...)
```

Inherits `add_transition`, `forward`, `backward`, `reset`, `_init_trans_buffer` from `MA2C_NC` — so it uses the **single** `MultiAgentOnPolicyBuffer` and the single-policy training path. Only `_init_policy` is overridden (and `visualize_masks` is added).

### `visualize_masks` (lines 425–430)

```python
def visualize_masks(self, step, save_path=None, draw_whole=False):
    """Delegate visualization to the policy's visualize_masks method"""
    if hasattr(self.policy, 'visualize_masks'):
        self.policy.visualize_masks(step, save_path, draw_whole)
    else:
        logging.warning("Policy does not have visualize_masks method")
```

Pure delegation to [[policies.py#BayesianGraphCMultiAgentPolicy.visualize_masks]]. Guarded by `hasattr`. The training loop in [[main.py]] / [[utils.py#Trainer]] periodically calls this to plot the learned posterior adjacency.

### `_init_policy` (lines 432–447) — confirms `learn_mask` wiring

```python
def _init_policy(self):
    self.is_graph_nn = self.model_config.getboolean('is_graph_nn', False)
    self.unify_act_state_dim = self.model_config.getboolean('unify_act_state_dim', False)
    if self.is_graph_nn:
        # Add GAT-specific parameters
        self.gnn_type = self.model_config.get('gnn_type', 'gcn')
        self.heads = self.model_config.getint('n_attention_heads', 4)  # default 4 heads
        self.learn_mask = self.model_config.getboolean('learn_mask', True)
        return BayesianGraphCMultiAgentPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                              self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm,
                                              n_s_ls=self.n_s_ls, n_a_ls=self.n_a_ls,
                                              identical=self.identical_agent, unify_act_state_dim=self.unify_act_state_dim,
                                              n_heads=self.heads, gnn_type=self.gnn_type, learn_mask=self.learn_mask)

    else:
        raise ValueError("Bayesian Graph is not supported for non-graph models")
```

Confirmed:

- `learn_mask` defaults to `True` and is read from config (`getboolean('learn_mask', True)`).
- Forwarded as the kwarg `learn_mask=self.learn_mask` into `BayesianGraphCMultiAgentPolicy.__init__`.
- The class **hard-fails** when `is_graph_nn=False` — `BayesianGraph` is meaningless without graph machinery.
- Unlike `MA2C_NC._init_policy`, this branch **always** passes `n_s_ls`, `n_a_ls`, `identical=self.identical_agent` (i.e. doesn't gate on heterogeneity — the downstream policy is expected to handle both).
- Default `gnn_type` here is `'gcn'`, but the [[main.py]] dispatcher overrides with config fallback `'gat'` — see "Bugs / oddities".

Cross-reference: [[policies.py#BayesianGraphCMultiAgentPolicy]] section "learn_mask branch".

---

## class MA2C_DIAL

```python
class MA2C_DIAL(MA2C_NC):
    def __init__(...):
        self.name = 'ma2c_dial'
        self._init_algo(...)
```

### Only override: `_init_policy` (lines 456–463)

```python
if self.identical_agent:
    return DIALMultiAgentPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm)
else:
    return DIALMultiAgentPolicy(..., n_s_ls=..., n_a_ls=..., identical=False)
```

Mirrors `IA2C_CU._init_policy` but with `DIALMultiAgentPolicy`. See [[policies.py#DIALMultiAgentPolicy]]. Differentiable Inter-Agent Learning is the "messages-with-gradients" baseline.

Inherits everything else from `MA2C_NC`.

---

## class MA2C_CNET

```python
class MA2C_CNET(MA2C_NC):
    def __init__(...):
        self.name = 'ma2c_ic3'
        self._init_algo(...)
```

Class is `MA2C_CNET` (CommNet) but `self.name = 'ma2c_ic3'` — another naming inconsistency. [[main.py]] dispatches this under the algo key `'CommNet'`.

### Only override: `_init_policy` (lines 472–479)

Returns `CommNetMultiAgentPolicy` (identical or hetero). See [[policies.py#CommNetMultiAgentPolicy]].

---

## class IA2C_LToS

**Most complex class in the file** — 339 lines (482–821). Suspicious by virtue of size alone; the gradient-exchange `backward` is what the user flagged.

### Inheritance & `__init__` (lines 482–513)

```python
class IA2C_LToS(IA2C):
    def __init__(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                 total_step, model_config, seed=0, use_gpu=False):
        # Initialize shared_dim before calling parent's __init__
        self.shared_dim = model_config.getint('shared_dim', fallback=64)
        # Ensure n_fc matches shared_dim to maintain consistent dimensions
        self.n_fc = self.shared_dim  # Set n_fc equal to shared_dim
        self.use_lstm = model_config.getboolean('use_lstm', fallback=True)
        self.tau = model_config.getfloat('tau', fallback=0.01)
        self.grad_clip = model_config.getfloat('grad_clip', fallback=10.0)
        self.update_frequency = model_config.getint('update_frequency', fallback=1)
        self.gradient_steps = model_config.getint('gradient_steps', fallback=1)
        self.gamma = model_config.getfloat('gamma', fallback=0.99)

        # Initialize epsilon parameters
        self.epsilon = model_config.getfloat('epsilon', fallback=1.0)
        self.epsilon_min = model_config.getfloat('epsilon_min', fallback=0.01)
        self.epsilon_decay = model_config.getfloat('epsilon_decay', fallback=0.995)
        self.current_epsilon = self.epsilon

        # Initialize loss tracking
        self.q_losses = [0.0] * len(n_s_ls)
        self.policy_losses = [0.0] * len(n_s_ls)

        self.step_count = 0

        # Call parent's __init__ after initializing our attributes
        super().__init__(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                        total_step, model_config, seed, use_gpu)

        self.w_in  = [[] for _ in range(self.n_agent)]   # Store input weights
        self.w_out = [[] for _ in range(self.n_agent)]   # Store output weights
```

Several things that differ from every other class:

1. **It actually uses `super().__init__()`** instead of re-calling `_init_algo` (the only subclass that does).
2. **Sets attributes BEFORE `super().__init__`** because `_init_policy` (called from inside `_init_algo`) needs `self.shared_dim`, `self.use_lstm`, etc.
3. **Overwrites `self.n_fc = self.shared_dim`** — clobbers the value `_init_algo` will later read from `model_config.getint('num_fc')`. Effectively `n_fc` is forced to `shared_dim` for LToS regardless of config.
4. Caches `gamma` directly (`self.gamma = …`) instead of relying on `OnPolicyBuffer`'s — needed because `backward` does its own Bellman update.
5. **Epsilon-greedy state**: `self.current_epsilon` is initialised but **never decayed** anywhere in this file (only read). Likely decay happens externally or this is a latent bug.
6. `self.w_in` / `self.w_out` are per-agent **lists of tensors** representing the high-level LToS weight vectors exchanged between agents.

### `_init_policy` (lines 515–532)

```python
for i in range(self.n_agent):
    policy = LToSMultiAgentPolicy(
        n_s_ls=self.n_s_ls,
        n_a_ls=self.n_a_ls,
        neighbor_mask=self.neighbor_mask,
        n_step=self.n_step,
        shared_dim=self.shared_dim,
        n_fc=self.n_fc,
        use_lstm=self.use_lstm,
        identical=self.identical_agent,
        agent_id=i
    ).to(self.device)
    policies.append(policy)
return nn.ModuleList(policies)
```

One [[policies.py#LToSMultiAgentPolicy]] per agent, all wrapped in `nn.ModuleList`. Note: each policy is told its `agent_id` but is also handed the full `n_s_ls`/`n_a_ls`/`neighbor_mask` — i.e. each agent has a view of the whole graph (consistent with LToS's gradient-exchange design).

Also: each policy is `.to(self.device)` here, then `_init_algo` later does `self.policy = self.policy.to(self.device)` — redundant but harmless.

### `_init_trans_buffer` (lines 534–545)

```python
self.trans_buffer = []
for i in range(self.n_agent):
    self.trans_buffer.append(LToSPolicyBuffer(
        gamma, coop_gamma, distance_mask[i]))
```

One `LToSPolicyBuffer` per agent (back to the **list** model like `IA2C`, not `MA2C_NC`'s single buffer). See [[utils.py#LToSPolicyBuffer]] — the difference vs `OnPolicyBuffer` is that it also stores `w_in`.

### `add_transition` (lines 547–557)

```python
def add_transition(self, ob, naction, action, reward, value, done):
    if self.reward_norm > 0:
        reward = reward / self.reward_norm
    if self.reward_clip > 0:
        reward = np.clip(reward, -self.reward_clip, self.reward_clip)
    for i in range(self.n_agent):
        self.trans_buffer[i].add_transition(
            ob[i], naction[i], action[i], reward, value[i], done, self.w_in[i])
```

Extra positional arg: **`self.w_in[i]`** — the list of neighbour-output-weight tensors received by agent `i` in the *most recent* `forward` call. This is the entire point of LToS: the credit-assignment weights are recorded with the transition.

Reward shaping uses `np.clip` (not `torch.clamp`), and unlike `IA2C.add_transition`, reward is **not** wrapped to a tensor here.

### `forward` (lines 559–605)

```python
def forward(self, obs, done, naction=None, out_type='p'):
    if done:
        self.reset()

    # Compute output weights (high-level policy)
    self.w_out = []
    for i in range(self.n_agent):
        w = self.policy[i].compute_w_out(obs[i], agent_id=i, epsilon=self.current_epsilon)
        self.w_out.append(w)

    # Compute input weights from neighbors
    self.w_in = [[] for _ in range(self.n_agent)]
    for i in range(self.n_agent):
        neighbors = torch.where(self.neighbor_mask[i] == 1)[0]
        for j in neighbors:
            self.w_in[i].append(self.w_out[j])
```

Three-phase forward:

1. **Reset on done.** Note this is called **before** action selection on a done step, but `done` is also passed into `compute_actions` — minor double-handling.
2. **High-level**: each agent computes its `w_out` from its own observation (`policy[i].compute_w_out`).
3. **Wire neighbours**: `w_in[i] = [w_out[j] for j in neighbours_of(i)]`. This is the message-passing step. It's a **shallow copy** — `w_in[i]` holds references to the tensors in `w_out[j]`, so gradients can flow back across agents through these references.

Then the `out_type` branch (the 'p' vs 'v' branch):

```python
if out_type == 'p':
    actions = []
    for i in range(self.n_agent):
        if self.use_lstm:
            self.policy[i].states_fw        = self.policy[i].states_fw[:1]
            self.policy[i].target_states_fw = self.policy[i].target_states_fw[:1]
        action = self.policy[i].compute_actions(obs[i], self.w_in[i], done=done, epsilon=self.current_epsilon)
        actions.append(action)
    return actions
else:  # 'v'
    values = []
    for i in range(self.n_agent):
        q_values = self.policy[i].compute_critic(self.w_out[i], None)  # None → all Q-values
        value = torch.max(q_values).item()
        values.append(value)
    return values
```

Differences from `IA2C.forward`:

- `'p'` calls `compute_actions(obs, w_in, done, epsilon)` instead of `policy[i](obs, done, naction, 'p')`.
- `'v'` returns `max(q_values).item()` — Python floats, **not gradient-tracking tensors**. So the value branch is for bootstrap-only, never backpropagated.
- The `naction` argument is accepted but ignored.
- The `self.policy[i].states_fw = self.policy[i].states_fw[:1]` reaches inside the policy module and truncates its hidden state to the first row each step — looks like a workaround for an off-by-one growth bug in the LSTM state cache.

### `backward` (lines 607–796) — careful walkthrough

This is the suspicious one. It implements LToS's gradient-exchange update.

```python
def backward(self, Rends, dt, summary_writer=None, global_step=None):
    self.step_count += 1
    if self.step_count % self.update_frequency != 0:
        return
    for _ in range(self.gradient_steps):
        self.optimizer.zero_grad()
        for i in range(self.n_agent):
            obs, nactions, actions, dones, Rs, Advs, stored_w_in = self.trans_buffer[i].sample_transition(Rends[i], dt)
            if len(obs) == 0:
                continue
```

So far so good — `update_frequency` gates updates, `gradient_steps` loops the update, `zero_grad` once per gradient step, then a per-agent inner loop. `LToSPolicyBuffer.sample_transition` returns 7 things (one more than the on-policy version: `stored_w_in`).

```python
batch_size = len(obs)
obs     = torch.from_numpy(obs).float().to(self.device)
actions = torch.from_numpy(actions).long().to(self.device)
dones   = torch.from_numpy(dones).float().to(self.device)
Rs      = torch.from_numpy(Rs).float().to(self.device)
Advs    = torch.from_numpy(Advs).float().to(self.device)
```

Promote numpy arrays to tensors on device.

```python
non_empty_w_in = [w_list for w_list in stored_w_in if w_list]
if non_empty_w_in:
    stored_w_in = torch.stack([torch.stack(w_list) for w_list in non_empty_w_in]).to(self.device)
    stored_w_in = stored_w_in.squeeze(2)  # [batch_size, n_neighbors, w_dim]
else:
    n_neighbors = int(torch.sum(self.neighbor_mask[i]))
    stored_w_in = torch.zeros((batch_size, n_neighbors, self.shared_dim), device=self.device)
```

**Suspicion #1**: this filters out empty `w_in` entries (timesteps where for some reason the agent had no recorded weights). The resulting outer-stack therefore has dim 0 = **count-of-non-empty-rows**, not `batch_size`. Downstream code then uses `batch_size = len(obs)` to iterate, creating a mismatch when filtering actually happens.

`.squeeze(2)` assumes each stored `w` was shape `[1, w_dim]` (i.e. that `compute_w_out` returns a single-batch tensor).

```python
w_out = self.policy[i].phi[0](obs)  # [720, 32]
```

The shape comment `[720, 32]` suggests obs is batched to 720 rows here — but `batch_size = len(obs)` would be 120 (the rollout length) for a single agent. The 720 likely came from a debugging session with `n_agent=6`. The comment is stale.

```python
with torch.no_grad():
    neighbors = torch.where(self.neighbor_mask[i] == 1)[0]
    n_neighbors = int(torch.sum(self.neighbor_mask[i]))

    if n_neighbors == 0:
        next_w_in = torch.zeros((obs.size(0), 0), device=self.device)
        next_actions = self.policy[i].compute_actions(obs, next_w_in, epsilon=0.0)
        continue                # ← exits the agent loop after computing only next_actions

    valid_next_w_in = []
    for j in neighbors:
        neighbor_obs, _, _, _, _, _, _ = self.trans_buffer[j].sample_transition(Rends[j], dt)
        if len(neighbor_obs) == 0:
            continue
        neighbor_obs = torch.from_numpy(neighbor_obs).float().to(self.device)
        next_w_out = self.policy[j].target_phi[0](neighbor_obs)  # [720, 32]
        valid_next_w_in.append(next_w_out)

    if len(valid_next_w_in) < n_neighbors:
        padding = torch.zeros((obs.size(0), self.shared_dim), device=self.device)
        for _ in range(n_neighbors - len(valid_next_w_in)):
            valid_next_w_in.append(padding)

    try:
        next_w_in = torch.stack(valid_next_w_in, dim=1)
        next_w_in = next_w_in.reshape(next_w_in.shape[0], -1)  # [720, n_neighbors*32]
    except RuntimeError as e:
        print(...)
        raise e

    next_actions = self.policy[i].compute_actions(obs, next_w_in, epsilon=0.0)
    if isinstance(next_actions, np.ndarray):
        next_actions = torch.from_numpy(next_actions).long().to(self.device)

    target_q_vals = self.policy[i].compute_target_critic(next_w_out, next_actions)
    y = Rs + self.gamma * target_q_vals * (1 - dones)
```

**Suspicion #2** (line 657): when `n_neighbors == 0` the code `continue`s — but it does so **before** `q_vals` is computed and before `self.optimizer.step()` runs for this agent. That's fine for isolated agents, but the `continue` is inside `with torch.no_grad()`, so we leave the `no_grad` context cleanly.

**Suspicion #3** (`sample_transition` called twice per neighbour, line 663): inside the agent-`i` loop we re-sample the neighbour's buffer. `sample_transition` on `LToSPolicyBuffer` may **mutate / pop** the buffer (depending on impl in [[utils.py]]), so calling it once per (i, j) pair is dangerous. Worse, when the outer agent loop reaches `j`, it will call its own `sample_transition` again. Net effect: each agent's buffer is sampled `n_neighbors_of_self + 1` times per gradient step. If the buffer is on-policy this is benign; if it pops, this is catastrophic.

**Suspicion #4** (line 670 vs 703): `target_q_vals = self.policy[i].compute_target_critic(next_w_out, next_actions)`. But `next_w_out` was last assigned **inside the for-loop over neighbours**, so it equals the **last** neighbour's target-phi output — *not* the neighbour-aggregated `next_w_in`. This is almost certainly a bug: the target critic should consume the aggregated next-step neighbour weights (or this agent's own `next_w_out`), not the last neighbour's. The aggregated `next_w_in` is built but only used to compute `next_actions`, then discarded.

**Suspicion #5** (line 704): `y = Rs + self.gamma * target_q_vals * (1 - dones)`. Standard Bellman, but the shapes likely don't broadcast cleanly — `Rs` is `[batch]`, `target_q_vals` from `compute_target_critic` will be `[batch, 1]` or `[batch]` depending on the policy implementation, and `dones` is `[batch]`. If `compute_target_critic` returns `[batch, 1]`, broadcasting produces `[batch, batch]` (an outer product). This depends entirely on [[policies.py#LToSMultiAgentPolicy.compute_target_critic]].

```python
q_vals = self.policy[i].compute_critic(w_out, actions).squeeze(-1)

q_loss = F.mse_loss(q_vals, y.detach())
self.q_losses[i] = q_loss.item()

policy_loss = -(Advs * q_vals).mean()
self.policy_losses[i] = policy_loss.item()

total_loss = q_loss + policy_loss
total_loss.backward()
```

**Suspicion #6** (line 714, policy loss): `policy_loss = -(Advs * q_vals).mean()`. This is **not** a standard policy-gradient loss. In actor-critic you would compute `-log_prob(action) * Adv`. Multiplying advantage by the Q-value is dimensionally odd: it's `Adv · Q`, which depends on the magnitude of Q, not the action probability. Either the LToS paper formulates it this way (it does not — LToS uses Q-based credit assignment but still has a policy-gradient term), or this is wrong. Likely a paper-implementation mismatch.

```python
w_in_grads = self.policy[i].compute_gradients(obs, stored_w_in, actions)
# w_in_grads tensor [120, 4, 32]

neighbors = torch.where(self.neighbor_mask[i] == 1)[0]
w_out = self.policy[i].phi[0](obs)  # [120, 32]
theta_grad = None

for t in range(batch_size):
    w_out_t = w_out[t]
    w_out_t.requires_grad_(True)        # ← see suspicion #7

    neighbor_grads = []
    for n, neighbor_idx in enumerate(neighbors):
        g_in = w_in_grads[t, n]
        neighbor_grads.append(g_in)

    neighbor_grads = torch.stack(neighbor_grads)

    phi_grad = torch.autograd.grad(
        w_out_t,
        self.policy[i].phi[0].parameters(),
        grad_outputs=neighbor_grads.sum(dim=0),
        create_graph=True,
        retain_graph=True
    )

    if theta_grad is None:
        theta_grad = phi_grad
    else:
        theta_grad = [g1 + g2 for g1, g2 in zip(theta_grad, phi_grad)]

theta_grad = [g / batch_size for g in theta_grad]

for param, grad in zip(self.policy[i].phi[0].parameters(), theta_grad):
    if param.grad is None:
        param.grad = grad
    else:
        param.grad += grad
```

**Suspicion #7** (line 745): `w_out_t.requires_grad_(True)`. This sets `requires_grad` on a single row of `w_out` — but `w_out` was produced by `self.policy[i].phi[0](obs)`, so it already carries a gradient graph back to `phi[0]`'s parameters. Setting `requires_grad_` on a non-leaf tensor is a no-op at best and an error at worst (PyTorch typically raises `RuntimeError: you can only change requires_grad flags of leaf variables`). It's possible this *silently works* because `w_out_t` is a view, but it's suspicious.

**Suspicion #8** (line 757–763): `torch.autograd.grad(w_out_t, self.policy[i].phi[0].parameters(), grad_outputs=neighbor_grads.sum(dim=0), create_graph=True, retain_graph=True)`. This re-traverses the `phi[0]` computation graph **per timestep** for `batch_size` iterations. With `create_graph=True` this builds a second-order graph each time, and with `retain_graph=True` nothing is freed. Memory grows linearly with `batch_size` per backward call. For `batch_size = 120` and a small `phi[0]` this may be tolerable but it's wasteful — the same computation could be done with a single batched autograd call using `grad_outputs=neighbor_grads.sum(dim=1)` over the full `w_out`.

**Suspicion #9** (line 775–779): the grad accumulation manually adds `theta_grad` into `param.grad`. But `total_loss.backward()` two stanzas earlier already populated `param.grad` for `phi[0]` (since `q_loss + policy_loss` flows through `q_vals → w_out → phi[0]`). So the manual accumulation is **double-counting** the gradient from the q-loss path through `phi[0]`. This is almost certainly a bug — the LToS algorithm's gradient-exchange term should be the **only** gradient on `phi[0]` (the high-level policy), not added on top of the Q-learning gradient.

```python
self.optimizer.step()

self.policy[i].soft_update_target_phi()
self.policy[i].soft_update_target_policy()

self._update_lr()

if summary_writer is not None:
    self.policy[i]._update_tensorboard(summary_writer, global_step)
```

**Suspicion #10** (line 782): `self.optimizer.step()` is called **inside the per-agent loop**. The base `IA2C.backward` accumulates grads for all agents then steps once at the end. Here, each agent steps individually — meaning agent 0's update changes the parameters before agent 1's gradient computation reads them. With `n_agent` agents and `gradient_steps` outer iterations, the optimiser steps `n_agent × gradient_steps` times per `backward()` call. Whether this is intentional (sequential agent updates) or a bug depends on the algorithm spec.

**Suspicion #11**: `self.max_grad_norm` clipping (the standard `nn.utils.clip_grad_norm_` line) is **missing** from `IA2C_LToS.backward`. Every other algorithm clips before `optimizer.step()`. Also, `self.grad_clip` (set in `__init__`) is read but never used.

**Suspicion #12** (line 791): `_update_lr` is called every agent inside every gradient step, regardless of `self.lr_decay`. Other classes only call it when `self.lr_decay != 'constant'`.

### `reset` (lines 797–804)

```python
def reset(self):
    super().reset()
    self.w_in  = [[] for _ in range(self.n_agent)]
    self.w_out = [[] for _ in range(self.n_agent)]
    if self.use_lstm:
        for i in range(self.n_agent):
            self.policy[i].states_fw        = torch.zeros(1, self.policy[i].shared_dim * 2, device=self.device)
            self.policy[i].target_states_fw = torch.zeros(1, self.policy[i].shared_dim * 2, device=self.device)
```

Calls `IA2C.reset` (per-agent `_reset`), then zeros out the message caches and explicitly rebuilds the LSTM hidden states. The `shared_dim * 2` factor suggests `states_fw` packs `(h, c)` for an LSTM cell into one tensor.

### `_update_tensorboard` (lines 806–820)

Logs `current_epsilon`, `avg_q_loss`, `avg_policy_loss` to TensorBoard. Per-agent scalars are commented out.

---

## Algorithm registry summary table

| Class | `name` | Parent | Policy class | Buffer | `add_transition` extra arg | Single vs per-agent policy | Notable override |
|---|---|---|---|---|---|---|---|
| `IA2C` | `'ia2c'` | — | `LstmPolicy` × N | `OnPolicyBuffer` × N | `naction` (neighbour actions) | per-agent `nn.ModuleList` | base impl |
| `IA2C_FP` | `'ia2c_fp'` | `IA2C` | `FPPolicy` × N | `OnPolicyBuffer` × N | `naction` (inherited) | per-agent `nn.ModuleList` | `_init_policy` augments state with neighbour-action probs |
| `MA2C_NC` | `'ma2c_nc'` | `IA2C` | `NCMultiAgentPolicy` / `NCMultiAgentPolicy_MLP` / `GraphCMultiAgentPolicy` (config-driven) | `MultiAgentOnPolicyBuffer` (single) | `p` (neighbour policy fingerprints) | single policy | `add_transition`, `forward`, `backward`, `reset` all overridden |
| `IA2C_CU` | `'ma2c_cu'` ⚠ | `MA2C_NC` | `ConsensusPolicy` | inherits | inherits | single | `backward` appends `consensus_update()` |
| `BayesianGraph` ⭐ | `'bayesian_graph'` | `MA2C_NC` | `BayesianGraphCMultiAgentPolicy` (only if `is_graph_nn=True`) | inherits | inherits | single | `_init_policy` wires `learn_mask`; adds `visualize_masks` |
| `MA2C_DIAL` | `'ma2c_dial'` | `MA2C_NC` | `DIALMultiAgentPolicy` | inherits | inherits | single | only `_init_policy` |
| `MA2C_CNET` | `'ma2c_ic3'` ⚠ | `MA2C_NC` | `CommNetMultiAgentPolicy` | inherits | inherits | single | only `_init_policy` |
| `IA2C_LToS` | (no `self.name` set ⚠) | `IA2C` | `LToSMultiAgentPolicy` × N | `LToSPolicyBuffer` × N | `naction` (signature) + `self.w_in[i]` passed in body | per-agent `nn.ModuleList` | bespoke `forward` (w_in/w_out exchange), bespoke Q-learning `backward` |

Cells marked ⚠ are naming inconsistencies — e.g. class `MA2C_CNET` self-identifies as `'ma2c_ic3'`. `IA2C_LToS.__init__` notably never sets `self.name`, so it inherits `'ia2c'` from any prior `IA2C` instance — but since `self.name` is set inside `__init__`, and `IA2C_LToS.__init__` doesn't set it, `self.name` is whatever the **super class** would set… except `super().__init__` calls `_init_algo`, which doesn't touch `self.name`. So `IA2C_LToS` instances have **no `self.name` attribute at all**. Confirmed bug.

---

## Bugs / oddities

1. **`Large_city` / `coop_gamma = -1` special-casing is NOT in `models.py`.** The user prompt asked me to quote it from this file, but it does not appear here — it lives in `/tmp/BayesG/main.py` lines 92–93, 105–106, 118–119 inside `init_agent()`:

   ```python
   if env_name == "Large_city":
       coop_gamma = -1
       adj_order = 30
       gnn_type = config.get('MODEL_CONFIG', 'gnn_type', fallback='gat')
       return MA2C_NC(..., coop_gamma, ..., gnn_type=gnn_type)
   ```

   `models.py` itself contains no `Large_city` references and no `gnn_type` override logic of its own. See [[main.py]] for the dispatch.

2. **`BayesianGraph` ignores the `gnn_type` override**. In [[main.py]] line 111–113 the `Large_city` branch reads `gnn_type` from config but the `BayesianGraph(...)` constructor call **does not pass it** (unlike `MA2C_NC(..., gnn_type=gnn_type)` on line 100). Inside `BayesianGraph._init_policy`, `gnn_type` is read from `model_config` independently, so the override is functionally moot — but it shows intent didn't make it to code.

3. **`IA2C_LToS` never sets `self.name`**. Every other class sets it as the first line of `__init__`. Confirmed by inspection — line 482 jumps straight into reading config.

4. **`MA2C_CNET` is named `'ma2c_ic3'`** (line 468) despite the class being `CNET` — historical name from when this was based on IC3Net.

5. **`IA2C_CU` is named `'ma2c_cu'`** (line 401) despite the `IA2C` prefix.

6. **`IA2C` constructor sets `self.n_s = n_s_ls[0]`** when `identical_agent=True` — but identicality is checked on **action** dims only (line 140). If state dims differ but action dims are equal, `n_s` silently truncates to agent 0's state dim.

7. **`MA2C_NC.__init__` accepts `gnn_type='gat'` kwarg that is unused** — it's overwritten in `_init_policy` by reading `model_config`. Vestigial parameter.

8. **`load()` checks `isinstance(self.policy, GraphCMultiAgentPolicy)` but not `BayesianGraphCMultiAgentPolicy`** — checkpoint resumption for the paper's algorithm falls into the generic branch. May or may not be intentional.

9. **`IA2C_LToS.backward` issues** (recap of suspicions in walkthrough):
   - L657 `continue` skips Q-loss / step entirely for agents with no neighbours (probably intentional but undocumented).
   - L663 calls `sample_transition` on neighbour buffers while iterating agent `i` — if the buffer is destructive this is broken.
   - L703 uses `next_w_out` (the last neighbour's target-phi output) as input to `compute_target_critic` instead of the aggregated `next_w_in` or this agent's own next-step output. Likely a bug.
   - L714 policy loss is `-(Advs * q_vals).mean()` — multiplies advantage by Q-value, not by `log_prob`. Non-standard.
   - L745 `w_out_t.requires_grad_(True)` on a non-leaf tensor — at best a no-op.
   - L757–763 `torch.autograd.grad(..., create_graph=True, retain_graph=True)` is called inside a per-timestep loop, blowing up memory unnecessarily.
   - L775–779 manually adds `theta_grad` into `phi[0]`'s `param.grad` on top of what `total_loss.backward()` already deposited there — **double-counts** the q-loss path through `phi[0]`.
   - L782 `self.optimizer.step()` inside the per-agent inner loop — agents update sequentially, not jointly. Different from every other algorithm in this file.
   - **No `max_grad_norm` clipping** — every other algorithm clips; `self.grad_clip` is set in `__init__` but never read.
   - L791 `_update_lr` called every agent every gradient step, ignoring `lr_decay`.
   - `current_epsilon` is initialised but never decayed in this file.

10. **`MA2C_NC._init_policy` debug print** (line 356) `print(f"Config is_graph_nn value: …")` is left in production code.

11. **`is_graph_nn` is stored on `self` in `MA2C_NC._init_policy` and again in `BayesianGraph._init_policy`** — duplicate state, easy to drift if config is ever mutated.

12. **`MA2C_NC._init_policy` reads `'gnn_type'` default `'gcn'`**, but [[main.py]] line 98 reads it with default `'gat'`. Inconsistent defaults across layers.

13. **The four-line block at line 590–591** truncating `self.policy[i].states_fw[:1]` reaches inside the policy module's internals from the wrapper — violates encapsulation; looks like a workaround for an LSTM hidden-state growth bug in [[policies.py#LToSMultiAgentPolicy]].

14. **Heterogeneous padding in `MA2C_NC._convert_hetero_states`** uses zeros — agents with smaller `n_s_ls[i]` get zero-padded observations and neighbours see zero policy probabilities for the unused action slots. This is the source of any silent dim-mismatch debugging in mixed-agent runs.

15. **Inheritance pattern inconsistency**: every subclass except `IA2C_LToS` re-implements `__init__` by calling `self._init_algo(...)` rather than `super().__init__(...)`. The pattern works because `_init_algo` is the actual constructor, but it means subclass `__init__` blocks are pure boilerplate — a `super().__init__()` call would be cleaner everywhere.
