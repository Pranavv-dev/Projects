# `agents/utils.py` — Line-by-Line Walkthrough

Source: `/tmp/BayesG/agents/utils.py` (373 lines). This file contains four logical blocks: **initializers**, **layer helpers**, **transition buffers** (the workhorse of the on-policy training loop), and a **linear scheduler**. Related notes: [[ma2c]], [[ic3net]], [[lto_s]], [[policies]].

---

## `init_layer` (fc, lstm) — orthogonal init choice

```python
def init_layer(layer, layer_type):
    if layer_type == 'fc':
        nn.init.orthogonal_(layer.weight.data)
        nn.init.constant_(layer.bias.data, 0)
    elif layer_type == 'lstm':
        nn.init.orthogonal_(layer.weight_ih.data)
        nn.init.orthogonal_(layer.weight_hh.data)
        nn.init.constant_(layer.bias_ih.data, 0)
        nn.init.constant_(layer.bias_hh.data, 0)
```

- **Lines 8–11 (`fc` branch)**: orthogonally initializes a fully-connected layer's weight matrix and zeros its bias. Orthogonal init keeps singular values at 1, which preserves the norm of activations on the forward pass and the norm of gradients on the backward pass — a well-known stabilizer for deep nets and for actor/critic heads where exploding/vanishing signal would corrupt early training.
- **Lines 12–16 (`lstm` branch)**: applies orthogonal init separately to the input→hidden (`weight_ih`) and hidden→hidden (`weight_hh`) matrices, then zeros both biases. Orthogonal init of the recurrent matrix `W_hh` is the standard recipe for fighting vanishing gradients through time in vanilla LSTMs (the recurrent eigenvalues sit on the unit circle at t=0). Biases at 0 means the forget gate starts neutral; some codebases bias the forget gate to +1 instead, but this file does not.
- Notably, no `if isinstance(layer, nn.LSTMCell)` check — the caller must pass a literal `'lstm'` or `'fc'` string. Anything else is a silent no-op.

---

## `batch_to_seq`, `run_rnn` — chunked-LSTM execution and done-masking

```python
def batch_to_seq(x):
    n_step = x.shape[0]
    if len(x.shape) == 1:
        x = torch.unsqueeze(x, -1)
    return torch.chunk(x, n_step)
```

- **Line 22**: takes the leading dim as the sequence length `n_step`.
- **Lines 23–24**: if `x` is 1-D (e.g. a `(T,)` done vector), promote to `(T, 1)` so chunking yields per-timestep `(1, 1)` tensors, matching the per-timestep `(1, feat)` shape expected by `nn.LSTMCell`.
- **Line 25**: `torch.chunk(x, n_step)` along dim 0 produces a tuple of `n_step` tensors each of shape `(1, feat)` — i.e. one minibatch-of-1 per timestep. This is the "convert a rollout batch into a Python-loopable sequence" idiom.

```python
def run_rnn(layer, xs, dones, s):
    xs = batch_to_seq(xs)
    dones = batch_to_seq(dones)
    n_in = int(xs[0].shape[1])
    n_out = int(s.shape[0]) // 2
    s = torch.unsqueeze(s, 0)
    h, c = torch.chunk(s, 2, dim=1)
    outputs = []
    for ind, (x, done) in enumerate(zip(xs, dones)):
        c = c * (1-done)
        h = h * (1-done)
        h, c = layer(x, (h, c))
        outputs.append(h)
    s = torch.cat([h, c], dim=1)
    return torch.cat(outputs), torch.squeeze(s)
```

- **Lines 29–31**: convert both the observation rollout `xs` and the per-step done flags `dones` into chunked, per-timestep tensors.
- **Line 32**: `n_in` is captured but unused locally — looks like a leftover.
- **Line 33**: `n_out` derived from the flat hidden state by halving (the state stores `[h; c]` concatenated). Also captured but unused (kept for clarity / future use).
- **Lines 34–35**: state `s` arrives as a 1-D `(2*n_out,)` vector; `unsqueeze(0)` makes it `(1, 2*n_out)` and `chunk(..., 2, dim=1)` splits into `h` and `c`, each `(1, n_out)`.
- **Lines 37–41 — the done-masking core**: for each timestep,
  - **Line 38–39**: `c = c * (1-done)` and `h = h * (1-done)` zero out the carried state *before* the cell call **iff** the previous step ended an episode. This is how the LSTM "forgets" across episode boundaries inside a single batched rollout — without resetting, hidden state from episode k would leak into episode k+1.
  - **Line 40**: `layer(x, (h, c))` is `nn.LSTMCell.__call__`, producing the new `(h, c)`.
  - **Line 41**: collect `h` (not `c`) as the timestep output.
- **Lines 42–43**: re-pack the final `(h, c)` into a flat state vector and return both `(T, n_out)` outputs and the squeezed `(2*n_out,)` carry-out state.
- **Important convention**: `dones` here are *pre-step* done flags — `dones[t]` says "did the *previous* episode end before step t". That's why masking is applied *before* the cell call. This matches the `self.dones = [done]` initialization in `OnPolicyBuffer.reset` (line 98), where the buffer pre-seeds a single pre-step done.

---

## `one_hot` — arbitrary-dim handling check

```python
def one_hot(x, oh_dim, dim=-1):
    device = x.device
    oh_shape = list(x.shape)
    if dim == -1:
        oh_shape.append(oh_dim)
    else:
        oh_shape = oh_shape[:dim+1] + [oh_dim] + oh_shape[dim+1:]
    x_oh = torch.zeros(oh_shape,  device=device)
    x = torch.unsqueeze(x, -1)
    if dim == -1:
        x_oh = x_oh.scatter(dim, x, 1)
    else:
        x_oh = x_oh.scatter(dim+1, x, 1)
    return x_oh
```

- **Line 47**: preserves device (CPU vs CUDA).
- **Lines 48–52**: builds the output shape. For `dim=-1` it appends `oh_dim` at the end; for an explicit `dim`, it splices `oh_dim` immediately *after* index `dim` (so the one-hot axis lands at position `dim+1`).
- **Line 54**: `unsqueeze(x, -1)` adds a trailing singleton so the indices tensor has the same rank as `x_oh`.
- **Lines 55–58**: `scatter` writes 1.0 along the inserted axis. For non-`-1` `dim`, scattering at `dim+1` is consistent with the shape splice on line 52.
- **Arbitrary-dim verification**:
  - Input `x` of any shape `(d0, d1, ..., dk)` becomes output of shape `(d0, ..., d_dim, oh_dim, d_{dim+1}, ..., dk)` when `dim` is non-negative, or `(d0, ..., dk, oh_dim)` when `dim == -1`. Both cases handled.
  - **Caveat**: only `dim == -1` is treated specially. Other negative values (e.g. `dim = -2`) will follow the `else` branch with `dim+1`, which still yields a valid (but counter-intuitive) negative-index scatter. Not a bug per se, but the documented contract is effectively "use `-1` or a non-negative `dim`."
  - **Caveat 2**: `x` must be `torch.long` (scatter requires `int64` indices); not enforced here. Caller's job.

---

## `class TransBuffer` (base) — interface

```python
class TransBuffer:
    def reset(self):
        self.buffer = []

    @property
    def size(self):
        return len(self.buffer)

    def add_transition(self, ob, a, r, *_args, **_kwargs):
        raise NotImplementedError()

    def sample_transition(self, *_args, **_kwargs):
        raise NotImplementedError()
```

- A minimal abstract base. Only `add_transition` and `sample_transition` are required overrides; `reset` here sets `self.buffer = []` but **subclasses override `reset` entirely** and never touch `self.buffer` — so the base `reset`/`size` are effectively dead code in the on-policy subclasses. `size` would return `len([])` since no subclass populates `self.buffer`.

---

## `class OnPolicyBuffer`

### `__init__` (lines 81–89)

```python
def __init__(self, gamma, alpha, distance_mask):
    self.gamma = gamma
    self.alpha = alpha
    if alpha > 0:
        self.distance_mask = distance_mask
        self.max_distance = np.max(distance_mask, axis=-1)
    self.reset()
```

- **`gamma`**: standard temporal discount factor in `[0, 1)`.
- **`alpha`**: **dual-purpose** scalar:
  - When `alpha < 0` it's a **sentinel** telling `sample_transition` to use *vanilla* reward-to-go (`_add_R_Adv`); no spatial reward shaping at all (see line 109).
  - When `alpha >= 0` it's an actual **spatial-discount factor** — the per-hop multiplicative attenuation for rewards received by neighbors. Higher `alpha` means farther neighbors' rewards count more.
  - Note the asymmetry: `alpha > 0` (strict) gates the distance-mask storage, but `alpha < 0` (strict) gates the algorithm choice — `alpha == 0` falls through to `_add_s_R_Adv` *without* having `distance_mask` set, which would crash on line 170 (`self.max_distance`). So `alpha == 0` is effectively unsupported.
- **`distance_mask`**: a per-agent matrix of hop-distances between agents/cells. `max_distance` precomputes the largest distance per row so the spatial loop can be bounded.

### `reset` / `add_transition` (lines 91–106)

```python
def reset(self, done=False):
    self.obs = []
    self.acts = []
    self.rs = []
    self.vs = []
    self.adds = []
    self.dones = [done]

def add_transition(self, ob, na, a, r, v, done):
    self.obs.append(ob)
    self.adds.append(na)
    self.acts.append(a)
    self.rs.append(r)
    self.vs.append(v)
    self.dones.append(done)
```

- `reset` seeds `self.dones` with a single element (the *pre-step* done from the previous rollout's last step). This is why `self.dones` is always length `T+1` after `T` `add_transition` calls — index 0 is "pre-step done before step 0".
- `add_transition` is straightforward append. The `na` slot ("neighbor actions" or "additional info") is renamed to "policy logits" or "input weights" in subclasses.

### `sample_transition` flow (lines 108–121)

```python
def sample_transition(self, R, dt=0):
    if self.alpha < 0:
        self._add_R_Adv(R)
    else:
        self._add_s_R_Adv(R)
    obs = np.array(self.obs, dtype=np.float32)
    nas = np.array(self.adds, dtype=np.int32)
    acts = np.array(self.acts, dtype=np.int32)
    Rs = np.array(self.Rs, dtype=np.float32)
    Advs = np.array(self.Advs, dtype=np.float32)
    dones = np.array(self.dones[:-1], dtype=np.bool)
    self.reset(self.dones[-1])
    return obs, nas, acts, dones, Rs, Advs
```

- **Branching (lines 109–112)**: `alpha < 0` → `_add_R_Adv` (vanilla reward-to-go). `alpha >= 0` → `_add_s_R_Adv` (spatial-only reward, no temporal distance term). Note `_add_st_R_Adv` (the spatial-*and*-temporal variant) is **never reached from this dispatcher** — it's defined but unused in `OnPolicyBuffer.sample_transition`.
- **Lines 113–117**: collect rollouts into numpy arrays in standard `(T, ...)` layout.
- **Line 119**: `self.dones[:-1]` is the **pre-step** dones (length T), drops the trailing post-step done that's saved for the next rollout's seed.
- **Line 120**: `self.reset(self.dones[-1])` — re-seeds the next rollout with the final post-step done as the new pre-step done.
- **Returns**: `(obs, nas, acts, dones, Rs, Advs)`.

### `_add_R_Adv` — standard reward-to-go (lines 123–136)

```python
def _add_R_Adv(self, R):
    Rs = []
    Advs = []
    for r, v, done in zip(self.rs[::-1], self.vs[::-1], self.dones[:0:-1]):
        R = r.cpu().numpy() if isinstance(r, torch.Tensor) else r + self.gamma * R * (1.-done)
        Adv = R - v
        Rs.append(R)
        Advs.append(Adv)
    Rs.reverse()
    Advs.reverse()
    self.Rs = Rs
    self.Advs = Advs
```

- The intended recurrence is the classic reward-to-go bootstrap with terminal masking:
  - `R_t = r_t + gamma * R_{t+1} * (1 - done_{t+1})`
  - `Adv_t = R_t - V(s_t)`
  - Iterate from `t = T-1` down to `t = 0` with `R_T` = the bootstrap value passed in as the `R` argument.
- **BUG — line 129 operator precedence**: the ternary
  ```python
  R = r.cpu().numpy() if isinstance(r, torch.Tensor) else r + self.gamma * R * (1.-done)
  ```
  parses as `R = (r.cpu().numpy() if isinstance(r, torch.Tensor) else (r + self.gamma * R * (1.-done)))`. So when `r` is a tensor, **the gamma-discounted return term is silently discarded** and `R` is overwritten with the raw current-step reward as a numpy array. The intended code is almost certainly
  ```python
  r = r.cpu().numpy() if isinstance(r, torch.Tensor) else r
  R = r + self.gamma * R * (1.-done)
  ```
  See [[Bugs / oddities]] below.
- **Reverse iteration**: `self.rs[::-1]`, `self.vs[::-1]`, `self.dones[:0:-1]`. The dones slice `[:0:-1]` excludes index 0 — which is exactly the pre-step done — so we iterate the post-step dones `[done_T, done_{T-1}, ..., done_1]` aligned with `[r_{T-1}, ..., r_0]`. That's correct.
- After accumulating in reverse, `reverse()` puts them back in chronological order.

### `_add_st_R_Adv` — spatial + temporal reward with distance mask (lines 138–160)

```python
def _add_st_R_Adv(self, R, dt):
    Rs = []
    Advs = []
    tdiff = dt
    for r, v, done in zip(self.rs[::-1], self.vs[::-1], self.dones[:0:-1]):
        R = self.gamma * R * (1.-done)
        if done:
            tdiff = 0
        tmax = min(tdiff, self.max_distance)
        for t in range(tmax + 1):
            rt = torch.sum(r[self.distance_mask == t])
            R += (self.gamma * self.alpha) ** t * rt
        Adv = R - v
        tdiff += 1
        Rs.append(R)
        Advs.append(Adv)
    Rs.reverse()
    Advs.reverse()
    self.Rs = Rs
    self.Advs = Advs
```

- Walks backward through the rollout like `_add_R_Adv`, but the per-step current reward is decomposed by **hop-distance** via `self.distance_mask == t` masking.
- **Line 145**: bootstrap-only term first: `R <- gamma * R * (1 - done)`. The reward at this step is *not yet added*; it'll be added by the inner loop at `t=0`.
- **Lines 146–147**: a per-iteration `tdiff` counter resets on episode boundaries. This bounds how far backward the spatial-temporal coupling can propagate within an episode.
- **Line 149**: `tmax = min(tdiff, self.max_distance)` — only sum up to the smaller of "how many steps we've been in the current episode" and "the largest hop-distance present in the mask". Capping at `tdiff` means at step 0 of an episode you only get the t=0 (own) reward; one step in, you add t=0 and t=1; etc. — a *forward-in-time* spatial reach.
- **Lines 150–152**: inner loop adds `(gamma * alpha)^t * sum_of_rewards_at_hop_t`. So both temporal (`gamma`) and spatial (`alpha`) discounts compound multiplicatively per hop. This is the "spatial-temporal" coupling the function name implies.
- **`tdiff += 1`** (line 154) inside the *reversed* loop — and `dt` is the initial value passed by the caller — means as we step backward in the rollout, `tdiff` grows. (Slightly counter-intuitive: backward iteration but increasing `tdiff`.) Combined with the episode reset on line 147, this models how far we are from the **start** of the episode.
- **Never called by the base `OnPolicyBuffer.sample_transition`** (only by callers passing `dt`); only the multi-agent variant calls a similar function with `dt`.

### `_add_s_R_Adv` — spatial-only reward (lines 162–181)

```python
def _add_s_R_Adv(self, R):
    Rs = []
    Advs = []
    for r, v, done in zip(self.rs[::-1], self.vs[::-1], self.dones[:0:-1]):
        R = self.gamma * R * (1.-done)
        for t in range(self.max_distance + 1):
            if isinstance(r, np.ndarray):
                r = torch.from_numpy(r).float()
            rt = torch.sum(r[self.distance_mask == t])
            R += (self.alpha ** t) * rt
        Adv = R - v
        Rs.append(R)
        Advs.append(Adv)
    Rs.reverse()
    Advs.reverse()
    self.Rs = Rs
    self.Advs = Advs
```

- Same backward sweep, but **no `tdiff` bookkeeping**: every step sums over all hops up to `self.max_distance`.
- **Lines 171–172**: lazy numpy-to-torch conversion (only `_add_s_R_Adv` and `_add_st_R_Adv` do this, not `_add_R_Adv`).
- **Line 174 — the `(alpha^t * rt)` term**: at hop `t`, sum the rewards of all agents at distance `t` from the focal agent, weight by `alpha^t`. So:
  - `t=0`: own reward, weight 1.
  - `t=1`: immediate-neighbor rewards, weight `alpha`.
  - `t=2`: 2-hop neighbor rewards, weight `alpha^2`.
  - …
- Crucially, `gamma` is **only** applied to the *bootstrap value* `R` on line 168, **not** mixed into the spatial discount. Contrast with `_add_st_R_Adv` where the per-hop weight is `(gamma * alpha)^t`. Hence "spatial-only".
- The bootstrap from the next-step value `R` is still temporally discounted by `gamma`, so this is really "temporal-discount on the bootstrap + flat per-hop spatial discount on the immediate reward decomposition."

### Which `sample_transition` path runs when?

| `alpha` value | Called from `sample_transition` | Method dispatched | What happens |
|---|---|---|---|
| `alpha < 0`  | line 109 (`if self.alpha < 0`) | `_add_R_Adv` | Vanilla reward-to-go; no `distance_mask` needed; spatial structure ignored. |
| `alpha == 0` | line 111 (`else`) | `_add_s_R_Adv` | **Crashes** — `__init__` skipped storing `self.distance_mask` and `self.max_distance` because the `if alpha > 0` check on line 84 is strict. |
| `alpha > 0`  | line 111 (`else`) | `_add_s_R_Adv` | Spatial-only shaping. `_add_st_R_Adv` exists but **never reached** in this class. |

---

## `class MultiAgentOnPolicyBuffer` — what changes

```python
class MultiAgentOnPolicyBuffer(OnPolicyBuffer):
    def __init__(self, gamma, alpha, distance_mask):
        super().__init__(gamma, alpha, distance_mask)
```

- Same init, same dispatcher branching (`alpha < 0` ⇒ `_add_R_Adv`; else `_add_s_R_Adv`).
- **Key differences in `sample_transition` (lines 208–213)**:
  - `obs = np.transpose(np.array(self.obs, dtype=np.float32), (1, 0, 2))` — original shape per-step is `(N_agents, obs_dim)`, so the array of T steps is `(T, N_agents, obs_dim)`. The transpose `(1, 0, 2)` reshapes to `(N_agents, T, obs_dim)` — i.e. the **leading axis is now agent**, not time. Downstream code can then index `obs[i]` to get agent i's full rollout.
  - `policies = np.transpose(np.array(self.adds, dtype=np.float32), (1, 0, 2))` — same trick: `self.adds` here stores per-step policy distributions/logits (overloading the slot used for "neighbor actions" in the single-agent buffer), and gets reshaped to `(N_agents, T, action_dim)`.
  - `acts = np.transpose(np.array(self.acts, dtype=np.int32))` — `self.acts` has shape `(T, N_agents)`, so `np.transpose` with no axes argument reverses to `(N_agents, T)`.
  - `dones = np.array(self.dones[:-1], dtype=bool)` — uses Python's `bool` rather than `np.bool` (which is the deprecated alias used in `OnPolicyBuffer.sample_transition`, see [[Bugs / oddities]]).
- **Per-agent return/advantage computation**: `_add_R_Adv` (lines 220–240), `_add_st_R_Adv`, and `_add_s_R_Adv` all gain an outer `for i in range(vs.shape[1])` loop iterating over agents. The bootstrap `R` is now a per-agent vector indexed as `R[i]`. Each agent's reward-to-go is computed independently using that agent's value estimates `vs[::-1, i]` and (for spatial variants) `distance_mask[i]` / `max_distance[i]`. Results stack into `(N_agents, T)` arrays.
- **Per-agent reward-to-go is correctly written** in `_add_R_Adv` here (line 231: `cur_R = r + self.gamma * cur_R * (1.-done)`), avoiding the operator-precedence bug present in the single-agent version's line 129. So MAOP's `_add_R_Adv` is the *correct* reference implementation.
- **`r` is *not* tensor-typed in `_add_R_Adv`** here — no `.cpu().numpy()` branch — implying the multi-agent buffer's caller always feeds numpy arrays. The spatial variants still have the lazy `torch.from_numpy` conversion (lines 265–266, 295–296).

---

## `class LToSPolicyBuffer` — `w_in` storage and non-reset

```python
class LToSPolicyBuffer(OnPolicyBuffer):
    def __init__(self, gamma, alpha, distance_mask):
        super().__init__(gamma, alpha, distance_mask)
        self.w_in = []
```

- Adds a `w_in` list that stores **input weights from neighbors** for the LToS (Learning to Share) algorithm. See [[lto_s]].
- **`reset` (lines 316–326)**: explicitly enumerates the fields to clear — `obs, adds, acts, rs, Rs, Advs, vs, dones` — and **deliberately omits `w_in`**. The comment `# Do NOT reset self.w_in here if you intend to keep its history` is explicit about the choice. This means `w_in` is a *persistent log* that survives across rollouts.
  - Note `reset` also adds `self.Rs = []` and `self.Advs = []` that weren't in the base `reset` — needed because `sample_transition` here doesn't always overwrite them.
- **`add_transition` (lines 328–333)**: appends `w_in` (if provided) then delegates to the base. The cloned-detach comment on line 331 is dead code (commented out); the live code appends the raw `w_in` tensor list, which could be a gradient-graph leak depending on caller behavior.
- **`sample_transition` (lines 335–354)**: 
  - Same dispatcher as base (`alpha < 0` ⇒ `_add_R_Adv`, else `_add_s_R_Adv`).
  - Lines 340–346 build the same numpy returns as the base.
  - Line 346 also returns `vs` as a numpy array — but it's collected and **never returned** by the function (defined but unused).
  - **Lines 347–351 — manual `w_in` book-keeping** (this is the unique part):
    ```python
    stored_w_in = self.w_in[:len(obs)]
    self.w_in = self.w_in[len(obs):]
    ```
    Take the first `T` (= `len(obs)`) elements as the current batch, then keep the remainder. So `w_in` acts as a FIFO queue across rollouts. In practice `len(self.w_in) == len(obs)` after each rollout (since `add_transition` appends one `w_in` per call), so the suffix is empty — making the persistence theoretically meaningful but practically a no-op unless `add_transition` is called with `w_in=None` for some steps.
  - Line 353: `self.reset(self.dones[-1])` re-seeds dones but, per `reset`'s definition, **does not touch `self.w_in`** — the queue logic on lines 350–351 is the only thing managing it.
  - Returns the extended tuple `(obs, nas, acts, dones, Rs, Advs, stored_w_in)`.

---

## `class Scheduler` — linear decay (lines 359–372)

```python
class Scheduler:
    def __init__(self, val_init, val_min=0, total_step=0, decay='linear'):
        self.val = val_init
        self.N = float(total_step)
        self.val_min = val_min
        self.decay = decay
        self.n = 0

    def get(self, n_step):
        self.n += n_step
        if self.decay == 'linear':
            return max(self.val_min, self.val * (1 - self.n / self.N))
        else:
            return self.val
```

- Stateful scheduler. `self.n` accumulates how many environment steps have elapsed.
- **Linear formula**: `value(n) = max(val_min, val_init * (1 - n / N))`.
  - At `n = 0`: returns `val_init`.
  - At `n = N`: returns `max(val_min, 0)` = `val_min`.
  - Beyond `n = N`: factor goes negative, clipped by `max(val_min, ...)`. So `val_min` is a hard floor.
- **`else` branch** (line 372): for any `decay != 'linear'`, returns the *initial* `self.val` unchanged — i.e. no decay at all. Constant fallback, not e.g. exponential. This means `decay='exp'` would silently be a constant schedule.
- **Divide-by-zero**: if `total_step=0` (the default!), the linear branch computes `self.val * (1 - self.n / 0.0)` which produces `-inf` (or `nan` when `n=0`), then `max(val_min, -inf)` returns `val_min`. So calling `get` on a default-constructed `Scheduler('linear')` instantly snaps to `val_min`. Caller must supply a real `total_step`.

---

## Bugs / oddities

1. **`np.bool` is removed in NumPy ≥ 1.20** (deprecated) and gone in NumPy ≥ 1.24. Lines `119` and `345` use `dtype=np.bool`, which will raise `AttributeError: module 'numpy' has no attribute 'bool'` on modern NumPy. Compare with `MultiAgentOnPolicyBuffer.sample_transition` line 213, which correctly uses `dtype=bool` (Python builtin). Fix: change both occurrences to `dtype=bool`.

2. **Operator-precedence bug in `OnPolicyBuffer._add_R_Adv` (line 129)**:
   ```python
   R = r.cpu().numpy() if isinstance(r, torch.Tensor) else r + self.gamma * R * (1.-done)
   ```
   When `r` is a tensor, the bootstrap term `self.gamma * R * (1.-done)` is **silently dropped** because the ternary binds tighter than `+`. The intended logic is "convert `r` to numpy if needed, then accumulate." Compare with `MultiAgentOnPolicyBuffer._add_R_Adv` line 231 (`cur_R = r + self.gamma * cur_R * (1.-done)`) which has no such branch and is correct. Fix:
   ```python
   r_np = r.cpu().numpy() if isinstance(r, torch.Tensor) else r
   R = r_np + self.gamma * R * (1.-done)
   ```

3. **`alpha == 0` crashes in `OnPolicyBuffer`**: `__init__` line 84 uses `if alpha > 0` (strict) to gate `distance_mask` storage, but `sample_transition` line 109 uses `if self.alpha < 0` (strict) to gate the algorithm. So `alpha == 0` falls into `_add_s_R_Adv` without `self.max_distance` ever being set, raising `AttributeError`. Either the gates should match (`>= 0` vs `< 0`) or `alpha == 0` should be explicitly documented as unsupported.

4. **`_add_st_R_Adv` is unreachable in `OnPolicyBuffer`**: it's defined (lines 138–160) but never called by `sample_transition`. The `dt` parameter on `sample_transition` (line 108) is accepted but unused. Same dead-code pattern in the multi-agent class — `MultiAgentOnPolicyBuffer.sample_transition` dispatches only to `_add_R_Adv` or `_add_s_R_Adv` despite accepting `dt`.

5. **Commented-out code in `LToSPolicyBuffer.add_transition` (line 331)**:
   ```python
   # cloned_w_in = [w.clone().detach() for w in w_in]
   ```
   The live code stores `w_in` directly. If `w_in` is a list of tensors still attached to the autograd graph, this leaks gradients across rollouts and inflates memory. The commented-out clone-detach was probably the safer original implementation; whoever uncommented it (or rather *re*commented it) should verify.

6. **`vs` collected but never returned in `LToSPolicyBuffer.sample_transition`** (line 346) — dead variable.

7. **`n_in` (line 32) and `n_out` (line 33) in `run_rnn` are computed but unused**.

8. **`TransBuffer.reset`/`size` are dead code** — subclasses override `reset` entirely and don't maintain `self.buffer`.

9. **`Scheduler` default `total_step=0`** is a footgun — instantly snaps the schedule to `val_min` due to division by zero.

10. **`one_hot` doesn't enforce that `x` is `torch.long`**; `scatter` will fail at runtime with a less obvious error if a float index tensor is passed.

---

## Cross-references

- Used by [[ma2c]] (multi-agent A2C policy/value rollouts) via `MultiAgentOnPolicyBuffer`.
- Used by [[lto_s]] (Learning to Share) via `LToSPolicyBuffer`.
- The `Scheduler` is used in [[policies]] / training loops for things like entropy-coefficient and learning-rate decay.
- `init_layer`, `run_rnn`, `batch_to_seq`, `one_hot` are imported across the [[policies]] module for actor/critic networks.
