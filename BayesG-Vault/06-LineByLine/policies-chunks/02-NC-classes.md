# NeurComm-style Multi-Agent Policies — Line-by-Line

**Source file:** `/tmp/BayesG/agents/policies.py`
**Range covered:** lines 193 – 871 (679 source lines)
**Classes covered:**
- `NCMultiAgentPolicy`  (lines 193 – 514)
- `NCMultiAgentPolicy_MLP`  (lines 516 – 870)

Both classes inherit from `Policy` (defined at line 15). The shared base provides `_run_loss` (lines 82–91) and `_update_tensorboard` (lines 93–102) — these are reused verbatim, so the walkthrough refers back to them but does not re-explain.

Helper utilities (from `agents/utils.py`):
- `init_layer(layer, 'fc'|'lstm')` — orthogonal init for weights, zeros for bias.
- `batch_to_seq(x)` — `torch.chunk(x, n_step)` along axis 0. Returns a tuple of `n_step` tensors.
- `one_hot(x, oh_dim)` — one-hot encode an index tensor; preserves device.
- `run_rnn` — exists but is NOT used by `NCMultiAgentPolicy` (NC re-implements the per-step loop manually inside `_run_comm_layers`).

---

## class NCMultiAgentPolicy  (lines 193 – 514)

### Docstring (lines 194 – 203)

Describes the class as a "centralized meta-DNN" where:
- All agents share parameters of a single model but process inputs independently (per-agent `ModuleList` entries, not literally shared weights — see `_init_net`).
- Communication is via a learned interaction model over the `neighbor_mask`.
- Architecture: actor-critic.

Note: the docstring claim that "all input and output dimensions are identical among all agents" is only true when `identical=True`. The `n_s_ls`/`n_a_ls` machinery is the escape hatch for heterogeneous agents.

### `__init__`  (lines 204 – 222)

```
def __init__(self, n_s, n_a, n_agent, n_step, neighbor_mask, n_fc=64, n_h=64,
             n_s_ls=None, n_a_ls=None, identical=True):
```

Parameters:
- **`n_s`** — per-agent observation dim (used when `identical=True`).
- **`n_a`** — per-agent action dim (discrete, used when `identical=True`).
- **`n_agent`** — total number of agents `N`.
- **`n_step`** — rollout horizon (used by the parent class to size buffers; stored as `self.n_step` by the parent).
- **`neighbor_mask`** — `N × N` adjacency-style mask. Each row `i` has 1s in columns corresponding to agent `i`'s neighbors. Stored as a tensor; line 216 prints its per-row sums (degree of each node).
- **`n_fc=64`** — hidden width of the three fan-in MLPs (`fc_x`, `fc_p`, `fc_m`).
- **`n_h=64`** — LSTM hidden state size. Also the per-agent message dimension (since messages = neighbors' `h`).
- **`n_s_ls`** / **`n_a_ls`** — per-agent obs/action dim lists, only populated when `identical=False`.
- **`identical`** — branches all heterogeneity logic.

Line 206: calls `super().__init__(n_a, n_s, n_step, 'nc', None, identical)`. The 4th arg `'nc'` becomes `self.name`. The 5th arg `None` (agent_name) means no suffix is appended to `self.name`.
Line 208: device selection — CUDA if available, else CPU. **Important:** `neighbor_mask` is passed in by the caller and never explicitly moved to `self.device` here, but it is read inside `_run_comm_layers` after being indexed — the device of the mask matters (see Open Questions).
Lines 211 – 213: only store `n_s_ls`/`n_a_ls` when not identical.
Lines 214 – 218: store the remaining args.
Line 220: `self._init_net()` — builds all `ModuleList`s.
Line 222: `self._reset()` — zero-init the forward/backward LSTM states.

### `backward`  (lines 228 – 260)

Signature: `backward(self, obs, fps, acts, dones, Rs, Advs, e_coef, v_coef, summary_writer=None, global_step=None)`.

Line 230: `obs = torch.from_numpy(obs).float().transpose(0, 1).to(self.device)`. Inline comment says final shape `[120, 25, 12]`. Reading literally: the incoming numpy `obs` is `[25, 120, 12]` (N, T, n_s), and `transpose(0, 1)` swaps to `[T=120, N=25, n_s=12]`.
Line 231: `dones` shape `[120]` (one done per timestep, broadcast across all agents).
Line 232: `fps` — fingerprints (neighbor policy distributions). Same transpose: `[T=120, N=25, n_a=5]`.
Line 233: `acts` shape `[N=25, T=120]` — note it is NOT transposed; later code indexes `acts[i]` per agent and gets a `[T]` vector.

Line 234: `hs, new_states = self._run_comm_layers(obs, dones, fps, self.states_bw)`.
- `hs` returned shape is `[N, T, n_h]` (we'll prove this in the `_run_comm_layers` walkthrough — `outputs.transpose(0, 1)`).
- `new_states` shape `[N, 2*n_h]` — packed `(h, c)`.

Line 236: `self.states_bw = new_states.detach()` — truncate BPTT across minibatches.
Line 237: actor logits per agent.
Line 238: critic values per agent.
Lines 239 – 241: zero the three loss accumulators.
Lines 242 – 243: cast `Rs`, `Advs` to tensors. Expected shape `[N, T]` (indexed as `Rs[i]`, `Advs[i]` per agent).
Line 246: **advantage normalization across the entire `[N, T]` batch**: `(Advs - Advs.mean()) / (Advs.std() + 1e-8)`. Note this is a single mean/std over all agents and steps — not per-agent. (This is a deviation from the MLP version: line 584 has no such normalization step.)

Lines 248 – 256: per-agent loss accumulation. For each agent `i`:
- Build a `Categorical(logits=ps[i])`. Recall `ps[i]` is `log_softmax`-ed (see line 425), so passing as `logits=` is consistent.
- `_run_loss` (inherited, lines 82 – 91) returns `(policy_loss, value_loss, entropy_loss)`:
  - `policy_loss = -(log_probs * Advs).mean()`
  - `entropy_loss = -(entropy()).mean() * e_coef`  (sign: negative entropy → minimizing this **encourages** high entropy)
  - `value_loss = (Rs - vs).pow(2).mean() * v_coef`

Line 257: total loss = sum of the three.
Line 258: `self.loss.backward()` — populates gradients on all parameters; the optimizer step is performed externally (likely in `models.py` after this call).
Lines 259 – 260: optional tensorboard logging.

### `forward`  (lines 263 – 282)

Used for inference (rollout collection), one step at a time.

Line 265: `ob = torch.from_numpy(np.expand_dims(ob, axis=0)).float()`. Input numpy `ob` is `[N=25, n_s=12]`; expanded to `[T=1, N=25, n_s=12]`. **Note:** unlike `backward`, this tensor is NOT moved to `self.device` here. It will be moved inside `_run_comm_layers` via `torch.as_tensor(..., device=self.device)` (line 435).
Line 267: `done` expand to `[1]`. (Single scalar wrapped.)
Line 269: `fp` to `[1, 25, 5]`.
Line 272: `h, new_states = self._run_comm_layers(ob, done, fp, self.states_fw)`.
Line 275 – 278: if asking for policy:
- Cache `new_states` (detach) for next forward step.
- Return `_run_actor_heads(h, detach=True)` — numpy probability arrays.
Lines 279 – 282: else (value):
- `action` numpy shape `[N]`; expanded to `[N, 1]`.
- Note: action tensor is built but `self.states_fw` is **NOT** updated in this branch. This is intentional — the value path is called immediately after the policy path with the same observation, so the state has already been advanced.
- Returns numpy value array per agent.

### `_get_comm_s`  (lines 288 – 327) — the heart of NeurComm

Called per agent per timestep from `_run_comm_layers`. Produces the LSTM input `s_i` for agent `i`.

Args: `i` (agent idx), `n_n` (# neighbors of `i`), `x` (`[N, n_s]` current obs), `h` (`[N, n_h]` previous hidden states of all agents), `p` (`[N, n_a]` current fingerprints).

Line 292: `js = torch.nonzero(self.neighbor_mask[i]).squeeze(1)` — neighbor indices of agent `i`. Shape `[n_n]`.
Line 298: `m_i = torch.index_select(h, 0, js).view(1, self.n_h * n_n)` — **message tensor**: concatenated hidden states of all neighbors, flattened. Shape `[1, n_h * n_n]`.

**Identical branch (lines 300 – 309):**
- Line 303: `x_i = x[i].unsqueeze(0)` — own observation, shape `[1, n_s]`.
- Line 306: `p_i = torch.index_select(p, 0, js).view(1, self.n_a * n_n)` — concatenated neighbor fingerprints, shape `[1, n_a * n_n]`.
- Line 309: `nx_i = torch.index_select(x, 0, js).view(1, self.n_s * n_n)` — concatenated neighbor observations, shape `[1, n_s * n_n]`.

**Non-identical branch (lines 310 – 318):**
- Line 311: own obs sliced to `self.n_s_ls[i]` (raw obs may be zero-padded to a uniform width — this strips the padding).
- Lines 314 – 316: per-neighbor slicing using `self.na_ls_ls[i][j]` and `self.ns_ls_ls[i][j]` (note: these are indexed `[i][j]`, where `j` is the local neighbor index within agent `i`'s neighborhood, not the global agent index — see `_init_net` and `_get_neighbor_dim`).
- Lines 317 – 318: `torch.cat(...)` of variable-length pieces, then `.unsqueeze(0)`.

**Three-stream aggregation (lines 322 – 326):**
- Line 322: `x_combined = torch.cat([x_i, nx_i], dim=1)`. Shape `[1, n_s + n_s*n_n] = [1, n_s*(n_n+1)]` (for the identical case).
- Line 323: `F.relu(self.fc_x_layers[i](x_combined))` — stream X: own + neighbor observations → `[1, n_fc]`.
- Line 324: `F.relu(self.fc_p_layers[i](p_i))` — stream P: neighbor policy fingerprints → `[1, n_fc]`.
- Line 325: `F.relu(self.fc_m_layers[i](m_i))` — stream M: neighbor hidden-state messages → `[1, n_fc]`.
- Line 327: `torch.cat(s_i, dim=1)` → `[1, 3*n_fc]` = `[1, 192]` for default `n_fc=64`.

**This is "NeurComm"**: the agent's LSTM input is the concatenation of three relu-activated affine transforms — one for observations (self+neighbors), one for neighbor policies, one for neighbor hidden-state messages. The three FCs act as type-aware encoders so that the LSTM doesn't have to disentangle modalities itself.

### `_get_neighbor_dim`  (lines 329 – 342)

Returns `(n_n, n_ns, n_na, ns_ls, na_ls)` for one agent.

- Line 331: `n_n = int(torch.sum(self.neighbor_mask[i_agent]).item())`.
- Identical (line 333): `n_ns = n_s * (n_n+1)` because `fc_x` consumes own obs + `n_n` neighbor obs concatenated. `n_na = n_a * n_n`. `ns_ls` and `na_ls` are just repeated lists used downstream by `_get_comm_s` non-identical branch (unused in identical case).
- Non-identical (lines 335 – 342): builds per-neighbor `ns_ls`/`na_ls` by indexing `self.n_s_ls`/`self.n_a_ls`. `n_ns = n_s_ls[i_agent] + sum(neighbor_ns)`, `n_na = sum(neighbor_na)`. Note: uses `np.where(mask)[0]` after `cpu().numpy()` so it works on GPU tensors.

### `_init_actor_head`  (lines 344 – 348)

Builds a single `nn.Linear(self.n_h, n_a)`, orthogonal init, appends to `self.actor_heads`. **The actor head consumes only the LSTM hidden state** — it does NOT see the raw observation or neighbor info directly; all that info is summarized in `h`.

### `_init_comm_layer`  (lines 350 – 381)

The 4-layer block per agent: `fc_x`, `fc_p`, `fc_m`, `lstm`.

Line 361: `n_lstm_in = 3 * self.n_fc`. **This is hardcoded to 3 streams.**
Lines 362 – 365: `fc_x_layer = nn.Linear(n_ns, self.n_fc)` — always appended.

If `n_n > 0` (line 366):
- Lines 368 – 369: `fc_p_layer = nn.Linear(n_na, self.n_fc)`.
- Lines 371 – 372: `fc_m_layer = nn.Linear(self.n_h * n_n, self.n_fc)`.
- Line 373: `self.fc_m_layers.append(fc_m_layer)`.
- Line 374: `self.fc_p_layers.append(fc_p_layer)`.
- Line 375: `lstm_layer = nn.LSTMCell(n_lstm_in, self.n_h)` (input = `3*n_fc`).

Else (no neighbors, lines 376 – 379):
- Append `None` placeholders for `fc_m` and `fc_p`.
- `lstm_layer = nn.LSTMCell(self.n_fc, self.n_h)` (input = `n_fc` only — just the obs stream).

Line 380 – 381: init the LSTM (orthogonal on both `weight_ih` and `weight_hh`, zero biases) and append.

**Subtle bug / asymmetry:** the order of `append` on lines 373 – 374 swaps `m` and `p` relative to the else branch (lines 377 – 378). The else branch appends to `fc_m_layers` BEFORE `fc_p_layers`. In the `if` branch line 373 appends `fc_m_layer` and line 374 appends `fc_p_layer` — so it's actually the same order (m first, then p). OK, no bug. Just unusual that the `if`-branch insertion order is `m → p` while the variables are declared `p → m` two lines earlier.

### `_init_critic_head`  (lines 383 – 390)

`critic_head = nn.Linear(self.n_h + n_na, 1)` — the critic sees `h_i` concatenated with **one-hot neighbor actions** (`n_na` = sum of neighbor action dimensions). For identical agents `n_na = n_a * n_n`. This is a centralized critic conditioned on neighbor actions only (not all agents) — consistent with the cooperative-MARL approach where each agent learns a local Q-value over its neighborhood.

### `_init_net`  (lines 392 – 410)

Lines 393 – 398: declare six `nn.ModuleList`s.
Lines 399 – 401: three Python lists `ns_ls_ls`, `na_ls_ls`, `n_n_ls` — these are not parameters; they're per-agent dim metadata used at runtime by `_get_comm_s`.

Per agent (lines 402 – 410):
1. `_get_neighbor_dim(i)` → `(n_n, n_ns, n_na, ns_ls, na_ls)`.
2. Cache `ns_ls`, `na_ls`, `n_n` in the meta lists (`self.ns_ls_ls[i]` is the list of neighbor obs-dims for agent i).
3. `_init_comm_layer(n_n, n_ns, n_na)` — appends to `fc_x_layers`, `fc_p_layers`, `fc_m_layers`, `lstm_layers`.
4. `n_a = self.n_a if self.identical else self.n_a_ls[i]`.
5. `_init_actor_head(n_a)` — agent-specific output dim.
6. `_init_critic_head(n_na)` — input width depends on summed neighbor action dims.

After this loop, every `ModuleList` has length `n_agent` (with `None` placeholders in `fc_p_layers`/`fc_m_layers` for isolated agents).

### `_reset`  (lines 412 – 414)

Both `states_fw` and `states_bw` initialized to `torch.zeros(n_agent, n_h * 2, device=self.device)` — the last dim is `2*n_h` because we pack `(h, c)` side by side and split with `torch.chunk(..., 2, dim=1)`.

### `_run_actor_heads`  (lines 416 – 427)

Iterates over agents.
- `detach=True` path (line 423): `F.softmax(...).squeeze().detach().cpu().numpy()` — used in `forward` for action sampling. Note `.squeeze()` here would drop a singleton batch dim (`T=1` case during rollout), returning a 1D numpy array of probabilities.
- `detach=False` path (line 425): `F.log_softmax(...)` — log-probabilities, used by `backward` to feed `Categorical(logits=...)`.

Returns a Python list of length `n_agent`. Mixing tensor / numpy types depending on `detach`.

### `_run_comm_layers`  (lines 433 – 485) — the temporal driver

Args:
- `obs`: `[T, N, n_s]`
- `dones`: `[T]` (or `[T, 1]` after `batch_to_seq`)
- `fps`: `[T, N, n_a]`
- `states`: `[N, 2*n_h]` packed `(h, c)`

Lines 435 – 437: ensure all are tensors on `self.device`.
Lines 441 – 445: `batch_to_seq(obs)` → tuple of `T` tensors each shape `[1, N, n_s]`. Similarly `dones` → tuple of `T` tensors `[1, 1]` (since 1-D input, `batch_to_seq` unsqueezes to 2D first at utils.py:24). And `fps` → tuple of `T` tensors `[1, N, n_a]`.
Line 448: `h, c = torch.chunk(states, 2, dim=1)` — split into `h` `[N, n_h]` and `c` `[N, n_h]`.

Per timestep `t` (lines 452 – 483):
- Line 455: `x = x.squeeze(0)` → `[N, n_s]`.
- Line 456: `p = p.squeeze(0)` → `[N, n_a]`.
- `done` is **not** squeezed; remains shape `[1, 1]` and broadcasts later.
- Per agent `i` (lines 458 – 480):
  - Line 459: `n_n = self.n_n_ls[i]`.
  - If `n_n > 0`: `s_i = self._get_comm_s(i, n_n, x, h, p)` → `[1, 3*n_fc]`.
  - Else (lines 463 – 472): isolated agent.
    - If identical, `x_i = x[i].unsqueeze(0)` → `[1, n_s]`.
    - Else, narrow to `n_s_ls[i]`. Line 468 references `self.unify_act_state_dim` — **this attribute does not exist on `NCMultiAgentPolicy`!** It only exists on `NCMultiAgentPolicy_MLP`. Hitting this branch on `NCMultiAgentPolicy` with `identical=False` and `n_n==0` would crash with `AttributeError`. See "Open Questions / Bugs".
    - Line 472: `s_i = F.relu(self.fc_x_layers[i](x_i))` → `[1, n_fc]`.
  - Lines 475 – 476: reset gating via `(1 - done)`. Since `done` is `[1, 1]`, this broadcasts across `[1, n_h]`. So a single shared `done` flag resets all agents simultaneously (this is appropriate when episode resets are global).
  - Line 478: `next_h_i, next_c_i = self.lstm_layers[i](s_i, (h_i, c_i))`. Note `s_i` is `[1, 3*n_fc]` (with neighbors) or `[1, n_fc]` (without) — must match the LSTM cell's `input_size` from `_init_comm_layer`.
  - Lines 479 – 480: stash per-agent outputs.
- Line 481: `h = torch.cat(next_h)` — concat `N` tensors of shape `[1, n_h]` → `[N, n_h]`.
- Line 482: same for `c`.
- Line 483: `outputs.append(h.unsqueeze(0))` — `[1, N, n_h]`.

After all timesteps:
- Line 484: `outputs = torch.cat(outputs)` → `[T, N, n_h]`.
- Line 485: `return outputs.transpose(0, 1), torch.cat([h, c], dim=1)`.
  - First return: `[N, T, n_h]` (transposed).
  - Second return: `[N, 2*n_h]` — the new packed state for next call.

### `_run_critic_heads`  (lines 487 – 514)

Args:
- `hs`: shape `[N, T, n_h]` from `_run_comm_layers`.
- `actions`: `[N, T]` (long) — the agents' actual actions over the rollout.

Per agent (lines 491 – 513):
- Line 492: `n_n`.
- If `n_n > 0`:
  - Line 495: `mask = self.neighbor_mask[i].cpu()`. **Subtle:** the `.cpu()` is unnecessary for `torch.nonzero` — `nonzero` works on GPU tensors. Probably leftover from `np.where` debugging.
  - Line 497: `js = torch.nonzero(mask).squeeze(1).to(self.device)`.
  - Line 499: `na_i = torch.index_select(actions, 0, js)` → shape `[n_n, T]`.
  - Lines 502 – 504: per neighbor `j`, one-hot encode `na_i[j]` (shape `[T]`) to width `self.na_ls_ls[i][j]` (the neighbor's action dim). Result shape `[T, na_dim]`.
  - Line 506: `h_i = torch.cat([hs[i]] + na_i_ls, dim=1)`. `hs[i]` is `[T, n_h]`; each one-hot is `[T, na_dim_j]`. Result: `[T, n_h + sum(na_dim_j)]` = `[T, n_h + n_na]`, matching the critic head's input size.
- Else: `h_i = hs[i]` — but the critic head was built with input size `n_h + n_na`. For an isolated agent `n_na = 0`, so input is just `n_h`. Consistent.
- Line 509: `v_i = self.critic_heads[i](h_i).squeeze()` → shape `[T]` after the final squeeze of dim-1.
- Lines 510 – 513: detach/numpy or tensor.

Returns list of length `N`.

### Methods listed in the prompt but NOT present

The prompt mentions `_convert_hetero_states`. **This method does not exist on `NCMultiAgentPolicy` or `NCMultiAgentPolicy_MLP`.** A grep across the entire `policies.py` returns no matches for `convert_hetero`. There is `convert_state_linears` / `convert_action_linears` on the `_MLP` variant (lines 543, 547), and a private `_create_linear` helper at line 559, but no method named `_convert_hetero_states`. Either the prompt is mistaken or this refers to a different class elsewhere in the codebase.

---

## class NCMultiAgentPolicy_MLP  (lines 516 – 870)

Sister class with one core addition: **per-agent MLPs that project heterogeneous obs/action dims to a common width**, enabling shared downstream layers even when agents differ.

### `__init__`  (lines 529 – 557)

New parameter: **`unify_act_state_dim=False`**.
Line 531: `super().__init__(...)` with `policy_name='nc_mlp'`.
Line 535: store the flag.
Lines 537 – 548: **only when `not self.identical`**:
- Line 542: `self.convert_state_shape = 16` — hardcoded target obs width.
- Line 543: `self.convert_state_linears = [self._create_linear(input_shape, 16) for input_shape in n_s_ls]` — one `nn.Linear` per agent mapping `n_s_ls[i] → 16`. **These are stored as a plain Python list, NOT `nn.ModuleList`.** Consequence: `self.parameters()` will NOT include these layers' weights. The optimizer won't update them. Bug — see Open Questions.
- Line 546: `self.convert_action_shape = max(n_a_ls)` — pad neighbor action representations to the maximum action dim across all agents.
- Line 547: `self.convert_action_linears` — same plain-list pattern. Same bug.
- Lines 544, 548, 562: noisy `print` statements showing the created layers.

Note: this block executes only if `not self.identical` AND `self.unify_act_state_dim`. If `identical=True` and `unify_act_state_dim=True`, the `convert_*` attributes are never created — code at lines 469 / 825 references them only behind a `self.unify_act_state_dim` check, but the references at lines 543/547 leave the attributes undefined when `identical=True`.

Lines 550 – 557: same as `NCMultiAgentPolicy.__init__` from there on.

### `_create_linear`  (lines 559 – 563)

Helper that builds `nn.Linear(in, out).to(self.device)` and prints what it created. Used only by the convert linears in `__init__`.

### `backward`  (lines 569 – 597)

Identical structure to `NCMultiAgentPolicy.backward` with **two differences**:
1. **No advantage normalization step** — there is no equivalent of line 246. `Advs` is fed raw into `_run_loss`.
2. Internally uses the MLP variant's `_run_comm_layers` and `_get_comm_s`, which apply `convert_state_linears` / `convert_action_linears` (only in the non-identical + unify branch).

Otherwise identical: per-agent loss accumulation, `.backward()`, tensorboard.

### `forward`  (lines 600 – 619)

Byte-for-byte identical to `NCMultiAgentPolicy.forward`. The differences are all inside `_run_comm_layers` / `_get_comm_s`.

### `_get_comm_s`  (lines 625 – 673)

Differs from the base class only in the **non-identical** branch (lines 647 – 662):

- Line 648: own obs narrowed to `n_s_ls[i]`.
- Line 649: **`x_i = self.convert_state_linears[i](x_i)`** — own obs projected to `convert_state_shape=16`. **This happens unconditionally in the non-identical branch — `self.unify_act_state_dim` is NOT checked here, yet `convert_state_linears` only exists when `unify_act_state_dim=True`.** This is an inconsistency with the neighbor branch (lines 655 – 660), which DOES gate on `unify_act_state_dim`. See Open Questions.
- Lines 650 – 660: per neighbor `j`:
  - `p_i_temp` / `nx_i_temp` narrowed to that neighbor's true dim.
  - If `unify_act_state_dim`: apply `convert_action_linears[js[j]]` and `convert_state_linears[js[j]]` — projecting EACH neighbor's obs/action through that NEIGHBOR'S own conversion layer (uses global agent index `js[j]`, NOT the local `j`). This makes sense — each agent has its own size, so we look up its dedicated converter.
  - Else: keep raw dims (variable per neighbor).

Lines 661 – 662: `cat` and `unsqueeze`.

Lines 669 – 672: same three-stream concat as the base class. **No behavior difference from the base class — the difference is purely in input preprocessing.**

### `_get_neighbor_dim`  (lines 675 – 688)

Same as base for identical case.

Non-identical case (line 688): `return n_n, self.convert_state_shape * (n_n+1), sum(na_ls), ns_ls, na_ls`.
**Key change:** `n_ns = convert_state_shape * (n_n+1) = 16 * (n_n+1)` — because every obs (own + neighbors) has been projected to width 16 before being concatenated and fed into `fc_x_layer`.

Note: this is only correct if all obs are actually projected, i.e. `unify_act_state_dim=True`. If `unify_act_state_dim=False`, `fc_x_layer`'s declared input is wrong relative to the actual cat'd tensor. See Open Questions.

### `_init_actor_head`  (lines 690 – 695)

Adds a print. Otherwise identical.

### `_init_comm_layer`  (lines 697 – 733)

Signature renames the third arg to `convert_n_a` (computed in `_init_net` as `convert_action_shape * n_n`). The body is otherwise identical to the base class: same `3*n_fc` LSTM input, same `n_h*n_n` for `fc_m`. Most prints commented out (lines 711, 717, 721, 726, 731).

### `_init_critic_head`  (lines 735 – 743)

Same as base, plus a print.

### `_init_net`  (lines 745 – 766)

Same loop, with one critical change: **line 761** computes `convert_n_a = self.convert_action_shape * n_n` and passes that as the third arg to `_init_comm_layer`. This means `fc_p_layer`'s input width is `max(n_a_ls) * n_n` — sized for the projected, unified-width neighbor policies.

**Implication:** If `identical=True`, `self.convert_action_shape` is undefined (never set in `__init__`). Calling this with `identical=True` would crash at line 761. The `_MLP` class is only usable with `identical=False` AND `unify_act_state_dim=True` — matching the gate in `models.py:389` `if not self.identical_agent and self.unify_act_state_dim`.

### `_reset`, `_run_actor_heads`, `_run_comm_layers`, `_run_critic_heads`  (lines 768 – 870)

Functionally identical to the base class. `_run_comm_layers` lines 824 – 825 contains the conditional projection `self.convert_state_linears[i](x_i_temp)` for the isolated-agent + non-identical + unify path — this is now reachable (unlike the same code in the base class) because the attribute exists.

### When is `_MLP` used vs the base class?

From `models.py` (line 388 – 393):
```
self.unify_act_state_dim = self.model_config.getboolean('unify_act_state_dim', False)
if not self.identical_agent and self.unify_act_state_dim:
    return NCMultiAgentPolicy_MLP(**policy_params)
else:
    return NCMultiAgentPolicy(**policy_params)
```

**Rule:** `_MLP` is selected only when agents have heterogeneous state/action dims AND the config explicitly opts in (`unify_act_state_dim=True`). For all other configurations (identical agents, or heterogeneous-but-no-unification), the base class is used and zero-padding handles the dimension mismatch.

---

## Tensor shape table (default `n_s=12, n_a=5, n_h=64, n_fc=64, N=25, T=120, n_n_i=2`)

### `backward` path

| Variable | Shape | Notes |
|---|---|---|
| input `obs` (numpy) | `(N, T, n_s) = (25, 120, 12)` | from rollout buffer |
| `obs` after `.transpose(0,1)` | `(T, N, n_s) = (120, 25, 12)` | line 230 |
| `dones` | `(T,) = (120,)` | line 231 |
| `fps` | `(T, N, n_a) = (120, 25, 5)` | line 232 |
| `acts` | `(N, T) = (25, 120)` | NOT transposed; line 233 |
| inside `_run_comm_layers` after `batch_to_seq(obs)` | tuple of `T` tensors, each `(1, N, n_s) = (1, 25, 12)` | line 441 |
| `h, c` after `chunk(states, 2)` | each `(N, n_h) = (25, 64)` | line 448 |
| Per-step `x` (after `.squeeze(0)`) | `(N, n_s) = (25, 12)` | line 455 |
| Per-step `p` (after `.squeeze(0)`) | `(N, n_a) = (25, 5)` | line 456 |
| Per-step `done` (NOT squeezed) | `(1, 1)` | broadcasts in line 475 |
| `js` (neighbor indices of agent `i`) | `(n_n_i,) = (2,)` | line 292 |
| `m_i` (neighbor hidden states flat) | `(1, n_h * n_n_i) = (1, 128)` | line 298 |
| `x_i` (own obs) | `(1, n_s) = (1, 12)` | line 303 |
| `p_i` (neighbor fingerprints flat) | `(1, n_a * n_n_i) = (1, 10)` | line 306 |
| `nx_i` (neighbor obs flat) | `(1, n_s * n_n_i) = (1, 24)` | line 309 |
| `x_combined = cat([x_i, nx_i])` | `(1, n_s*(n_n_i+1)) = (1, 36)` | line 322 |
| After `fc_x_layers[i]` | `(1, n_fc) = (1, 64)` | line 323 |
| After `fc_p_layers[i]` | `(1, n_fc) = (1, 64)` | line 324 |
| After `fc_m_layers[i]` | `(1, n_fc) = (1, 64)` | line 325 |
| `s_i` = `cat([...3 streams...])` | `(1, 3*n_fc) = (1, 192)` | line 327 |
| LSTM cell output `next_h_i`, `next_c_i` | each `(1, n_h) = (1, 64)` | line 478 |
| Per-step `h` after `cat(next_h)` | `(N, n_h) = (25, 64)` | line 481 |
| Per-step entry of `outputs` | `(1, N, n_h) = (1, 25, 64)` | line 483 |
| `outputs` after final `cat` | `(T, N, n_h) = (120, 25, 64)` | line 484 |
| Returned `hs` (after transpose) | `(N, T, n_h) = (25, 120, 64)` | line 485 |
| Returned `new_states` | `(N, 2*n_h) = (25, 128)` | line 485 |
| `ps[i]` (log_softmax) | `(T, n_a) = (120, 5)` | line 425 |
| `na_i_ls[j]` (one-hot neighbor action) | `(T, n_a) = (120, 5)` | line 504 |
| `h_i` (critic input) | `(T, n_h + n_a*n_n_i) = (120, 64+10) = (120, 74)` | line 506 |
| `v_i` after critic + squeeze | `(T,) = (120,)` | line 509 |

### `forward` path (single timestep, T=1)

| Variable | Shape |
|---|---|
| `ob` (after expand_dims) | `(1, N, n_s) = (1, 25, 12)` |
| `done` | `(1,)` |
| `fp` | `(1, N, n_a) = (1, 25, 5)` |
| Returned `h` | `(N, T=1, n_h) = (25, 1, 64)` |
| `new_states` | `(N, 2*n_h) = (25, 128)` |
| Each `ps[i]` (softmax, squeezed) | `(n_a,) = (5,)` — note the `.squeeze()` collapses the T=1 dim |

---

## Cross-references

The classes consumed elsewhere in the project:

- **[[MA2C_NC]]** (`models.py:292`) — the canonical "Multi-Agent A2C with NeurComm" model. Its `_init_policy` (line 354) returns either `NCMultiAgentPolicy_MLP` (heterogeneous + unify) or `NCMultiAgentPolicy` (default), gated by the `unify_act_state_dim` config flag.
- **[[IA2C_CU]]** (`models.py:398`) — Consensus Update variant. Subclasses `MA2C_NC` but overrides `_init_policy` (line 405) to return `ConsensusPolicy`, which itself subclasses `NCMultiAgentPolicy` (`policies.py:1686`).
- **[[BayesianGraph]]** (`models.py:418`) — extends `MA2C_NC`; uses `BayesianGraphCMultiAgentPolicy` (a graph-NN variant, not the plain NC).
- **[[MA2C_DIAL]]** (`models.py:449`) — uses `DIALMultiAgentPolicy` (`policies.py:2004`), which subclasses `NCMultiAgentPolicy`.
- **[[MA2C_CNET]]** (`models.py:465`) — uses `CommNetMultiAgentPolicy` (`policies.py:1840`), also a subclass of `NCMultiAgentPolicy`.
- **[[GraphCMultiAgentPolicy]]** (`policies.py:872`) — **direct subclass of `NCMultiAgentPolicy`**, overrides communication layers with GNNs.

Inheritance tree rooted at NC:

```
NCMultiAgentPolicy
 ├── GraphCMultiAgentPolicy           (policies.py:872)
 ├── ConsensusPolicy                  (policies.py:1686)
 ├── CommNetMultiAgentPolicy          (policies.py:1840)
 └── DIALMultiAgentPolicy             (policies.py:2004)

NCMultiAgentPolicy_MLP   (sibling, NOT a subclass)
```

See also [[01-base-Policy-class]] for the parent `Policy` (lines 15 – 102), which defines `_run_loss` and `_update_tensorboard` consumed here. See [[utils-helpers]] for `batch_to_seq`, `init_layer`, `one_hot`, `run_rnn`.

---

## Open questions / bugs

1. **`unify_act_state_dim` referenced on the base class.** Line 468 (`NCMultiAgentPolicy._run_comm_layers`) reads `if self.unify_act_state_dim:` but `NCMultiAgentPolicy.__init__` never sets this attribute. This branch is only reachable when `n_n == 0` AND `not self.identical`. An isolated heterogeneous agent will crash with `AttributeError`. The code looks like a copy-paste from the `_MLP` version that was incompletely scrubbed.

2. **`convert_state_linears` / `convert_action_linears` are plain Python lists, not `nn.ModuleList`.** Lines 543 and 547 in `NCMultiAgentPolicy_MLP.__init__`. PyTorch's `Module.parameters()` only walks `nn.Module`-tracked attributes (including `ModuleList`); a plain `list` of `nn.Linear`s is invisible. Therefore:
   - The optimizer will NOT update these layers.
   - `state_dict()` / `load_state_dict()` will NOT serialize them.
   - These linear projections are effectively frozen at their initial random init.
   This is almost certainly a bug. The fix is `nn.ModuleList(...)` on both lines.

3. **Asymmetric `unify_act_state_dim` gating in `_MLP._get_comm_s`.** Line 649 applies `convert_state_linears[i]` to own obs **unconditionally** in the non-identical branch, while lines 655 – 660 apply the conversion only when `unify_act_state_dim=True`. If a user constructs `NCMultiAgentPolicy_MLP` with `identical=False, unify_act_state_dim=False`, the attribute `convert_state_linears` does not exist (lines 540 – 543 only run if `unify_act_state_dim`), so line 649 crashes. Mitigated in practice by `models.py:389` only selecting `_MLP` when both flags are true — but the class is locally unsafe.

4. **Advantage normalization only in base `backward`, not `_MLP.backward`.** Line 246 normalizes advantages per minibatch; the `_MLP` equivalent (line 584) has no such step. Whether this is intentional differential behavior or a forgotten port is unclear.

5. **`self.neighbor_mask` device.** Stored on whatever device the caller passes; not explicitly moved to `self.device` in `__init__`. `torch.nonzero(self.neighbor_mask[i])` works on either device, but mixing devices later (e.g., line 497 does `.cpu()` then `.to(self.device)`) suggests the original code was unsure where the mask lives. Worth pinning to `self.device` once at construction.

6. **`done` shape in per-step loop.** `done` retains shape `(1, 1)` because lines 455 – 456 only squeeze `x` and `p`, not `done`. The multiplication `h[i].unsqueeze(0) * (1-done)` then broadcasts `(1, n_h) * (1, 1) → (1, n_h)`. Functional, but a single global done flag means the LSTM state of EVERY agent resets together — there is no per-agent done. For environments where agents finish independently this would be incorrect (but the project's environments appear to use synchronous resets).

7. **`_run_actor_heads` returns mixed types.** With `detach=True` returns numpy arrays; with `detach=False` returns tensors. Callers must know which mode they're in. Minor design wart.

8. **The prompt mentions `_convert_hetero_states`.** No such method exists on either class (`grep "convert_hetero" /tmp/BayesG/agents/policies.py` returns nothing). Possibly confused with `convert_state_linears` / `convert_action_linears`, or with a method on a different class.

9. **`fc_p_layers` / `fc_m_layers` may hold `None`.** When an agent has zero neighbors, lines 377 – 378 append `None`. Any subsequent code that iterates over these `ModuleList`s without checking `is not None` will crash. In the current `_run_comm_layers`, the `n_n == 0` branch (line 463) avoids calling these — but downstream subclasses must respect this contract.

10. **`actor_heads` always uses `nn.Linear(n_h, n_a)`.** The actor never sees neighbor info directly — it relies entirely on the LSTM hidden state to encode neighborhood context. Whether this is optimal or whether a critic-style concat of neighbor actions/fingerprints would help is an open empirical question for the project.
