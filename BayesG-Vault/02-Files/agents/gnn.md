# gnn.py — Hand-rolled GNN Layers

File: `/tmp/BayesG/agents/gnn.py` (123 lines)

Three custom GNN layer implementations: [[GATLayer]], [[GCNLayer]], [[SAGELayer]]. All work on dense adjacency matrices (no edge-index / sparse format). Imports: `torch`, `torch.nn`, `torch.nn.functional`.

Related: [[models]], [[policies]].

---

## GATLayer

Multi-head Graph Attention layer using **dense** adjacency masking.

### `__init__` (lines 5–19)

```python
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, n_heads, dropout=0.1, alpha=0.2):
```

- **Line 7**: `super(GATLayer, self).__init__()` — initialize `nn.Module`.
- **Lines 8–11**: store hyperparameters `in_features`, `out_features`, `n_heads`, `dropout` as attributes.
- **Line 14**: `self.W = nn.Parameter(torch.empty(in_features, out_features * n_heads))` — a single fused projection matrix that produces all heads' outputs at once. Shape `[F_in, F_out * H]`.
- **Line 15**: `self.a = nn.Parameter(torch.empty(2 * out_features, 1))` — attention vector of shape `[2 * F_out, 1]`. Note it is sized for **one head only** (not `2 * F_out * n_heads`). This is suspicious; see [[#Bugs / oddities]].
- **Lines 16–17**: Xavier-uniform init for `W` and `a`.
- **Line 19**: `self.leakyrelu = nn.LeakyReLU(alpha)` — alpha=0.2 (the GAT default).

### `forward(self, x, adj)` (lines 21–62)

Inputs: `x` of shape `[N, F_in]`, `adj` of shape `[N, N]` (dense, presumably `{0,1}` or non-negative weights).

- **Line 23**: `h = torch.mm(x, self.W).view(x.size(0), self.n_heads, self.out_features)` — project all heads in one matmul then reshape to `[N, H, F_out]`. Example with `N=3, H=4, F_out=64`: shape becomes `[3, 4, 64]`.

#### Broadcasting for pairwise attention

- **Line 26**: `h_i = h.unsqueeze(1)` → `[N, 1, H, F_out]` = `[3, 1, 4, 64]` (source dim is axis 0, target slot inserted at axis 1).
- **Line 27**: `h_j = h.unsqueeze(0)` → `[1, N, H, F_out]` = `[1, 3, 4, 64]`.
- **Line 30**: `h_i = h_i.expand(-1, h.size(0), -1, -1)` → `[N, N, H, F_out]` = `[3, 3, 4, 64]`.
- **Line 31**: `h_j = h_j.expand(h.size(0), -1, -1, -1)` → `[N, N, H, F_out]` = `[3, 3, 4, 64]`.
- **Line 34**: `h_j = h_j.transpose(-1, -2)` → `[N, N, F_out, H]` = `[3, 3, 64, 4]`.
- **Line 37**: `e = self.leakyrelu(torch.matmul(h_i, h_j))` — `[3, 3, 4, 64] @ [3, 3, 64, 4]` → `[N, N, H, H]` = `[3, 3, 4, 4]`.

> Notice: this **never actually uses `self.a`**! The attention vector `a` initialized on line 15 is dead weight. Instead, attention is computed as `h_i @ h_j^T` (a dot-product attention over feature dimension), then LeakyReLU'd. This is closer to a Transformer/Luong-style attention than the original GAT formulation (`a^T [Wh_i || Wh_j]`).

- **Line 40**: `e = e.mean(dim=-1)` → `[N, N, H]` = `[3, 3, 4]`.
- **Lines 43–47**: masked-fill softmax setup. `adj.unsqueeze(2)` is `[N, N, 1]`, broadcast against `e` of shape `[N, N, H]`. Where `adj>0`, keep `e`; else `-inf`. This is the standard dense-GAT trick.
- **Lines 50–54**: softmax over `dim=1` (over source nodes for each target). Then dropout (only during `training`).
- **Lines 57–60**: aggregate.
  - `attention.permute(2, 0, 1)` → `[H, N, N]`.
  - `h.permute(1, 0, 2)` → `[H, N, F_out]`.
  - `torch.bmm(...)` → `[H, N, F_out]` (per-head weighted sum of neighbor features).
  - `.permute(1, 0, 2)` → `[N, H, F_out]`.
  - `.mean(dim=1)` → `[N, F_out]` — averages heads (rather than concatenating, which is the canonical GAT choice for intermediate layers).
- **Line 62**: `return F.elu(out)` — final activation `[N, F_out]`.

---

## GCNLayer

Standard Kipf & Welling GCN: `H' = ReLU(D^(-1/2) A D^(-1/2) X W)`.

### `__init__` (lines 65–78)

```python
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
```

- **Lines 68–69**: store dims.
- **Line 72**: `self.linear = nn.Linear(in_features, out_features)` — the learnable `W` (+ bias).
- **Line 73**: `self.reset_parameters()`.
- **Lines 75–78**: Xavier-uniform on weight, zero on bias.

### `forward(self, x, adj)` (lines 80–91)

Symmetric normalization `D^(-1/2) A D^(-1/2)`:

- **Line 82**: `deg = torch.sum(adj, dim=1)` — row sums = node degrees, shape `[N]`. Note: `adj` is **not** augmented with self-loops here (`A_tilde = A + I` is missing from canonical GCN).
- **Line 83**: `deg = torch.clamp(deg, min=1)` — avoid divide-by-zero for isolated nodes.
- **Line 84**: `deg_inv_sqrt = deg.pow(-0.5)` → `D^(-1/2)`, shape `[N]`.
- **Line 85**: `deg_inv_sqrt_mat = torch.diag(deg_inv_sqrt)` → dense `[N, N]` diagonal matrix. Inefficient for large N (should use elementwise multiply), but readable.
- **Line 86**: `norm_adj = torch.mm(torch.mm(deg_inv_sqrt_mat, adj), deg_inv_sqrt_mat)` — `D^(-1/2) A D^(-1/2)`, shape `[N, N]`.
- **Line 89**: `support = self.linear(x)` → `X W + b`, shape `[N, F_out]`.
- **Line 90**: `output = torch.mm(norm_adj, support)` → propagate, shape `[N, F_out]`.
- **Line 91**: `return F.relu(output)`.

---

## SAGELayer

GraphSAGE with **mean aggregator**: `h_v' = ReLU(W_self h_v + W_neigh mean(h_u for u in N(v)))`.

### `__init__` (lines 93–110)

```python
class SAGELayer(nn.Module):
    def __init__(self, in_features, out_features):
```

- **Lines 96–97**: store dims.
- **Line 100**: `self.linear_self = nn.Linear(in_features, out_features)` — `W_self`.
- **Line 101**: `self.linear_neigh = nn.Linear(in_features, out_features)` — `W_neigh`.
- **Line 102**: `self.reset_parameters()`.
- **Lines 104–110**: Xavier-uniform on both weights; zero on both biases.

### `forward(self, x, adj)` (lines 112–122)

- **Line 114**: `neigh_mean = torch.mm(adj, x) / (torch.sum(adj, dim=1, keepdim=True) + 1e-6)` — sum neighbor features then divide by degree, giving mean. The `+1e-6` guards against isolated nodes. Shape `[N, F_in]`.
- **Line 117**: `from_self = self.linear_self(x)` → `[N, F_out]`.
- **Line 118**: `from_neighs = self.linear_neigh(neigh_mean)` → `[N, F_out]`.
- **Line 121**: `output = F.relu(from_self + from_neighs)` — **additive** combination (the original GraphSAGE paper concatenates then projects; this is the simpler "additive" variant).
- **Line 122**: `return output`.

---

## Differences from `torch_geometric` equivalents

| Concern | This file | `torch_geometric` |
|---|---|---|
| Adjacency format | Dense `[N, N]` tensor | Sparse `edge_index [2, E]` |
| GAT attention | `h_i @ h_j^T` (dot product), `self.a` never used | `LeakyReLU(a^T [Wh_i || Wh_j])` (the GAT paper) |
| GAT head combine | Mean over heads | Concat (default) or mean (final layer) |
| GCN self-loops | Not added (raw `A` only) | `A + I` injected by default |
| GCN normalization | Built from scratch each forward | Precomputed / cached in `GCNConv` |
| SAGE combine | Additive (`W_s h + W_n m`) | Concat `[h ‖ m]` then linear in the canonical paper variant |
| Sparse efficiency | O(N²) memory and compute | O(E) via scatter |
| Numerical guards | `clamp(min=1)` for GCN deg, `+1e-6` for SAGE | Same idea, handled internally |

These layers are pedagogical / "from scratch" reimplementations, fine for the small grids in this project but unsuitable for large graphs.

---

## Bugs / oddities

1. **`self.a` is never used** ([[GATLayer]] lines 15–17 declare it, but `forward` never references it). Dead parameter — wastes init and shows up in `.parameters()` (so it gets gradient updates from any regularizer but no signal from the loss). This means the "GAT" is not actually a GAT — it's a multi-head dot-product attention layer.

2. **`e.mean(dim=-1)` averages over output features, not heads** (line 40). After `torch.matmul(h_i, h_j)` the shape is `[N, N, H, H]`. The author's intent (from the comment "Average over the last dimension to get [3, 3, 4]") is to collapse to a per-head score `[N, N, H]`. But the last dim of `[N, N, H, H]` is also a head dim (specifically, head index from `h_j`). So this reduces `(i, j, h_i, h_j) → mean over h_j`, which is structurally wrong:
   - Each "head's" attention logit ends up as a sum of dot products against **all** heads of the neighbor, rather than each head attending in isolation.
   - This breaks the multi-head independence that is the entire point of GAT's parallel heads.
   - A correct formulation here would use `h_i * h_j` elementwise (per-head) then sum over `F_out`, e.g. `e = (h_i * h_j_pre_transpose).sum(dim=-1)` to get `[N, N, H]` directly. Flag this.

3. **GCN missing self-loops**: line 82 sums `adj` without `+ I`. Canonical Kipf-GCN uses `A_tilde = A + I_N`. Without it, a node's own previous representation does not propagate through `norm_adj @ support`; it only enters via the bias of `self.linear`. For graphs with self-loops already baked into `adj`, fine; otherwise this hurts.

4. **GCN uses a dense `torch.diag` matmul** (lines 85–86): O(N³) and O(N²) memory for what could be `norm_adj = deg_inv_sqrt[:, None] * adj * deg_inv_sqrt[None, :]` (an elementwise broadcast).

5. **GAT head combine uses mean rather than concat** (line 60 `.mean(dim=1)`). This silently halves the effective output dimensionality interpretation vs. canonical GAT and may degrade expressiveness; if the caller expects `out_features` to mean per-head, the mean is fine; if it expects concat behavior, this is a footgun.

6. **GAT softmax `dim=1`**: softmaxing over neighbors (axis 1, the source-node axis of `adj`). This is correct **assuming** the convention that `adj[i, j]` means "j → i" (j is a neighbor of i, normalize over j for fixed i). If the project's adjacency convention is the opposite, attention is normalized along the wrong axis.
