# Large_city.py — Line-by-Line Walkthrough

File: `/tmp/BayesG/envs/Large_city.py` (630 lines)

Environment for the **NewYork33 / NewYork51 / NewYork167** scenarios (also referred to as "Large city" or `Large_city`). This file is the odd one out in [[atsc_env]] — it does **not** inherit from `TrafficSimulator` (see [[atsc_env]]), instead it subclasses `gym.Wrapper` directly and re-implements most of the SUMO/traffic plumbing inline.

The companion factory `Large_city_Env(...)` at the bottom returns a `Large_city_net` instance — callers in [[main]] / agent code therefore see a `Large_city_net` object behind the `Large_city_Env` name.

---

## Imports

Lines 1–42:

```python
from collections import deque

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
parent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
import sys
import optparse
import random
import sumolib
import numpy as np
import gym
from gym.spaces import Box, Discrete
import configparser
import os
import pdb
# we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa
from copy import deepcopy as dp


import json
```

Notes:
- `deque` is used by the BFS in `get_neighbor_matrix_khop` and friends.
- `current_dir` / `parent_dir` are computed at import time but never used (a leftover from the commented-out path-prefixing logic in `Large_city_Env`).
- `os` is imported twice (lines 15 and 27). `optparse`, `random`, `configparser`, `pdb`, `Box` are imported but never used.
- `dp` is `copy.deepcopy` (used once to clone `neighbor_mask` into `distance_mask`).
- `SUMO_HOME` must be set or the module hard-`sys.exit`s at import — this means even *importing* `Large_city` will kill the process if the env var is missing.
- `traci` is imported as a module here (not via a `TrafficSimulator` wrapper as in [[atsc_env]]).

---

## class `Large_city_net` (gym.Wrapper — NOT TrafficSimulator-based)

Line 44:

```python
class Large_city_net(gym.Wrapper):
```

Unlike every other env in this repo (e.g. [[real_net_env]], [[large_grid_env]], [[atsc_env]]), `Large_city_net` does **not** inherit from `TrafficSimulator` / `PhaseMap` / `PhaseSet` infrastructure. It inherits from `gym.Wrapper` — but note that `super().__init__()` is never called, and no `self.env` is wrapped, so the `gym.Wrapper` semantics are nominal only. In practice this class is a standalone object with hand-rolled `step` / `reset` / `get_state` / `get_reward` methods that talk to `traci` directly.

---

## `__init__` (lines 45–161)

Signature:

```python
def __init__(self, net_path, sim_path, folder_name=None, reward_scale=10):
```

Parameters:
- `net_path` — absolute path to the SUMO `.net.xml` file (e.g. `NewYork167/newyork167.net.xml`).
- `sim_path` — absolute path to the SUMO `.sumocfg`.
- `folder_name` — accepted but never used inside the class (stored on `self` only).
- `reward_scale` — divisor applied to the per-agent reward in `get_reward()`. Default `10`.

### Static configuration (lines 46–60)

```python
self.name = "Large_city"
self.obj = "queue"     # queue   wait  arrived
self.folder_name = folder_name
self.net_path = net_path
self.sim_path = sim_path
self.net = sumolib.net.readNet(self.net_path)
self.AdjacencyList = self.generateTopology()    # 获取邻接矩阵
print("self.AdjacencyList=", self.AdjacencyList)
self.Edges = self.net.getEdges()
self.VEH_LEN_M = 200
self.coop_gamma = -1
self.T = 500
# self.max_steps = self.T
# self.cur_step = 0
self.reward_scale = reward_scale
```

Key constants:
- `self.obj = "queue"` — selects the reward branch in `get_reward`. Alternatives commented in the source are `"wait"` and `"arrived"`.
- `self.VEH_LEN_M = 200` — used as the divisor for lane capacity normalization in the state (`cur_cap / VEH_LEN_M`). The unusual large value (200 m) yields very small "capacity in vehicle-equivalents" numbers — this looks like an intentional rescaling so that `getLastStepVehicleNumber(lane)/CAP` lies in a useful range, but it is also possibly a bug (`VEH_LEN_M` semantically should be ~5 m per vehicle).
- `self.coop_gamma = -1` — sentinel indicating "no cooperative discount" (consistent with [[atsc_env]] where `coop_gamma < 0` disables spatial reward weighting).
- `self.T = 500` — total episode horizon (steps). But note: `step()` never increments a step counter and never sets `done = True` based on `T`, so this is unused at runtime; the episode length is enforced externally (likely by the SUMO `.sumocfg` end-time).

### SUMO startup (lines 65–71)

```python
self.gui = False
if self.gui:
    self.sumoBinary = checkBinary('sumo-gui')
else:
    self.sumoBinary = checkBinary('sumo')
self.traci = traci
self.traci.start([self.sumoBinary, "-c", sim_path, "--no-warnings"])
```

**No port is specified.** `traci.start([...])` launches SUMO as a subprocess and connects on a default port (chosen by traci). This means:
- Only **one** instance of `Large_city_net` can exist in the same Python process at a time (no parallel envs).
- There is no analogue to the `port=` argument seen in [[atsc_env]] / [[TrafficSimulator]].
- The simulator is started inside `__init__` purely to query metadata (number of TLS, phases, controlled lanes, lane lengths), then `self.traci.close()` is called at line 160. The simulator is restarted in `reset()` (line 523).

### Inferring agent count from the net (lines 75–94)

```python
self.n_agent = self.traci.trafficlight.getIDCount()
self.n_agents = self.n_agent
if self.n_agent < 500:
    self.single_step_second = 20
else:
    self.single_step_second = 40
print("n_agent= ", self.n_agent)
self.id_list = list(self.traci.trafficlight.getIDList())
print("id_list= ", self.id_list)

self.A = []
for i in range(self.n_agent):
    phase = self.traci.trafficlight.getAllProgramLogics(self.id_list[i])
    a = len(phase[0].phases)
    self.A.append(a)

self.n_action = max(self.A)
self.action_space = Discrete(self.n_action)
```

- `n_agent` is **inferred from the SUMO net** by counting traffic-light objects — no hard-coded constant. For NewYork33 this yields 33, for NewYork51 → 51, for NewYork167 → 167 (and the `__main__` block at the bottom references "436 agents" in a stale comment).
- Both `self.n_agent` and `self.n_agents` are set (defensive against agent code that uses either name).
- `single_step_second` (the number of SUMO seconds per RL step) is `20` for nets with < 500 agents and `40` otherwise. So all three NewYork variants use a 20-second control interval.
- `self.A` is a per-agent list of phase counts (heterogeneous — each TLS has its own number of phases).
- `self.n_action = max(self.A)` is the **shared** discrete action-space size, padded up to the largest TLS's phase count. Agents with fewer phases handle "out-of-range" actions via modulo wrap in `step()`. The `gym.Discrete(self.n_action)` action space is therefore the same across all agents (the per-agent variability is hidden via the wrap-around).

### Neighbor mask (line 97)

```python
self.neighbor_mask = self.get_neighbor_matrix_khop(k=1)
```

A **1-hop BFS** along the network topology — see the dedicated section below.

### Graph stats printing (lines 99–116)

```python
adj = self.neighbor_mask
num_nodes = adj.shape[0]
adj_no_self = adj.copy()
np.fill_diagonal(adj_no_self, 1)   # <-- ODDITY: forces diagonal to 1 even when it might be 0
num_edges = int(np.sum(adj_no_self))
degrees = np.sum(adj_no_self, axis=1)
avg_degree = np.mean(degrees)
max_degree = np.max(degrees)
print(f"Graph stats: nodes={num_nodes}, edges={num_edges}, avg_degree={avg_degree:.2f}, max_degree={max_degree}")
print("self.neighbor_mask=", self.neighbor_mask.sum(1))
print("Total TLS:", len(self.id_list))
print("Connected TLS nodes:")
```

**Oddity**: the comment says `# Remove self-loops for edge counting`, but the code does `np.fill_diagonal(adj_no_self, 1)` — this *sets* the diagonal to 1, the opposite of removing self-loops. So `num_edges` includes `N` self-loops. (Almost certainly intended to be `np.fill_diagonal(adj_no_self, 0)`.)

### Distance / edge matrices (lines 123–124)

```python
self.distance_mask = dp(self.neighbor_mask)
self.E_id, self.E_from, self.E_to, self.E_to_state, self.E_from_state = self.get_edge_matrix()
```

- `self.distance_mask` is just an alias for the neighbor mask — not a real distance matrix. (In [[atsc_env]] / [[real_net_env]] this is a multi-hop hop-count matrix.)
- Edge matrix builds `E_to_state[i]` / `E_from_state[i]` — lists of edge IDs that terminate at / originate from agent `i`'s junction (see "Edge matrix pre-computation" section).

### Per-agent observation construction (lines 126–148)

```python
self.node_names = self.id_list
self.ILDS_in = []
self.CAP = []
ss = []
for node_name in self.node_names:
    lanes_in = self.traci.trafficlight.getControlledLanes(node_name)
    ilds_in = []
    lanes_cap = []
    for lane_name in lanes_in:
        cur_ilds_in = [lane_name]
        ilds_in.append(cur_ilds_in)
        cur_cap = 0
        for ild_name in cur_ilds_in:
            cur_cap += self.traci.lane.getLength(ild_name)
        lanes_cap.append(cur_cap / float(self.VEH_LEN_M))
    ss.append(len(lanes_cap))
    self.ILDS_in.append(ilds_in)
    self.CAP.append(lanes_cap)
self.n_s = max(ss)
print("ss=", ss)
self.n_s_ls = [self.n_s] * self.n_agent
self.n_a_ls = [self.n_action] * self.n_agent
print("state_space=", self.n_s)
self.fp = np.ones((self.n_agent, self.n_action)) / self.n_action
```

- For each TLS, `getControlledLanes` returns the list of incoming lane IDs (with duplicates — SUMO returns one entry per signal-link, so a single physical lane controlled by multiple links appears multiple times). Each lane is wrapped in a singleton list `[lane_name]` (`cur_ilds_in`) for compatibility with the "induction-loop detector list" pattern in [[atsc_env]] — but here there is exactly one entry per "ILD".
- `lanes_cap[j] = lane_length / 200` — this is the normalization divisor for state computation.
- `ss[i]` = number of controlled lanes at agent `i` (with duplicates).
- `self.n_s = max(ss)` — **shared, padded** observation dimension. All agents report a vector of length `n_s` even when their true observation is shorter — `get_state` right-pads with zeros.
- `self.n_s_ls = [self.n_s] * self.n_agent` — homogeneous (after padding).
- `self.n_a_ls = [self.n_action] * self.n_agent` — homogeneous (after wrap-around).
- `self.fp` — uniform initial fingerprint distribution, shape `(n_agent, n_action)`. Used by `get_fingerprint` / `update_fingerprint` for MA2C-style fingerprint communication.

**Note**: although `n_s_ls` and `n_a_ls` are uniform here (because of padding), the underlying TLS phases ARE heterogeneous — preserved in `self.A` and `self.state_heterogeneous_space` (see below).

### Heterogeneous state-space size (lines 151–157)

```python
self.state_heterogeneous_space = ss
for i in range(self.n_agent):
    self.state_heterogeneous_space[i] = ss[i]
    indices = np.where(self.neighbor_mask[i] == 1)[0]
    for j in range(len(indices)):
        self.state_heterogeneous_space[i] += ss[j]
```

For each agent `i`, the "heterogeneous" state dim is `ss[i]` plus, for each neighbor (including self, since `neighbor_mask` has diag=1), `ss[j]`. The accumulation is meant to represent the size of the concatenated own-state + neighbor-states.

**Bug**: the inner loop uses `j` as the loop index over `range(len(indices))` instead of indexing into `indices`. So it sums `ss[0] + ss[1] + ... + ss[k-1]` where `k = len(indices)`, not `ss[indices[0]] + ss[indices[1]] + ...`. The list `state_heterogeneous_space` therefore does not actually reflect the per-agent neighborhood — it accidentally sums the first `k` agents' observation sizes regardless of which agents are the neighbors. (See "Bugs / oddities" below.)

**Second bug**: `self.state_heterogeneous_space = ss` is an alias assignment, not a copy. The augmentation loop then mutates `ss` in place, which means the printed `ss=` value (printed earlier on line 145) is the only place the un-augmented `ss` was observed; downstream uses of `ss` (none after this point) would see the augmented values.

### Close (lines 160–161)

```python
self.traci.close()
sys.stdout.flush()
```

SUMO is shut down at the end of `__init__`; it gets restarted in `reset()`.

---

## Action / observation space construction (summary)

| Quantity | Source | Notes |
|---|---|---|
| `n_agent` | `traci.trafficlight.getIDCount()` after starting SUMO with `net_path` | Inferred from net file; not hard-coded. |
| `n_action` | `max(self.A)` where `A[i] = len(phase[0].phases)` for TLS `i` | Shared, padded. Per-agent action count `A[i]` is preserved on `self.A`. |
| `n_a_ls` | `[n_action] * n_agent` | Homogeneous after padding. |
| `n_s` | `max(ss)` where `ss[i] = len(controlled_lanes(i))` | Shared, padded. |
| `n_s_ls` | `[n_s] * n_agent` | Homogeneous after padding. |
| `state_heterogeneous_space` | `ss[i] + sum of neighbor ss` | Buggy (see above) — but used in `reset()` only as a no-op line. |
| `action_space` | `Discrete(n_action)` | Shared `gym.spaces.Discrete`. No `observation_space` is set. |

Note: there is **no** `observation_space` attribute, despite this being a `gym.Wrapper`.

---

## `neighbor_mask` construction — k-hop BFS

The mask used at runtime is `self.neighbor_mask = self.get_neighbor_matrix_khop(k=1)` (line 97).

Definition at lines 300–340:

```python
def get_neighbor_matrix_khop(self, k=2):
    """
    Builds a k-hop neighbor matrix among traffic light-controlled junctions.
    Handles cases where traffic light IDs do not directly map to node IDs.
    """
    tls_list = self.net.getTrafficLights()  # sumolib.TLS objects
    tl_node_ids = [tls.getID() for tls in tls_list]  # true junction node IDs

    id_map = {tl_id: idx for idx, tl_id in enumerate(tl_node_ids)}
    n = len(tl_node_ids)
    neighbor_mask = np.eye(n)

    for i, from_node in enumerate(tl_node_ids):
        visited = set()
        queue = deque([(from_node, 0)])
        visited.add(from_node)

        while queue:
            current_node, depth = queue.popleft()
            if depth >= k:
                continue

            try:
                outgoing_edges = self.net.getNode(current_node).getOutgoing()
            except KeyError:
                continue  # Skip if node not found in net

            for edge in outgoing_edges:
                to_node = edge.getToNode().getID()

                if to_node in visited:
                    continue
                visited.add(to_node)

                if to_node in id_map and to_node != from_node:
                    j = id_map[to_node]
                    neighbor_mask[i][j] = 1

                queue.append((to_node, depth + 1))

    return neighbor_mask
```

Mechanics:
1. Enumerate all sumolib `TrafficLight` objects; their IDs are used directly as junction node IDs (this assumes TLS-ID = node-ID, which is true for most SUMO nets where the TLS is named after the junction it controls).
2. For each agent `i`, BFS outward following `Node.getOutgoing()` (outgoing edges in the network graph).
3. A neighbor is recorded if its junction is **also** a TLS node (`to_node in id_map`) and is within `k` hops (the `if depth >= k: continue` gate ensures only nodes reached at depth ≤ k are added — note that since the depth check is on the popped node and a new node is added on the recursive step, the actual recorded depth is `depth + 1`, so neighbors are added when `depth + 1 ≤ k`, i.e. up to `k` hops).
4. The diagonal is initialized to 1 (`np.eye(n)`), so each agent is its own neighbor — this matters for the `get_neighbor_action` indexing and downstream uses that include self in the neighborhood.

**Important**: BFS uses **outgoing** edges only, so the resulting mask is **asymmetric** in general (directed network). If agent A has a one-way edge to B but no return edge, then `neighbor_mask[A][B] = 1` but `neighbor_mask[B][A] = 0` (unless reachable via another path within `k` hops).

There are also two sibling methods that are defined but unused at construction time:
- `get_neighbor_matrix_tls_khop(k=3)` (lines 209–255) — maps each TLS to a representative junction via `getControlledLanes(...)[0]` and the lane's edge `getToNode()`. Tracks visited-with-depth (allows revisits if a shorter path appears).
- `get_neighbor_matrix_khop_2(k=2)` (lines 262–296) — similar but uses `visited` as a dict-of-depths (revisit-friendly).
- `get_neighbor_matrix()` (lines 195–205) — the **lane-graph-based** original: walks `self.AdjacencyList` (built in `generateTopology`) and marks any TLS that is a direct successor in the directed edge graph.

Only `get_neighbor_matrix_khop(k=1)` is used at runtime; the others are dead code / experimental.

---

## `cal_n_order_matrix` (lines 163–179)

```python
def cal_n_order_matrix(self, n_nodes, max_order, adj):
    def calculate_high_order_adj(max_order):
        result_matrix = np.zeros((n_nodes, n_nodes), dtype=int)
        for i in range(n_nodes):
            for j in range(n_nodes):
                if abs(j - i) <= max_order:
                    result_matrix[i][j] = 1
        return result_matrix

    adjacency_matrix = np.eye(n_nodes)
    result = calculate_high_order_adj(max_order)
    for q in range(n_nodes):
        for k in range(n_nodes):
            if adj[q][k] == 1:
                result[q][k] = 1
    return result - np.eye(n_nodes)
```

What it computes:
- A bandwidth-`max_order` mask in the agent-**index** space (i.e. `result[i][j] = 1` iff `|i - j| <= max_order`) — this is a banded matrix that has nothing to do with graph topology, only the **ordering of agents in `id_list`**.
- ORed with the actual adjacency `adj` passed in.
- Diagonal is subtracted off at the end (`- np.eye`).

In other words, the "n-order matrix" is a **union of the topological adjacency with an index-distance band**. The result is intended to be the communication mask for MA2C_NC / BayesianGraph variants when topology alone is too sparse — it adds extra connectivity to "nearby" agents in the `id_list` order. The local index ordering only reflects topology if the `id_list` was sorted that way (which is not the case here — `id_list = list(traci.trafficlight.getIDList())` returns IDs in insertion order from the net file).

The method is defined on `self` but is **not called inside this file**. It is expected to be called externally by the MA2C_NC / BayesianGraph training code, which is why this is included in the file at all. (The agent code uses it to build a wider receptive field.)

`adjacency_matrix = np.eye(n_nodes)` on line 173 is computed but never used — dead variable.

---

## Edge matrix pre-computation (`get_edge_matrix`, lines 343–366)

```python
def get_edge_matrix(self):
    E_id = []
    E_from = []
    E_to = []

    for i in range(len(self.Edges)):
        E_id.append(self.Edges[i].getID())
        E_from.append(self.Edges[i].getFromNode().getID())
        E_to.append(self.Edges[i].getToNode().getID())

    E_to_state = []
    E_from_state = []
    for p in range(self.n_agent):
        E_index_to = [index for index, value in enumerate(E_to) if value == self.id_list[p]]
        E_index_from = [index for index, value in enumerate(E_from) if value == self.id_list[p]]
        e_to = []
        e_from = []
        for q in E_index_to:
            e_to.append(E_id[q])
        for w in E_index_from:
            e_from.append(E_id[w])
        E_to_state.append(e_to)
        E_from_state.append(e_from)
    return E_id, E_from, E_to, E_to_state, E_from_state
```

What it builds, per agent `p` (= TLS at `id_list[p]`):
- `E_to_state[p]` — list of edge IDs whose **to-node** is this TLS junction (i.e. incoming edges).
- `E_from_state[p]` — list of edge IDs whose **from-node** is this TLS junction (i.e. outgoing edges).

These lists are stored on `self.E_to_state` / `self.E_from_state`. They are referenced only by the **commented-out** alternative `get_state(self, old_state)` (lines 443–458), which would have used `traci.edge.getLastStepVehicleNumber(self.E_to_state[k][o])` to count vehicles per incoming edge. The active `get_state` (lines 462–479) does not use them.

So `E_*` is effectively **unused at runtime**, but the cost is paid in `__init__` (one O(|edges| × n_agent) double-loop).

`E_id` / `E_from` / `E_to` are similarly stored for completeness; they are inputs to the above lists and not referenced elsewhere.

---

## `step()` (lines 556–599)

```python
def step(self, action):

    for i in range(self.n_agent):
        if action[i] <= self.A[i] - 1:
            self.traci.trafficlight.setPhase(self.id_list[i], action[i])
        else:
            a = action[i]
            while a > self.A[i] - 1:
                a = a - self.A[i]
            self.traci.trafficlight.setPhase(self.id_list[i], a)

    self.arrived = []
    for _ in range(self.single_step_second):
        self.traci.simulationStep()


        if self.obj == "arrived":
            arrived_vehicles = self.traci.simulation.getArrivedIDList()
            num_arrived_vehicles = len(arrived_vehicles)
            if num_arrived_vehicles > 0:
                self.arrived.append(num_arrived_vehicles)


    state_old = self.state
    state = self.get_state()


    if self.obj == "arrived":
        reward = np.repeat(sum(self.arrived), self.n_agent)
    else:
        reward = self.get_reward()



    reward = np.array(reward, dtype=np.float32)
    done = False  # Initialize done as False
    self.state = state

    reward = np.sum(reward)
    return state, reward, done, reward
```

Key behaviors:
- **Action wrap-around**: if `action[i]` exceeds `A[i] - 1`, it is mod-reduced via a `while` loop (functionally `action[i] % A[i]`, but written as iterative subtraction). This is the mechanism by which the shared `n_action`-wide action space accommodates heterogeneous per-TLS phase counts.
- **No yellow-phase handling**: unlike [[atsc_env]] / [[real_net_env]], there is no yellow-phase / transition logic. The new phase is set immediately via `setPhase`, and then `single_step_second` (20 or 40) SUMO steps are simulated. This is a much simpler control model — actions correspond directly to SUMO phase indices, including any yellow phases that already exist in the program logic.
- **Reward modes**:
  - `obj == "queue"` (default): negative sum of `getLastStepHaltingNumber` over all controlled lanes, per agent, divided by `reward_scale`.
  - `obj == "wait"`: a (rather odd) per-vehicle wait-time accumulation — see `get_reward` discussion below.
  - `obj == "arrived"`: per-step accumulator of `getArrivedIDList` vehicles, broadcast to all agents (`np.repeat`).
- **`done` is always `False`**. The wrapper never terminates an episode from inside `step`. Termination must be enforced externally (likely by SUMO's `.sumocfg` end-time, after which `traci.simulationStep` becomes a no-op).
- **Return shape oddity**: `return state, reward, done, reward` — the 4-tuple is `(state, reward_scalar, done, info)` where `info = reward` (the same scalar). The conventional gym contract expects `info` to be a dict. Here `info` is just the scalar reward repeated. Also note `reward = np.sum(reward)` on line 598 — the per-agent reward array is **collapsed to a single scalar** before being returned. This is a deviation from [[atsc_env]], which returns per-agent reward arrays.
- `state_old = self.state` (line 583) is dead — assigned but never used.
- `self.arrived` is only populated under `obj == "arrived"` but is initialized unconditionally each step.

`reward_scale` usage: applied in `get_reward()`:

```python
reward = -np.array(queues) / float(self.reward_scale)
```

Larger `reward_scale` → smaller magnitude of (negative) reward signal. Default 10. The final `np.sum(reward)` then aggregates across all agents.

---

## `get_state()` (lines 462–479)

```python
def get_state(self):
    cur_state = []
    for k, ild in enumerate(self.ILDS_in):

        cur_wave = []
        for j, ild_seg in enumerate(ild):
            cur_wave.append(self.traci.lane.getLastStepVehicleNumber(ild_seg[0]) / self.CAP[k][j])
        cur_state.append(cur_wave)
    # cur_state = np.array(cur_state)

    # Pad each sublist to length self.n_s
    cur_state_padded = np.array([
        np.pad(sublist, (0, self.n_s - len(sublist)), constant_values=0.0)
        for sublist in cur_state
    ])

    return cur_state_padded
```

- Per-agent observation: list of (vehicles-on-lane / lane-capacity) for each controlled lane.
- Each agent's vector is right-padded with zeros to length `self.n_s`.
- Return shape: `(n_agent, n_s)`. Numpy array.

This is a **density-per-lane** observation. Note that `CAP[k][j] = lane_length / 200`, so the ratio is `vehicles / (length / 200)` = `vehicles × 200 / length` — i.e. vehicles per (lane-length / 200 m). For a 100 m lane, the ratio is `vehicles / 0.5 = 2 × vehicles`. The normalization is unusual.

## `get_reward()` (lines 483–512)

Two implemented modes:

**`obj == "queue"`** (default):
```python
queues = []
for k, ild in enumerate(self.ILDS_in):
    cur_queue = 0
    for j, ild_seg in enumerate(ild):
        cur_queue += self.traci.lane.getLastStepHaltingNumber(ild_seg[0])
    queues.append(cur_queue)
reward = -np.array(queues) / float(self.reward_scale)
```
Per-agent: negative sum of halting (≈ stopped) vehicles across controlled lanes, divided by `reward_scale`. Shape `(n_agent,)`.

**`obj == "wait"`**:
```python
waits = []
for k, ild in enumerate(self.ILDS_in):
    for j, ild_seg in enumerate(ild):
        max_pos = 0
        cur_cars = self.traci.lane.getLastStepVehicleIDs(ild_seg[0])
        for vid in cur_cars:
            car_pos = self.traci.vehicle.getLanePosition(vid)
            if car_pos > max_pos:
                max_pos = car_pos
                car_wait = self.traci.vehicle.getWaitingTime(vid)
                waits.append(car_wait)
reward = -np.array(waits) / float(self.reward_scale)
```
This branch is **broken/buggy** as a per-agent reward:
- `waits` is appended to from every `(k, j, vid)` triple that increases `max_pos` — so the list length varies per step and per traffic state, not `== n_agent`.
- It is therefore not aligned with the agent index at all when reshaped.
- Downstream in `step()`, `reward = np.array(reward, dtype=np.float32)` will yield a 1-D array of arbitrary length, and `np.sum(reward)` will collapse it — so the "wait" mode does happen to produce a scalar that `step` can return, but it isn't a meaningful per-agent reward and won't have shape `(n_agent,)`.

Since the default `self.obj = "queue"`, this bug is dormant.

---

## `reset()` (lines 516–539)

```python
def reset(self):

    # if self.gui:
    #     self.sumoBinary = checkBinary('sumo-gui')
    # else:
    #     self.sumoBinary = checkBinary('sumo')
    # self.cur_step = 0  # reset step counter
    self.traci.start([self.sumoBinary, "-c", self.sim_path, "--no-warnings"])


    self.state_heterogeneous_space


    #state = np.zeros((self.n_agent, self.n_s))

    state = []
    for i in range(len(self.state_heterogeneous_space)):
        state.append(np.zeros((1, self.state_heterogeneous_space[i]))[0])

    state = np.zeros((self.n_agent, self.n_s))
    self.state = state


    return self.state
```

- Restarts SUMO via `traci.start`. There is no `traci.close()` here — it assumes the previous `traci` connection was already closed (which it is at the end of `__init__`, line 160). However, on subsequent resets (after `step`) there is **no `traci.close()` either**, which means calling `reset()` a second time without manually closing SUMO will fail or open a duplicate connection. The proper pattern (closing the existing connection before starting a new one) is missing.
- `self.state_heterogeneous_space` on line 526 is a no-op statement (attribute access without assignment).
- The first construction of `state` as a list of per-agent zero-vectors of sizes `state_heterogeneous_space[i]` is **immediately overwritten** by `state = np.zeros((self.n_agent, self.n_s))`. So the heterogeneous state plumbing is dead — every `reset()` returns a homogeneous `(n_agent, n_s)` zero matrix.
- Returns `self.state` (the zero matrix). No info dict.

---

## `init_data` / `output_data` / `init_test_seeds`

**These methods do not exist on `Large_city_net`.**

The standard [[atsc_env]] / [[TrafficSimulator]] interface includes:
- `init_data(...)` — open CSV writers for trip and traffic logging.
- `output_data(...)` — flush logging to disk.
- `init_test_seeds(seeds)` — set the deterministic seed sequence used by `reset()`.
- `update_fingerprint(...)` / `get_fingerprint()` — present here (lines 183–187).

`Large_city_net` only implements:
- `__init__`
- `cal_n_order_matrix`
- `get_fingerprint`
- `update_fingerprint`
- `get_neighbor_action`
- `get_neighbor_matrix` (lane-graph based; unused)
- `get_neighbor_matrix_tls_khop` (unused)
- `get_neighbor_matrix_khop_2` (unused)
- `get_neighbor_matrix_khop` (used, k=1)
- `get_edge_matrix`
- `generateTopology`
- `get_tls_positions`
- `write_neighbor_graph_poly`
- `get_state`
- `get_reward`
- `reset`
- `clear` (lines 541–550 — calls `traci.close()` and flushes stdout)
- `step`
- `get_state_` (returns `self.state`)

So **`init_data`, `output_data`, `init_test_seeds`, `terminate`, `update_fingerprint` (present), seed management, and trip-data logging are all missing** compared to [[atsc_env]]. There is no per-step CSV output and no deterministic-seed mechanism — every `reset()` restarts SUMO with the same `.sumocfg`, and the only randomness comes from SUMO's own RNG (typically also fixed via the cfg).

`clear()` is the rough analogue of `terminate()` in [[atsc_env]] but does not do trip-data flushing.

---

## Differences from `atsc_env.py` — interface divergences

Compared with the standard `TrafficSimulator` base in [[atsc_env]]:

| Aspect | [[atsc_env]] (`TrafficSimulator`) | `Large_city_net` |
|---|---|---|
| Base class | custom `TrafficSimulator` | `gym.Wrapper` (super().__init__ never called) |
| SUMO connection | per-instance `port` argument; multiple parallel instances supported | single, default port; one instance per process |
| Phase model | `PhaseMap`/`PhaseSet`; yellow-phase transitions; control_interval_sec / yellow_interval_sec | direct `setPhase`; no yellow handling; fixed `single_step_second` per net size |
| Action space | per-agent, heterogeneous via `n_a_ls` (true per-agent sizes) | shared `Discrete(max_phases)` with modulo wrap |
| Observation space | per-agent vector (own + neighbor wave/wait), `n_s_ls` heterogeneous | per-agent padded vector of lane densities, `n_s_ls = [n_s] * n_agent` |
| Reward return | per-agent `np.ndarray` shape `(n_agent,)` | **scalar** (`np.sum(reward)`) — agent-level reward is discarded |
| `step` return | `(state, reward_array, done, global_reward_or_info_dict)` | `(state, reward_scalar, done, reward_scalar)` — info is the scalar reward, not a dict |
| `done` | computed from `cur_episode_step >= episode_length_sec/control_interval_sec` | always `False` |
| Episode step counter | tracked | not tracked (commented out) |
| `init_data` / `output_data` | present; writes traffic + trip CSVs | **absent** |
| `init_test_seeds(seeds)` | present | **absent** |
| `terminate()` | present | replaced by `clear()` |
| `seed(s)` | present | **absent** |
| Coop reward weighting | via `coop_gamma >= 0` and `distance_mask` | `coop_gamma = -1` only; `distance_mask` is just an alias for `neighbor_mask` |
| `update_fingerprint` / `get_fingerprint` | present | present |
| Reward modes | hardcoded `queue` or `wait` per child class | runtime-selectable via `self.obj` (`queue` / `wait` / `arrived`) — but only `queue` is wired correctly |
| Logging | `init_data` writes per-step CSVs | none |
| Reset behavior | manages SUMO lifecycle, seeds, episode counter | restarts SUMO every call but never closes prior connection within `reset()` |
| `observation_space` attribute | set | **never set** |

Net effect: agent code that assumes the [[atsc_env]] interface must special-case `Large_city`. In particular:
- The reward signal is a single scalar, not a per-agent vector.
- There is no trip-CSV output, so evaluation pipelines that read those CSVs won't have anything to read.
- There is no seed setter, so determinism control is impossible from Python.

---

## Bugs / oddities

1. **`np.fill_diagonal(adj_no_self, 1)` on line 103** — the comment says "Remove self-loops for edge counting" but the code sets the diagonal to 1, not 0. So `num_edges` over-counts by `n_agent`.

2. **`self.state_heterogeneous_space = ss` alias** (line 151) — this is not a copy. The subsequent augmentation loop mutates `ss` in-place. After `__init__`, `ss` and `self.state_heterogeneous_space` are the same list, and both contain the augmented (own + neighbor) sizes.

3. **Wrong neighbor indexing on line 156**:
   ```python
   for j in range(len(indices)):
       self.state_heterogeneous_space[i] += ss[j]
   ```
   `j` should be `indices[j]` (or use `for idx in indices:`). Currently it adds `ss[0..k-1]` regardless of which agents are the actual neighbors. Combined with bug #2, the per-iteration mutation of `ss` makes the result quasi-quadratic and unrelated to topology.

4. **`adjacency_matrix = np.eye(n_nodes)` in `cal_n_order_matrix`** (line 173) — dead variable.

5. **`reset()` does not close the previous traci connection** (line 523). If called more than once after a prior `step` sequence, traci will refuse to start a second connection without `traci.close()` first.

6. **`reset()` heterogeneous state construction is overwritten** (lines 531–535). The non-trivial first attempt is dead code.

7. **`self.state_heterogeneous_space` on line 526** — bare attribute access with no effect (likely a leftover debug print or missing `print(...)`).

8. **`step()` returns the reward both as the 2nd and 4th element** (line 599), and collapses per-agent reward to a single scalar via `np.sum`. Agent code that expects per-agent rewards or an info dict will misbehave.

9. **`done` is hardwired to `False`** (line 595). No internal termination mechanism.

10. **`get_reward()` `wait` branch produces a misaligned array** (lines 496–508). The number of entries in `waits` is data-dependent (one per max-position vehicle per lane), not `n_agent`. The downstream `np.sum` masks this bug as long as nobody inspects shape.

11. **`get_state()` `n_s` padding can truncate** — actually, since `n_s = max(ss)`, `n_s - len(sublist) >= 0` always, so this is safe. But the heterogeneity is hidden from the agent.

12. **`VEH_LEN_M = 200`** — used as if vehicles were 200 m long. Conventional vehicle length is ~5 m. This makes the normalized "capacity" very small (`length/200`), which in turn makes the density observation = `vehicles / (length/200)` blow up for short lanes. May be intentional for scaling, but it's at odds with the variable name "vehicle length in meters."

13. **`E_to_state` / `E_from_state` / `E_id` / `E_from` / `E_to`** are computed in O(|edges| × n_agent) but never used by the active `get_state` (only by the commented-out variant).

14. **`AdjacencyList` and `get_neighbor_matrix` (lane-graph version)** are computed/defined but unused — the runtime mask comes from `get_neighbor_matrix_khop(k=1)`.

15. **`distance_mask = dp(neighbor_mask)`** — named "distance mask" but holds boolean adjacency. Code expecting hop-counts will misbehave.

16. **No `observation_space`** — `gym` wrappers usually expose one.

17. **`super().__init__` not called** — `gym.Wrapper.__init__(self, env)` is skipped, so `self.env`, `self.observation_space`, `self.action_space` from the wrapper layer are nominally unset (the local `self.action_space = Discrete(n_action)` does set the action space, though).

18. **Module-level `current_dir`/`parent_dir`** (lines 16–18) are dead. The factory `Large_city_Env` (line 606) has commented-out path-prefixing that would have used them.

19. **Asymmetric `neighbor_mask`** (directed BFS along outgoing edges) — most GNN / MA2C code assumes symmetric neighbor relations. The asymmetry may cause subtle bugs when iterating `naction = action[neighbor_mask[i] == 1]` from one side but not the other.

20. **`single_step_second = 20` vs `40`** — the cutoff at `n_agent < 500` is brittle; all three NewYork variants (33, 51, 167) use 20s, but the comment in `__main__` (line 617) about "436 agents" suggests a larger variant exists that would use 40s — different episodes-per-second across nets makes cross-net comparisons non-trivial.

21. **`__main__` block** (lines 614–629) hard-codes `NewYork167/newyork167.net.xml` and references "436 agents" in a stale comment. The example calls `env.reset()` but the actual `step` loop is commented out — so running the file directly only validates `__init__` + one reset.

---

## Factory function `Large_city_Env` (lines 606–610)

```python
def Large_city_Env(net_path, sim_path, folder_name, reward_scale):
    # print("parent_dir=",parent_dir) /home/wduan/Data
    # net_path = parent_dir + net_path   # 436 agents
    # sim_path = parent_dir + sim_path
    return Large_city_net(net_path, sim_path, folder_name, reward_scale)
```

A thin wrapper that constructs the `Large_city_net` object. Callers (training scripts) import `Large_city_Env`, not `Large_city_net` directly. The commented lines reveal an earlier convention where paths were relative and prepended with a `parent_dir` rooted at `/home/wduan/Data`.

---

## Summary

`Large_city_net` is a **standalone, simplified env** that talks to SUMO directly without going through the [[atsc_env]] / [[TrafficSimulator]] machinery. It supports the NewYork33/51/167 nets by inferring `n_agent`, `n_action`, and `n_s` directly from the SUMO net + program logics, padding heterogeneous quantities to common sizes, and using modulo wrap-around to handle per-TLS phase-count differences in a shared `Discrete(n_action)` action space. The reward is collapsed to a single scalar before being returned (a major divergence from [[atsc_env]]). Several attributes (`distance_mask`, `state_heterogeneous_space`, `E_*`, `AdjacencyList`) are computed but unused or buggy. Episode termination is external (SUMO `.sumocfg` end-time); there is no `init_data`/`output_data`/`init_test_seeds`/`seed` plumbing.
