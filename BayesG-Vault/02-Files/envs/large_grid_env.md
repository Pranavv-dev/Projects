# large_grid_env.py — Line-by-line Walkthrough

Path: `/tmp/BayesG/envs/large_grid_env.py` (198 lines).

This module defines the synthetic 5x5 SUMO grid environment used in MA2C / [[BayesianGraph]] experiments. It is a concrete `TrafficSimulator` (see [[atsc_env]]) with 25 traffic-light nodes arranged in a regular Manhattan grid (`nt1`..`nt25`), all sharing the same 5-phase signal program. Compare with the real-network counterpart [[real_net_env]].

## Imports

```python
import configparser
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import time
from envs.atsc_env import PhaseMap, PhaseSet, TrafficSimulator
from envs.large_grid_data.build_file import gen_rou_file

sns.set_color_codes()
```

- `configparser`, `os`, `time`, `logging` — standard scaffolding for the `__main__` greedy runner block.
- `numpy`, `matplotlib.pyplot`, `seaborn` — used for the `neighbor_mask` matrix and `plot_stat` / `plot_cdf` reward plotting at the bottom.
- `PhaseMap`, `PhaseSet`, `TrafficSimulator` are imported from [[atsc_env]] — `LargeGridEnv` subclasses `TrafficSimulator`, and `LargeGridPhase` subclasses `PhaseMap`.
- `gen_rou_file` from `envs.large_grid_data.build_file` is the SUMO route-file generator. It is called by `_init_sim_config` to materialize `exp_<thread>.rou.xml` and `exp_<thread>.sumocfg`.
- `sns.set_color_codes()` is a global side-effect at import time — applies seaborn color shortcuts (`'b'`, `'r'`, etc.) even when the module is only imported for its env class.

## Module-level constants

```python
STATE_NAMES = ['wave']
```

> Only the `'wave'` (lane-level vehicle count, a.k.a. number-of-vehicles) state is exposed by this env. No `'wait'` here, in contrast with what [[atsc_env]]'s base `_init_state` machinery can do. The single-element list is assigned to `self.state_names` in `_init_map`, which downstream controls which features each node observes.

```python
PHASE_NUM = 5
```

> Every traffic-light node is hard-wired to phase-set id `5` (i.e. there are 5 discrete phases per intersection). `_get_node_phase_id` always returns this constant — see below.

There is no explicit `NEIGHBOR_MAP` constant at module level in this file — neighbor info is built imperatively inside `_init_neighbor_map`. See the [[#Bugs / oddities]] section.

## class LargeGridPhase(PhaseMap)

```python
class LargeGridPhase(PhaseMap):
    def __init__(self):
        phases = ['GGgrrrGGgrrr', 'rrrGrGrrrGrG', 'rrrGGrrrrGGr',
                  'rrrGGGrrrrrr', 'rrrrrrrrrGGG']
        self.phases = {PHASE_NUM: PhaseSet(phases)}
```

A `PhaseMap` (from [[atsc_env]]) is a registry mapping a `phase_id` to a `PhaseSet`. For the large grid:

- Every intersection has 12 controllable signal heads (string length 12 — 3 lanes per approach x 4 approaches).
- Five phases are defined:
  1. `'GGgrrrGGgrrr'` — N-S straight + N-S right (a permissive right-turn lower-case `g`).
  2. `'rrrGrGrrrGrG'` — E-W and W-E protected lefts only on the inner left lanes.
  3. `'rrrGGrrrrGGr'` — E-W through + left.
  4. `'rrrGGGrrrrrr'` — eastbound only (E-direction full green).
  5. `'rrrrrrrrrGGG'` — westbound only.
- The dict key is `PHASE_NUM` (= 5), so `phase_map.phases[5]` returns the `PhaseSet`. The action space at each node therefore has cardinality 5 — consistent with `LargeGridController.greedy` returning `argmax` over a length-5 `flows` vector.

## class LargeGridController

This is a hand-coded *greedy* baseline, not part of any learner.

```python
class LargeGridController:
    def __init__(self, node_names):
        self.name = 'greedy'
        self.node_names = node_names

    def forward(self, obs):
        actions = []
        for ob, node_name in zip(obs, self.node_names):
            actions.append(self.greedy(ob, node_name))
        return actions

    def greedy(self, ob, node_name):
        # hard code the mapping from state to number of cars
        flows = [ob[0] + ob[3], ob[2] + ob[5], ob[1] + ob[4],
                 ob[1] + ob[2], ob[4] + ob[5]]
        return np.argmax(np.array(flows))
```

- `forward` simply broadcasts `greedy` across all 25 nodes.
- `greedy` assumes the observation vector has 6 lane-wave entries `ob[0..5]` and computes a *demand proxy* per phase by summing the wave counts of the lanes that the phase actually serves. It then picks the phase index that maximizes demand.
- The mapping is consistent with the phase strings above (N-S pair, E-W pair, etc.) but it is hand-aligned to the lane ordering produced by SUMO's `getControlledLanes()` — there are no asserts, so silently corrupted lane ordering would silently degrade the baseline.
- The `node_name` argument is unused — every node is treated identically (which is valid for the homogeneous grid).

## class LargeGridEnv(TrafficSimulator)

### `__init__`

```python
def __init__(self, config, port=0, output_path='', is_record=False, record_stat=False):
    self.peak_flow1 = config.getint('peak_flow1')
    self.peak_flow2 = config.getint('peak_flow2')
    self.init_density = config.getfloat('init_density')
    super().__init__(config, output_path, is_record, record_stat, port=port)
```

- Pulls three SUMO-flow tuning knobs from the `[ENV_CONFIG]` section of the ini:
  - `peak_flow1`, `peak_flow2` — the two peak demand levels (veh/hr) that `output_flows` in `build_file.py` consumes for the two OD pulse waves.
  - `init_density` — the initial routes density (used by `init_routes` in `build_file`).
- Calls the parent `TrafficSimulator.__init__` (see [[atsc_env]]) which, per the comment block on lines 52-58, performs in order:
  1. `_init_map()` — overridden below; builds node list, neighbor mask, distance mask, and the `LargeGridPhase`.
  2. `init_data(is_record, record_stats, output_path)`.
  3. `init_test_seeds(test_seeds)`.
  4. `_init_sim(self.seed)` — which calls `_init_sim_config(seed)` (overridden below).
  5. `_init_nodes()` — instantiates per-TLS node objects using the neighbor map.
  6. `terminate()`.

The comment block is documentation-only and does not execute.

### `_get_node_phase_id`

```python
def _get_node_phase_id(self, node_name):
    return PHASE_NUM
```

All 25 nodes share phase-set id 5. The base [[atsc_env]] uses this hook when looking up `phase_map.phases[phase_id]` to size action spaces and decode phase strings.

### `_init_neighbor_map`

This method does **not** use a precomputed `NEIGHBOR_MAP` dict — it hand-codes corner + edge node neighbors and then computes interior node neighbors with a loop.

```python
neighbor_map = {}
# corner nodes
neighbor_map['nt1'] = ['nt6', 'nt2']
neighbor_map['nt5'] = ['nt10', 'nt4']
neighbor_map['nt21'] = ['nt22', 'nt16']
neighbor_map['nt25'] = ['nt20', 'nt24']
```

Numbering is row-major across the 5x5 grid: `nt1..nt5` is the bottom (or top) row, `nt21..nt25` the opposite row. Corner nodes have only 2 neighbors. The four corners hit here are `nt1`, `nt5`, `nt21`, `nt25`.

```python
# edge nodes
neighbor_map['nt2'] = ['nt7', 'nt3', 'nt1']
neighbor_map['nt3'] = ['nt8', 'nt4', 'nt2']
neighbor_map['nt4'] = ['nt9', 'nt5', 'nt3']
neighbor_map['nt22'] = ['nt23', 'nt17', 'nt21']
neighbor_map['nt23'] = ['nt24', 'nt18', 'nt22']
neighbor_map['nt24'] = ['nt25', 'nt19', 'nt23']
neighbor_map['nt10'] = ['nt15', 'nt5', 'nt9']
neighbor_map['nt15'] = ['nt20', 'nt10', 'nt14']
neighbor_map['nt20'] = ['nt25', 'nt15', 'nt19']
neighbor_map['nt6'] = ['nt11', 'nt7', 'nt1']
neighbor_map['nt11'] = ['nt16', 'nt12', 'nt6']
neighbor_map['nt16'] = ['nt21', 'nt17', 'nt11']
```

Twelve edge (non-corner perimeter) nodes, each with 3 neighbors. Note the *ordering convention*: roughly `[north, east/along-edge, south/west]`, but it's not strictly uniform — for the top-edge nodes (`nt2..nt4`) the order is `[N, E, W]`, while for the right-edge (`nt10`, `nt15`, `nt20`) it's `[N, S, W]`. This ordering matters when downstream code uses positional indexing of neighbors (e.g. message-passing channels in [[BayesianGraph]]). Most consumers use `self.neighbor_mask` (a symmetric 0/1 matrix), so the per-node ordering is not load-bearing for that.

```python
# internal nodes
for i in [7, 8, 9, 12, 13, 14, 17, 18, 19]:
    n_node = 'nt' + str(i + 5)
    s_node = 'nt' + str(i - 5)
    w_node = 'nt' + str(i - 1)
    e_node = 'nt' + str(i + 1)
    cur_node = 'nt' + str(i)
    neighbor_map[cur_node] = [n_node, e_node, s_node, w_node]
self.neighbor_map = neighbor_map
```

The 9 interior nodes get the canonical `[N, E, S, W]` 4-neighborhood via index arithmetic on the row-major grid (`+5` = next row up, `-5` = row down, `+/-1` = column).

```python
# neighbor_mask is a 25x25 matrix (adjacent matrix), where each element is 0 or 1.
self.neighbor_mask = np.zeros((self.n_node, self.n_node))
for i in range(self.n_node):
    for nnode in neighbor_map['nt%d' % (i+1)]:
        ni = self.node_names.index(nnode)
        self.neighbor_mask[i, ni] = 1
logging.info('neighbor mask:\n %r' % self.neighbor_mask)
```

The 25x25 adjacency `neighbor_mask` is then constructed by iterating each node and setting `mask[i, ni] = 1` for every neighbor name. The matrix is symmetric because the hand-coded `neighbor_map` is symmetric (every edge appears in both endpoints' lists). This mask is the graph used by [[BayesianGraph]] / MA2C neighbor-communication layers.

### `_init_distance_map`

The 5x5 grid distance matrix is built block-wise using row-major Manhattan distance.

```python
block0 = np.array([[0,1,2,3,4],[1,0,1,2,3],[2,1,0,1,2],[3,2,1,0,1],[4,3,2,1,0]])
block1 = block0 + 1
block2 = block0 + 2
block3 = block0 + 3
block4 = block0 + 4
```

- `block0` is the 5x5 intra-row Manhattan-distance matrix (just column-index difference between nodes in the same row).
- `blockK = block0 + K` represents the inter-row block where the row gap is `K` — so the total Manhattan distance between node `(r1, c1)` in row `r1` and `(r2, c2)` in row `r2` is `|c1-c2| + |r1-r2|`, which equals `block0[c1, c2] + |r1-r2|`.

```python
row0 = np.hstack([block0, block1, block2, block3, block4])
row1 = np.hstack([block1, block0, block1, block2, block3])
row2 = np.hstack([block2, block1, block0, block1, block2])
row3 = np.hstack([block3, block2, block1, block0, block1])
row4 = np.hstack([block4, block3, block2, block1, block0])
self.distance_mask = np.vstack([row0, row1, row2, row3, row4])
```

- Each `rowK` is the 5x25 horizontal stack representing distances from all 5 nodes in row `K` to all 25 nodes in the grid. The block at column-position `j` uses `block_{|K-j|}` — i.e. the inter-row offset.
- `vstack`-ing row0..row4 produces a 25x25 symmetric Manhattan-distance matrix.

The docstring after the assignment quotes the exact matrix:

```python
'''[[0 1 2 3 4 1 2 3 4 5 2 3 4 5 6 3 4 5 6 7 4 5 6 7 8]
    ...
    [8 7 6 5 4 7 6 5 4 3 6 5 4 3 2 5 4 3 2 1 4 3 2 1 0]]'''
```

Max value is 8 = 4 + 4 (opposite corners `nt1` <-> `nt25`), which lines up with `self.max_distance = 8` set in `_init_map`. The distance matrix is used by [[atsc_env]]'s spatial-discount machinery to reweight reward / neighbor contributions by graph distance.

### `_init_map`

```python
def _init_map(self):
    self.node_names = ['nt%d' % i for i in range(1, 26)]
    self.n_node = 25
    self._init_neighbor_map()
    # for spatial discount
    self._init_distance_map()
    self.max_distance = 8
    self.phase_map = LargeGridPhase()
    self.state_names = STATE_NAMES
```

This is the central wiring method invoked from `TrafficSimulator.__init__` (see [[atsc_env]]). In order:

1. `self.node_names` = `['nt1', 'nt2', ..., 'nt25']` — row-major naming consistent with the SUMO net file (`exp.net.xml`) and the connection/edge generator in `build_file.py`.
2. `self.n_node = 25` — referenced inside `_init_neighbor_map` for the mask allocation.
3. `_init_neighbor_map()` — builds `self.neighbor_map` (dict) and `self.neighbor_mask` (25x25 numpy).
4. `_init_distance_map()` — builds `self.distance_mask` (25x25 numpy).
5. `self.max_distance = 8` — used for normalization of the distance mask.
6. `self.phase_map = LargeGridPhase()` — the 5-phase signal program registry.
7. `self.state_names = STATE_NAMES` (= `['wave']`) — only lane-wave features are exposed.

### Greedy controller methods

There are no greedy / heuristic controller methods *on the env class* itself. The `LargeGridController` class above is a sibling, used only by the `__main__` block.

### `_init_sim_config` (route file selection)

```python
def _init_sim_config(self, seed):
    return gen_rou_file(self.data_path,
                        self.peak_flow1,
                        self.peak_flow2,
                        self.init_density,
                        seed=seed,
                        thread=self.sim_thread)
```

- `self.data_path` is set by the parent `TrafficSimulator` from the `data_path` config key.
- `self.sim_thread` is the simulator-thread index (default ~0 if `port=0`); it disambiguates parallel SUMO instances by file name.
- `seed` parameterizes the *random demand profile* inside `output_flows` — it controls how OD pairs are sampled, the noise on peak-flow envelopes, and `init_routes(density)`'s random orig-dest selection.

`gen_rou_file` from `envs/large_grid_data/build_file.py` is:

```python
def gen_rou_file(path, peak_flow1, peak_flow2, density, seed=None, thread=None):
    if thread is None:
        flow_file = 'exp.rou.xml'
    else:
        flow_file = 'exp_%d.rou.xml' % int(thread)
    write_file(path + flow_file, output_flows(peak_flow1, peak_flow2, density, seed=seed))
    sumocfg_file = path + ('exp_%d.sumocfg' % thread)
    write_file(sumocfg_file, output_config(thread=thread))
    return sumocfg_file
```

So the function *writes* the route file and the sumocfg, and returns the sumocfg path that the parent then hands to SUMO.

### Plotting helpers

```python
def plot_stat(self, rewards):
    self.state_stat['reward'] = rewards
    for name, data in self.state_stat.items():
        fig = plt.figure(figsize=(8, 6))
        plot_cdf(data)
        plt.ylabel(name)
        fig.savefig(self.output_path + self.name + '_' + name + '.png')


def plot_cdf(X, c='b', label=None):
    sorted_data = np.sort(X)
    yvals = np.arange(len(sorted_data))/float(len(sorted_data)-1)
    plt.plot(sorted_data, yvals, color=c, label=label)
```

- `plot_stat` is meant to be called after a run; it appends the reward series to `self.state_stat` and saves a per-statistic empirical CDF figure to `output_path`.
- `plot_cdf` is a free function — it sorts the data and plots cumulative fraction `i / (N-1)` vs sorted value. Note the `N-1` denominator: the largest sample lands at `1.0`, the smallest at `0.0`. If `len(X) == 1` this divides by zero.

## How the seed parameterizes which `exp_NNNN.rou.xml` gets used

Important subtlety: the on-disk filename is **not** controlled by the seed — it is controlled by `self.sim_thread`. The seed only changes the *contents* of the file (which OD pairs and timing samples get emitted by `output_flows`).

Concretely:

- `_init_sim_config(seed)` always writes to `exp_<self.sim_thread>.rou.xml` and `exp_<self.sim_thread>.sumocfg`. Successive resets with different seeds overwrite the same files (per-thread).
- The pre-generated `exp_<N>.rou.xml` and `exp_<N>.sumocfg` files visible in `envs/large_grid_data/` (e.g. `exp_0`, `exp_8000`, `exp_8001`, `exp_8380`, `exp_8519`, `exp_8816`, `exp_10244`, ...) are *leftover artifacts* from prior runs at various `sim_thread` indices, **not** a seed-indexed lookup table. The numbering in their filenames corresponds to past thread/port assignments — not to the `seed` argument.
- The test-seed mechanism comes from `init_test_seeds` in [[atsc_env]]: at evaluation time `env.reset(test_ind=i)` picks the i-th element of `self.test_seeds` and feeds it into `_init_sim`, which calls `_init_sim_config(seed)`, which re-generates the route file from scratch with that seed.

So: seeds are *content-level* randomness, threads are *file-level* identity. The naming `exp_NNNN.rou.xml` does **not** map seed→file.

## Bugs / oddities

- **No module-level `NEIGHBOR_MAP` constant despite the contract** — neighbor relationships are hand-coded inside `_init_neighbor_map`. The header comment block at lines 52-58 mentions documentation-only methods; the file is otherwise minimal. The prompt assumed a `NEIGHBOR_MAP` constant exists; it does not.
- **Inconsistent neighbor ordering** between edge nodes (variable `[N, E, W]` vs `[N, S, W]` orderings depending on which side of the grid the node lives on) and interior nodes (uniform `[N, E, S, W]`). The `neighbor_mask` matrix masks out ordering, so downstream graph-conv layers are unaffected; but any code path that indexes into `neighbor_map['ntX'][0]` expecting a fixed cardinal direction will silently misalign.
- **Method name mismatch in the docstring** — the in-code comment on line 102 says "distance_mask is a 25x25 matrix (adjacent matrix)" but it is *not* an adjacency matrix, it is a Manhattan-distance matrix. Misleading comment.
- **Method is `_init_distance_map` but conventionally [[atsc_env]] calls it `_init_distance_mask` in some siblings** — confirm what the parent class actually calls. Here, `_init_map` calls `_init_distance_map()` directly, so the inconsistent name does not break this env, but copy-pasting to another env subclass could.
- **`plot_cdf` divides by `len(sorted_data) - 1`** — zero-division when called with a single-element array.
- **`plot_stat` constructs file paths with `self.output_path + self.name + '_' + name + '.png'`** — uses string concatenation rather than `os.path.join`. If `self.output_path` does not end with `/` the file is dropped into the parent directory with a mangled filename.
- **`LargeGridController.greedy` ignores `node_name`** — fine for the homogeneous grid, but the signature suggests heterogeneity that isn't implemented.
- **`sns.set_color_codes()` runs at import time** — global mutation, surprising for downstream code that imports `LargeGridEnv` purely for env access without intending to use seaborn.
- **Hard-coded `PHASE_NUM = 5` returned for every node** — `_get_node_phase_id` is a constant function. If the phase program ever needs to differ across nodes (e.g. on-ramp vs intersection), the env will silently apply the wrong action mask.
- **The `__main__` block** at lines 172-198 calls `env.train_mode = False` and runs the greedy controller for `env.test_num` episodes, but it never reseeds explicitly between episodes — relies on `reset(test_ind=i)` to drive seed selection via `init_test_seeds`. If `test_seeds` is empty (config-dependent), this loop is a silent no-op.
- **`exp_0.rou.xml` is also present in the data dir** — and gets *overwritten* every time someone runs with `thread=0` (default `port=0` -> `sim_thread=0` typically). Workers sharing a `sim_thread` will race on the same file.

See also [[atsc_env]] for parent-class semantics, [[BayesianGraph]] for downstream use of `neighbor_mask` / `distance_mask`, and [[real_net_env]] for the corresponding real-network counterpart.
