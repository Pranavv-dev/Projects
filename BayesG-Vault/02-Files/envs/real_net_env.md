# real_net_env.py — Monaco Real-World Network Env

Line-by-line walkthrough of [[real_net_env.py]] (255 lines). This env wires a real Monaco SUMO scenario into the [[atsc_env|TrafficSimulator]] base class. Unlike the synthetic [[large_grid_env|5x5 grid]], topology is irregular: 28 traffic-light agents with **heterogeneous phase counts** (2 to 6 phases per node) and **multi-segment lane detectors**.

## Imports

```python
import configparser
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import time
from collections import deque
from envs.atsc_env import PhaseMap, PhaseSet, TrafficSimulator
from envs.real_net_data.build_file import gen_rou_file

sns.set_color_codes()
```

Standard imports. Three things to note:

- `deque` is used for the BFS distance map (line 154).
- The env inherits `PhaseMap`, `PhaseSet`, and `TrafficSimulator` from [[atsc_env]] — the variable-phase machinery lives in `PhaseSet`.
- `gen_rou_file` from [[build_file]] is the per-seed route file generator (used in `_init_sim_config`).
- `sns.set_color_codes()` is a side-effect on import; only `plot_cdf` actually uses seaborn, and even then only indirectly.

## Module constants

### `STATE_NAMES`

```python
STATE_NAMES = ['wave']
```

Only `'wave'` is enabled — i.e. `getLastStepVehicleNumber` normalized by lane capacity (see [[atsc_env]] `_measure_state_step`, line ~558). `'wait'` is not used here even though `atsc_env` supports it.

### `NODES` — node → (phase_key, neighbors)

Lines 18-45. 28 traffic-light controlled intersections. Each value is a `(phase_key, neighbor_list)` tuple. Examples:

```python
NODES = {'10026': ('6.0', ['9431', '9561', 'cluster_9563_9597', '9531']),
         '8794':  ('4.0', ['cluster_8985_9609', '9837', '9058', 'cluster_9563_9597']),
         '8940':  ('2.1', ['9007', '9429']),
         ...
         'joinedS_1': ('3.2', ['9531', '9429'])}
```

Key observations:

- Node names are a mix of plain IDs (`'8794'`), SUMO junction clusters (`'cluster_9563_9597'`), and manual joins (`'joinedS_0'`, `'joinedS_1'`).
- Three nodes have **empty neighbor lists**: `'8996'`, `'9433'`, `'9480'`, `'cluster_8751_9630'` — these are isolated leaf agents (no edges in the agent communication graph). BFS from them yields distance only to self; max-distance calculation still works because `_bfs` is called for every starting node.
- Wait, `'cluster_8751_9630'` IS referenced as a neighbor of `'cluster_9389_9689'` (line 42) — so it's asymmetrically connected (others reach it, but its own neighbor list is empty). See [[#bugs-oddities]].
- The phase key (e.g. `'4.0'`, `'2.1'`, `'6.1'`) decimal suffix disambiguates same-#-phases nodes that have differing controlled-lane counts. `'4.0'`, `'4.1'`, `'4.2'` are all 4-phase but have 12, 11, 12 lanes respectively.

### `PHASES` — phase signal strings per phase key

```python
PHASES = {'4.0': ['GGgrrrGGgrrr', 'rrrGGgrrrGGg', 'rrGrrrrrGrrr', 'rrrrrGrrrrrG'],
          '4.1': ['GGgrrGGGrrr', 'rrGrrrrrrrr', 'rrrGgrrrGGg', 'rrrrGrrrrrG'],
          '4.2': ['GGGGrrrrrrrr', 'GGggrrGGggrr', 'rrrGGGGrrrrr', 'grrGGggrrGGg'],
          '2.0': ['GGrrr', 'ggGGG'],
          '2.1': ['GGGrrr', 'rrGGGg'],
          '2.2': ['Grr', 'gGG'],
          '2.3': ['GGGgrr', 'GrrrGG'],
          '2.4': ['GGGGrr', 'rrrrGG'],
          '2.5': ['Gg', 'rG'],
          '2.6': ['GGGg', 'rrrG'],
          '2.7': ['GGg', 'rrG'],
          '3.0': ['GGgrrrGGg', 'rrGrrrrrG', 'rrrGGGGrr'],
          '3.1': ['GgrrGG', 'rGrrrr', 'rrGGGr'],
          '3.2': ['GGGGrrrGG', 'rrrrGGGGr', 'GGGGrrGGr'],
          '5.0': ['GGGGgrrrrGGGggrrrr', 'grrrGrrrrgrrGGrrrr', 'GGGGGrrrrrrrrrrrrr',
                  'rrrrrrrrrGGGGGrrrr', 'rrrrrGGggrrrrrggGg'],
          '6.0': ['GGGgrrrGGGgrrr', 'rrrGrrrrrrGrrr', 'GGGGrrrrrrrrrr', 'rrrrrrrrrrGGGG',
                  'rrrrGGgrrrrGGg', 'rrrrrrGrrrrrrG'],
          '6.1': ['GGgrrGGGrrrGGGgrrrGGGg', 'rrGrrrrrrrrrrrGrrrrrrG', 'GGGrrrrrGGgrrrrGGgrrrr',
                  'GGGrrrrrrrGrrrrrrGrrrr', 'rrrGGGrrrrrrrrrrrrGGGG', 'rrrGGGrrrrrGGGgrrrGGGg']}
```

**Heterogeneous phase counts**: the first digit in the key is the number of phases (the action-space size for that node). The distribution:

| #phases | keys | count |
|---|---|---|
| 2 | 2.0–2.7 | 8 |
| 3 | 3.0–3.2 | 3 |
| 4 | 4.0–4.2 | 3 |
| 5 | 5.0 | 1 |
| 6 | 6.0–6.1 | 2 |

So each agent's discrete action space size `n_a` ∈ {2, 3, 4, 5, 6}. This is non-uniform — see [[#heterogeneous-phase-implications]].

Each phase is a SUMO-style signal string (`G` = green priority, `g` = green low-prio, `r` = red). String length = number of controlled lanes/links for that TLS. The string length differs per key (e.g. `'2.5'` has 2 links, `'6.1'` has 22).

### `EXTENDED_LANES` — multi-segment detector aggregation

Lines 68-100. A dict keyed by `(node_name, lane_name)` mapping to a list of additional lane names that should be aggregated into the same logical "detector".

```python
EXTENDED_LANES = {('9431', '10099#3_1'): ['10099#1_1', '10099#2_1'],
                  ('10026', '-10046#0_1'): ['-10046#1_1'],
                  ...
                  ('joinedS_1', '10181#2_1'): ['10181#1_1']
                  }
```

**Purpose**: in the real-Monaco network, what's logically a "single approach lane" is often split across multiple SUMO edges (because of mid-block junctions, ramps, or upstream segments without intersections). When measuring the `wave` state (vehicle count) or `queue` reward, the env sums vehicle counts across all segments listed for each `(node, lane)` pair. This makes the detection range longer than just the immediate stop-line lane.

Consumed in [[atsc_env]] `_init_nodes` lines 381-388:

```python
cur_ilds_in = [lane_name]
if (node_name, lane_name) in self.extended_lanes:
    cur_ilds_in += self.extended_lanes[(node_name, lane_name)]
ilds_in.append(cur_ilds_in)
cur_cap = 0
for ild_name in cur_ilds_in:
    cur_cap += self.sim.lane.getLength(ild_name)
lanes_cap.append(cur_cap/float(VEH_LEN_M))
```

So `node.ilds_in[k]` is a **list of lane names**, not a single string (whereas in the synthetic grid env it's a single string). Capacity is the sum of segment lengths divided by vehicle length. Per-step wave count is the sum of `getLastStepVehicleNumber` over all segments (see `_measure_state_step` lines 558-563), then normalized by `lanes_capacity[k]`.

Note: the key `('9431', '10099#3_1')` is **defined twice** (lines 68 and 76) — the later overrides the earlier. The first gives 2 extensions, the second gives 3 (adds `'gneE14_0'`). See [[#bugs-oddities]].

## class `RealNetPhase(PhaseMap)`

```python
class RealNetPhase(PhaseMap):
    def __init__(self):
        self.phases = {}
        for key, val in PHASES.items():
            self.phases[key] = PhaseSet(val)
```

Wraps `PHASES` into a `{key: PhaseSet}` dict where each `PhaseSet` (from [[atsc_env]]) precomputes `num_phase`, `num_lane`, and `red_lanes`. The parent `PhaseMap.get_phase_num(phase_id)` then returns the phase count for a given key — this is how the env discovers each node's `n_a`.

## class `RealNetController` (greedy baseline)

Lines 109-142. Not used during RL training; only invoked in the `__main__` block as a baseline.

```python
def greedy(self, ob, node_name):
    phases = PHASES[NODES[node_name][0]]
    flows = []
    node = self.nodes[node_name]
    for phase in phases:
        wave = 0
        visited_ilds = set()
        for i, signal in enumerate(phase):
            if signal == 'G':
                lane = node.lanes_in[i]
                ild = lane
                if ild not in visited_ilds:
                    j = node.ilds_in.index(ild)
                    wave += ob[j]
                    visited_ilds.add(ild)
        flows.append(wave)
    return np.argmax(np.array(flows))
```

For each candidate phase, sum wave observations over lanes flagged with capital `G` (priority green) — lowercase `g` is ignored. Picks the phase with max total wave (greedy phase selection).

**Bug**: `node.ilds_in.index(ild)` will fail in `atsc_real_net` mode because `node.ilds_in` is a list of *lists*, not a list of lane strings. See [[#bugs-oddities]].

## class `RealNetEnv(TrafficSimulator)`

### `__init__` (146-148)

```python
def __init__(self, config, port=0, output_path='', is_record=False, record_stat=False):
    self.flow_rate = config.getint('flow_rate')
    super().__init__(config, output_path, is_record, record_stat, port=port)
```

Reads `flow_rate` (vehicles/hour scaling factor for [[build_file]]'s `output_flows`) and delegates to [[atsc_env|TrafficSimulator]]. The parent `__init__` calls `_init_map` → which builds neighbor map, distance map, phase map. Note `port` is passed as a kwarg.

### `_init_neighbor_map` (171-178) — hardcoded, not BFS

```python
def _init_neighbor_map(self):
    self.neighbor_map = dict([(key, val[1]) for key, val in NODES.items()])
    self.neighbor_mask = np.zeros((self.n_node, self.n_node)).astype(int)
    for i, node_name in enumerate(self.node_names):
        for nnode in self.neighbor_map[node_name]:
            ni = self.node_names.index(nnode)
            self.neighbor_mask[i, ni] = 1
    logging.info('neighbor mask:\n %r' % self.neighbor_mask)
```

**Hardcoded** — pulls neighbors straight from the `NODES` dict (column 1 of each tuple). Builds an `(n_node, n_node)` adjacency mask. `node_names.index(...)` is O(n) per lookup but fine at n=28.

### `_init_distance_map` (180-185) + `_bfs` (150-166) — BFS

```python
def _bfs(self, i):
    d = 0
    self.distance_mask[i, i] = d
    visited = [False]*self.n_node
    que = deque([i])
    visited[i] = True
    while que:
        d += 1
        for _ in range(len(que)):
            node_name = self.node_names[que.popleft()]
            for nnode in self.neighbor_map[node_name]:
                ni = self.node_names.index(nnode)
                if not visited[ni]:
                    self.distance_mask[i, ni] = d
                    visited[ni] = True
                    que.append(ni)
    return d

def _init_distance_map(self):
    self.distance_mask = -np.ones((self.n_node, self.n_node)).astype(int)
    self.max_distance = 0
    for i in range(self.n_node):
        self.max_distance = max(self.max_distance, self._bfs(i))
    logging.info('distance mask:\n %r' % self.distance_mask)
```

Standard level-order BFS. Distance entries default to `-1` (unreachable) and `distance_mask[i,i] = 0`. Returns the final `d` value reached by each BFS — used to track `max_distance`.

**Subtle bug** in `_bfs`: the final `d` increment happens once after the last level's nodes are popped but before checking that anything was added — so `_bfs` returns `max_dist + 1` rather than `max_dist`. The `while que:` loop only re-enters if new nodes were queued, so if no new nodes were added in the last `for _ in range(len(que))` iteration, the final `d` was incremented but no level was actually processed. See [[#bugs-oddities]].

Also: since the neighbor graph may be asymmetric (e.g. `cluster_8751_9630` has empty out-neighbors), `_bfs` from that node only finds itself; `max_distance` is still set globally via the max over all roots.

### `_init_map` (187-195) — node setup orchestrator

```python
def _init_map(self):
    self.node_names = sorted(list(NODES.keys()))
    self.n_node = len(self.node_names)
    self._init_neighbor_map()
    self._init_distance_map()
    self.phase_map = RealNetPhase()
    self.phase_node_map = dict([(key, val[0]) for key, val in NODES.items()])
    self.state_names = STATE_NAMES
    self.extended_lanes = EXTENDED_LANES
```

- `node_names` are **sorted alphabetically** — so e.g. `'10026'` comes before `'8794'` lexically (digits sort before letters). This determines the canonical agent ordering across the env, neighbor mask, distance mask, action list, and obs list.
- Builds `phase_node_map: node_name -> phase_key` (e.g. `'10026' -> '6.0'`). Used by `_get_node_phase_id` which the parent class calls to find each node's `n_a`.
- Sets `self.extended_lanes` so the parent `_init_nodes` (in [[atsc_env]]) can aggregate.

### `_get_node_phase_id` (168-169)

```python
def _get_node_phase_id(self, node_name):
    return self.phase_node_map[node_name]
```

Just a dict lookup — returns the phase key string (e.g. `'6.0'`).

### `_init_sim_config` (197-202) — route file selection per seed

```python
def _init_sim_config(self, seed):
    # comment out to call build_file.py
    return gen_rou_file(self.data_path,
                        self.flow_rate,
                        seed=seed,
                        thread=self.sim_thread)
```

Delegates to [[build_file]] `gen_rou_file`, which:

1. Writes a fresh `most_<thread>.rou.xml` with traffic flows scaled by `flow_rate`, randomized by `seed`.
2. Writes a `most_<thread>.sumocfg` referencing that route file.
3. Returns the sumocfg path for the parent class to launch SUMO.

So **every reset with a new seed regenerates routes**. This is unlike pre-cooked scenarios that ship fixed `.rou.xml` files. Thread ID enables parallel rollouts without file collisions.

### State/reward — inherited, no overrides

`RealNetEnv` does **not** override `_measure_state_step` or `_measure_reward_step`. The parent [[atsc_env]] dispatches on `self.name == 'atsc_real_net'` (must be set in the config) and handles:

- **Multi-segment wave**: `sum(getLastStepVehicleNumber(seg) for seg in ild) / lanes_capacity[k]` (lines 560-563 of atsc_env).
- **Multi-segment queue**: `sum(getLastStepHaltingNumber(seg) for seg in ild)` (lines 613-615).
- **Wait** (if used): only `ild[0]` (the head segment) is queried for vehicle IDs (line 575), because waiting time only matters at the stop line.

### `plot_stat` (204-210) and `plot_cdf` (213-216)

Plots empirical CDFs of the per-state stats and rewards over the run. Minor: `plot_cdf` divides by `len(sorted_data) - 1` (Bessel-like) — would crash on single-element input.

### `__main__` (219-255)

Stand-alone test harness:

- Loads `./config/config_test_real.ini`.
- Instantiates `RealNetEnv` with `port=2`, recording on.
- Sets test seeds `[10000, 20000, ..., 100000]`.
- Runs the greedy `RealNetController` baseline for 10 episodes.
- Calls `plot_stat`, `collect_tripinfo`, `output_data`.

Calls `env.terminate()` twice (lines 248 and 252) with a 2-second sleep between — defensive cleanup.

## Heterogeneous-phase implications

`n_a` varies per agent (∈ {2, 3, 4, 5, 6}). The codebase handles this via:

1. **`n_a_ls`** in [[atsc_env]] (line 348-355): the parent's `_init_nodes` builds a per-agent list of action-space sizes. Each `Node.n_a = phase_num` is set from `phase_map.get_phase_num(phase_id)`.
2. **`_init_policy`** (line 406): returns a *list* of uniform distributions of varying length: `[np.ones(self.n_a_ls[i]) / self.n_a_ls[i] for i in range(self.n_agent)]`. So policies are stored per-agent, not as a fixed-shape tensor.
3. **Action application** (`_set_phase`, lines 644-648): iterates per-node, calling `phase_map.get_phase(phase_id, action)` for *that* node's phase_id. The action is an integer index into the heterogeneous phase list — so the overall action vector has type `list[int]` where each element's valid range differs.
4. **Agent networks** (likely in [[agents]]): downstream PPO/IA2C/etc. must allocate per-agent policy heads sized to `n_a_ls[i]`. There is no shared categorical head with fixed size. Search [[agents]] for `n_a_ls` to confirm.
5. **Multi-segment ilds** (`ilds_in`) means **`num_state` varies per agent too** (see line 475: `node.num_state = len(node.ilds_in)`). Observation lengths are per-agent — so state-space is also heterogeneous, not just action-space.

This makes `RealNetEnv` a useful stress test for any agent class that assumed homogeneous shapes (as the synthetic grid env provides). Algorithms that share parameters across agents must either pad to `max(n_a_ls)` and mask, or use parameter-free architectures (e.g. independent learners).

## Bugs / oddities

1. **Duplicate `EXTENDED_LANES` key**. Lines 68 and 76 both define `('9431', '10099#3_1')`:

    ```python
    ('9431', '10099#3_1'): ['10099#1_1', '10099#2_1'],   # line 68
    ...
    ('9431', '10099#3_1'): ['10099#1_1', '10099#2_1', 'gneE14_0'],  # line 76
    ```

    The second entry wins (Python dict literal evaluation). The first is dead code. Likely an authoring slip when adding `'gneE14_0'`.

2. **Asymmetric neighbor graph**. `'cluster_8751_9630'` is named as a neighbor of `'cluster_9389_9689'` (line 42), but its own neighbor list (line 39) is empty. Also `'8996'`, `'9433'`, `'9480'` have empty neighbor lists — they appear nowhere as neighbors either, so they are truly isolated nodes. The `neighbor_mask` is therefore *not symmetric*. Downstream code that assumes symmetry (e.g. for undirected message passing) may behave inconsistently.

3. **`_bfs` returns `max_dist + 1`**. In the loop, `d` is incremented at the top of each `while que:` iteration before processing that level. After processing the last reachable level, `que` becomes empty and the loop exits — but the last increment is "wasted" because no new nodes were added. So if the true graph diameter from `i` is `D`, this returns `D+1`. Since `max_distance` is only used for normalization (likely), the off-by-one may not matter functionally but is technically incorrect.

4. **`RealNetController.greedy` indexing assumes flat ilds_in**. In real-net mode, `node.ilds_in` is a list-of-lists, but the controller does:

    ```python
    lane = node.lanes_in[i]
    ild = lane
    j = node.ilds_in.index(ild)
    wave += ob[j]
    ```

    `node.ilds_in.index(ild)` searches for a string in a list of lists — will always raise `ValueError`. The baseline can never have run as-is in real-net mode (it may have been written for the synthetic grid case and never updated). Also `node.lanes_in` isn't set anywhere in [[atsc_env]] that I've seen (it uses `lanes_in` only as a temporary variable inside `_init_nodes`, never stored on the node).

5. **`sns.set_color_codes()` at module import**. Side-effect on a global library. Not breaking, but un-hygienic.

6. **No yellow-phase override**. The env relies on the parent's `_get_node_phase` (atsc_env line 268+) to insert yellow transitions by comparing `prev_phase` and `cur_phase` strings of differing lengths. Because phase strings are *per-node* (different lengths across nodes), but `prev_action` and current action are both indices into the *same* node's phase set, the comparison is intra-node — so this is fine.

7. **`'most.rou.xml'` vs `'most_<thread>.rou.xml'` branch is unreachable from `RealNetEnv`** since `self.sim_thread` is always set (parent passes thread). The `if thread is None` branch in `gen_rou_file` is dead code in this context.

8. **`flow_rate` config**: read as int. `output_flows` in [[build_file]] multiplies base flows by `flow_rate / 1000.0` typically — so `flow_rate=1000` means baseline. Confirm in [[build_file]].

9. **`is_record` and `record_stat` argument order swap**. The parent's `__init__` signature is `(config, output_path, is_record, record_stat, port=port)` — passed positionally here. Make sure the parent class hasn't reordered these. Worth grepping [[atsc_env]] `class TrafficSimulator.__init__`.

## Related notes

- [[atsc_env]] — parent class providing SUMO/TraCI integration, multi-segment ild handling, yellow-phase insertion.
- [[large_grid_env]] — synthetic 5x5 grid counterpart; homogeneous phases.
- [[build_file]] — Monaco route/config file generation.
- [[agents]] — downstream policies that must accommodate `n_a_ls`.
