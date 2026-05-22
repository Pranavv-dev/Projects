# Line-by-Line: `atsc_env.py` (lines 1–330) — Init / Step / Reset

> Source: `/tmp/BayesG/envs/atsc_env.py`
> Scope: lines 1–330 — the module preamble, `PhaseSet`, `PhaseMap`, `Node` skeleton, and the upper half of `TrafficSimulator` (everything up through the start of `_get_state`).

Related notes: [[atsc_env-overview]], [[PhaseSet-PhaseMap]], [[TrafficSimulator-init]], [[TraCI-cheatsheet]], [[Reward-construction]].

---

## Module imports & globals

```python
import logging
import numpy as np
import pandas as pd
import subprocess
from sumolib import checkBinary
import time
import traci
import xml.etree.cElementTree as ET
import random
import socket
```

- `logging` — used for `logging.info`, `logging.warning`, `logging.error` throughout (e.g. line 92, 119, 174).
- `numpy as np` — for `np.ceil`, `np.sum`, and (later) state vectors.
- `pandas as pd` — used in `output_data()` (lines 180–185) to dump CSVs.
- `subprocess` — used to (a) `rm` trip XML files (line 137), and (b) kill leftover SUMO via `fuser` in `terminate()` (line 257).
- `from sumolib import checkBinary` — imported but **not used in lines 1–330** (it's typically used to resolve the SUMO binary path inside `_init_sim`, which is below the read window).
- `time` — imported but not visibly used in this slice (likely used in `_init_sim` or `_simulate`).
- `traci` — the SUMO Python control API. The actual `traci.*` calls live below line 330, but the module is the workhorse used by `_set_phase`, `_simulate`, `_measure_state_step`, etc.
- `xml.etree.cElementTree as ET` — parses `<scenario>_<agent>_trip.xml` in `collect_tripinfo` (line 123).
- `random` — imported but not used in this slice.
- `socket` — used by the (referenced but not visible) `_find_free_port` helper at line 87.

### Globals

```python
DEFAULT_PORT = 8000
SEC_IN_MS = 1000
VEH_LEN_M = 7.5 # effective vehicle length
QUEUE_MAX = 10
```

- `DEFAULT_PORT = 8000` — base TraCI port. When a `port` index is passed to `__init__`, the actual port becomes `DEFAULT_PORT + port` (line 89), so e.g. `port=3` -> TCP 8003. Used to avoid collisions in parallel rollouts.
- `SEC_IN_MS = 1000` — millisecond conversion (not used in this slice; likely used in `_simulate`).
- `VEH_LEN_M = 7.5` — "effective vehicle length" including spacing. Standard SUMO heuristic for converting queue length in meters to vehicle count. Used downstream in lane-capacity / wave-density computations.
- `QUEUE_MAX = 10` — maximum queue length normalizer / cap. Not referenced in this slice.

---

## `PhaseSet` class — phase encoding

```python
class PhaseSet:
    def __init__(self, phases):
        self.num_phase = len(phases)
        self.num_lane = len(phases[0])
        self.phases = phases
        self._init_phase_set()
```

- `phases` is a list of phase **strings**, one per discrete action. Each string is a per-link signal code using SUMO's link-state alphabet — `G`/`g` = green, `r` = red, `y` = yellow, `s` = priority straight, etc.
- `num_phase` = number of available actions at this intersection.
- `num_lane` = length of any phase string (assumed uniform), which equals the number of controlled links at the junction. (Note the comment in `PhaseMap.get_lane_num` later: *"the lane number is link number"* — they're using "lane" loosely for SUMO's link index.)

```python
@staticmethod
def _get_phase_lanes(phase, signal='r'):
    phase_lanes = []
    for i, l in enumerate(phase):
        if l == signal:
            phase_lanes.append(i)
    return phase_lanes
```

- Returns the list of link indices in `phase` that have the given `signal` character. Default `'r'` -> indices of red lanes.
- Note: this does case-sensitive matching, so `'G'` and `'g'` are not interchangeable (this is important — SUMO uses lowercase `g` for "minor green / yield" and uppercase `G` for "major green").

```python
def _init_phase_set(self):
    self.red_lanes = []
    for phase in self.phases:
        self.red_lanes.append(self._get_phase_lanes(phase))
```

- Precomputes `red_lanes[action_idx]` -> list of link indices that are red under that phase. Used to know which lanes are blocked (for wait / queue accumulation) per discrete action.

---

## `PhaseMap` class — phase mapping per-node

```python
class PhaseMap:
    def __init__(self):
        self.phases = {}
```

- A registry of `phase_id -> PhaseSet`. The `phase_id` is the intersection's *topology type* (e.g. `"4-arm-2-lane"`), allowing multiple nodes that share geometry to share a `PhaseSet`. Subclasses (in the scenario-specific env, e.g. Grid) populate `self.phases` keys.

```python
def get_phase(self, phase_id, action):
    # phase_type is either green or yellow
    return self.phases[phase_id].phases[int(action)]
```

- Returns the SUMO phase **string** for `(phase_id, action)`. Note the docstring comment is misleading — this returns the *green* phase string; `_get_node_phase` synthesizes yellow strings on the fly.

```python
def get_phase_num(self, phase_id):
    return self.phases[phase_id].num_phase

def get_lane_num(self, phase_id):
    # the lane number is link number
    return self.phases[phase_id].num_lane

def get_red_lanes(self, phase_id, action):
    # the lane number is link number
    return self.phases[phase_id].red_lanes[int(action)]
```

- Thin pass-throughs. Note `int(action)` casting — actions can arrive as `np.int64`, `float`, etc.

---

## `Node` class — per-intersection bookkeeping (skeleton only)

```python
class Node:
    def __init__(self, name, neighbor=[], control=False):
        self.control = control # disabled
        self.ilds_in = [] # for state
        self.lanes_capacity = []
        self.fingerprint = [] # local policy
        self.name = name
        self.neighbor = neighbor
        self.num_state = 0 # wave and wait should have the same dim
        self.wave_state = [] # local state
        self.wait_state = [] # local state
        self.phase_id = -1
        self.n_a = 0
        self.prev_action = -1
```

- `control = False` — flagged as "disabled" in the comment; probably a legacy flag for whether this node is actuated.
- `ilds_in` — list of **induction loop detector** IDs (SUMO `e1` detectors) feeding into the node. These are how `wave` and `wait` are measured downstream via `traci.lanearea.*` / `traci.inductionloop.*`.
- `lanes_capacity` — per-lane capacity in vehicles (used to normalize density into [0,1] for the wave feature).
- `fingerprint` — the "policy snapshot" used by IA2C-FP for partial observability of neighbors' policies.
- `neighbor` — list of neighbor node names (graph adjacency).
- `num_state` — dimensionality of *one* feature stream; "wave and wait should have the same dim" hints both features are vectors of length `len(ilds_in)`.
- `phase_id` — key into `PhaseMap`.
- `n_a` — number of discrete actions for this node.
- `prev_action = -1` — sentinel for "no previous action" (the yellow logic in `_get_node_phase` treats `< 0` as "first step, no transition needed").

> Mutable default argument trap: `neighbor=[]` is a Python footgun (a shared list across instances). Not exercised here because `neighbor` is reassigned per construction, but worth flagging — see [[Bugs / oddities]] below.

---

## `TrafficSimulator.__init__` — every config field

```python
def __init__(self, config, output_path, is_record, record_stats, port=None):
    self.name = config.get('scenario')
    self.seed = config.getint('seed')
    self.control_interval_sec = config.getint('control_interval_sec') # Grid : control_interval_sec = 5
    self.yellow_interval_sec = config.getint('yellow_interval_sec') # Grid : yellow_interval_sec = 2
    self.episode_length_sec = config.getint('episode_length_sec')
    self.T = np.ceil(self.episode_length_sec / self.control_interval_sec)
```

- `name` — scenario string (e.g. `"large_grid"`), used for output filenames.
- `seed` — base RNG seed (incremented every reset — see line 200).
- `control_interval_sec` (Grid default: 5s) — total decision interval; **yellow + green = control interval**.
- `yellow_interval_sec` (Grid default: 2s) — duration the yellow buffer phase runs.
- `episode_length_sec` — total simulated seconds per episode.
- `T = ceil(episode_length_sec / control_interval_sec)` — number of agent decision steps per episode. Stored as a `numpy` float (it's a `np.ceil` return), which is a minor type oddity — downstream loops that compare integers will still work but it's not strictly `int`.

```python
# Use random port if none specified
if port is None:
    self.port = self._find_free_port()
else:
    self.port = DEFAULT_PORT + port

self.sim_thread = self.port  # Use port as thread identifier
logging.info(f"Initializing SUMO environment with port {self.port}")
```

- `_find_free_port()` is not shown in this slice — almost certainly uses the imported `socket` module to grab a kernel-assigned ephemeral port.
- `sim_thread` is just an alias of `port`; the legacy name implies an older API used thread indices instead of ports.

```python
self.obj = config.get('objective')
self.data_path = config.get('data_path')
self.agent = config.get('agent')
self.coop_gamma = config.getfloat('coop_gamma')
self.cur_episode = 0
self.norms = {'wave': config.getfloat('norm_wave'),
              'wait': config.getfloat('norm_wait')}
self.clips = {'wave': config.getfloat('clip_wave'),
              'wait': config.getfloat('clip_wait')}
self.coef_wait = config.getfloat('coef_wait')
self.train_mode = True
test_seeds = config.get('test_seeds').split(',')
test_seeds = [int(s) for s in test_seeds]
```

- `obj` — reward objective code (e.g. `queue`, `wait`, mixed); consumed by `_measure_reward_step`.
- `data_path` — directory holding the SUMO `.net.xml` / `.rou.xml` files.
- `agent` — `'greedy'`, `'ia2c'`, `'ia2c_fp'`, etc. Branches state construction (lines 319–329) and reward aggregation (line 247).
- `coop_gamma` — neighbor reward mixing coefficient; **`< 0` means "use global reward"** per the branch at line 247.
- `norms = {'wave': ..., 'wait': ...}` — divisors applied to raw measurements before clipping.
- `clips = {'wave': ..., 'wait': ...}` — upper bounds after normalization.
- `coef_wait` — scalar weight balancing the "wait" term against "wave" in the composite reward.
- `train_mode = True` — set on the instance and flipped externally by the trainer when evaluating.
- `test_seeds` — comma-separated string in config, parsed into a list of ints; the eval harness picks `self.test_seeds[test_ind]` (line 193).

```python
self._init_map()  # define the map, neighbor relationships, and phases.
self.init_data(is_record, record_stats, output_path)
self.init_test_seeds(test_seeds)
self._init_sim(self.seed)
self._init_nodes() # Initializes traffic light nodes based on the simulation map and neighbors.
self.terminate()
```

- `_init_map()` is abstract here — subclasses (e.g. `LargeGridEnv`) define which nodes exist, neighbors, and `phase_map`. Must populate `self.node_names`, `self.neighbor_mask`, etc.
- `init_data(...)` — sets up record/stat buffers (see [[init_data]] below).
- `init_test_seeds(test_seeds)` — stores test seeds (lines 168–170).
- `_init_sim(self.seed)` — boots SUMO and connects via TraCI on `self.port`.
- `_init_nodes()` — populates the `Node` objects (presumably reading from `traci.trafficlight.getControlledLinks`, etc.); not visible in this slice.
- `self.terminate()` at the end of `__init__` — **boots SUMO just to introspect topology, then immediately tears it down.** The real episode starts later via `reset()`. See [[Bugs / oddities]] — this is intentional but easy to misread.

> Notice: `_init_neighbor_map` and `_init_distance_map` are **not defined in lines 1–330** of this file. They live either in subclasses or below the read window. What we can infer from line 152 (`self.neighbor_mask[i] == 1`) is that `neighbor_mask` is a 2D 0/1 matrix `[n_agent, n_agent]` indicating direct adjacency. A `distance_mask` typically holds integer hop counts for `coop_gamma` decay. The graph representation is implicit: `Node.neighbor` (list of names) is the per-node adjacency, and `neighbor_mask` is the matrix form for vectorized indexing in `get_neighbor_action`.

---

## `collect_tripinfo` — XML trip log ingestion

```python
def collect_tripinfo(self):
    if self.output_path is None:
        logging.warning('Output path is None, skipping trip info collection')
        return

    trip_file = self.output_path + ('%s_%s_trip.xml' % (self.name, self.agent))
    tree = ET.ElementTree(file=trip_file)
    for child in tree.getroot():
        cur_trip = child.attrib
        cur_dict = {}
        cur_dict['episode'] = self.cur_episode
        cur_dict['id'] = cur_trip['id']
        cur_dict['depart_sec'] = cur_trip['depart']
        cur_dict['arrival_sec'] = cur_trip['arrival']
        cur_dict['duration_sec'] = cur_trip['duration']
        cur_dict['wait_step'] = cur_trip['waitingCount']
        cur_dict['wait_sec'] = cur_trip['waitingTime']
        self.trip_data.append(cur_dict)
    # delete the current xml
    cmd = 'rm ' + trip_file
    subprocess.check_call(cmd, shell=True)
```

- Reads SUMO's `--tripinfo-output` XML, written when SUMO exits. The function must be called *after* SUMO terminates (i.e. between episodes), hence the docstring "has to be called externally to get complete file."
- Per-trip fields scraped: `id`, `depart`, `arrival`, `duration`, `waitingCount` (number of times waiting started), `waitingTime` (seconds waiting).
- Deletes the file via shell `rm`. Brittle — if path contains spaces or special chars this breaks; `os.remove` would be safer. See [[Bugs / oddities]].

---

## `get_fingerprint` / `get_neighbor_action` / `update_fingerprint`

```python
def get_fingerprint(self):
    policies = []
    for node_name in self.node_names:
        policies.append(self.nodes[node_name].fingerprint)
    return policies
```

- Returns the list of stored fingerprints (one per node), in `self.node_names` order.

```python
def get_neighbor_action(self, action):
    naction = []
    for i in range(self.n_agent):
        naction.append(action[self.neighbor_mask[i] == 1])
    return naction
```

- For each agent `i`, returns the actions of its *direct* graph neighbors (boolean-mask indexing on `neighbor_mask[i]`). Used by IA2C variants to condition on neighbor actions.
- Implicitly requires `action` to be a numpy array (boolean mask indexing on a python list would fail).

```python
def update_fingerprint(self, policy):
    for node_name, pi in zip(self.node_names, policy):
        self.nodes[node_name].fingerprint = pi
```

- Pushes the trainer's current per-node policy vectors back into the node objects, so neighbors can read them in `_get_state`.

---

## `init_data` — recording buffers

```python
def init_data(self, is_record, record_stats, output_path):
    self.is_record = is_record
    self.record_stats = record_stats
    self.output_path = output_path
    if self.is_record:
        self.traffic_data = []
        self.control_data = []
        self.trip_data = []
    if self.record_stats:
        self.state_stat = {}
        for state_name in self.state_names:
            self.state_stat[state_name] = []
```

- Two independent recording modes: `is_record` (per-episode CSV dumps) vs `record_stats` (running state-vector stats for normalization studies).
- `self.state_names` is assumed to be set by `_init_map`/subclass — not visible here.

---

## `init_test_seeds` / `output_data`

```python
def init_test_seeds(self, test_seeds):
    self.test_num = len(test_seeds)
    self.test_seeds = test_seeds

def output_data(self):
    if not self.is_record:
        logging.error('Env: no record to output!')
        return
    if self.output_path is None:
        logging.warning('Output path is None, skipping data output')
        return

    control_data = pd.DataFrame(self.control_data)
    control_data.to_csv(self.output_path + ('%s_%s_control.csv' % (self.name, self.agent)))
    traffic_data = pd.DataFrame(self.traffic_data)
    traffic_data.to_csv(self.output_path + ('%s_%s_traffic.csv' % (self.name, self.agent)))
    trip_data = pd.DataFrame(self.trip_data)
    trip_data.to_csv(self.output_path + ('%s_%s_trip.csv' % (self.name, self.agent)))
```

- Dumps three CSVs after a run: control actions/rewards, traffic measurements, and parsed trip info.
- Note `output_path` is concatenated directly with the filename — the caller must include the trailing `/`.

---

## `reset()` — episode reset

```python
def reset(self, gui=False, test_ind=0):
    # have to terminate previous sim before calling reset
    self._reset_state()
    if self.train_mode:
        seed = self.seed
    else:
        seed = self.test_seeds[test_ind]
    self._init_sim(seed, gui=gui)
    self.cur_sec = 0
    self.cur_episode += 1
    # initialize fingerprint
    self.update_fingerprint(self._init_policy())
    # next environment random condition should be different
    self.seed += 1
    return self._get_state()
```

- Caller-responsibility comment: `terminate()` must be called externally before `reset()`. There is no defensive guard here, so calling `reset()` twice in a row will leak a SUMO process.
- Picks training seed (auto-incrementing) vs deterministic test seed list.
- `_init_sim(seed, gui=gui)` — launches SUMO with the seed and optional GUI.
- `cur_sec = 0` — episode clock; `cur_episode += 1` — global episode counter.
- `update_fingerprint(self._init_policy())` — seeds each node's fingerprint with a default policy (e.g. uniform), so the very first `_get_state()` for IA2C-FP doesn't read stale data from a prior episode.
- `self.seed += 1` — *training* seed monotonically increases. Note this means the "deterministic for a given run" guarantee depends on never reordering episodes. Test seeds are unaffected.
- Returns the initial state — a list of per-node feature vectors.

---

## `step(action)` — yellow phase, green phase, measurement

```python
def step(self, action):
    self._set_phase(action, 'yellow', self.yellow_interval_sec)
    self._simulate(self.yellow_interval_sec)

    rest_interval_sec = self.control_interval_sec - self.yellow_interval_sec
    self._set_phase(action, 'green', rest_interval_sec)
    self._simulate(rest_interval_sec)
```

The control loop within one decision step:

1. Compute the *yellow transition* string for each node (via `_get_node_phase` with `phase_type='yellow'`), push it via TraCI, advance SUMO `yellow_interval_sec` seconds.
2. Compute the remaining time `rest = control_interval - yellow`. For Grid defaults: `5 - 2 = 3` seconds of green.
3. Push the *green* phase string, advance SUMO another `rest_interval_sec` seconds.

> Important: yellow + green durations are fixed by config; the agent doesn't choose duration, only the *next* phase index. The yellow transition is auto-derived by comparing current vs previous phase strings (see `_get_node_phase` lines 268–292).

```python
state = self._get_state()
reward = self._measure_reward_step()

done = False
if self.cur_sec >= self.episode_length_sec:
    done = True
global_reward = np.sum(reward)
```

- State is measured *after* the full control interval (yellow+green) — i.e. it reflects the post-action equilibrium.
- `reward` is a vector (per-node); `global_reward` is its scalar sum.
- `done` flag flips when simulated clock reaches `episode_length_sec`. Note this assumes `_simulate` increments `self.cur_sec` (its definition is below the window).

```python
if self.is_record:
    action_r = ','.join(['%d' % a for a in action])
    cur_control = {'episode': self.cur_episode,
                   'time_sec': self.cur_sec,
                   'step': self.cur_sec / self.control_interval_sec,
                   'action': action_r,
                   'reward': global_reward}
    self.control_data.append(cur_control)
```

- Logs the action vector as a comma-joined string of ints and the global reward.

```python
# use original rewards in test
if not self.train_mode:
    return state, reward, done, global_reward
if (self.agent == 'greedy') or (self.coop_gamma < 0):
    reward = global_reward
return state, reward, done, global_reward
```

- Test mode: always returns the per-node reward vector.
- Train mode: `greedy` agent and any agent with `coop_gamma < 0` use the **scalar global** reward in the second tuple slot; other (cooperative) agents get the per-node vector.
- `global_reward` is always returned as the fourth element for logging.

> Subtle: this means the second return value's shape (vector vs scalar) depends on agent + mode. Downstream code must be aware. See [[Bugs / oddities]].

---

## `terminate()` — SUMO cleanup + fuser fallback

```python
def terminate(self):
    """Cleanup when environment is closed"""
    if hasattr(self, 'sim'):
        try:
            self.sim.close()
            # Try to cleanup the port
            subprocess.run(['fuser', '-k', f'{self.port}/tcp'],
                         stderr=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL)
        except:
            pass
```

- `self.sim` is the TraCI connection handle (set in `_init_sim`; presumably `self.sim = traci.connect(self.port)` or `traci.start(...)`). `hasattr(self, 'sim')` guards the very first `terminate()` call.
- `self.sim.close()` — gracefully closes the TraCI connection (SUMO should then exit).
- `fuser -k {port}/tcp` — Linux-only nuclear option that kills any process still holding the TCP port. Necessary because SUMO sometimes hangs and doesn't release the port, causing the next `_init_sim` to fail.
- `except:` (bare) — swallows all exceptions including `KeyboardInterrupt`. See [[Bugs / oddities]].

---

## `_get_node_phase` — yellow transition synthesis

```python
def _get_node_phase(self, action, node_name, phase_type):
    node = self.nodes[node_name]
    cur_phase = self.phase_map.get_phase(node.phase_id, action)
    if phase_type == 'green':
        return cur_phase
    prev_action = node.prev_action
    node.prev_action = action
    if (prev_action < 0) or (action == prev_action):
        return cur_phase
    prev_phase = self.phase_map.get_phase(node.phase_id, prev_action)
    switch_reds = []
    switch_greens = []
    for i, (p0, p1) in enumerate(zip(prev_phase, cur_phase)):
        if (p0 in 'Gg') and (p1 == 'r'):
            switch_reds.append(i)
        elif (p0 in 'r') and (p1 in 'Gg'):
            switch_greens.append(i)
    if not len(switch_reds):
        return cur_phase
    yellow_phase = list(cur_phase)
    for i in switch_reds:
        yellow_phase[i] = 'y'
    for i in switch_greens:
        yellow_phase[i] = 'r'
    return ''.join(yellow_phase)
```

- For `phase_type == 'green'`: return the target phase string as-is.
- For `'yellow'`: synthesize a transition phase.
  - First call (`prev_action < 0`) or no-change (`action == prev_action`): no yellow needed; return the new phase directly.
  - **Side effect:** `node.prev_action = action` is written **before** the early return. So `prev_action` records what the controller *intends* to set next, regardless of whether a yellow transition was actually inserted. Good — that matches the semantics of "previous action".
  - Otherwise: walk the per-link signal characters in lockstep:
    - Link going green->red: mark `y` (yellow) for the transition.
    - Link going red->green: temporarily force `r` (don't release until yellow elapses).
  - If no green-to-red transitions exist (`switch_reds` empty), no yellow is required — just return the target phase.

> Edge case: what if `prev_phase` has a `y` from a previously-synthesized yellow? The `in 'Gg'` and `in 'r'` checks ignore `y`. But this is fine because `prev_action` always holds an *action index*, not a yellow phase — the yellow phase is never stored as the "previous phase."

```python
def _get_node_phase_id(self, node_name):
    # needs to be overwriteen
    raise NotImplementedError()
```

- Abstract method — subclass must map `node_name -> phase_id` (the PhaseSet topology key). Typo: `"overwriteen"`.

---

## `_get_state` — start of state construction (lines 308–329 visible)

```python
def _get_state(self):
    # hard code the state ordering as wave, wait, fp
    state = []
    self._measure_state_step()

    for node_name in self.node_names:
        node = self.nodes[node_name]
        if self.agent == 'greedy':
            state.append(node.wave_state)
        else:
            cur_state = [node.wave_state]

            # include wave states of neighbors
            if self.agent.startswith('ia2c'):
                for nnode_name in node.neighbor:
                    cur_state.append(self.nodes[nnode_name].wave_state)
```

- Calls `_measure_state_step()` **once** at the top — populates `node.wave_state` and `node.wait_state` for every node by reading TraCI. Then assembles per-node observation vectors.
- Hard-coded ordering by comment: `wave, wait, fp` (the rest is in the un-read portion below line 330).
- Agent branching:
  - `'greedy'` -> just the local wave vector. Greedy controllers don't need neighbor information.
  - Anything else -> start with `[node.wave_state]` and grow.
  - `agent.startswith('ia2c')` -> append each direct neighbor's wave state.

> The `_get_state` function continues past line 330 — it presumably appends the local `wait_state` and (for IA2C-FP variants) the neighbors' fingerprints, then `np.concatenate`s and clips/normalizes.

---

## TraCI interactions — what's directly visible

In lines 1–330, **no `traci.*` call is made directly**; they all live in helper methods below or in subclasses. However, the design implies:

| Helper | Likely TraCI calls |
|---|---|
| `_init_sim` | `traci.start([...sumo_binary, '-c', cfg, ...])` or `traci.init(port)` after `subprocess.Popen` of `sumo`/`sumo-gui` |
| `_init_nodes` | `traci.trafficlight.getIDList`, `traci.trafficlight.getControlledLinks`, `traci.trafficlight.getControlledLanes`, `traci.lane.getLength` |
| `_set_phase` | `traci.trafficlight.setRedYellowGreenState(node_name, phase_string)` |
| `_simulate` | repeated `traci.simulationStep()` calls; tracks `self.cur_sec += 1` (or by sim step size) |
| `_measure_state_step` | `traci.lane.getLastStepVehicleNumber`, `traci.lane.getLastStepHaltingNumber`, `traci.lanearea.*` (e2 detectors) for wave/wait |
| `_measure_reward_step` | similar lane/detector calls aggregated per node, optionally negated and weighted |
| `terminate` | `self.sim.close()` where `self.sim` is the TraCI connection |

(See [[TraCI-cheatsheet]] for the full vocabulary; concrete bindings will be confirmed when we read past line 330.)

---

## State construction — `wave` / `wait` features and neighbor stitching

From what's visible:

- **`wave`**: stored per-node as `node.wave_state`. Conceptually the *flow / density* feature — number of vehicles (or vehicles per lane capacity) on each inbound lane. Normalized by `self.norms['wave']` and clipped by `self.clips['wave']`.
- **`wait`**: stored per-node as `node.wait_state`. The *queue / accumulated waiting time* on red-blocked lanes. Normalized/clipped via the `'wait'` entries.
- Both vectors have length `len(ilds_in)` — one entry per inbound detector. The comment on line 68 confirms: *"wave and wait should have the same dim"*.

**Neighbor stitching** (the part visible in 308–329):

1. Always start with the local `wave_state`.
2. For `agent.startswith('ia2c')`: append each neighbor's `wave_state` (in the order of `node.neighbor`).
3. Below line 330 (not in this window): likely appends local `wait_state` (since the comment says ordering is *wave, wait, fp*), and for the `_fp` variants, the neighbors' fingerprints.

The resulting per-node observation is a list of arrays which presumably gets `np.concatenate`d before being returned to the agent.

---

## Bugs / oddities

1. **`Node(neighbor=[])` mutable default argument.** Classic Python footgun — all `Node` instances created without an explicit `neighbor` would share the same list. Not triggered in current usage (neighbors are always passed in), but a latent landmine.

2. **`self.terminate()` at the end of `__init__` (line 112).** Boots SUMO purely to enumerate topology in `_init_nodes`, then tears it down. Functional, but counterintuitive — anyone reading `__init__` expects the env to be "ready" after construction. The contract is actually: *you must call `reset()` before `step()`.*

3. **`reset()` has no defensive `terminate()` call.** Line 188 comment says "have to terminate previous sim before calling reset" — but nothing enforces this. Forgetting to terminate leaks a SUMO process and (because of port collisions) breaks the next `_init_sim`.

4. **Bare `except:` in `terminate()` (line 260).** Swallows `KeyboardInterrupt`, `SystemExit`, and any unexpected error. Should be `except Exception:` at minimum.

5. **`fuser` is Linux-only.** Calling `terminate()` on macOS/Windows silently no-ops (the `subprocess.run` exits non-zero into the bare except). Fine if the project is Linux-only, but worth flagging.

6. **`subprocess.check_call('rm ' + trip_file, shell=True)` in `collect_tripinfo` (line 137).** Shell injection vector if `output_path`, `name`, or `agent` ever contain user input. Use `os.remove(trip_file)`.

7. **Return-type inconsistency in `step()`** (lines 244–249). Second slot is a numpy vector (per-node) in test mode but a scalar in some train-mode branches (`agent == 'greedy'` or `coop_gamma < 0`). Downstream callers must dispatch on agent type, which is fragile.

8. **`self.T = np.ceil(...)` is a float**, not an int (line 83). Anywhere that uses `self.T` for indexing/looping should `int()` it explicitly.

9. **`self.seed += 1` (line 200) means seeds are not reproducible across reorderings.** If episodes are skipped or shuffled, you can't reproduce a specific episode by knowing only the run's base seed.

10. **`prev_action` write happens before the early return** (lines 274–276). Intentional but subtle: even if the action repeats and no yellow is inserted, `node.prev_action` still gets overwritten with the same value (harmless). However, on the path where `prev_action < 0` (first step ever), the write `node.prev_action = action` makes future steps treat *this* action as the baseline. That's the desired behavior — just be aware that the first call to `_get_node_phase` with `phase_type='yellow'` mutates state even though it returns the green phase directly.

11. **Typo: `"overwriteen"` in `_get_node_phase_id`** (line 295).

12. **Misleading inline comment on `PhaseMap.get_phase` (line 45).** Says *"phase_type is either green or yellow"* but the function takes no `phase_type` argument — it always returns the green phase string. The comment is leftover documentation.

13. **`from sumolib import checkBinary` and `import time` / `import random`** are imported but unused in this slice — probably used below line 330 (in `_init_sim` etc.).

14. **`self.state_names` referenced in `init_data` (line 165) before any visible definition.** Must be set by `_init_map()` in a subclass, but the dependency is implicit — easy to break by reordering `__init__` calls.

---

Continued in [[02-state-reward-simulate]] (lines 330–end).
