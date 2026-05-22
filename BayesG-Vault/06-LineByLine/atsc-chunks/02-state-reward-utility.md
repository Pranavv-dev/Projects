# ATSC Env — Line-by-Line: State, Reward, and Utility Methods (lines 330–664)

This note walks meticulously through the second chunk of [[atsc_env.py]] (lines 330–664), covering state and reward measurement, episode-level data collection, port allocation, seeding, and several oddities. The first chunk is documented at [[01-init-and-state-construction]].

File under analysis: `/tmp/BayesG/envs/atsc_env.py`. Class context: `TrafficSimulator` (begins around line 76).

> Note: a few helpers mentioned in the structural outline (`_add_action_log`, an explicit greedy controller class/method) do not exist in this file — those sections explain what is present in lieu of them. See [[#The _add_action_log helper]] and [[#Greedy controller logic if present]].

---

## State measurement (`_measure_state_step`, `_measure_state_obs`)

There is **no method named `_measure_state_obs`** in this file. The only state-measurement method is `_measure_state_step` (lines 552–594). The observation-building wrapper that *consumes* its output is `_get_state` (line 308, covered in [[01-init-and-state-construction]]) — it calls `self._measure_state_step()` then assembles per-agent vectors (wave + optional neighbor wave + optional neighbor fingerprint + optional wait). So when the outline says "obs", the relevant code is `_get_state`, not a missing `_measure_state_obs`.

### `_measure_state_step` — lines 552–594

```python
def _measure_state_step(self):
    for node_name in self.node_names:
        node = self.nodes[node_name]
        for state_name in self.state_names:
```

- Line 553: iterates every controlled intersection name in `self.node_names`.
- Line 554: grabs the [[Node]] object — this carries `ilds_in` (inductive-loop detector ids), `lanes_capacity`, `wave_state`, `wait_state`, etc. (see Node defn at lines 60–73).
- Line 555: iterates each state component the env declares (`self.state_names` is set per scenario in `_init_map`; typical values are `['wave']` or `['wave', 'wait']`).

#### Wave branch (lines 556–568)

```python
if state_name == 'wave':
    cur_state = []
    for k, ild in enumerate(node.ilds_in):
        if self.name == 'atsc_real_net':
            cur_wave = 0
            for ild_seg in ild:
                cur_wave += self.sim.lane.getLastStepVehicleNumber(ild_seg)
            cur_wave /= node.lanes_capacity[k]
            # cur_wave = min(1.5, cur_wave / QUEUE_MAX)
        else:
            cur_wave = self.sim.lanearea.getLastStepVehicleNumber(ild)
        cur_state.append(cur_wave)
    cur_state = np.array(cur_state)
```

- Line 558: `enumerate(node.ilds_in)`. Recall that for the `atsc_real_net` scenario each entry is a *list* of lane ids (a primary lane plus optional extension segments from `self.extended_lanes`, see lines 379–391 of `_init_nodes`). For other scenarios each entry is a single lane-area detector id (`lanearea`).
- Lines 559–563 (real net):
  - Sums `getLastStepVehicleNumber` across every segment in the composite detector. This is a **vehicle count**, not a density — until the next step.
  - Divides by `node.lanes_capacity[k]`, which `_init_nodes` computed as `sum(lane.getLength) / VEH_LEN_M` (so it’s "how many 7.5 m vehicles fit"). Result is occupancy fraction in [0, 1+] of an idealized jam capacity.
- Line 564: commented-out alternative normalization (`min(1.5, cur_wave / QUEUE_MAX)`). Left in as a breadcrumb — see [[#Bugs oddities]].
- Lines 565–566 (synthetic nets like Grid/Monaco): uses `lanearea` (E2 detector). Returns the vehicle count over the detector’s extent — **not normalized here**; normalization is deferred to `_norm_clip_state`.
- Line 568: assembles into a NumPy array of length `node.num_state`.

#### Wait branch (lines 569–584)

```python
elif state_name == 'wait':
    cur_state = []
    for ild in node.ilds_in:
        max_pos = 0
        car_wait = 0
        if self.name == 'atsc_real_net':
            cur_cars = self.sim.lane.getLastStepVehicleIDs(ild[0])
        else:
            cur_cars = self.sim.lanearea.getLastStepVehicleIDs(ild)
        for vid in cur_cars:
            car_pos = self.sim.vehicle.getLanePosition(vid)
            if car_pos > max_pos:
                max_pos = car_pos
                car_wait = self.sim.vehicle.getWaitingTime(vid)
        cur_state.append(car_wait)
    cur_state = np.array(cur_state)
```

- Line 571: per-detector loop. `max_pos` / `car_wait` reset *per detector*.
- Lines 574–577: in `atsc_real_net` only the **first segment** `ild[0]` is queried — the extension segments are silently ignored for the wait observation. This is asymmetric with the wave branch above (which sums *all* segments). See [[#Bugs oddities]].
- Lines 578–582: scans every vehicle on the detector and keeps the **single waiting-time value belonging to the vehicle with the largest lane position** (i.e., the one nearest the stop line). This is a leading-vehicle proxy for queue head wait; vehicles behind it contribute nothing.
- Line 583: appends that scalar.

#### Bookkeeping & normalization (lines 585–594)

```python
if self.record_stats:
    self.state_stat[state_name] += list(cur_state)
# normalization
norm_cur_state = self._norm_clip_state(cur_state,
                                       self.norms[state_name],
                                       self.clips[state_name])
if state_name == 'wave':
    node.wave_state = norm_cur_state
else:
    node.wait_state = norm_cur_state
```

- Line 585–586: if stats recording is on, raw (pre-norm) values are appended to a per-component flat list. Used later to calibrate `norm_wave` / `norm_wait` config values.
- Lines 588–590: divides by the configured norm and clamps using `_norm_clip_state` (defined at line 633).
- Lines 591–594: writes the result into the node attribute that `_get_state` will read next.

> Side effect note: the same `_measure_state_step` is called once per env step from `_get_state` (line 312). It overwrites `wave_state`/`wait_state` in place, so any code holding a stale reference loses it.

---

## Reward measurement (`_measure_reward_step`) — queue / wait / hybrid objectives

`_measure_reward_step` lives at lines 490–548. Despite the comment at line 485–489 claiming it "measures the current state of traffic", it does **not** read or write node state — it only queries the simulator and returns per-agent rewards.

```python
def _measure_reward_step(self):
    rewards = []
    for node_name in self.node_names:
        queues = []
        waits = []
        for ild in self.nodes[node_name].ilds_in:
```

- Lines 491–499: per-node, per-detector iteration.

### Queue measurement (lines 506–515)

```python
if self.obj in ['queue', 'hybrid']:
    if self.name == 'atsc_real_net':
        cur_queue = self.sim.lane.getLastStepHaltingNumber(ild[0])
        cur_queue = min(cur_queue, QUEUE_MAX)
    else:
        cur_queue = self.sim.lanearea.getLastStepHaltingNumber(ild)
    queues.append(cur_queue)
```

- Line 510 (real net): only `ild[0]` (the first segment) — extension lanes are ignored for the *reward* queue exactly as they were for the *wait state*. Compare with `_measure_traffic_step` (line 615) which **does** loop over all `ild_seg`. Inconsistency flagged in [[#Bugs oddities]].
- Line 512: hardcoded cap `QUEUE_MAX = 10` (constant defined at line 16). Applied **only in the real-net branch**, not in the synthetic-net `lanearea` branch on line 514. So a Grid/Monaco intersection can report queues > 10 in its reward while a real-net one cannot — see [[#Bugs oddities]].

### Wait measurement (lines 517–535)

```python
if self.obj in ['wait', 'hybrid']:
    max_pos = 0
    car_wait = 0
    if self.name == 'atsc_real_net':
        cur_cars = self.sim.lane.getLastStepVehicleIDs(ild[0])
    else:
        cur_cars = self.sim.lanearea.getLastStepVehicleIDs(ild)

    for vid in cur_cars:
        car_pos = self.sim.vehicle.getLanePosition(vid)
        if car_pos > max_pos:
            max_pos = car_pos
            car_wait = self.sim.vehicle.getWaitingTime(vid)

    waits.append(car_wait)
```

- Mirror of the wait branch in `_measure_state_step` — same single-leading-vehicle proxy. Same `ild[0]`-only quirk on real nets.
- Note `max_pos`/`car_wait` are reset per detector (line 519–520), unlike `queues`/`waits` which are reset per node (lines 493–494).

### Aggregation and sign (lines 536–548)

```python
queue = np.sum(np.array(queues)) if len(queues) else 0
wait = np.sum(np.array(waits)) if len(waits) else 0

if self.obj == 'queue':
    reward = - queue
elif self.obj == 'wait':
    reward = - wait
else:
    reward = - queue - self.coef_wait * wait
rewards.append(reward)
return np.array(rewards)
```

- Lines 537–538: empty-list guards prevent `np.sum([])` from being a NumPy scalar; they fall back to plain `0`. Mixing of Python ints and NumPy ints in the same `rewards` list is fine because `np.array` will upcast.
- Lines 541–546: sign convention — rewards are always **non-positive** (negative queue / negative wait). The "hybrid" branch is `-queue - coef_wait * wait`, with `coef_wait` from config (line 103: `self.coef_wait = config.getfloat('coef_wait')`).
- Returned as `np.array(rewards)` of length `n_agent`. Used in `step` (line 225) and may be collapsed to `global_reward = np.sum(reward)` (line 232) for greedy / coop_gamma<0 policies (line 247).

---

## `init_data` / `output_data` methods — what data is collected per episode for analysis

These methods are actually at lines 155–185, **above** the chunk under review, but they are essential context for `_measure_traffic_step` and `step` data flow.

### `init_data` (lines 155–166)

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

- Two orthogonal flags:
  - `is_record` — enables three running CSV-bound lists: `traffic_data`, `control_data`, `trip_data`.
  - `record_stats` — enables `state_stat`, a dict of flat lists keyed by `state_name` (`'wave'`, `'wait'`). These are pre-normalization samples used for offline calibration.
- Called from `__init__` (line 108), so it runs **once per env construction**, not per episode. Consequence: the lists accumulate across episodes unless externally reset (see [[#Bugs oddities]]).

### What gets appended where

- `traffic_data` ← `_measure_traffic_step` (line 631), one dict per simulation second when `is_record` is true. Keys: `episode, time_sec, number_total_car, number_departed_car, number_arrived_car, avg_wait_sec, avg_speed_mps, std_queue, avg_queue` (lines 622–630).
- `control_data` ← `step` (line 242). Keys: `episode, time_sec, step, action, reward`, where `step = self.cur_sec / self.control_interval_sec`. One row per control interval.
- `trip_data` ← `collect_tripinfo` (lines 115–137). Parsed from SUMO's tripinfo XML *after* the episode; keys include `id, depart_sec, arrival_sec, duration_sec, wait_step, wait_sec`.
- `state_stat[<name>]` ← `_measure_state_step` (line 586), per-detector pre-norm values, one per detector per call.

### `output_data` (lines 172–185)

```python
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

- Three CSV outputs named `{scenario}_{agent}_{control|traffic|trip}.csv`. Filename does **not** include the episode index — repeated calls overwrite. Callers typically invoke this once at end-of-evaluation.
- Note `state_stat` is **not** written by `output_data`; its consumer must access `self.state_stat` directly.
- No flushing/reset is done here; the dataframes are rebuilt each call from the entire list.

---

## `init_test_seeds`, `_set_seed`

### `init_test_seeds` (lines 168–170)

```python
def init_test_seeds(self, test_seeds):
    self.test_num = len(test_seeds)
    self.test_seeds = test_seeds
```

- Records the test-evaluation seeds and their count. `test_seeds` itself is built in `__init__` (lines 105–106):

  ```python
  test_seeds = config.get('test_seeds').split(',')
  test_seeds = [int(s) for s in test_seeds]
  ```

  i.e., a comma-separated string in the config. Used by `reset(test_ind)` (line 193) to pick `seed = self.test_seeds[test_ind]` when `train_mode` is False.

### `_set_seed`

**There is no `_set_seed` method in this file.** Seeding happens in three implicit places:

1. `self.seed = config.getint('seed')` at line 79 — the training seed.
2. After each `reset` in train mode the seed is incremented: `self.seed += 1` at line 200, so each episode gets a fresh seed.
3. In `_init_sim` the seed is passed to SUMO via the CLI: `command += ['--seed', str(seed)]` (line 426). NumPy/Python random are *not* seeded here — if any agent uses `random` / `np.random` reproducibility is unguaranteed.

---

## Port allocation (the fuser-based cleanup)

Three pieces work together: `__init__` port choice, `_find_free_port`, and the `fuser -k` cleanup blocks in `_init_sim` and `terminate`.

### Port chosen in `__init__` (lines 86–91)

```python
if port is None:
    self.port = self._find_free_port()
else:
    self.port = DEFAULT_PORT + port

self.sim_thread = self.port  # Use port as thread identifier
```

- If the caller doesn't pass a port, an ephemeral one is grabbed from the OS. If a small integer is passed, it's added to `DEFAULT_PORT = 8000` (line 13) — caller convention.

### `_find_free_port` (lines 658–664)

```python
def _find_free_port(self):
    """Find a free port by creating and closing a temporary socket"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))  # Bind to any available port
        s.listen(1)
        port = s.getsockname()[1]
        return port
```

- Classic "bind to 0 to get a free ephemeral port" trick. The socket is closed by the `with` exit before the port number is returned, leaving the port in `TIME_WAIT` momentarily. There's a race window between this return and SUMO actually binding it — see [[#Bugs oddities]].

### `_init_sim` cleanup (lines 437–445)

```python
# Try to cleanup any existing process on this port
try:
    subprocess.run(['fuser', '-k', f'{self.port}/tcp'],
                 stderr=subprocess.DEVNULL,
                 stdout=subprocess.DEVNULL)
except:
    pass

time.sleep(0.1)  # Short delay to ensure port is free
```

- Uses Linux `fuser -k <port>/tcp` to kill any process listening on the target port. The `try/except` swallows missing-`fuser` errors silently (e.g., on macOS).
- The 0.1 s sleep is a heuristic wait for the kernel to release the socket. Not strictly synchronous with `fuser`.

### Retry loop (lines 408–464)

```python
for retry in range(max_retries):
    try:
        if retry > 0:
            self.port = self._find_free_port()
            logging.info(f"Retrying with new port {self.port}")
        sumocfg_file = self._init_sim_config(seed)
        ...
        subprocess.Popen(command)
        time.sleep(1.0)  # Wait for SUMO to start
        self.sim = traci.connect(port=self.port)
        ...
    except (traci.exceptions.FatalTraCIError, socket.error) as e:
        last_exception = e
        ...
        time.sleep(0.5)
        continue
```

- Up to 5 attempts. On retry it picks a *new* free port and tries again.
- The 1.0 s sleep is the only "wait for SUMO to come up" mechanism — fragile under heavy load.

### `terminate` cleanup (lines 251–261)

```python
def terminate(self):
    if hasattr(self, 'sim'):
        try:
            self.sim.close()
            subprocess.run(['fuser', '-k', f'{self.port}/tcp'],
                         stderr=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL)
        except:
            pass
```

- Closes the TraCI connection, then `fuser -k`s the port. The bare `except` masks any error including programming bugs. Called once at end of `__init__` (line 112) to free the bootstrap simulator, and externally by training/test harnesses.

---

## The `_add_action_log` helper

**No method called `_add_action_log` exists in this file.** The closest functionality is inlined inside `step` (lines 234–242):

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

So per-step action and global reward are logged inline rather than via a helper. Notes:

- `action` is the joint action vector across agents, serialized as a CSV-in-a-field string (`"1,0,2,3,..."`).
- Only `global_reward` (sum over agents) is recorded — per-agent rewards are not persisted. Anyone wanting per-agent reward traces from the CSV is out of luck.

If you were looking for the abstraction, it doesn't exist — see [[#Bugs oddities]].

---

## Greedy controller logic (if present)

**There is no `Greedy` class or `greedy` controller method in this file.** Two greedy *branches* exist, both inside otherwise-shared methods:

### Branch 1: state shape (line 319, inside `_get_state`)

```python
if self.agent == 'greedy':
    state.append(node.wave_state)
else:
    cur_state = [node.wave_state]
    ...
```

- Greedy agents see *only their own wave state*, no neighbor info and no fingerprint. This makes the per-agent observation dim equal to `node.num_state`.
- Side effect: `_init_state_space` (lines 470–483) still computes `n_s_ls` including neighbor contributions for non-`ma2c` agents:
  ```python
  if not self.agent.startswith('ma2c'):
      for nnode_name in node.neighbor:
          num_wave += self.nodes[nnode_name].num_state
  ```
  Since `'greedy'` neither starts with `ma2c` nor with `ia2c`, `n_s_ls` will *over-report* the observation length for greedy. The actual observation in `_get_state` is shorter than `n_s_ls[i]` says. See [[#Bugs oddities]].

### Branch 2: reward shape (line 247, inside `step`)

```python
if (self.agent == 'greedy') or (self.coop_gamma < 0):
    reward = global_reward
return state, reward, done, global_reward
```

- Greedy and "no cooperation" agents receive a scalar global reward instead of the per-agent reward vector — appropriate for centralized greedy decision-making.

The actual greedy *policy* (which action to take from a given observation) lives outside this file — see [[policies-chunks]] notes.

---

## Other utility methods

### `_measure_traffic_step` (lines 596–631)

Per-second simulation telemetry, called from `_simulate` (line 656) when `is_record` is true.

```python
cars = self.sim.vehicle.getIDList()
num_tot_car = len(cars)
num_in_car = self.sim.simulation.getDepartedNumber()
num_out_car = self.sim.simulation.getArrivedNumber()
if num_tot_car > 0:
    avg_waiting_time = np.mean([self.sim.vehicle.getWaitingTime(car) for car in cars])
    avg_speed = np.mean([self.sim.vehicle.getSpeed(car) for car in cars])
else:
    avg_speed = 0
    avg_waiting_time = 0
```

- Iterates **all vehicles** in the network and computes mean wait/speed. O(N_vehicles) per second; for Monaco this is noticeable.
- Queues are gathered separately:

  ```python
  for ild in self.nodes[node_name].ilds_in:
      if self.name == 'atsc_real_net':
          cur_queue = 0
          for ild_seg in ild:
              cur_queue += self.sim.lane.getLastStepHaltingNumber(ild_seg)
      else:
          cur_queue = self.sim.lane.getLastStepHaltingNumber(ild)
      queues.append(cur_queue)
  ```

  Two interesting quirks here:
  - Real-net branch sums over **all segments** — *unlike* the reward path (`ild[0]` only). So `_measure_traffic_step`'s queue is more honest than the queue used to train the agent.
  - Non-real-net branch calls `self.sim.lane.getLastStepHaltingNumber(ild)` — `ild` is a `lanearea` (E2 detector) id, not a lane id, yet the code passes it to the **lane** API, not the `lanearea` API. Contrast with `_measure_reward_step` line 514 which correctly uses `lanearea.getLastStepHaltingNumber`. Either both work in this TraCI version because synthetic detector ids happen to coincide with lane ids, or this is a latent bug. See [[#Bugs oddities]].

### `_norm_clip_state` (lines 633–636)

```python
@staticmethod
def _norm_clip_state(x, norm, clip=-1):
    x = x / norm
    return x if clip < 0 else np.clip(x, 0, clip)
```

- Divides element-wise by `norm`, clips to `[0, clip]` if `clip >= 0`.
- Negative `clip` is the "no clipping" sentinel. The minimum is hardcoded to `0` — *negative* normalized values are silently zeroed. Doesn't matter for wave/wait (both non-negative) but worth noting.

### `_reset_state` (lines 638–642)

```python
def _reset_state(self):
    for node_name in self.node_names:
        node = self.nodes[node_name]
        node.prev_action = 0
```

- Called from both `_init_state_space` (line 471) and `reset` (line 189).
- Resets only `prev_action`. Note: resets it to `0`, not `-1` (the initial Node value at line 73). The `-1` sentinel is meaningful in `_get_node_phase` (line 275: `if (prev_action < 0)`), so after first reset the env can never re-enter the "no previous action" branch via this code path. Possibly intentional, possibly a latent bug.

### `_set_phase` (lines 644–648)

```python
def _set_phase(self, action, phase_type, phase_duration):
    for node_name, a in zip(self.node_names, list(action)):
        phase = self._get_node_phase(a, node_name, phase_type)
        self.sim.trafficlight.setRedYellowGreenState(node_name, phase)
        self.sim.trafficlight.setPhaseDuration(node_name, phase_duration)
```

- Single helper used for both yellow and green phases by `step` (lines 209 and 218). `_get_node_phase` (line 268) decides whether to inject a yellow when the new action differs from the previous.

### `_simulate` (lines 650–656)

```python
def _simulate(self, num_step):
    for _ in range(num_step):
        self.sim.simulationStep()
        self.cur_sec += 1
        if self.is_record:
            self._measure_traffic_step()
```

- Wall-clock-per-second loop. `cur_sec` is incremented in lockstep with SUMO's internal seconds. Logging is on the *per-second* grid; control decisions are on the `control_interval_sec` grid.

---

## Bugs / oddities

A consolidated list of suspicious or inconsistent things found in this chunk:

### 1. `QUEUE_MAX = 10` cap is asymmetric

- Constant defined at line 16: `QUEUE_MAX = 10`.
- Applied in `_measure_reward_step` **only** for `atsc_real_net` (line 512):
  ```python
  cur_queue = self.sim.lane.getLastStepHaltingNumber(ild[0])
  cur_queue = min(cur_queue, QUEUE_MAX)
  ```
  Not applied in the synthetic-net branch on line 514.
- Implications:
  - Synthetic networks (Grid, Monaco) can yield reward magnitudes far larger than 10 per detector — different reward scale than real-net.
  - The cap is also **not** applied to `_measure_traffic_step` queues (line 615 just sums), so logged "avg_queue" diverges from the queue used in the reward signal.
- The commented line 564 (`# cur_wave = min(1.5, cur_wave / QUEUE_MAX)`) hints `QUEUE_MAX` was once used as a wave-normalization constant too. Currently it has a single in-use site.

### 2. Hardcoded yellow_interval defaults

- `yellow_interval_sec` is read from config (line 81):
  ```python
  self.yellow_interval_sec = config.getint('yellow_interval_sec') # Grid : yellow_interval_sec = 2
  ```
  with no `fallback` argument, so it will **crash** if missing from the config rather than defaulting. The inline comment encodes 2 s as the Grid default — that "default" is actually a comment, not a fallback. Anyone porting to a new scenario must remember to add this key.
- `step` (line 217) computes `rest_interval_sec = self.control_interval_sec - self.yellow_interval_sec`. If `yellow_interval_sec >= control_interval_sec`, `rest_interval_sec` becomes zero or negative and `_simulate(0)` silently does nothing — the env will tick yellow only and produce degenerate behavior, with no warning.

### 3. `ild[0]`-only on real net (wait state & reward), but sum-of-segments elsewhere

- `_measure_state_step` wait branch (line 575), `_measure_reward_step` queue branch (line 510), `_measure_reward_step` wait branch (line 523): only the first segment is read.
- `_measure_state_step` wave branch (line 562) and `_measure_traffic_step` queue (line 615): all segments are summed.
- This asymmetry means an agent on `atsc_real_net` is rewarded based on the head of the link only, but its wave observation reflects the full queue spillback. Likely a latent bug.

### 4. `lane` vs `lanearea` API mix-up in `_measure_traffic_step`

- Line 617 (synthetic branch):
  ```python
  cur_queue = self.sim.lane.getLastStepHaltingNumber(ild)
  ```
  but on synthetic nets `ild` is a `lanearea` detector id (see `_init_nodes` line 390). The reward path correctly uses `self.sim.lanearea.getLastStepHaltingNumber(ild)` (line 514). Either this is an actual bug (TraCI raises) or the detector ids happen to alias lane ids in the bundled `.net.xml` files, which is the only explanation for the code working at all.

### 5. `_find_free_port` races

- The port returned is no longer bound at the moment of return. Between this and the `subprocess.Popen` that launches SUMO there is a race; the kernel can hand the same port to a different process. The 0.1 s sleep and 1.0 s SUMO startup wait try to paper over this. The retry loop in `_init_sim` provides a backstop.

### 6. `fuser -k` is Linux-only; bare `except`

- Lines 437–443 and 257–260 use `fuser`, which doesn't exist on macOS by default. The `try/except: pass` silently swallows the resulting `FileNotFoundError`. On non-Linux this means the env *appears* to clean up but doesn't.
- Bare `except:` also masks `KeyboardInterrupt` and any programming bugs (e.g., a typo in command construction).

### 7. `init_data` is called once at construction, not per episode

- `traffic_data`, `control_data`, `trip_data`, `state_stat` accumulate across episodes within a single env lifetime. There is **no method to flush them between episodes**. Long training runs with `is_record=True` will see memory grow without bound. Callers must externally clear these lists after `output_data` or accept growing CSVs.

### 8. `output_data` filenames collide across episodes/runs

- Filenames `{scenario}_{agent}_control.csv` etc. include no run-id, no episode index, no timestamp. Two parallel evaluations on the same `output_path` will clobber each other's CSVs. A simple `cur_episode` suffix would fix it.

### 9. Greedy `n_s_ls` mismatch

- `_init_state_space` (lines 478–483) inflates `num_wave` by neighbor contributions for any non-`ma2c` agent:
  ```python
  if not self.agent.startswith('ma2c'):
      for nnode_name in node.neighbor:
          num_wave += self.nodes[nnode_name].num_state
  ```
  But `_get_state` for greedy emits only `node.wave_state`. So a greedy agent constructed against `self.n_s_ls` will have a model input dim **larger** than the actual observation. Downstream policies that introspect `n_s_ls` to build a network will be wrong; the silent fix is that greedy doesn't actually use a learned network.

### 10. Per-agent reward not persisted

- `control_data` only logs `global_reward` (line 241). For analysis of credit assignment in multi-agent training this is insufficient. There is no per-agent reward CSV.

### 11. `_reset_state` resets `prev_action` to 0, not -1

- Node initializes `prev_action = -1` (line 73), but `_reset_state` sets it to `0` (line 642). The `< 0` guard in `_get_node_phase` (line 275) — which suppresses the yellow-injection logic when there is no previous phase — never fires after a reset. So immediately after `reset`, the env may inject a yellow even though there is no real "previous" phase to transition from. Probably benign because the first action will set `prev_action` before the next call sees it.

### 12. Pre-norm `state_stat` collected inside per-detector loop, but not per-node grouped

- `self.state_stat[state_name] += list(cur_state)` (line 586) is appended per node call. The flat list loses node identity, so the calibration data assumes IID across detectors. Fine for setting a global `norm_wave`, less fine if per-intersection norms were ever wanted.

### 13. No `_set_seed` method despite outline expectation

- Seed handling is fragmented across `__init__`, `reset`, and `_init_sim` rather than centralized in a `_set_seed`. There is no Python-side `random.seed` / `np.random.seed` call anywhere in this file; only SUMO is seeded.

### 14. No `_add_action_log` helper despite outline expectation

- The action/reward logging lives inline in `step` (lines 234–242). Without a helper, any subclass that overrides `step` (e.g., for hierarchical control) must remember to duplicate the logging block.

---

## Cross-references

- First chunk: [[01-init-and-state-construction]] — covers `__init__`, `_init_map`, `_init_nodes`, `_init_action_space`, `_init_state_space`, `_get_state` (the consumer of `_measure_state_step`).
- Policies that consume these envs: [[policies-chunks]].
- Related utilities & helpers: [[utils-chunks]].
- File: `/tmp/BayesG/envs/atsc_env.py`.
