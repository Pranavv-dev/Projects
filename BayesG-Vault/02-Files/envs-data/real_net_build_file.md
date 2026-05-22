# real_net_data/build_file.py — Real-Network (MoST) Route Generator

File: `/tmp/BayesG/envs/real_net_data/build_file.py` (167 lines)

Generates SUMO route/config files for the **MoST** (Monaco SUMO Traffic) scenario or a similar real-world map. Unlike [[large_grid_build_file]] this file does **not** build a network from scratch — it assumes a pre-existing `in/most.net.xml` and only emits route flows, a sumocfg, and (in commented-out code) an additional file with induction-loop detectors.

Related: [[large_grid_build_file]], [[real_net_env]], [[draw_net]].

---

## Top-level constants and helpers (lines 1–13)

```python
import configparser
import logging
import numpy as np
import os
# from envs.real_net_env import RealNetEnv  # commented out
ILD_POS = 50
```

`ILD_POS = 50` is the (signed positive) detector-placement distance from the stop line; used by `output_ild` later as `-ILD_POS` to put it 50 m upstream.

`write_file(path, content)` (lines 10–12) — same trivial writer as in [[large_grid_build_file]].

---

## OD flow generation — `output_flows(flow_rate, seed=None)` (lines 15–105)

Builds a `routes` XML by stitching together four hand-curated source/sink/via groups, then sequencing them across time using two volume schedules.

### The four flow groups

Each group `flows[j]` is a list of 4 tuples `(src_edge, sink_edge, via_edges_str)`. The longer 6-tuple variants (commented out, lines 22–27, 33–38, 45–50, 58–63) are an earlier richer routing; the active code keeps a slimmed-to-four variant.

#### Group 0 — flow1 (active lines 27–31)

```python
flows1.append(('-10114#1', '-10079',  '10115#2 -10109'))
flows1.append(('-10114#1', '-10079',  '-10114#0 10108#0 gneE5'))
flows1.append(('-10114#1', '-10079',  '-10114#0 10108#0 10102'))
flows1.append(('-10114#1', '10076',   '-10114#0 10107 10102'))
```

All share source `-10114#1`; sinks split between `-10079` (×3) and `10076` (×1). Different `via` strings route through different interior corridors.

#### Group 1 — flow2 (active lines 39–42)

```python
flows1.append(('10096#1',  '10063',     '10089#3'))
flows1.append(('-10185#1', '-10071#3',  'gneE20'))
flows1.append(('10096#1',  '10063',     '10109'))
flows1.append(('-10185#1', '-10061#5',  'gneE19'))
```

Two source/sink pairs alternated with different vias.

#### Group 2 — flow3 (active lines 51–54)

```python
flows1.append(('10052#1',  '10104',  '10181#1 -10089#3'))
flows1.append(('-10064#9', '10104',  '-10068 10102'))
flows1.append(('-10051#2', '10043',  '10181#1 gneE4'))
flows1.append(('-10064#9', '-10110', '-10064#4 -10064#3'))
```

#### Group 3 — flow4 (active lines 64–67)

```python
flows1.append(('10061#4',  '-10085', '10065#2 10102'))
flows1.append(('10071#3',  '10085',  '10065#2 -10064#3'))
flows1.append(('-10070#1', '-10086', 'gneE9'))
flows1.append(('-10063',   '10085',  'gneE8'))
```

Groups 0 and 1 are "direction A" (morning peak); groups 2 and 3 are "direction B" (afternoon peak), per the indexing logic below.

---

## Time-varying volumes (lines 70–105)

```python
# vols_a = [2, 3, 4, 6, 4, 2, 1, 0, 0, 0, 0]    # old, larger profile
# vols_b = [0, 0, 0, 1, 2, 3, 5, 4, 3, 2, 1]
vols_a = [1, 2, 4, 4, 4, 4, 2, 1, 0, 0, 0]
vols_b = [0, 0, 0, 1, 2, 4, 4, 4, 4, 2, 1]
times  = np.arange(0, 3301, 300)
```

- `times` has 12 timestamps → **11 windows of 300s each, total horizon 3300s = 55 min** (the sumocfg below still ends at 3600s; the last 5 min has no flow).
- `vols_a[i]` is the number of group-0 + group-1 routes activated in window `i`. Ramps `1→4→4→4→4→2→1→0…`.
- `vols_b[i]` is the number of group-2 + group-3 routes activated. Ramps `0→0→0→1→2→4→4→4→4→2→1`, peaking later than `vols_a`.
- Maximum per-window volume is 4 routes per direction, capped by the fact that each group only has 4 entries (so `inds = np.arange(vol)` would index out of range if `vol > 4`). The 6-vehicle entry in the commented-out `vols_a` is what would have used the 6-entry route lists above — they were trimmed together.

```python
flow_str = '  <flow id="f%s" departPos="random_free" from="%s" to="%s" via="%s" begin="%d" end="%d" vehsPerHour="%d" type="car"/>\n'
output  = '<routes>\n'
output += '  <vType id="car" length="5" accel="5" decel="10" speedDev="0.1"/>\n'
```

Vehicle type "car" (`speedDev=0.1` adds 10% speed heterogeneity — absent from [[large_grid_build_file]]).

### Per-window loop (lines 80–103)

```python
for i in range(len(times) - 1):
    t_begin, t_end = times[i], times[i + 1]
    k = 0
    for j in [0, 1]:               # direction-A groups
        vol = vols_a[i]
        if vol > 0:
            inds = np.arange(vol)  # NOT np.random.choice — comment shows that was the prior intent
            for ind in inds:
                src, sink, via = flows[j][ind]
                output += flow_str % (cur_name, src, sink, via, t_begin, t_end, flow_rate)
                k += 1
    for j in [2, 3]:               # direction-B groups
        vol = vols_b[i]
        ...
```

For each 5-minute window:

- Activate the first `vols_a[i]` route templates from groups 0 and 1 (so 2 * `vols_a[i]` flows from direction A).
- Activate the first `vols_b[i]` route templates from groups 2 and 3 (so 2 * `vols_b[i]` flows from direction B).
- Each activated flow has constant `vehsPerHour = flow_rate` (caller-supplied — passed through unchanged).

Total flows in window `i` = `2 * (vols_a[i] + vols_b[i])`. Peak: window 5 has `2 * (4 + 4) = 16` concurrent flows.

The commented-out `np.random.choice(FLOW_NUM, vol, replace=False)` (lines 87, 97) shows that in an earlier version routes were sampled randomly from each group of 6; the current deterministic `np.arange(vol)` keeps experiments reproducible without needing the `seed`.

`FLOW_NUM = 6` (line 18) is dead — left over from the 6-route variant.

---

## Config emission — `output_config(thread=None)` (lines 108–120) and `gen_rou_file` (lines 123–131)

```python
def output_config(thread=None):
    out_file = 'most.rou.xml' if thread is None else 'most_%d.rou.xml' % int(thread)
    str_config  = '<configuration>\n  <input>\n'
    str_config += '    <net-file value="in/most.net.xml"/>\n'
    str_config += '    <route-files value="in/%s"/>\n' % out_file
    # str_config += '    <additional-files value="in/most.add.xml"/>\n'   # COMMENTED
    str_config += '  </input>\n  <time>\n'
    str_config += '    <begin value="0"/>\n    <end value="3600"/>\n'
    str_config += '  </time>\n</configuration>\n'
    return str_config
```

Note `additional-files` is commented out, matching the fact that `output_ild` (below) is never invoked. The simulation runs for 3600s but flows stop after 3300s.

`gen_rou_file(path, flow_rate, seed=None, thread=None)` (lines 123–131) writes the route file to `<path>in/most_<thread>.rou.xml` and the sumocfg to `<path>most_<thread>.sumocfg`, returning the sumocfg path. Mirrors the API of [[large_grid_build_file]] `gen_rou_file`, but signature is `(flow_rate)` instead of `(peak_flow1, peak_flow2, density)` — single intensity knob.

---

## Detectors — `output_ild(env, ild)` (lines 134–152)

Different from the grid version: takes a live `env` argument (with `env.node_names`, `env.nodes[name].ilds_in`, and `env.sim.lane.getLength(name)`), so it needs **traci** running.

```python
for node_name in env.node_names:
    node = env.nodes[node_name]
    for ild_name in node.ilds_in:
        lane_name = ild_name           # ild name == lane name
        l_len = env.sim.lane.getLength(lane_name)
        i_pos = min(ILD_POS, l_len - 1)
        if lane_name in ['gneE4_0', 'gneE5_0']:
            str_adds += ild % (ild_name, lane_name, -63, -13)
        elif lane_name == 'gneE18_0':
            str_adds += ild % (ild_name, lane_name, -116, -66)
        elif lane_name == 'gneE19_0':
            str_adds += ild % (ild_name, lane_name, 1, 50)
        else:
            str_adds += ild % (ild_name, lane_name, -i_pos, -1)
```

Hand-tuned offsets for four special lanes (`gneE4_0`, `gneE5_0`, `gneE18_0`, `gneE19_0`) — presumably because their geometry (short lengths or atypical orientation) doesn't fit the default `[-50, -1]` placement; one (`gneE19_0`) is even placed *downstream* (positive position) of the start, suggesting it's a one-way out-edge needing a different detection window.

This function is **not called** anywhere in the active code path — see the commented-out `if __name__ == '__main__'` block (lines 155–167) where it would have been wired up after constructing a `RealNetEnv` from `./config/config_ia2c_real.ini`.

---

## Assumed pre-existing files (most.net.xml)

This script is a *route generator only*, so it assumes the SUMO network is already provided. Specifically:

- **`in/most.net.xml`** — the compiled MoST network. Referenced by `output_config` (line 114). Must contain every edge id used in the four flow groups: `-10114#1, -10079, 10076, 10115#2, -10109, -10114#0, 10108#0, gneE5, 10102, 10107, 10096#1, 10063, 10089#3, -10185#1, -10071#3, gneE20, 10109, -10061#5, gneE19, 10052#1, 10104, 10181#1, -10089#3, -10064#9, -10068, -10051#2, 10043, gneE4, -10110, -10064#4, -10064#3, 10061#4, -10085, 10065#2, 10071#3, 10085, -10070#1, -10086, gneE9, -10063, gneE8` and all the longer commented-out edges (`gneE4, gneE7, gneE9, gneE10, gneE12, gneE13, -10046#0, -10046#5, ...`) if uncommented. These are real MoST-network edges (the numeric ones look OSM-derived; `gneE*` are netedit-generated patches).
- **`in/most.add.xml`** — referenced only in the commented-out `additional-files` line. Would be produced by an `output_ild(env, ild)` call inside a live `RealNetEnv` instance (the `__main__` block at lines 155–167, also commented out). Currently the project simulates without explicit additional file — any detectors needed by the environment must be added by the env itself at run time via traci.
- **`./config/config_ia2c_real.ini`** — referenced inside the commented `__main__` (line 159) and is the launch parameters file for the real-net IA2C agent. Not strictly required for `gen_rou_file`, but documents how `RealNetEnv` would have been wired.

What this script does **not** produce: nodes, edges, types, connections, tll, netconvert calls — all of those are pre-built for MoST (and presumably distributed with the repo under `envs/real_net_data/in/`). Compare with [[large_grid_build_file]] which generates everything.

---

## Differences from large_grid_build_file at a glance

| Concern | large_grid | real_net |
|---|---|---|
| Network | Generated by netconvert from `.nod/.edg/.typ/.con/.tll` | Pre-existing `in/most.net.xml` |
| Flow params | `(peak_flow1, peak_flow2, density)` | `(flow_rate)` |
| Demand pattern | Triangular ratios × peak | Integer "number of active route templates" `vols_a/vols_b` × fixed `flow_rate` |
| Routing | `from`/`to` only; routes computed by SUMO | Explicit `via` corridors |
| Vehicle type | `length=5 accel=5 decel=10` | same + `speedDev=0.1` |
| Detectors | `output_ild` static, emitted by `main()` | `output_ild` runtime, needs `env`/traci, currently not wired |
| Horizon | 3000s of flow inside 3600s sim | 3300s of flow inside 3600s sim |
