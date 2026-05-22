# large_grid_data/build_file.py — Large 5x5 Grid SUMO Scenario Generator

File: `/tmp/BayesG/envs/large_grid_data/build_file.py` (449 lines)

Procedurally builds the entire SUMO scenario for a 5x5 traffic-light grid: nodes, edge types, edges, connections, traffic-light logic, induction-loop detectors, route flows, and config files. Run as `python build_file.py` to emit everything from scratch, then invokes `netconvert` to produce `exp.net.xml`.

Related: [[real_net_build_file]], [[large_grid_env]], [[draw_net]].

---

## Top-level constants (lines 10–15)

```python
MAX_CAR_NUM   = 30          # cap on initial vehicles per init-flow stretch
SPEED_LIMIT_ST = 20         # m/s on "streets" (type "a", 2 lanes)
SPEED_LIMIT_AV = 11         # m/s on "avenues" (type "b", 1 lane)
L0     = 200                # spacing (m) between adjacent intersections
L0_end = 75                 # stub length (m) from boundary intersection out to the priority node
N      = 5                  # grid size (unused inside the file but documents intent)
```

The grid is 5x5 traffic-light nodes (`nt1`…`nt25`) at coordinates `(dx, dy)` with `dx, dy in {0, 200, 400, 600, 800}`. Around it sit 20 boundary priority nodes (`np1`…`np20`) at offset `L0_end=75` past the grid edge.

Helper `write_file(path, content)` at lines 18–20 — trivial UTF-8 writer.

---

## What XML files are output

All written by `main()` (lines 407–446) to the current working directory `./`:

| File | Producer | Purpose |
|---|---|---|
| `exp.nod.xml` | `output_nodes(node)` | 25 traffic-light + 20 priority node definitions |
| `exp.typ.xml` | `output_road_types()` | Two edge types: `a` (street, 2 lanes, 20 m/s) and `b` (avenue, 1 lane, 11 m/s) |
| `exp.edg.xml` | `output_edges(edge)` | Bidirectional edges for boundary stubs + interior grid streets and avenues |
| `exp.con.xml` | `output_connections(con)` | Per-junction movement permissions (go-through / left / right) for boundary and interior junctions |
| `exp.tll.xml` | `output_tls(tls, phase)` | 3-phase static traffic-light logic for each of the 25 lights |
| `exp.netccfg` | `output_netconfig()` | Netconvert configuration glue file pointing at the five above + emitting `exp.net.xml` |
| `exp.net.xml` | `os.system('netconvert -c exp.netccfg')` (line 432) | Compiled network (external tool call) |
| `exp.rou.xml` | `output_flows(1000, 2000, 0.2)` (line 435) | Route + flow file with default peak flows and density |
| `exp.add.xml` | `output_ild(ild)` | `laneAreaDetector` definitions on every traffic-light approach lane |
| `exp.sumocfg` | `output_config()` | Top-level SUMO scenario config (net + route + additional + 0..3600s time window) |

The optional `jtrrouter` call on line 438 is commented out.

For training/multithread invocations, `gen_rou_file(path, peak_flow1, peak_flow2, density, seed, thread)` (lines 325–333) emits `exp_<thread>.rou.xml` and `exp_<thread>.sumocfg` instead and returns the sumocfg path.

---

## Grid generation

### Nodes — `output_nodes(node)` (lines 23–46)

Two loops fill the file with `<node id="..." x="..." y="..." type="..."/>` lines:

- **Traffic-light nodes (lines 26–30)**: nested `dy` then `dx` over `np.arange(0, L0*5, L0)` = `[0, 200, 400, 600, 800]`. Produces `nt1`..`nt25` row-major (so `nt1` is at `(0,0)`, `nt5` at `(800,0)`, `nt6` at `(0,200)`, ..., `nt25` at `(800,800)`). Type `"traffic_light"`.
- **Priority (boundary) nodes (lines 32–44)**: stubs at `-L0_end = -75` past each boundary, walking clockwise:
  - Bottom row: `np1`..`np5` at `(0, -75)`..`(800, -75)` (south of `nt1`..`nt5`).
  - Right column: `np6`..`np10` at `(875, 0)`..`(875, 800)` (east of `nt5, nt10, nt15, nt20, nt25`).
  - Top row (right-to-left): `np11`..`np15` at `(800, 875)`..`(0, 875)` (north of `nt25, nt24, nt23, nt22, nt21`).
  - Left column (top-to-bottom): `np16`..`np20` at `(-75, 800)`..`(-75, 0)` (west of `nt21, nt16, nt11, nt6, nt1`).

### Road types — `output_road_types()` (lines 49–54)

```xml
<types>
  <type id="a" priority="2" numLanes="2" speed="20.00"/>
  <type id="b" priority="1" numLanes="1" speed="11.00"/>
</types>
```

Type `a` is "street" (wide, fast, 2 lanes; horizontal grid lines through the avenues' index space — see edges below). Type `b` is "avenue" (1 lane, slower).

### Edges — `output_edges(edge)` (lines 62–94)

Each edge id is `<from>_<to>`; both directions are emitted.

- **External street stubs (lines 65–71)**: pairs `(in=nt5, out=np6)`, `(nt10, np7)`, `(nt15, np8)`, `(nt20, np9)`, `(nt25, np10)` and `(nt21, np16)`, `(nt16, np17)`, `(nt11, np18)`, `(nt6, np19)`, `(nt1, np20)`. Type `a`. These are the east-west boundary connectors.
- **External avenue stubs (lines 73–79)**: pairs `(nt1, np1)`..`(nt5, np5)` and `(nt25, np11)`..`(nt21, np15)`. Type `b`. North-south boundary connectors.
- **Internal streets (lines 81–86)**: rows `i in {1, 6, 11, 16, 21}`, `j in 0..3` — edges `nt(i+j) <-> nt(i+j+1)`. Type `a`. Five east-west rows of four streets each = 20 streets each direction = 40 edges.
- **Internal avenues (lines 87–92)**: columns `i in 1..5`, `j in {0, 5, 10, 15}` — edges `nt(i+j) <-> nt(i+j+5)`. Type `b`. Five north-south columns of four avenues each = 40 edges.

### Connections — `output_connections(con)` via `get_con_str_set` (lines 97–178)

Per junction, `get_con_str_set(cur_node, n, s, w, e)` (lines 103–120) produces twelve `<connection .../>` lines covering all turning movements:

- Go-through: S→N, N→S, W→E, E→W (lane 0 → lane 0).
- Left-turn: S→W, N→E (lane 0 → lane 1); W→N, E→S (lane 1 → lane 0). Note the asymmetric `fromLane`/`toLane`: lefts from a street (type `a`, 2 lanes) use lane 1, lefts from an avenue (`b`, 1 lane) use lane 0.
- Right-turn: S→E, N→W, W→S, E→N (lane 0 → lane 0).

`output_connections(con)` walks three groups:

- **East-west boundary junctions (lines 126–150)** — `i in [5, 10, 15, 20, 25, 21, 16, 11, 6, 1]` with `j` the boundary `np` id. Branches handle the four corners: `i=1` gets `s_node='np1'`, `i=5` also `'np5'` (but `j=6` so `e_node='np6'`), `i=25` gets `n_node='np11'`, `i=21` gets `n_node='np15'`. Then `e_node`/`w_node` are derived from `i%5`.
- **North-south boundary junctions (lines 152–166)** — `i in [2, 3, 4, 24, 23, 22]`. The "top and bottom middles" of the grid border (corners already handled above).
- **Interior junctions (lines 168–175)** — the 9 nodes `{7, 8, 9, 12, 13, 14, 17, 18, 19}` (the 3x3 inner block); all four neighbors are `nt` nodes.

### Traffic lights — `output_tls(tls, phase)` (lines 390–404)

Static 6-phase program applied to every `nt1`..`nt25`:

```
phase 0: 'GGgrrrGGgrrr'   30s   # NS go (through + right green, left permissive)
phase 1: 'yyyrrryyyrrr'    3s   # NS yellow
phase 2: 'rrrGrGrrrGrG'   30s   # EW go (through + right green, left protected on G)
phase 3: 'rrrGryrrrGry'    3s   # EW yellow on lefts
phase 4: 'rrrGGrrrrGGr'   30s   # (a second EW-go-ish state?)
phase 5: 'rrryyrrrryyr'    3s   # yellow
```

Phase durations alternate `30, 3, 30, 3, 30, 3` via `phase_duration = [30, 3]` and `phase_duration[k % 2]`. Every `nt` light shares the same identical program (no offset / no coordination between adjacent lights).

### Detectors — `output_ild(ild)` (lines 351–387)

`laneAreaDetector`s placed on approach lanes, 50m upstream of each stop line (`pos="-50" endPos="-1"`).

- **Boundary stubs (lines 363–369)**: detector on `np<j> → nt<i>` lane 0, plus lane 1 if it's a street (first 10 stubs, i.e. `k<10`, the east-west street stubs which are type `a`).
- **Internal streets (lines 371–378)**: detectors on both directions, both lanes (lane 0 and lane 1).
- **Internal avenues (lines 380–385)**: detectors on both directions, lane 0 only (single-lane).

`get_ild_str(from_node, to_node, ild, lane_i=0)` (lines 351–353) just formats `id="<from>_<to>_<lane>"` on `lane="<from>_<to>_<lane>"`.

---

## Flow generation (uniform + peaks)

### Uniform initial fill — `init_routes(density)` (lines 219–262)

When `density > 0` (e.g. 0.2 default), pre-seeds every internal edge with `MAX_CAR_NUM * density = 6` (when density=0.2) parked-then-released vehicles per lane direction.

```python
init_flow = '  <flow id="i%s" departPos="random_free" from="%s" to="%s" begin="0" end="1" departLane="%d" departSpeed="0" number="%d" type="type1"/>\n'
```

Each interior street is filled in lane 0 and lane 1, both directions, with a randomly chosen sink edge from `sink_edges` (any boundary outflow). Avenues only get lane 0 (single-lane). Uses `np.random.choice` over the precomputed `sink_edges` list; `seed` controls reproducibility.

### Time-varying peak demand — `output_flows(peak_flow1, peak_flow2, density, seed)` (lines 264–322)

Comment-block (lines 265–270) documents four directional OD flows:

```
flow1: x11, x12, x13, x14, x15 -> x1, x2, x3, x4, x5      (north → south, top→bottom)
flow2: x16, x17, x18, x19, x20 -> x6, x7, x8, x9, x10     (west  → east, left→right)
flow3: x1,  x2,  x3,  x4,  x5  -> x15, x14, x13, x12, x11 (south → north, return)
flow4: x6,  x7,  x8,  x9,  x10 -> x20, x19, x18, x17, x16 (east  → west, return)
```

(The `x*` names refer to `np*` boundary nodes.)

Sources and sinks are built from `get_external_od([...], dest=False)` and `get_external_od([...], dest=True)` (lines 194–207), which use the lookup `edge_maps = [0, 1, 2, 3, 4, 5, 5, 10, 15, 20, 25, 25, 24, 23, 22, 21, 21, 16, 11, 6, 1]` to invert "outer np index → inner nt anchor".

#### Peak profiles (quote)

```python
# create volumes per 5 min for flows
ratios1 = np.array([0.4, 0.7, 0.9, 1.0, 0.75, 0.5, 0.25])  # start from 0
ratios2 = np.array([0.3, 0.8, 0.9, 1.0, 0.8, 0.6, 0.2])    # start from 15min
flows1 = peak_flow1 * 0.6 * ratios1   # less-heavy direction of morning peak
flows2 = peak_flow1 * ratios1          # heavy direction of morning peak
flows3 = peak_flow2 * 0.6 * ratios2   # less-heavy direction of afternoon peak
flows4 = peak_flow2 * ratios2          # heavy direction of afternoon peak
flows = [flows1, flows2, flows3, flows4]
times = np.arange(0, 3001, 300)        # 11 timestamps → 10 intervals of 5 min
id1 = len(flows1)                       # = 7
id2 = len(times) - 1 - id1              # = 3
```

- `times` has 11 stamps, so 10 windows of 300s each (covers `[0, 3000]`s of the 3600s scenario).
- `flows1`/`flows2` ride a triangular morning peak ramping up `[0.4 … 1.0]` over the first 4 windows then back down to `0.25`. They are active for windows `i in [0, id1)` i.e. `[0, 7)`.
- `flows3`/`flows4` ride the afternoon peak `[0.3 … 1.0 … 0.2]`. They are active for windows `i in [id2, 10) = [3, 10)`. The offset of `id2=3` means the afternoon peak starts at minute 15 (`3*300s = 900s`), overlapping the morning ramp-down.
- Flow lines look like:

```xml
<flow id="f<window>_<k>" departPos="random_free" from="<src>" to="<sink>"
      begin="t_begin" end="t_end" vehsPerHour="<flow>" type="type1"/>
```

Default invocation `output_flows(1000, 2000, 0.2)` (line 435) means morning peak ~600/1000 veh/h per direction and afternoon peak ~1200/2000 veh/h per direction at maximum.

The vehicle type is declared once at the top of the route file: `<vType id="type1" length="5" accel="5" decel="10"/>`.

---

## Phase durations

Lights run a `[30, 3]` alternation across the 6 phases (3 long greens of 30s, 3 short yellows of 3s) — total cycle = `3 * 30 + 3 * 3 = 99s`. See [[#Traffic lights]] above. All 25 intersections share the same program with **offset=0**, so this is *uncoordinated* fixed-time control — the baseline against which RL controllers in [[policies]] are compared.

---

## Dependencies

- **`numpy`** — `np.arange`, `np.random.choice`, `np.random.seed`, `np.array`. Pure numerics, no scientific deps.
- **`os`** — only for `os.system('netconvert -c exp.netccfg')` on line 432.
- **`sumolib`** — **not directly imported**. The script generates SUMO XML by string formatting and shells out to the external `netconvert` binary. The SUMO installation (and `netconvert` on `PATH`) is required at run time, but no Python `sumolib`/`traci` bindings are used in this file.

(`sumolib` / `traci` are used elsewhere in [[envs]] for live simulation — this file is only network/scenario authoring.)
