# draw_net.py — SUMO Network Overlay Visualizer

File: `/tmp/BayesG/envs/draw_net.py` (110 lines)

A **standalone script** (not a module / not imported anywhere) that renders a SUMO `.net.xml` road network with an overlaid "neighbor / environment" graph from an `.add.xml` polylines file, plus traffic-light markers from a JSON file. Output is `network_overlay.png` plus an interactive matplotlib window.

Related: [[real_net_build_file]], [[draw_net_real]] (likely sibling), [[policies]] (also contains draw_networkx code for a different purpose).

---

## What it draws

Three layers, in z-order:

1. **Base road network** (gray, thin, `zorder=1`): every non-internal `<edge>` in `newyork33.net.xml`. Geometry preference: edge `shape` attribute → first lane's `shape` → straight line between `from`/`to` junction coordinates.
2. **Environment graph edges** (orange `#e46d4c`, thicker, `zorder=3`): polylines parsed from `neighbor_graph.add.xml`, each with a `shape` and optional `color`/`width` attributes.
3. **Traffic lights** (blue `#1f77b4` circles, markersize 15, `zorder=4`): one dot per entry in `tls_positions.json`.

Final figure includes:
- Equal aspect ratio (`ax.set_aspect('equal', 'box')`).
- Title `"NewYork33: SUMO Network with Environment Graph"`.
- X/Y axis labels.
- Legend with two entries: "Road network" (gray) and "Edge in Environment Graph" (orange). **No legend entry for the traffic-light dots** — minor oversight.
- Saved to `NewYork33/network_overlay.png`; also `plt.show()`'d.

Toolchain: **`xml.etree.ElementTree`** (streaming `iterparse` with `elem.clear()` for memory frugality on large `.net.xml`), **`matplotlib.pyplot`**, **`matplotlib.lines`**, plus `json` for the TLS positions. No `networkx` — despite the structure being graph-like, drawing is direct matplotlib `ax.plot` for every polyline.

### Notable quirks

- **`path = "NewYork33"`** is hardcoded at module level (line 5) — not parameterized. To run on a different scenario you'd edit the source.
- **Color parsing for polylines** is implemented (lines 80–88, supporting comma-separated RGB in `[0,1]` or `[0,255]`) but then **ignored**: line 89 hardcodes `color='#e46d4c'`. So `color_val` is dead.
- The script reads `newyork33.net.xml` **twice** — once to gather junction coords (lines 8–14), once to gather edges (lines 18–47). Could be one pass.

---

## Where it's called from

```
$ grep -rn "draw_net" /tmp/BayesG/
/tmp/BayesG/agents/policies.py:1539:   nx.draw_networkx_nodes(...)    # different — matches "draw_net*" prefix
/tmp/BayesG/agents/policies.py:1545:   nx.draw_networkx_edges(...)
/tmp/BayesG/agents/policies.py:1551:   nx.draw_networkx_labels(...)
/tmp/BayesG/agents/policies.py:1626:   nx.draw_networkx_nodes(...)
/tmp/BayesG/agents/policies.py:1629:   nx.draw_networkx_edges(...)
/tmp/BayesG/agents/policies.py:1636:   nx.draw_networkx_edges(...)
/tmp/BayesG/agents/policies.py:1644:   nx.draw_networkx_nodes(...)
/tmp/BayesG/agents/policies.py:1650:   nx.draw_networkx_labels(...)
```

All grep hits are false positives — they match `draw_networkx_*` inside [[policies]], **not** the `draw_net` script. The script is **never imported** by any Python in the repo. It is run manually, e.g. `python envs/draw_net.py`. This makes it a one-off visualization utility, not part of the simulation pipeline.

---

## Function signatures

There are **no functions and no classes defined** — `draw_net.py` is a top-to-bottom imperative script. Sections, by line range:

| Section | Lines | Purpose |
|---|---|---|
| Imports + `path = "NewYork33"` | 1–5 | `ElementTree`, `pyplot`, `matplotlib.lines as mlines` |
| Parse junction coords | 6–14 | Streaming `iterparse(end)` over `newyork33.net.xml`, collects `{junction_id: (x, y)}` |
| Parse edges → `edges_geometry` | 16–47 | Streaming pass; skips `function="internal"`; tries `edge.shape` → `lane.shape` → straight from/to fallback |
| Parse polylines → `polylines` | 49–65 | `ET.parse(neighbor_graph.add.xml).getroot().findall("polyline")` — each item dict `{coords, color, width}` |
| Plot base roads | 67–73 | `for coords in edges_geometry: ax.plot(xs, ys, gray, lw=0.5, z=1)` |
| Plot neighbor polylines | 75–89 | Per-polyline plot; color parsing computed but hard-overridden to `#e46d4c` |
| Load + plot TLS positions | 91–96 | `json.load` then `ax.plot(x, y, 'o', size 15, '#1f77b4', z=4)` |
| Finalize + save + show | 99–110 | Aspect, legend, title, axis labels, `tight_layout`, `savefig`, `show` |

### Implicit "signatures" (module-level inputs)

The script implicitly expects (relative to its working directory):

- `NewYork33/newyork33.net.xml` — SUMO network.
- `NewYork33/neighbor_graph.add.xml` — `<polyline shape="..." color="..." width="..."/>` entries inside the root.
- `NewYork33/tls_positions.json` — JSON dict `{tls_id: [x, y]}`.

And produces:

- `NewYork33/network_overlay.png`.
