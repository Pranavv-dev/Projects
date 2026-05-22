# main.py — CLI Entrypoint Walkthrough

`main.py` is the top-level command-line entrypoint for the BayesG project. It parses CLI arguments, loads `.ini` configs, builds environments via [[Large_city_Env]] / [[LargeGridEnv]] / [[RealNetEnv]], instantiates an agent from [[models]] (the registry in `init_agent`), and dispatches to either [[Trainer]] or [[Evaluator]] from [[utils]]. There are two sub-commands: `train` and `evaluate`.

---

## Imports

```python
import argparse
import configparser
import logging
import threading
from torch.utils.tensorboard.writer import SummaryWriter
from envs.large_grid_env import LargeGridEnv
from envs.real_net_env import RealNetEnv
from envs.Large_city import Large_city_Env
from agents.models import IA2C, IA2C_FP, MA2C_NC, IA2C_CU, MA2C_CNET, MA2C_DIAL, BayesianGraph, IA2C_LToS
from utils import (Counter, Trainer, Tester, Evaluator,
                   check_dir, copy_file, find_file,
                   init_dir, init_log, init_test_flag)

import torch
import os
import datetime
import random
import numpy as np
```

| Import | Purpose |
| --- | --- |
| `argparse` | CLI flag parsing in `parse_args()`. |
| `configparser` | Reads `.ini` config files in `train()` and `evaluate_fn()`. |
| `logging` | Standard logging; initialised by `init_log` from [[utils]]. |
| `threading` | Imported but **never used** — dead import (the comment "disable multi-threading for safe SUMO implementation" hints it was previously needed). |
| `SummaryWriter` | TensorBoard event writer; only used in `train()` at line 206. |
| `LargeGridEnv` | 5x5 synthetic grid SUMO env, used for `env_name == "Grid_ATSC"`. See [[large_grid_env]]. |
| `RealNetEnv` | Real-world Monaco SUMO env, used for `env_name == "Monaco"`. See [[real_net_env]]. |
| `Large_city_Env` | Large city env (different constructor signature — note the lowercase `city`), used for `env_name == "Large_city"`. See [[Large_city]]. |
| `IA2C, IA2C_FP, MA2C_NC, IA2C_CU, MA2C_CNET, MA2C_DIAL, BayesianGraph, IA2C_LToS` | Agent classes from [[models]] used in the `init_agent` algo registry. |
| `Counter` | Step / log / test interval counter from [[utils]]. |
| `Trainer` | Driver loop for training (`trainer.run()`). |
| `Tester` | Imported but **never used** in this file. |
| `Evaluator` | Driver loop for evaluation (`evaluator.run()`). |
| `check_dir` | Verifies directory exists (used in `evaluate_fn`). |
| `copy_file` | Copies the config into `dirs['data']` for reproducibility. |
| `find_file` | Finds the saved config inside a checkpoint dir. |
| `init_dir` | Creates `log/`, `data/`, `model/` (+ `eva_data`, `eva_log`) under `base_dir`. |
| `init_log` | Configures Python logging to write to `log/` directory. |
| `init_test_flag` | Imported but **never used**. |
| `torch` | Imported but only used implicitly via the SummaryWriter and agent models — never referenced directly. |
| `os` | Path manipulation. |
| `datetime` | Imported but **never used** directly here. |
| `random` | Random port selection in `init_env` and seeding context. |
| `numpy as np` | Imported but **never used** directly here (the underlying envs/models do). |

Dead/unused imports: `threading`, `Tester`, `init_test_flag`, `torch`, `datetime`, `numpy`.

---

## `parse_args()` (lines 23-59)

```python
def parse_args():
    default_base_dir = '/BayesG'
    default_config_dir = './config/config_BayesG_grid.ini'
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str, required=False,
                        default=default_base_dir, help="experiment base dir")
```

- **`--base-dir`** (top-level, applies to both subcommands): defaults to `'/BayesG'`, an **absolute path at filesystem root** that will not exist on most machines. This silently leads to permission errors when `init_dir` tries to `mkdir` under it. See Bugs section.
- The two defaults are declared as locals on lines 24-25.

```python
    subparsers = parser.add_subparsers(dest='option', help="train or evaluate")
```

- `dest='option'` is the dispatch key, read at line 290 (`if args.option == 'train'`).
- **No `required=True` on the subparser**, so omitting the subcommand yields `args.option is None`, which is caught manually at lines 56-58 (prints help and `exit(1)`).

### Train subparser (lines 34-45)

```python
sp_train = subparsers.add_parser('train', help='train a single agent under base dir')
sp_train.add_argument('--config-dir', type=str, required=False,
                default=default_config_dir, help="experiment config path")
sp_train.add_argument('--n-heads', type=int, default=4,
                help='Number of attention heads for GAT')
sp_train.add_argument('--gnn-type', type=str, default='gcn',
                choices=['gat', 'gcn', 'sage'],
                help='Type of GNN to use (gat, gcn, or sage)')
sp_train.add_argument('--is-mlp-gnn', type=bool, default=False,
                help='Flag to use MLP for GNN (true or false)')
sp_train.add_argument('--n-neighbor-hops', type=int, default=1,
                help='Number of neighbor hops for GNN')
```

| Flag | Type | Default | Choices | Used? |
| --- | --- | --- | --- | --- |
| `--config-dir` | str | `./config/config_BayesG_grid.ini` | — | Yes — read into `args.config_dir` at `train()` line 141. |
| `--n-heads` | int | `4` | — | **NO** — never referenced anywhere in `main.py`. Unwired. |
| `--gnn-type` | str | `'gcn'` | `gat`/`gcn`/`sage` | **NO** — `init_agent` reads `gnn_type` from the `MODEL_CONFIG` section of the .ini file (lines 98, 111) with fallback `'gat'`, not from `args.gnn_type`. Unwired. |
| `--is-mlp-gnn` | bool | `False` | — | **NO** — never referenced. Also, `type=bool` is a classic argparse bug: `bool("False") == True`. Unwired. |
| `--n-neighbor-hops` | int | `1` | — | **NO** — never referenced. The hop count is effectively hardcoded as `adj_order = 30` inside `init_agent` (line 94/107/120) when `env_name == "Large_city"`. Unwired. |

**Confirmation of un-wiring:** In `train(args)` only `args.base_dir` and `args.config_dir` are read — the four GNN-related flags are dropped on the floor. They are not passed into `config['MODEL_CONFIG']`, not into `init_agent`, and not into any model constructor. They exist only as argparse metadata.

### Evaluate subparser (lines 47-53)

```python
sp_eval = subparsers.add_parser('evaluate', help="evaluate and compare agents under base dir")
sp_eval.add_argument('--evaluation-seeds', type=str, required=False,
                default=','.join([str(i) for i in range(2000, 2500, 10)]),
                help="random seeds for evaluation, split by ,")
sp_eval.add_argument('--demo', action='store_true', help="shows SUMO gui")
sp_eval.add_argument('--checkpoint', type=str, required=False, default=None,
                help="Path to a specific model checkpoint to evaluate")
```

| Flag | Type | Default |
| --- | --- | --- |
| `--evaluation-seeds` | comma-string | `"2000,2010,2020,...,2490"` (50 seeds) |
| `--demo` | flag (store_true) | `False` — enables SUMO GUI and suppresses log-dir creation |
| `--checkpoint` | str (path) | `None` — when set, triggers timestamp-based checkpoint resolution in `evaluate_fn` |

Final lines:

```python
args = parser.parse_args()
if not args.option:
    parser.print_help()
    exit(1)
return args
```

Manual subcommand validation — see Bugs section about using `required=True` on the subparser instead.

---

## `init_env(config, port)` (lines 62-77)

```python
def init_env(config, port=None):
    env_name = config.get('env_name')
    # Generate random port between 0 and 50000 if not specified
    if port is None:
        port = random.randint(0, 50000)
    if env_name == "Grid_ATSC":# Adaptive traffic signal control
        return LargeGridEnv(config, port=port) # 5*5 synthetic traffic grid 
    elif env_name == "Monaco":
        return RealNetEnv(config, port=port) # Real-world Monaco City
    elif env_name == "Large_city":
        net_path = config.get('net_path')
        sim_path = config.get('sim_path')
        reward_scale = config.get('reward_scale')
        return Large_city_Env(net_path, sim_path,None,reward_scale) 
    else:
        raise ValueError(f"Invalid environment name: {env_name}")
```

`config` here is the `ENV_CONFIG` *section proxy* (callers pass `config['ENV_CONFIG']`), so `config.get('env_name')` is the `configparser` section method, not a dict-`get` with a default.

### Branch 1 — `Grid_ATSC`

Returns [[LargeGridEnv]]`(config, port=port)` — the 5x5 synthetic traffic grid.

### Branch 2 — `Monaco`

Returns [[RealNetEnv]]`(config, port=port)` — the real-world Monaco scenario.

### Branch 3 — `Large_city`

```python
net_path = config.get('net_path')
sim_path = config.get('sim_path')
reward_scale = config.get('reward_scale')
return Large_city_Env(net_path, sim_path, None, reward_scale)
```

Special-cased: pulls three named keys (`net_path`, `sim_path`, `reward_scale`) out of `ENV_CONFIG` and passes them positionally. The third positional argument is hardcoded `None` (presumably `output_path` or similar — look at [[Large_city]] constructor). Critically, **no `port` is passed** to `Large_city_Env`, so the random port computed at line 66 is wasted for this branch.

Also note: `reward_scale` is fetched as a **string** (`.get` on a section returns str), not a float — relies on `Large_city_Env` to cast it.

### Port randomization (lines 65-66)

If no `port` is provided, a fresh `random.randint(0, 50000)` is generated. Used to avoid SUMO TraCI port collisions when running multiple instances. `train()` does not specify a port (line 172 → random), `evaluate()` always specifies `port=1` (see Bugs).

### Else branch

Unknown `env_name` raises `ValueError(f"Invalid environment name: {env_name}")`.

---

## `init_agent(env, config, total_step, seed, use_gpu)` (lines 80-136)

Registry table mapping the `algo` string (read from `config['MODEL_CONFIG']['algo']`) to a concrete model constructor from [[models]].

```python
def init_agent(env, config, total_step, seed,use_gpu):
    print("config:",config)
    # Read algo from MODEL_CONFIG section
    algo = config.get('MODEL_CONFIG', 'algo')
    env_name = config.get('ENV_CONFIG', 'env_name')
```

Note: unlike `init_env`, this one receives the *full* `ConfigParser` object (not a section proxy), and uses the two-argument `config.get(section, key)` form.

### Algorithm registry

| `algo` string | Class | Special cases |
| --- | --- | --- |
| `'IA2C'` | [[IA2C]] | none |
| `'IA2C_FP'` | [[IA2C_FP]] | none |
| `'NeurComm'` | [[MA2C_NC]] | Large_city: `coop_gamma = -1`, reads `gnn_type` from MODEL_CONFIG (fallback `'gat'`), passes `gnn_type=` to constructor. Comment "env.agent == 'ma2c_nc'" is a stale hint. |
| `'BayesianGraph'` | [[BayesianGraph]] | Large_city: `coop_gamma = -1`, reads `gnn_type` (fallback `'gat'`) **but does not forward it** to the constructor — the local variable is discarded. |
| `'CommNet'` | [[MA2C_CNET]] | Large_city: `coop_gamma = -1`, **actually calls** `env.cal_n_order_matrix(env.n_agent, adj_order, env.neighbor_mask)` with `adj_order=30` and uses the resulting `neighbor_mask` in the constructor. This is the ONLY branch where the multi-hop adjacency is actually applied. |
| `'Consensus'` | [[IA2C_CU]] | none |
| `'ma2c_dial'` | [[MA2C_DIAL]] | none (note: lowercased algo string, unlike the others) |
| `'IA2C_LToS'` | [[IA2C_LToS]] | none |

All constructors share the call signature:

```python
Class(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, coop_gamma,
      total_step, config['MODEL_CONFIG'], seed=seed, use_gpu=use_gpu)
```

Falls through to `return None` (line 136) for unknown algos — which `evaluate_fn` checks at line 255 but `train` does **not** (`train` would crash on `model.load`/`model.save`).

### Large_city special cases — full quote

For `NeurComm` (lines 92-103):

```python
if env_name == "Large_city":
    coop_gamma = -1
    adj_order = 30
    # neighbor_mask = env.cal_n_order_matrix(env.n_agent,adj_order,env.neighbor_mask) 
    # print("neighbor_mask:",env.neighbor_mask.sum(axis=1))
    # print("high order matrix:",neighbor_mask.sum(axis=1))
    gnn_type = config.get('MODEL_CONFIG', 'gnn_type', fallback='gat')
    return MA2C_NC(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, coop_gamma,
               total_step, config['MODEL_CONFIG'], seed=seed, use_gpu=use_gpu, gnn_type=gnn_type)
```

For `BayesianGraph` (lines 104-116) — identical structure but `gnn_type` is computed and then **silently dropped** because the constructor call omits it:

```python
gnn_type = config.get('MODEL_CONFIG', 'gnn_type', fallback='gat')
return BayesianGraph(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, coop_gamma,
           total_step, config['MODEL_CONFIG'], seed=seed, use_gpu=use_gpu)
```

Three observations on the Large_city branches:

1. **`coop_gamma = -1`** overrides whatever the environment has in `env.coop_gamma`. The `-1` sentinel is the cooperative-discount-disabled convention used internally by the models — see [[models]].
2. **`adj_order = 30`** is set but the call to `env.cal_n_order_matrix(env.n_agent, adj_order, env.neighbor_mask)` is **commented out** in the NeurComm and BayesianGraph branches. So `adj_order` is dead code there. The CommNet branch is the only one that actually uses it (uncommented at line 121).
3. `gnn_type` is only effective for `NeurComm`; for `BayesianGraph` it is computed-then-discarded.

---

## `train(args)` (lines 139-213)

### 1. Config loading (lines 140-149)

```python
base_dir = args.base_dir
config_dir = args.config_dir

# Load config first
config = configparser.ConfigParser()
config.read(config_dir)

# Update config with command line arguments
if not 'MODEL_CONFIG' in config:
    config.add_section('MODEL_CONFIG')
```

The comment "Update config with command line arguments" promises something that never happens — the CLI args (`--n-heads`, etc.) are **never written back into** `config['MODEL_CONFIG']`. The `add_section` call only guarantees the section exists for downstream `config.get('MODEL_CONFIG', ...)` calls.

### 2. Directory setup (lines 151-162)

```python
config_name = os.path.splitext(os.path.basename(args.config_dir))[0]
env_name = config['ENV_CONFIG']['env_name']
log_dir = os.path.join(base_dir, 'log', env_name, config_name)

dirs = init_dir(base_dir, custom_log_dir=log_dir, config=config)

init_log(dirs['log'])
copy_file(config_dir, dirs['data'])
```

- `config_name` = stem of the config filename, e.g. `config_BayesG_grid`.
- `env_name` is re-read from `config['ENV_CONFIG']` (note: it was already needed earlier, but is fetched here for the first time).
- `log_dir` = `<base_dir>/log/<env_name>/<config_name>`. Note **two** stub-levels of nesting.
- `init_dir` returns a dict with keys at least `'log'`, `'data'`, `'model'` — see [[utils]].
- `init_log` redirects Python `logging` to file under `dirs['log']`.
- `copy_file` archives the `.ini` into `dirs['data']` so the run is reproducible from the artefacts.

### 3. Config logging (lines 165-169)

```python
for section in config.sections():
    config_str = f"[{section}]\n"
    for key, value in config.items(section):
        config_str += f"{key} = {value}\n"
    logging.info(config_str.strip())  # Log the entire section at once
```

Dumps the full effective `.ini` into the log file, one `logging.info` call per section.

### 4. Env init + mask print (lines 172-178)

```python
env = init_env(config['ENV_CONFIG'])
print("Adjacency matrix:",env.neighbor_mask)
print("Node degree:",env.neighbor_mask.sum(axis=1))
# logging.info('Adjacency matrix:',env.neighbor_mask)
# logging.info('Node degree:',env.neighbor_mask.sum(axis=1))
logging.info('Training: a dim %r, agent dim: %d' % (env.n_a_ls, env.n_agent))
logging.info('Adjacency matrix Degree: %s', str(env.neighbor_mask.sum(axis=1)))
```

Calls `init_env` with **no port** → random port. Then prints the neighbour adjacency. The commented-out `logging.info` calls are correct to comment out — they used the bad `,`-arg form which `logging` does not splat the way `print` does.

`env.n_a_ls` is per-agent action-space dims (a list), `env.n_agent` is the agent count.

### 5. Counter init (lines 181-184)

```python
total_step = int(config.getfloat('TRAIN_CONFIG', 'total_step'))
test_step = int(config.getfloat('TRAIN_CONFIG', 'test_interval'))
log_step = int(config.getfloat('TRAIN_CONFIG', 'log_interval'))
global_counter = Counter(total_step, test_step, log_step)
```

Reads three `TRAIN_CONFIG` values as floats then casts to int — supports scientific notation like `1e6` in the .ini. `Counter` from [[utils]] tracks training progress.

### 6. Agent init (lines 187-194)

```python
# init centralized or multi agent
use_gpu = True
seed = config.getint('ENV_CONFIG', 'seed')
# print("MODEL_CONFIG contents:")
# for key, value in config['MODEL_CONFIG'].items():
#     print(f"    {key} = {value}")
algo = config.get('MODEL_CONFIG', 'algo')
env_name = config.get('ENV_CONFIG', 'env_name')
model = init_agent(env, config, total_step, seed, use_gpu)
```

**`use_gpu = True` is hardcoded** — there is no CLI flag, no env var, no config key consulted. On a CPU-only machine this will cascade into a CUDA initialisation failure inside the model constructors (or silently fall back depending on the model implementation). See Bugs section.

`algo` and `env_name` are extracted into locals for use by `Trainer` on line 207.

### 7. `load_model` branch (lines 196-203)

```python
load_model = config.getboolean('TRAIN_CONFIG', 'load_model')
logging.info(f'Loading pre-trained model: {load_model}')
if load_model:
    logging.info(f'Loading model from {dirs["model"]}')
    model.load(dirs['model'], train_mode=True)
    logging.info('Model loaded successfully')
else:
    logging.info('No pre-trained model loaded')
```

Reads `TRAIN_CONFIG.load_model` (boolean). When true, calls `model.load(dirs['model'], train_mode=True)` to resume from the same base directory's `model/` folder. There is no error-handling around `load` — it will crash if no checkpoint exists.

### 8. `SummaryWriter` (line 206)

```python
summary_writer = SummaryWriter(dirs['log'], flush_secs=10000)
```

TensorBoard writer pointed at the log dir. `flush_secs=10000` is unusually long (~2.8 h) — most runs that crash will lose events. The comment "disable multi-threading for safe SUMO implementation" (line 205) is a vestige.

### 9. `Trainer.run()` (lines 207-208)

```python
trainer = Trainer(env, env_name, algo, model, global_counter, summary_writer, output_path=dirs['data'], logger=logging)
trainer.run()
```

Hands off control to [[Trainer]] from [[utils]]. The trainer drives the rollout loop, periodic testing, and TensorBoard logging.

### 10. Final save (lines 211-213)

```python
final_step = global_counter.cur_step
model.save(dirs['model'], final_step)
summary_writer.close()
```

After `trainer.run()` returns, the final step counter is read off `Counter.cur_step` and the model is saved into `dirs['model']` with that step as the suffix. The `SummaryWriter` is closed last.

---

## `evaluate_fn(agent_dir, output_dir, seeds, port, demo, checkpoint=None)` (lines 216-268)

```python
def evaluate_fn(agent_dir, output_dir, seeds, port, demo, checkpoint=None):
    agent = agent_dir.split('/')[-1]
    if not check_dir(agent_dir):
        logging.error('Evaluation: %s does not exist!' % agent)
        return
```

- Extracts the trailing path component as `agent` (used only for the error message).
- Validates the directory exists via `check_dir` from [[utils]]; logs and returns silently on failure (no exception — caller cannot tell whether this succeeded).

### Checkpoint timestamp regex (lines 222-234)

```python
if checkpoint is not None:
    # Extract the timestamp from the checkpoint filename
    import re
    match = re.search(r'(\d{12})checkpoint', checkpoint)
    if match:
        timestamp = match.group(1)
        config_dir = find_file(os.path.join(agent_dir, 'data', timestamp))
        model_dir = os.path.join(agent_dir, 'model', timestamp)
    else:
        raise ValueError("Could not extract timestamp from checkpoint filename.")
else:
    config_dir = find_file(agent_dir + '/data/')
    model_dir = agent_dir + '/model/'
```

- `import re` is done **inside** the function — lazy import, but the module-level imports already pull in heavier deps so this is purely stylistic.
- The regex `r'(\d{12})checkpoint'` captures exactly **12 consecutive digits** immediately before the literal `checkpoint`. So a checkpoint filename like `202405010945checkpoint-12000.pt` would yield timestamp `202405010945` (YYYYMMDDHHMM). This is the timestamp under which the run was saved.
- `find_file(os.path.join(agent_dir, 'data', timestamp))` finds the config that was archived alongside the model for that specific run.
- When no checkpoint is provided: `find_file(agent_dir + '/data/')` is the fallback — the string concatenation here will produce paths like `foo//data/` if `agent_dir` already ends in `/`, but `os.path` tolerates this.

### Early-exit + config read (lines 236-247)

```python
if not config_dir:
    return
config = configparser.ConfigParser()
config.read(config_dir)

    # Debug: Print config sections and values
logging.info(f"Config file: {config_dir}")
logging.info(f"Config sections: {config.sections()}")
for section in config.sections():
    logging.info(f"Section {section}:")
    for key, value in config.items(section):
        logging.info(f"  {key} = {value}")
```

If `find_file` returns falsy (no config found), silently returns. Otherwise the config is loaded and fully echoed to the log. Note the indentation hiccup at line 241 (the `# Debug` comment is over-indented).

### Env + agent init (lines 249-256)

```python
env = init_env(config['ENV_CONFIG'], port=port)
env_name = config.get('ENV_CONFIG', 'env_name')
env.init_test_seeds(seeds)

# load model for agent
model = init_agent(env, config, 0, 0,use_gpu = False)
if model is None:
    return
```

- `init_env` is called **with** `port=port` here (unlike `train()`).
- `env.init_test_seeds(seeds)` seeds the env for deterministic evaluation.
- `init_agent` is called with `total_step=0`, `seed=0`, and **`use_gpu=False`** — so evaluation runs on CPU (an explicit choice, opposite of training).
- Guards against unknown `algo` returning `None`.

### Checkpoint loading (lines 258-264)

```python
if checkpoint is not None:
    checkpoint_path = checkpoint if os.path.isabs(checkpoint) else os.path.join(model_dir, os.path.basename(checkpoint))
    if not model.load(checkpoint_path):
        return
else:
    if not model.load(model_dir):
        return
```

- If `checkpoint` is an absolute path, use it as-is; otherwise resolve relative to `model_dir/<basename>`.
- `model.load(...)` is expected to return a truthy value on success; on failure the function returns silently.

### Evaluator (lines 267-268)

```python
evaluator = Evaluator(env, env_name, model, output_dir, gui=demo)
evaluator.run()
```

Hands off to [[Evaluator]]. `gui=demo` enables the SUMO GUI when running with `--demo`.

---

## `evaluate(args)` (lines 271-285)

```python
def evaluate(args):
    base_dir = args.base_dir
    if not args.demo:
        dirs = init_dir(base_dir, pathes=['eva_data', 'eva_log'])
        init_log(dirs['eva_log'])
        output_dir = dirs['eva_data']
    else:
        output_dir = None
    seeds = args.evaluation_seeds
    logging.info('Evaluation: random seeds: %s' % seeds)
    if not seeds:
        seeds = []
    else:
        seeds = [int(s) for s in seeds.split(',')]
    evaluate_fn(base_dir, output_dir, seeds, 1, args.demo, args.checkpoint)
```

- When **not** in `--demo` mode, creates `eva_data/` and `eva_log/` (note the misspelled keyword `pathes` — passed through to [[utils]]`.init_dir`, which presumably accepts it).
- In demo mode, `output_dir = None` so the `Evaluator` will not persist rollouts.
- `seeds` is parsed from the comma-separated string into a list of ints; empty string yields `[]`.
- Calls `evaluate_fn(base_dir, output_dir, seeds, 1, args.demo, args.checkpoint)` — **`port=1` is hardcoded**, which collides with privileged ports on Linux and conflicts with the `init_env` random-port fallback. See Bugs section.

---

## `__main__` dispatch (lines 288-293)

```python
if __name__ == '__main__':
    args = parse_args()
    if args.option == 'train':
        train(args)
    else:
        evaluate(args)
```

Simple if/else on `args.option`. Because of the manual check at line 56-58, by this point `args.option` is guaranteed to be either `'train'` or `'evaluate'`. (Any other value would only happen if someone called `parse_args()` programmatically with a weird argv.)

---

## Bugs / oddities

### Unwired CLI flags

`--n-heads`, `--gnn-type`, `--is-mlp-gnn`, `--n-neighbor-hops` are all defined on the `train` subparser but **never read** from `args` anywhere. The `gnn_type` actually used by [[MA2C_NC]] in Large_city mode is pulled from `config['MODEL_CONFIG']['gnn_type']` via `configparser`, not from the CLI flag. Users assuming `--gnn-type sage` will switch the GNN type will be silently wrong.

Additionally, `--is-mlp-gnn` declares `type=bool`, which is the well-known argparse pitfall: `bool("False")` is `True`, so passing `--is-mlp-gnn False` actually sets the flag to `True`. Should be `action='store_true'` (or use `argparse.BooleanOptionalAction`).

### Hardcoded `use_gpu = True` in `train()`

Line 187. No way to override from CLI or config. Causes failures on CPU-only hardware. Fix: read a `TRAIN_CONFIG.use_gpu` bool, or expose a `--cpu`/`--gpu` flag. Compare with `evaluate_fn` line 254 which hardcodes the opposite (`use_gpu=False`) — there is no symmetry between the two paths.

### Hardcoded `port=1` in `evaluate_fn` call

Line 285: `evaluate_fn(base_dir, output_dir, seeds, 1, args.demo, args.checkpoint)`. Port 1 is privileged (root-only on Linux) and also conflicts with anything else binding low ports. Since the only env that uses the port is SUMO/TraCI (and `Large_city_Env` ignores it anyway), this will at minimum break parallel evaluations. Should be a CLI flag or random as in `train`.

### Default `--base-dir = '/BayesG'`

Line 24: `default_base_dir = '/BayesG'`. This is an **absolute path at the filesystem root**, which:

1. Almost certainly does not exist on a fresh machine.
2. Cannot be created without `sudo`.
3. Has nothing to do with the project root — it leaks an author-specific assumption (a Docker/VM mount point perhaps).

A safer default would be `./BayesG` or `os.path.expanduser('~/BayesG')`, or making it `required=True`.

### `args.option` default behavior

`subparsers = parser.add_subparsers(dest='option', help="train or evaluate")` does not pass `required=True`. On Python 3.7+ subparsers are optional by default, so `args.option` is `None` when omitted. The manual check at line 56-58 prints help and exits with status 1, but the idiomatic fix is `subparsers = parser.add_subparsers(dest='option', required=True)`, which would emit a clearer error from argparse itself.

### Other smell-tests

- **Unused imports**: `threading`, `Tester`, `init_test_flag`, `torch`, `datetime`, `numpy`.
- **`adj_order = 30` is dead code** in the NeurComm and BayesianGraph branches of `init_agent` (the `cal_n_order_matrix` call is commented out). Only the `CommNet` branch actually applies it.
- **`gnn_type` is computed-then-discarded** in the `BayesianGraph` Large_city branch (line 111 → 112-113 constructor does not forward it). Bug or deliberate?
- **No error handling around `model.load` in `train()`** when `load_model=True` (line 200).
- **Print-vs-logging inconsistency** in `train()` lines 173-178: the adjacency matrix goes through `print` while the degree summary goes through `logging` — so logs and stdout diverge.
- **`Large_city_Env` ignores the random port** — the third positional arg is `None` and no port kwarg exists in `init_env`'s Large_city branch.
- **`evaluate_fn` swallows errors** by returning silently on every failure path (lines 220, 237, 256, 261, 264). This makes diagnosing eval failures very painful — no stderr, no non-zero exit code.

---

## End-to-end call graph

```
__main__
  |
  parse_args()                                     # main.py:23
  |
  +-- args.option == 'train' ----> train(args)     # main.py:139
  |     |
  |     configparser.read(args.config_dir)         # loads .ini
  |     init_dir(base_dir, custom_log_dir, config) # utils -> creates log/data/model
  |     init_log(dirs['log'])                      # utils -> configures logging
  |     copy_file(config_dir, dirs['data'])        # utils -> archive config
  |     init_env(config['ENV_CONFIG'])             # main.py:62
  |       |
  |       +-- env_name == 'Grid_ATSC' ----> LargeGridEnv(config, port)   # envs/large_grid_env.py
  |       +-- env_name == 'Monaco'    ----> RealNetEnv(config, port)     # envs/real_net_env.py
  |       +-- env_name == 'Large_city'----> Large_city_Env(net,sim,None,reward_scale)  # envs/Large_city.py
  |     |
  |     Counter(total_step, test_step, log_step)   # utils
  |     init_agent(env, config, total_step, seed, use_gpu=True)          # main.py:80
  |       |
  |       +-- algo dispatch on config['MODEL_CONFIG']['algo']:
  |             IA2C | IA2C_FP | MA2C_NC | BayesianGraph | MA2C_CNET |
  |             IA2C_CU | MA2C_DIAL | IA2C_LToS                          # agents/models.py
  |     |
  |     [optional] model.load(dirs['model'], train_mode=True)            # if load_model=True
  |     SummaryWriter(dirs['log'])                                       # torch.utils.tensorboard
  |     Trainer(env, env_name, algo, model, counter, writer, ...)        # utils
  |     trainer.run()                  # main rollout loop (rollout -> learn -> log -> test)
  |     model.save(dirs['model'], final_step)
  |     summary_writer.close()
  |
  +-- args.option == 'evaluate' --> evaluate(args)  # main.py:271
        |
        init_dir(base_dir, pathes=['eva_data', 'eva_log'])  # only if not --demo
        init_log(dirs['eva_log'])
        seeds = parse_csv(args.evaluation_seeds)
        evaluate_fn(base_dir, output_dir, seeds, port=1, demo, checkpoint)  # main.py:216
          |
          [optional] regex extract 12-digit timestamp from checkpoint filename
          find_file(<data dir>) -> config_dir
          configparser.read(config_dir)
          init_env(config['ENV_CONFIG'], port=1)
          env.init_test_seeds(seeds)
          init_agent(env, config, 0, 0, use_gpu=False)
          model.load(checkpoint_path | model_dir)
          Evaluator(env, env_name, model, output_dir, gui=demo)
          evaluator.run()              # eval rollout loop -> writes eva_data/
```

Cross-references:
- Agent classes: see [[models]] (registry described above).
- Env classes: [[large_grid_env]], [[real_net_env]], [[Large_city]].
- Training/evaluation infrastructure: [[utils]] (`Counter`, `Trainer`, `Evaluator`, `init_dir`, `init_log`, `find_file`, `copy_file`, `check_dir`).
- Configs consumed: see [[config]] folder for the `.ini` files; the default is `./config/config_BayesG_grid.ini`.
