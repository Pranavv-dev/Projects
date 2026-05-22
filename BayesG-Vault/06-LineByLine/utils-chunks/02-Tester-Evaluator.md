# utils.py — Tester & Evaluator (lines 460–735)

Line-by-line walkthrough of the two test-time driver classes in [[utils.py]]: `Tester` (online + offline batch testing during/after training) and `Evaluator` (single-shot evaluation runner that also dumps mask visualizations from [[BayesianGraphCMultiAgentPolicy]]).

Related notes: [[01-Trainer]], [[03-Misc-Utils]], [[BayesianGraphCMultiAgentPolicy]], [[models.py#visualize_masks]].

---

## class Tester(Trainer):

```python
class Tester(Trainer):
    def __init__(self, env, model, global_counter, summary_writer, output_path):
        super().__init__(env, model, global_counter, summary_writer)
        self.env.train_mode = False
        self.test_num = self.env.test_num
        self.output_path = output_path
        self.data = []
        logging.info('Testing: total test num: %d' % self.test_num)
```

### `__init__` (lines 544–550)

- **544**: signature accepts `env, model, global_counter, summary_writer, output_path` — five positional args (no `env_name`, no `algo_name`, no `logger`).
- **545**: `super().__init__(env, model, global_counter, summary_writer)` — calls `Trainer.__init__` with only four positionals. **This is a bug** (see [[#Bugs / oddities]]): `Trainer.__init__` (line 141) is declared as `(env, env_name, algo_name, model, global_counter, summary_writer, output_path=None, logger=None)`. The four positionals will be bound as `env=env, env_name=model, algo_name=global_counter, model=summary_writer`, leaving `global_counter` and `summary_writer` unset and triggering a `TypeError` (missing required `global_counter`). In practice `Tester` is never constructed in this codebase (only `Evaluator` is), which masks the bug.
- **546**: explicitly puts env into eval mode (`train_mode = False`), overriding `Trainer.__init__`'s `self.env.train_mode = True`.
- **547**: pulls `test_num` (count of test episodes / seeds) off the env.
- **548**: stores output dir on self.
- **549**: re-initialises `self.data = []` (would already be set by parent if super hadn't blown up).
- **550**: log line announcing the test count.

### `run_offline()` (lines 552–566)

```python
    def run_offline(self):
        # enable traffic measurments for offline test
        is_record = True
        record_stats = False
        self.env.cur_episode = 0
        self.env.init_data(is_record, record_stats, self.output_path)
        rewards = []
        for test_ind in range(self.test_num):
            rewards.append(self.perform(test_ind))
            self.env.terminate()
            time.sleep(2)
            self.env.collect_tripinfo()
        avg_reward = np.mean(np.array(rewards))
        logging.info('Offline testing: avg R: %.2f' % avg_reward)
        self.env.output_data()
```

- **553–555**: `is_record = True`, `record_stats = False`. The "record" flag tells the env to collect per-step traffic measurements; `record_stats` controls a separate (often heavier) statistics sink which is left off for offline runs.
- **556**: `self.env.cur_episode = 0` — resets episode counter on the env so file naming starts from 0.
- **557**: `self.env.init_data(is_record, record_stats, self.output_path)` — env-side hook that opens CSV writers / pre-allocates buffers under `output_path`. For ATSC envs this is where `traffic_log` and `control_log` files get prepared.
- **558**: local `rewards` list.
- **559**: `for test_ind in range(self.test_num):` — **this is the seed loop**: each `test_ind` is passed to `env.reset(test_ind=...)` deep inside `perform`, where it normally seeds the route/demand generation. The loop runs the env on `test_num` distinct seeds back-to-back.
- **560**: `rewards.append(self.perform(test_ind))` — calls **inherited** `Trainer.perform(test_ind)` (Tester doesn't override). Note `Trainer.perform` *returns a tuple* `(mean_reward, std_reward)`, so `rewards` is a list of tuples; the `np.mean` two lines down silently averages over both columns. See [[#Bugs / oddities]].
- **561**: `self.env.terminate()` — kills the SUMO process (or the analogous env subprocess) cleanly between seeds.
- **562**: `time.sleep(2)` — race-condition guard so SUMO ports are released before next reset.
- **563**: `self.env.collect_tripinfo()` — pulls SUMO's per-vehicle tripinfo XML into the env's data buffer for the just-finished episode.
- **564–565**: aggregate + log mean reward.
- **566**: `self.env.output_data()` — flush all accumulated buffers to disk (CSVs / JSON, env-specific).

No DataFrame is written by `run_offline` itself; **all CSV output is delegated to `env.output_data()`**.

### `run_online(coord)` (lines 568–590)

```python
    def run_online(self, coord):
        self.env.cur_episode = 0
        while not coord.should_stop():
            time.sleep(30)
            if self.global_counter.should_test():
                rewards = []
                global_step = self.global_counter.cur_step
                for test_ind in range(self.test_num):
                    cur_reward = self.perform(test_ind)
                    self.env.terminate()
                    rewards.append(cur_reward)
                    log = {'agent': self.agent,
                           'step': global_step,
                           'test_id': test_ind,
                           'reward': cur_reward}
                    self.data.append(log)
                avg_reward = np.mean(np.array(rewards))
                self._add_summary(avg_reward, global_step)
                logging.info('Testing: global step %d, avg R: %.2f' %
                             (global_step, avg_reward))
                # self.global_counter.update_test(avg_reward)
        df = pd.DataFrame(self.data)
        df.to_csv(self.output_path + 'train_reward.csv')
```

- **569**: reset `cur_episode`.
- **570**: `while not coord.should_stop()` — `coord` is the `tf.train.Coordinator`-style stop signal driven by the main training thread. Tester runs as a sibling thread.
- **571**: `time.sleep(30)` — polling interval; tester wakes every 30 s to check if it's time to test.
- **572**: `if self.global_counter.should_test()`: gated by `test_step` (set in `Counter.__init__`).
- **573–574**: snapshot the current global step.
- **575–583**: same seed loop as offline; each iteration: `perform(test_ind)` → terminate → append a log row with `agent / step / test_id / reward`. **`cur_reward` is the tuple `(mean, std)` from `perform`** so the `reward` column becomes a tuple — see [[#Bugs / oddities]].
- **584–585**: `avg_reward = np.mean(np.array(rewards))` then `self._add_summary(avg_reward, global_step)` — pushes scalar onto TensorBoard under `train_reward` because `_add_summary` defaults `is_train=True` (probably wrong here; the metric is a test metric).
- **587**: log the round.
- **588**: `# self.global_counter.update_test(avg_reward)` — dead code, commented-out best-checkpoint mechanism.
- **589–590**: when `coord.should_stop()` fires, dump `self.data` to `train_reward.csv` under `output_path`. Filename is misleading — these are *test* rewards collected during training. Also note `self.output_path + 'train_reward.csv'` uses string concatenation, so `output_path` must end with `/`.

### `perform()` override

**There is none.** `Tester` does *not* override `perform`; it inherits `Trainer.perform` (lines 369–470), which is the very same body that `Evaluator.perform` later duplicates verbatim (with two small tweaks). See [[#Bugs / oddities]].

---

## class Evaluator(Tester):

```python
class Evaluator(Tester):
    def __init__(self, env, env_name, model, output_path, gui=False):
        self.env = env
        self.env_name = env_name
        self.model = model
        self.agent = self.env.agent
        self.env.train_mode = False
        self.test_num = self.env.test_num
        self.gui = gui
        print(f"Model Name : {self.model.name}")
        
        # Create a default output directory if none provided
        if output_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
            self.output_path = os.path.join(os.getcwd(), 'demo_output', timestamp)
        else:
            self.output_path = output_path
            
        # Create directory for mask visualizations
        self.mask_viz_dir = os.path.join(self.output_path, 'mask_visualizations')
        os.makedirs(self.mask_viz_dir, exist_ok=True)
        logging.info(f'Saving mask visualizations to: {self.mask_viz_dir}')
```

### `__init__` (lines 594–614)

- **594**: signature is `(env, env_name, model, output_path, gui=False)`. **Notably skips `super().__init__()`** entirely — it manually sets the fields it needs (lines 595–601), so the `Tester.__init__` bug above is never triggered. Side effect: `global_counter`, `summary_writer`, `cur_step`, `n_step`, `algo_name`, `data`, `logger` are **never set**. Anything inherited that touches them (`_add_summary`, `_log_episode`, `Trainer.run`) would crash.
- **595–597**: stash env / env_name / model.
- **598**: `self.agent = self.env.agent` — note this differs from `Trainer.__init__`, which for `Pandemic` / `Large_city` derives `agent` from `algo_name`. Evaluator can't dispatch on algo_name because it never received one — fine for ATSC envs which set `env.agent` directly, but **broken for Pandemic / Large_city evaluation**.
- **599**: eval-mode env.
- **600**: cache `test_num`.
- **601**: GUI flag.
- **602**: `print(...)` (not `logging.info`) of the model name — debug leftover.
- **605–609**: default `output_path` to `cwd/demo_output/<YYYYMMDDHHMM>` if `None` was passed.
- **612**: `self.mask_viz_dir = output_path/mask_visualizations` — distinct subdir for mask PNGs.
- **613**: `os.makedirs(..., exist_ok=True)` — idempotent.
- **614**: log the viz dir.

### `perform(test_ind, gui=False)` override (lines 616–718)

`Evaluator.perform` is a **near-verbatim copy of `Trainer.perform`** (lines 369–470). Diffs are noted inline.

- **616**: `def perform(self, test_ind, gui=False):` — same signature.
- **617–625**: env reset branching by `env_name`. Identical to parent **except** Pandemic branch calls `self.env.reset()` then `get_state_()`, Large_city calls `clear()` → `reset()` → `get_state_()`, default calls `self.env.reset(gui=gui, test_ind=test_ind)`. The `gui=gui` here is the *parameter*, not `self.gui` — `run()` (below) passes `self.gui` in explicitly so it works out.
- **627–630**: standard prologue: `rewards = []`, `done = True`, `self.model.reset()`, `step = 0`.
- **632–647**: `control_interval` discovery, identical to `Trainer.perform`:
  - First check `self.env.sim.control_interval_sec`.
  - Else `self.env.config.getint('control_interval_sec')` inside try/except.
  - Else default to `1`.
- **649**: log the chosen interval.
- **651**: `while True:` — step loop.
- **652–653**: `agent == 'greedy'` → `action = self.model.forward(ob)`.
- **654–658**: else branch — chooses policy mode:
  - `env.name.startswith('atsc')` → `_get_policy(ob, done)` (no `mode` arg ⇒ defaults to `'train'` ⇒ samples from `pi`).
  - Otherwise → `_get_policy(ob, done, mode='test')` (argmax). This is **the only meaningful semantic difference from `Trainer.perform`**, which uses the same branch — actually identical. The semantic split (atsc samples even at eval time vs everything-else does argmax) is intentional: atsc policies are evaluated stochastically.
- **659**: `self.env.update_fingerprint(policy)` — keeps ma2c fingerprints fresh even at test time.

#### Mask visualization block (lines 661–679) — **the key difference**

- **662**: `current_time = getattr(self.env, 'cur_sec', step)` — prefers simulation seconds; falls back to local `step` (which is always `0`; never incremented in this method — see oddities).
- **663–665**: three `print()` debug lines (`current_time`, presence of `visualize_masks`, `current_time % (500 * control_interval)`). The printed modulus uses `500`, but the *actual* check on 666 uses `200`. Leftover debug.
- **666**: `if current_time % (200 * control_interval) == 0 and hasattr(self.model, 'visualize_masks'):` — fires when sim time hits an integer multiple of `200 * control_interval`. In `Trainer.perform` (line 418) the period is `500 * control_interval`, so **Evaluator visualizes 2.5× more often** than Trainer's online test path.
- **667**: log the trigger.
- **668**: `if self.mask_viz_dir is not None:` — always true in Evaluator (set in `__init__`), but defensive.
- **669–671**: `self.model.visualize_masks(current_time, save_path=self.mask_viz_dir, draw_whole=True)` — **draw_whole=True is unique to Evaluator**. `Trainer.perform` calls `visualize_masks(current_time, save_path=self.mask_viz_dir)` without `draw_whole`. See [[#How visualize_masks is triggered]].
- **672–673**: catch + log on failure.
- **674–679**: the `else` (`mask_viz_dir is None`) fallback path — calls `visualize_masks(current_time)` with no save_path, intended to display interactively. Unreachable from Evaluator because `mask_viz_dir` is always set.

#### Env step branching (lines 681–707)

Identical to `Trainer.perform`:
- **681–686**: CACC envs (`slowdown`, `catchup`) — unpack 4 values, take `done[0]`, sum reward for `global_reward`, scalarise reward if `coop_gamma <= 0`.
- **687–692**: `Grid`/`Monaco` — same shape.
- **693–699**: `Pandemic` — adds `done.astype(float64)` conversion.
- **700–705**: `Large_city` — env returns `global_reward` as 4th tuple element directly.
- **706–707**: default — env returns global_reward as 4th element.

#### Loop tail (lines 709–714)

- **709**: append `global_reward` to local list.
- **710**: comment confirms `step` is *not* incremented — relies on `env.cur_sec`. But `current_time` falls back to `step` on line 662 when `cur_sec` is missing, and `step` stays `0` forever, so the mod-test passes every iteration — see [[#Bugs / oddities]].
- **711–713**: termination on `done`.
- **714**: rotate observation.

#### Return (lines 716–718)

```python
        mean_reward = np.mean(np.array(rewards))
        std_reward = np.std(np.array(rewards))
        return mean_reward, std_reward
```

Returns the `(mean, std)` tuple — matches parent.

### `run()` (lines 720–735)

```python
    def run(self):
        if self.gui:
            is_record = False
        else:
            is_record = True
        record_stats = False
        self.env.cur_episode = 0
        self.env.init_data(is_record, record_stats, self.output_path)
        time.sleep(1)
        for test_ind in range(self.test_num):
            reward, _ = self.perform(test_ind, gui=self.gui)
            self.env.terminate()
            logging.info('test %i, avg reward %.2f' % (test_ind, reward))
            time.sleep(2)
            self.env.collect_tripinfo()
        self.env.output_data()
```

- **721–724**: GUI mode disables recording (`is_record = False`); headless mode records.
- **725**: `record_stats = False`.
- **726**: reset episode counter.
- **727**: `self.env.init_data(is_record, record_stats, self.output_path)` — env-side prep (same as `Tester.run_offline`). For atsc envs this configures `traffic_log` / `control_log`.
- **728**: `time.sleep(1)` — small pause before launching the seed loop, presumably for GUI window readiness.
- **729**: `for test_ind in range(self.test_num):` — seed loop.
- **730**: `reward, _ = self.perform(test_ind, gui=self.gui)` — **unpacks the tuple correctly** (unlike Tester); discards std. Passes `self.gui` explicitly so the GUI flag propagates into `env.reset`.
- **731**: terminate sim.
- **732**: log per-episode avg reward.
- **733**: 2 s pause.
- **734**: collect SUMO tripinfo.
- **735**: `self.env.output_data()` — flush all data buffers; this is where CSVs land.

#### Diff vs `Tester.run_offline`

| Aspect | `Tester.run_offline` | `Evaluator.run` |
| --- | --- | --- |
| `is_record` | hardcoded `True` | `False` if `gui`, else `True` |
| `time.sleep(1)` before loop | absent | present |
| `perform()` return handling | `rewards.append(self.perform(test_ind))` → list-of-tuples bug | `reward, _ = self.perform(test_ind, gui=self.gui)` → correct |
| Avg-reward aggregation | computes overall `avg_reward = np.mean(...)` and logs | logs per-episode only, no overall aggregate |
| Final write | none beyond `env.output_data()` | none beyond `env.output_data()` |
| GUI propagation | not plumbed | passes `gui=self.gui` into `perform` |

### `init_data` interactions

Both classes funnel through `self.env.init_data(is_record, record_stats, output_path)` exactly once before the seed loop. The signature `(is_record, record_stats, output_path)` is implemented on the env side (see [[atsc-chunks]] for `TrafficSimulator.init_data`). It:

1. Sets `self.is_record` / `self.record_stats` flags on the env.
2. Allocates `self.traffic_data = []`, `self.control_data = []`, etc.
3. Opens output handles or stashes `output_path` for later use in `output_data()`.

After the loop, `env.output_data()` writes all accumulated rows to CSVs under `output_path`.

---

## Output formats — CSV columns and files saved

### `Tester.run_online` → `train_reward.csv`

Written at `self.output_path + 'train_reward.csv'` (note: string concatenation, no separator). Columns (rows appended at line 579–582):

| column | source |
| --- | --- |
| `agent` | `self.agent` (e.g. `ma2c_cu`, `ia2c_fp`) |
| `step` | `self.global_counter.cur_step` |
| `test_id` | `test_ind` |
| `reward` | `cur_reward` — **a `(mean, std)` tuple** (bug) |

### `Tester.run_offline` and `Evaluator.run`

Neither calls `pd.DataFrame` itself. All output is via `env.output_data()`. Typical files (env-dependent, ATSC case):

- `traffic_{episode}.csv` — per-vehicle stats, written by `env.collect_tripinfo()`
- `control_{episode}.csv` — per-intersection action / reward log
- `trip_{episode}.csv` — SUMO tripinfo
- `summary.csv` — env-level aggregates

### `Evaluator` — mask visualizations

Under `<output_path>/mask_visualizations/`, the policy writes PNGs named by `step` (the `current_time` argument). See [[BayesianGraphCMultiAgentPolicy#visualize_masks]] for filename convention.

---

## How `visualize_masks` (from [[BayesianGraphCMultiAgentPolicy]]) is triggered

Full call chain inside `Evaluator.perform`:

1. **utils.py:662** — `current_time = getattr(self.env, 'cur_sec', step)` resolves simulation seconds.
2. **utils.py:666** — predicate `current_time % (200 * control_interval) == 0 and hasattr(self.model, 'visualize_masks')`.
3. **utils.py:670** — `self.model.visualize_masks(current_time, save_path=self.mask_viz_dir, draw_whole=True)`.
4. **models.py:425** — `BayesianGraphModel.visualize_masks(step, save_path, draw_whole)` (in [[models.py]]):
   ```python
   def visualize_masks(self, step, save_path=None, draw_whole=False):
       """Delegate visualization to the policy's visualize_masks method"""
       if hasattr(self.policy, 'visualize_masks'):
           self.policy.visualize_masks(step, save_path, draw_whole)
       else:
           logging.warning("Policy does not have visualize_masks method")
   ```
5. **policies.py:1476** — `BayesianGraphCMultiAgentPolicy.visualize_masks(step, save_path, draw_whole)` produces the per-agent mask heatmaps. The `draw_whole=True` flag (set only by Evaluator, not by Trainer's identical block at line 418) tells the policy to render the global graph in addition to per-agent local views.

The trigger fires once per **`200 * control_interval`** simulated seconds during Evaluator runs (vs `500 * control_interval` for Trainer's mid-training eval).

---

## Bugs / oddities

1. **Broken `Tester.__init__` `super()` call (line 545).** `super().__init__(env, model, global_counter, summary_writer)` passes 4 positionals to `Trainer.__init__`'s `(env, env_name, algo_name, model, global_counter, summary_writer, ...)`. Calling `Tester(...)` raises `TypeError: __init__() missing 2 required positional arguments`. Hidden because no code instantiates `Tester` directly — only `Evaluator` (which skips `super()` entirely) is used.

2. **`Evaluator.__init__` skips `super().__init__`.** `cur_step`, `n_step`, `algo_name`, `data`, `logger`, `global_counter`, `summary_writer` are never set. Any inherited method that touches them would crash. In practice `Evaluator` only invokes `perform` + `_get_policy` + (indirectly) `_get_value`, none of which need those fields.

3. **`Evaluator.agent` derivation is wrong for Pandemic / Large_city.** Line 598 sets `self.agent = self.env.agent` unconditionally. Trainer (line 154 onwards) has special handling that maps `algo_name` → `agent` for those two envs. Evaluating Pandemic/Large_city with this class will likely mis-route in `_get_policy` (which branches on `self.agent.startswith('ma2c')`).

4. **`Tester.run_offline` and `Tester.run_online` mis-handle `perform` return.** `perform` returns `(mean, std)`; both `run_offline` (line 560) and `run_online` (line 576) treat it as a scalar. `run_offline`'s `np.mean(np.array(rewards))` silently averages the column-stacked tuple. `run_online`'s per-row dict stores a tuple under `'reward'`. Only `Evaluator.run` (line 730) unpacks correctly.

5. **`step` is never incremented in `Evaluator.perform`.** The comment at line 710 (`# No need to increment step manually; rely on env.cur_sec`) explains the intent, but the fallback at line 662 (`getattr(self.env, 'cur_sec', step)`) means if `cur_sec` is absent, `current_time` is always 0 ⇒ `0 % anything == 0` ⇒ mask viz fires on **every** loop iteration. Identical issue in `Trainer.perform` line 417.

6. **Debug `print` statements in `Evaluator.perform`** at 663–665 (the modulus print uses `500` while the actual check uses `200`) — stale debug code, will spam stdout in production runs.

7. **Period inconsistency between `Trainer.perform` and `Evaluator.perform`.** Trainer uses `current_time % (500 * control_interval)` (line 418); Evaluator uses `current_time % (200 * control_interval)` (line 666). Likely intentional (more snapshots at eval time) but undocumented.

8. **`draw_whole` only passed by Evaluator** (line 670), not by Trainer (line 422). Two near-identical blocks have drifted.

9. **`Tester.run_online` writes test rewards to TensorBoard under `train_reward`.** Line 585 calls `self._add_summary(avg_reward, global_step)` without `is_train=False`, so it logs to the wrong scalar key.

10. **Misleading filename `train_reward.csv`** for what is actually a *test reward* log (`Tester.run_online` line 590, and `Trainer.run` line 540 — both reuse the same name).

11. **String concatenation for paths.** `self.output_path + 'train_reward.csv'` (line 540 / 590) assumes a trailing `/`. `Evaluator.__init__` correctly uses `os.path.join` for `mask_viz_dir` but the inherited path style is brittle.

12. **`time.sleep(30)` polling in `run_online`** is a coarse heartbeat; if `total_step / test_step` granularity is finer than 30 s of wall time, some `should_test()` windows could be missed.

13. **Massive duplication between `Trainer.perform` (369–470) and `Evaluator.perform` (616–718).** Two ~100-line blocks that differ only in (a) the visualization period (`500` vs `200`), (b) `draw_whole`, and (c) the three debug `print`s. Should be one method with parameters.

14. **`perform`'s `gui` parameter is shadowed.** `Evaluator.perform(self, test_ind, gui=False)` — the parameter exists, but `self.gui` is set in `__init__`. `run()` passes the right value (`gui=self.gui`); ad-hoc callers might forget and silently get `gui=False`.
