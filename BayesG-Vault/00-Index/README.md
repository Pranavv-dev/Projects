---
title: BayesG Vault Index
tags: [moc, index]
---

# BayesG Vault — Index

Open this vault in Obsidian (`Open folder as vault → BayesG-Vault/`) for live wiki-links and graph view.

A line-by-line analysis of the [`Wei9711/BayesG`](https://github.com/Wei9711/BayesG) repository — the NeurIPS 2025 paper *"Bayesian Ego-graph Inference for Networked Multi-Agent Reinforcement Learning"* (Duan, Lu, Xuan). Sources audited: `/tmp/BayesG` at commit `63c9f1d`. Analysis date: 2026-05-22.

**~10,500 lines of analysis across 23 notes covering ~7,300 lines of source code.**

## Quick links

- [[../05-Setup/MacM4-Feasibility|How long will it take on a MacBook Air M4?]] ← the user's question
- [[../05-Setup/Setup-Instructions|Step-by-step setup on macOS arm64]]
- [[../06-LineByLine/policies-chunks/04-BayesianGraph-KEY|⭐ The paper's main contribution: BayesianGraphCMultiAgentPolicy]]
- [[Bugs-Catalog|Bug catalog (every issue surfaced)]]

## Reading order

1. **Newcomer:** start with `BayesG_codebase_analysis.md` at the repo root (one-page overview), then jump into [[../05-Setup/MacM4-Feasibility]].
2. **Reproducer:** [[../05-Setup/Setup-Instructions]] → smoke test → full Grid run.
3. **Paper reviewer:** [[../06-LineByLine/policies-chunks/04-BayesianGraph-KEY]] for the core method, then [[../02-Files/agents/models|models.md]] for the wrapper, then [[../02-Files/config/configs-overview|configs]] for hyperparameters.
4. **Extender / forker:** read every file in `06-LineByLine/` in order.

## Vault layout

```
BayesG-Vault/
├── 00-Index/                 ← you are here
├── 01-Architecture/          ← system overview, data flow
├── 02-Files/                 ← one note per source file
│   ├── main.md
│   ├── agents/{models,policies,utils,gnn}.md
│   ├── envs/{atsc_env split,large_grid_env,real_net_env,Large_city,draw_net}.md
│   ├── envs-data/{large_grid,real_net}_build_file.md
│   └── config/configs-overview.md
├── 03-Algorithms/            ← per-algorithm summaries (stubs)
├── 04-Concepts/              ← ELBO, variational inference, GNNs, etc. (stubs)
├── 05-Setup/
│   ├── MacM4-Feasibility.md  ← runtime estimates ⭐
│   └── Setup-Instructions.md
└── 06-LineByLine/            ← chunk-by-chunk walkthroughs
    ├── policies-chunks/      ← 6 files covering policies.py (2,471 LOC)
    ├── utils-chunks/         ← 2 files covering utils.py (735 LOC)
    └── atsc-chunks/          ← 2 files covering envs/atsc_env.py (664 LOC)
```

## File inventory

| Note                                                                                  | Source coverage                                | LOC (analysis) |
| ------------------------------------------------------------------------------------- | ---------------------------------------------- | -------------: |
| [[../02-Files/main\|main.md]]                                                         | `main.py` (293 LOC)                            |            641 |
| [[../06-LineByLine/utils-chunks/01-Trainer\|01-Trainer]]                              | `utils.py` 1–460                               |            681 |
| [[../06-LineByLine/utils-chunks/02-Tester-Evaluator\|02-Tester-Evaluator]]            | `utils.py` 460–735                             |            343 |
| [[../06-LineByLine/policies-chunks/01-base-classes\|01-base-classes]]                 | `policies.py` 1–192 (Policy / LstmPolicy / FP) |            498 |
| [[../06-LineByLine/policies-chunks/02-NC-classes\|02-NC-classes]]                     | `policies.py` 193–871 (NeurComm)               |            455 |
| [[../06-LineByLine/policies-chunks/03-GraphC-and-MaskGen\|03-GraphC-and-MaskGen]]     | `policies.py` 872–1213                         |            323 |
| ⭐ [[../06-LineByLine/policies-chunks/04-BayesianGraph-KEY\|04-BayesianGraph-KEY]]    | `policies.py` 1214–1685 — **paper's method**   |            963 |
| [[../06-LineByLine/policies-chunks/05-Consensus-CommNet-DIAL\|05-Consensus-CommNet-DIAL]] | `policies.py` 1686–2046                    |            553 |
| [[../06-LineByLine/policies-chunks/06-LToS\|06-LToS]]                                 | `policies.py` 2047–2471                        |            571 |
| [[../02-Files/agents/models\|models.md]]                                              | `agents/models.py` (820 LOC)                   |            943 |
| [[../02-Files/agents/utils\|agents/utils.md]]                                         | `agents/utils.py` (373 LOC)                    |            419 |
| [[../02-Files/agents/gnn\|gnn.md]]                                                    | `agents/gnn.py` (123 LOC)                      |            149 |
| [[../06-LineByLine/atsc-chunks/01-init-step-reset\|atsc 01]]                          | `envs/atsc_env.py` 1–330                       |            621 |
| [[../06-LineByLine/atsc-chunks/02-state-reward-utility\|atsc 02]]                     | `envs/atsc_env.py` 330–664                     |            599 |
| [[../02-Files/envs/large_grid_env\|large_grid_env.md]]                                | `envs/large_grid_env.py` (198 LOC)             |            331 |
| [[../02-Files/envs/real_net_env\|real_net_env.md]]                                    | `envs/real_net_env.py` (255 LOC)               |            354 |
| [[../02-Files/envs/Large_city\|Large_city.md]]                                        | `envs/Large_city.py` (630 LOC)                 |            725 |
| [[../02-Files/envs/draw_net\|draw_net.md]]                                            | `envs/draw_net.py` (110 LOC)                   |             79 |
| [[../02-Files/envs-data/large_grid_build_file\|large_grid_build_file.md]]             | `envs/large_grid_data/build_file.py` (449)     |            196 |
| [[../02-Files/envs-data/real_net_build_file\|real_net_build_file.md]]                 | `envs/real_net_data/build_file.py` (167)       |            206 |
| [[../02-Files/config/configs-overview\|configs-overview.md]]                          | All 40 `.ini` files in `config/`               |            350 |
| [[../05-Setup/MacM4-Feasibility\|MacM4-Feasibility]]                                  | macOS arm64 portability + runtime              |            185 |
| [[../05-Setup/Setup-Instructions\|Setup-Instructions]]                                | Step-by-step install                           |            301 |
