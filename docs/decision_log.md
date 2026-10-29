# Decision log

Every change to the plan, research questions, metrics or data splits goes here with a date and a reason. Newest entries at the bottom.

## Entries

### 2026-10-29: Phase 0 environment

- Conda env `nettwin`, Python 3.11, all packages installed with pip inside the env (not base). Versions pinned in `requirements.txt`.
- Added `PyYAML` to the package list so scripts can read `config.yaml`.
- `topohub.get(name, use_names=True)` returns a node-link dict (keys `nodes` and `edges`), not a NetworkX graph. Convert with `nx.node_link_graph(g, edges="edges")`.
- Verified: `topozoo/Aarnet` 19 nodes / 24 edges, `topozoo/Geant2012` 37 / 58, `sndlib/germany50` 50 / 88, all connected. Matches PLAN.md section 3.
- The LightGBM pip wheel needs an OpenMP runtime on macOS. Installed `llvm-openmp` from conda-forge into the env (also listed in `environment.yml`).
- Known issue: torch bundles its own `libomp.dylib`. If `torch` is imported before `lightgbm` and LightGBM trains multithreaded, the process segfaults (exit 139). `KMP_DUPLICATE_LIB_OK=TRUE` does not help. Either import `lightgbm` before `torch`, or pass `n_jobs=1` to LightGBM. Only matters for Phase 8.
- Agent model IDs in `config.yaml` left as `null` on purpose; fill from current docs before Phase 6 (PLAN.md section 7).

## Parked ideas

Ideas that came up early, with the phase they belong to. Do not act on them before that phase.

- (none yet)
