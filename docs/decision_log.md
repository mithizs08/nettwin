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

### 2026-10-29: Phase 0 fresh-clone check: passed

- Cloned `origin/main` at commit `1a57a5d` into a temp folder. No `.env` in the clone.
- Built a new throwaway env `nettwin-freshcheck` from the clone's `environment.yml` (did not reuse `nettwin`). Took about 1 minute. Python 3.11.18, same pinned versions, `libomp.dylib` present.
- `pip install -e .`, then `python scripts/phase0_check_env.py` exited 0 and regenerated `results/phase0_env_check.json` byte-identical to the committed file (19/24, 37/58, 50/88, all connected).
- `pytest`: 5 passed. LightGBM also imports and trains in the fresh env.
- Temp clone and throwaway env deleted afterwards.

## Parked ideas

Ideas that came up early, with the phase they belong to. Do not act on them before that phase.

- (none yet)
