# NetTwin

A knowledge-graph digital twin of a telecom backbone, with GNN anomaly localisation and an LLM root-cause agent. Telemetry is synthetic and topologies are public (Internet Topology Zoo / SNDlib via `topohub`). None of this is Telstra's network.

Work in progress. See `PLAN.md` for the plan and `docs/decision_log.md` for decisions.

## Setup

```bash
conda env create -f environment.yml
conda activate nettwin
pip install -e .
python scripts/phase0_check_env.py
pytest
```

## Phase 1: topology and knowledge graph

```bash
python scripts/phase1_build_kg.py
```

Builds the typed knowledge graph (PoP, Link, RoutingPath, Service; Alert, Maintenance and Runbook as schema) for the dev network (AARNet) and the validation_real networks, exports Neo4j-style CSV to `data/kg/`, and writes `results/phase1_kg_summary.json`, `results/figures/aarnet_kg.png` and `results/figures/aarnet_map.png`. Services and SLAs are synthetic.

## Phase 2: telemetry simulator

```bash
python scripts/phase2_simulate.py
```

Simulates synthetic traffic, routing (via the Phase 1 KG), utilisation, latency and loss with six fault types and distractors, deterministic from seed. Writes sanity figures to `results/figures/phase2_*.png`, the pre-registered difficulty targets (validation networks only) to `results/phase2_summary.json`, and demo scenarios to `data/scenarios/phase2_demo/` (observable telemetry and ground truth in separate folders). All telemetry is synthetic.

## Topology splits and test-driven decisions

Splits are in `PLAN.md` section 3 and `config.yaml`. AARNet is the dev / demo network: it was used to develop and test the simulator in Phases 1 and 2, so its numbers are reported separately as "seen during development". The clean test set is Geant2012, germany50 and five Topology Zoo networks picked by a rule that was pre-registered in `docs/decision_log.md` before it was run (`scripts/select_topologies.py`).

Decisions that were made while looking at a test (or now-dev) topology, as PLAN.md section 3 requires us to disclose (audit IDs from `docs/audit_phases_0_2.md`):
- L2: capacity planning from a noisy history week was triggered by AARNet peaks (median 0.78). Also confirmed on a Gabriel graph.
- L3 / L4: the ECMP cap of 4 was checked against AARNet loads and set with AARNet, Geant2012 and germany50 ECMP counts in view.
- L5: the routing metric floor of 1 was motivated by AARNet's 0 km links.
- L6: the rule "0 km links get the largest adjacent capacity" was designed on AARNet. It now also runs on generated graphs with dual-router PoPs.
- L9: per-PoP time zones were built for real geography. Generated graphs now get pseudo-longitudes, so the feature is exercised in training and validation too.

## Known realistic signatures (read before Phases 4 to 6)

A down interface on a reachable router reads exactly 0 traffic and 0 discards, and its one-hop probe fails. So in the default (non-blind) telemetry, "util exactly 0 with latency missing" identifies a down link almost perfectly: 99.7% of such cells are down links in the audit's re-check (`docs/audit_phases_0_2_reverify.md`, N2). This is what real SNMP shows, so it stays, but it means link_down and node_down are easy to detect from link telemetry alone, even without status events. Telemetry also changes at the fault's start step, so the status-event lag does not delay onset for a telemetry-based detector. Results for link_down and node_down must therefore be reported twice: with status events and the default telemetry, and with the status-blind variant (`spec["status_blind"] = True`, where down links are fully missing and no status or reachability events exist).

## Phase 3: datasets

```bash
python scripts/phase3_build_dataset.py --estimate     # disk and time estimate first (about 8 GB, 3.6 CPU-hours)
python scripts/phase3_build_dataset.py --workers 6    # full build into data/datasets/v1 (gitignored)
```

Builds train (generated graphs), val (generated validation graphs plus validation_real), test (the 7 test networks) and dev (AARNet) scenarios, each in a `with_status` and a `status_blind` variant, with observable data and ground truth in separate trees. The simulator is frozen first (`results/simulator_freeze.json`), and the builder refuses to run if it changed without a logged reason. Counts are in `docs/dataset_card.md` and `results/phase3_dataset_summary.json`.

### Reproducing the datasets (needs the key)

Scenario seeds and ids are derived from a secret 32-byte key, so the repo alone cannot re-run the sampler and recover the faults behind the observable files (`docs/audit_phase3.md`, P1). The key lives only in `.nettwin_dataset_key` at the repo root, which is gitignored. Back it up somewhere safe outside the repo: without it, the datasets can be rebuilt with the same statistics but not byte for byte.

To reproduce `data/datasets/v1` exactly:
1. Clone the repo, create the env, and run `pip install -e .` (see Setup).
2. Copy your key file into the clone root as `.nettwin_dataset_key` (one line of hex, permissions 600).
3. Run `python scripts/phase3_build_dataset.py --workers 6`. The builder refuses to run without the key, and refuses if the frozen simulator files changed.
4. Each scenario's slot (topology and scenario number) and seed are in `data/datasets/v1/truth/<split>/<variant>/manifest.json`. `scenario_seed(key, split, topology, k)` in `src/nettwin/dataset.py` recomputes the seed from the slot.

To start a fresh dataset with a new key (different random draws, same distributions), write 32 random bytes as hex to `.nettwin_dataset_key`, for example with `python -c "import secrets; print(secrets.token_hex(32))" > .nettwin_dataset_key`.

## Phase 4: alerts, baselines and the evaluation harness (validation only)

```bash
python scripts/phase4_build_pretune.py --workers 6   # sensitivity data: val with the pre-tuning fault ranges (needs the key)
python scripts/phase4_baselines.py --workers 6       # B1, B2, B3, B3+ on val (and on the sensitivity copy if built)
python scripts/phase4_tables.py                      # results/phase4_tables.md: tables, claim and expectation verdicts
```

- `src/nettwin/alerts.py`: threshold alerts an operator would have (interface status, PoP reachability, high utilisation, discards, one-hop latency, probe failures, service SLA breaches), built from observable files only. Every field is causal; `alerts_at(alerts, t)` gives the list as known at step t.
- `src/nettwin/evaluation.py`: the shared harness for Phases 4 to 6.
  - Detection: link-level PR-AUC, root and symptomatic separately. Symptoms of concurrent faults are attributed by time and KG neighbourhood.
  - Localisation: root-cause top-1 / top-3 on an oracle window, with expected ranks under ties and set-based scoring for concurrent faults, under three window aggregations.
  - False alarms: rates on fault-free and maintenance-only hours.
  - Calibration: bins and thresholds come from each method's own validation scores. A supervised model is calibrated on validation, never on its training scenarios.
  - Breakdowns: every result is split by fault type, severity class, observability, variant, generated vs real network, root kind and concurrency, with scenario-bootstrap CIs.
  - Verdicts: claims and expectations get a paired-bootstrap verdict (supported, contradicted or inconclusive).
- `src/nettwin/baselines.py`: B1 EWMA z-score, B2 Isolation Forest, B3 tomography over alerts as pre-registered, and B3+, a disclosed variant whose link paths include the end routers. None uses time of day, time since the scenario start or file metadata.
- The definitions and expectations were committed before the first validation run. The revisions after the independent audit (`docs/audit_phase4.md`) are logged by ID in `docs/decision_log.md`.

## Phase 5: GNN

```bash
python scripts/phase5_gnn.py            # train on train (21 runs, about 2.5 h on CPU), select on a val subset, evaluate on val
python scripts/phase5_freeze.py         # validation only: freeze calibration and hashes, held-out headline, blanking check
python scripts/phase5_test.py           # test and dev, once: loads frozen files only, refuses on any hash mismatch
python scripts/phase5_tables.py         # results/phase5_tables.md
```

- `src/nettwin/gnn/`: line-graph data (links are nodes; inputs are the Phase 4 B2 window features plus static inventory features), a causal temporal encoder, GraphSAGE / GAT / GINE layers, and detection and localisation (link and PoP) heads.
- The tuning budget and the expectations X1 to X10 were committed before training (`docs/decision_log.md`, Phase 5 pre-registration).
- Results use the shared harness against B1, B2, B3 and B3+, with paired-bootstrap verdicts, seed variance, base rates, three window aggregations and per-network numbers.
- The MLP ablation shows that most of the gain over the baselines comes from supervised training, not from message passing. The MLP still uses link-to-PoP pooling and graph-derived static inputs, so this is "no message passing", not "no graph".
- The near-perfect link_down / node_down localisation is the model recognising the simulator's down-link rendering in the link telemetry, not hard detection (`docs/audit_phase5.md`, R3).
- The detection head also raises far more false alarms in maintenance hours than any baseline. All of this is reported in the decision log.
- No test-set run yet.
