# NetTwin: plan doc

A knowledge-graph digital twin of a telecom backbone, with GNN anomaly localisation and an LLM agent that does root-cause analysis. Built as a portfolio project for the UTS Telecom Research Unit Project D.A.R.T. internship (Telstra-partnered). Applications close **Mon 2 Nov 2026**.

This file is the source of truth. Claude Code: read it fully at the start of every session, work one phase at a time, and update `docs/decision_log.md` and the phase checklist as you go.

---

## 1. What we are building and why

Project D.A.R.T. builds a real-time digital twin of Telstra's production network using knowledge graphs, LLMs, agentic AI and GNNs, for topology discovery, error detection, anomaly detection and capacity management. This project builds a small, honest, fully measured version of that stack on public data:

1. **Topology layer**: real ISP backbones from the Internet Topology Zoo / SNDlib via the `topohub` package. AARNet (Australia's research and education network, 19 PoPs, 24 links) is the headline demo network. There is no Telstra topology in the Zoo, so we never claim one.
2. **Knowledge graph**: a typed, multi-layer graph: physical (PoP, Link) → logical (RoutingPath) → service (CustomerService with SLA) → operational (Alert, MaintenanceWindow, Runbook).
3. **Telemetry simulator**: traffic demands, routing, utilisation, latency and packet loss over time, with injected faults that propagate realistically (a failed link reroutes traffic and congests other links).
4. **Anomaly detection and root-cause localisation**: statistical baselines, a topology heuristic, and a GNN, compared on the same held-out incidents.
5. **LLM root-cause agent**: Claude with tools over the knowledge graph, telemetry, GNN scores and a small runbook RAG. It diagnoses incidents and recommends a runbook action. Measured against ground truth, with ablations.
6. **Capacity forecasting**: predict which links will breach a utilisation threshold within a horizon.
7. **Demo dashboard**: Streamlit app showing the network, live alerts, and the agent's reasoning trace.

**Honesty rules (non-negotiable):** telemetry is synthetic, topologies are public, and none of this is Telstra's network. Every number in the README comes from a script output. Negative results are reported, not hidden.

---

## 2. Pre-specified research questions

Fixed now, before any results. Changes go in the decision log with a reason.

- **RQ1 Detection.** Does a graph-aware model detect anomalous links better than per-link statistical baselines, especially for faults whose symptoms appear away from the faulty element?
- **RQ2 Localisation.** How accurately can each method identify the root-cause element (top-1 / top-3), and which fault types are hardest?
- **RQ3 Generalisation.** Trained only on synthetic topologies, does the GNN transfer to unseen real topologies (AARNet, GEANT, germany50)?
- **RQ4 Agent.** Does an LLM agent with knowledge-graph tools beat (a) the same LLM given only the raw alert list and (b) the best non-LLM localiser? Also: how grounded are its explanations?
- **RQ5 Capacity (stretch).** Can a simple forecaster flag links that will breach 80% utilisation within the next hour better than a naive persistence baseline?

Headline metrics: link-level PR-AUC for detection (RQ1); root-cause top-1 and top-3 accuracy for localisation (RQ2/RQ3/RQ4); agent groundedness rate and cost per incident (RQ4); breach precision/recall at fixed horizon (RQ5).

---

## 3. Data and splits

**Topologies** (revised 2026-10-29 after the Phase 0 to 2 audit, `docs/audit_phases_0_2.md` C1 / M4; selection rule pre-registered in `docs/decision_log.md`)
- Training pool: synthetic graphs from topohub's Gabriel generator, n between 15 and 60 (every size, not just multiples of 5), seeds outside topohub's shipped scheme, augmented so link lengths and structure resemble real backbones (edge thinning, leaf spurs, dual-router PoPs, backbone-km distances, pseudo time zones). Augmentation parameters come only from a reference pool of non-test, non-dev Topology Zoo networks.
- Validation: held-out generated graphs (split by graph, stratified by size, at least 60 graphs) plus `validation_real`: `topozoo/Aconet` (17/24), `topozoo/BeyondTheNetwork` (29/41), `topozoo/Fccn` (23/25), `topozoo/Quest` (20/31), `topozoo/Nextgen` (16/16), picked by the pre-registered rule.
- Dev / demo (seen during development): `topozoo/Aarnet` (19/24). It was used to develop and test the simulator in Phases 1 and 2, so its results are reported in their own column and never count as held-out evidence for RQ3 or RQ4.
- Test (never touched for design decisions): `topozoo/Geant2012` (37/58), `sndlib/germany50` (50/88, includes real demand matrix), plus five networks picked by the pre-registered rule: `topozoo/Garr201201` (48/62), `topozoo/Peer1` (16/20), `topozoo/York` (23/24), `topozoo/Belnet2010` (19/20), `topozoo/Ibm` (18/24).
- Skipped: near-duplicates of test networks (other GEANT snapshots, `sndlib/geant`, the DFN / NOBEL German networks) and anything that fails to load. The full list is `topologies.skip` in `config.yaml`.

**Leakage rules**
- Split by topology first, then by scenario seed. No test topology appears in training in any form.
- All tuning and diagnostics run on validation. If a test-set result drives a change, disclose it in the README (lesson from the link-prediction project).

---

## 4. Simulator spec

- **Time:** 5-minute steps, 7 simulated days per scenario (2016 steps), diurnal and weekly traffic patterns plus noise.
- **Demands:** gravity model on node population proxy (degree or a random weight); germany50 uses its real demand matrix scaled.
- **Routing:** shortest path on link weights (distance), ECMP where ties exist. Recompute on any topology change.
- **Capacity:** assign per-link capacity so baseline peak utilisation sits around 40 to 70%.
- **Metrics per link per step:** utilisation, latency (propagation from `dist` km plus a queueing term that blows up near saturation), packet loss (near zero until about 90% utilisation, then rises), plus measurement noise and occasional missing samples.
- **Path/service metrics:** end-to-end latency and loss per service = aggregate over its routing path.

**Fault types** (each with ground-truth root-cause element, start, end):
1. Link down (reroute, congestion elsewhere)
2. Node down (all incident links down)
3. Link degradation / gray failure (loss rises, utilisation looks normal)
4. Link flapping (intermittent down/up)
5. Demand surge at a node (genuine congestion, no broken element)
6. Routing weight misconfiguration (traffic shifts onto a bad path)

**Distractors** (to make it realistic and to test false positives):
- Planned maintenance windows logged in the change log (benign changes)
- Normal traffic peaks
- Telemetry gaps (missing data is not a fault)

Output: Parquet files per scenario plus a `scenarios.json` manifest with ground truth. Deterministic from seed.

---

## 5. Knowledge graph spec

Built in NetworkX as a typed `MultiDiGraph` (node attribute `kind`, edge attribute `rel`), with an export to Neo4j-style CSV for anyone who wants to load it.

- Node kinds: `PoP`, `Link`, `RoutingPath`, `Service`, `Alert`, `Maintenance`, `Runbook`
- Relations: `Link -CONNECTS-> PoP`, `RoutingPath -TRAVERSES-> Link` (ordered), `Service -USES-> RoutingPath`, `Alert -ON-> Link|Service`, `Maintenance -AFFECTS-> Link|PoP`, `Runbook -APPLIES_TO-> fault type`
- The KG updates when routing changes (paths are re-derived), which is the "dynamic, real-time" part. Keep a snapshot per time window so the agent can ask "what did the path look like at time t".
- Query layer: a small Python API (`kg.paths_through(link)`, `kg.services_on(element)`, `kg.neighbours(element, hops)`, `kg.shared_elements(alerted_paths)`, `kg.changes_between(t0, t1)`). The agent's tools wrap these.

---

## 6. Models

**Baselines (Phase 4)**
- B1: per-link rolling z-score / EWMA on each metric
- B2: Isolation Forest on per-link windowed features
- B3: topology heuristic (network tomography style): among all elements on alerted paths, rank by how many alerted paths share them, penalised by how many healthy paths also use them. Expect this to be strong. If it beats the GNN, say so.

**GNN (Phase 5)**
- PyTorch Geometric. Work on the line graph (links become nodes, adjacency = shared PoP) so link-level prediction is natural; include PoP-level features via aggregation.
- Input per link: window of the last W steps of normalised utilisation, latency, loss, missing-data flags, plus static features (capacity, length, number of paths through it).
- Architectures to compare: GraphSAGE, GAT, GINE with a temporal encoder (1D conv or GRU over the window). Keep it CPU-trainable.
- Two heads: (a) anomaly score per link (detection, RQ1), (b) root-cause score per element, softmax over elements (localisation, RQ2).
- Ablations: no graph (MLP on same features), no static path features, shuffled edges.
- Explainability: GNNExplainer or gradient attribution on a few incidents, used in the write-up and fed to the agent as evidence.

**Capacity forecasting (Phase 8, stretch)**
- Persistence and seasonal-naive baselines vs LightGBM on lag features vs the GNN encoder with a forecasting head. Metric: breach precision/recall at 80% within 60 minutes.

---

## 7. Agent spec

- Claude via the Anthropic API with tool use. Model names live in `config.yaml`; use a cheap model for development and the stronger one for the final eval. Check current model IDs and pricing in the docs before Phase 6; do not hardcode from memory.
- Tools:
  - `get_active_alerts(t0, t1)`
  - `get_metric_series(element, metric, t0, t1)`
  - `kg_paths_through(link)`, `kg_services_on(element)`, `kg_neighbours(element, hops)`, `kg_shared_elements(alerted_paths)`
  - `get_change_log(t0, t1)` (maintenance windows)
  - `get_gnn_scores(t, top_k)` and `explain_gnn(element, t)`
  - `search_runbooks(query)` (RAG over about 8 to 10 short runbooks we write ourselves, one per fault type plus general ones; simple embedding or BM25 retrieval)
  - `submit_diagnosis(root_cause, fault_type, confidence, evidence[], recommended_action)` (structured final answer)
- Agent loop capped at N tool calls and a token budget per incident. Log every call.
- **Conditions for RQ4:**
  - A0: LLM with alert list only, no tools
  - A1: LLM with KG and telemetry tools, no GNN tools
  - A2: full agent (KG + telemetry + GNN + runbooks)
  - Compare against B3 and the best GNN as non-LLM references.
- **Evaluation:** about 150 held-out incidents across test topologies, stratified by fault type, plus distractor-only cases (correct answer is "no fault" or "planned maintenance").
  - Root-cause top-1, fault-type accuracy, false-alarm rate on distractor cases
  - Groundedness: automatic check that every element and number cited in `evidence` exists in that run's tool outputs; report the rate of ungrounded claims
  - Cost and latency per incident
- Budget: estimate total API cost before the full run and log actuals. Run a 10-incident smoke test first.

---

## 8. Repo layout

```
nettwin/
  PLAN.md
  README.md
  config.yaml
  requirements.txt
  docs/decision_log.md
  docs/audit.md
  docs/report.md
  src/nettwin/topology.py
  src/nettwin/simulator.py
  src/nettwin/faults.py
  src/nettwin/kg.py
  src/nettwin/baselines.py
  src/nettwin/gnn/{data.py,models.py,train.py,explain.py}
  src/nettwin/forecast.py
  src/nettwin/agent/{tools.py,runbooks/,loop.py,eval.py}
  app/streamlit_app.py
  scripts/  (one entry script per phase, writes to results/)
  results/  (metrics JSON/CSV, figures)
  tests/
```

`data/` and large generated files are gitignored. Check the .gitignore does not hide source code (the link-prediction project lost model code from GitHub for three phases this way). End every phase with a fresh-clone check: clone, install, run that phase's script, confirm outputs.

---

## 9. Phases and timeline

Deadline is Mon 2 Nov. Target: code done Sunday night, write-up and application Monday.

| Phase | What | When |
|---|---|---|
| 0 | Repo, env, config, decision log, this plan committed; confirm topohub, PyG and Anthropic SDK install | Thu night |
| 1 | Topology loader + knowledge graph + query API + tests; AARNet KG figure | Fri |
| 2 | Simulator + fault injection + distractors + scenario generator; sanity plots (one incident end-to-end on AARNet) | Fri |
| 3 | Dataset build: train/val/test scenarios per split rules; dataset card in docs | Fri night |
| 4 | Baselines B1 to B3, evaluation harness, first results on validation | Sat |
| 5 | GNN: data pipeline, models, training, ablations, explanations; RQ1 to RQ3 on test | Sat to Sun |
| 6 | Agent: tools, runbooks + RAG, loop, smoke test, full eval A0/A1/A2; RQ4 | Sun |
| 7 | Streamlit demo: map of network, alert timeline, pick an incident, watch the agent's trace | Sun night |
| 8 | Capacity forecasting (RQ5), only if 0 to 7 are solid | Stretch |
| 9 | Independent audit in a fresh Claude Code session: leakage, metric bugs, claims vs outputs. Fix all major findings | Sun night / Mon early |
| 10 | Report + README + final figures + number checker script (every number in README traced to a results file); [VERIFY] tags on every citation for Zareen to check | Mon morning |
| 11 | Make repo public, update CV, answer form questions in her own words, submit | Mon |

If time runs short, cut in this order: Phase 8, then GINE and GRU variants, then Neo4j export. Never cut the audit, the baselines, or the honesty rules.

---

## 10. Working rules for Claude Code

- One phase per session where possible. Start each session by reading this file and the decision log.
- Pre-specified RQs and metrics stay fixed. Any change: decision log entry with date and reason.
- Validation for all design choices; test only for final numbers.
- Seeds everywhere; results reproducible from scripts.
- Keep a "parked ideas" list in the decision log so ideas land in the right phase.
- Every README number must come from `results/`; the number checker enforces this.
- Citations get `[VERIFY]` until Zareen checks them personally.
- Write plainly. No em dashes in any prose.

---

## 11. What the write-up should lead with

- The problem framing: a twin is only useful if it can explain incidents, so we measure explanation, not just detection.
- The head-to-head: tomography heuristic vs GNN vs agent, with whichever result actually holds.
- Generalisation from synthetic to real topologies.
- Agent groundedness and the no-tools ablation (shows the KG is doing real work, or that it isn't).
- Clear limitations: synthetic telemetry, small public topologies, no real Telstra data.
