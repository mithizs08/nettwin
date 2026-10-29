"""Simulator tests: physics, determinism, baseline sanity, and that every fault type propagates.

Runs on topohub's shipped `gabriel/20/1` (not part of any train, validation or test set) and on toy graphs, never on
the dev or test networks (audit C1). Elements are chosen by rule, not by name.
"""
import copy
import json

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import yaml

from nettwin.faults import check_fresh_seed, make_fault, sample_scenario
from nettwin.kg import FAULT_TYPES, pop_node
from nettwin.simulator import Simulator, congestion_loss, observed_kg, queue_delay_ms, to_frames, write_manifest, write_scenario
from nettwin.topology import from_node_link, load

T0, T1 = 516, 564                     # day 2, 19:00 to 23:00 sim time: peak traffic
FIXTURE = "gabriel/20/1"


@pytest.fixture(scope="module")
def cfg():
    with open("config.yaml") as f:
        c = copy.deepcopy(yaml.safe_load(f))
    c["simulator"]["days"] = 3        # shorter weeks keep the suite fast
    return c


@pytest.fixture(scope="module")
def sim(cfg):
    return Simulator(load(FIXTURE), cfg)


@pytest.fixture(scope="module")
def busy(sim):
    """Busiest non-bridge link that carries at least two services."""
    bridges = {sim.kg.G.nodes[l]["u"] + "|" + sim.kg.G.nodes[l]["v"] for l in sim.links} & set()
    br = {tuple(sorted(e)) for e in nx.bridges(sim.topo)}
    ok = [l for l in sim.links if (sim.kg.G.nodes[l]["u"], sim.kg.G.nodes[l]["v"]) not in br and len(sim.kg.services_on(l)) >= 2]
    assert ok and not bridges
    return max(ok, key=lambda l: (len(sim.kg.paths_through(l)), l))


def spec(sim, faults=(), seed=8101, **extra):
    s = {"scenario_id": "t", "topology": sim.name, "seed": seed, "n_steps": sim.n_steps, "step_minutes": 5,
         "faults": list(faults), "maintenance": [], "gaps": [], "event_days": []}
    s.update(extra)
    return s


def fault(sim, ftype, root, t0=T0, t1=T1, seed=0, **params):
    return make_fault(sim, ftype, root, t0, t1 - t0, np.random.default_rng(seed), **params)


def run(sim, *faults, **extra):
    res = sim.simulate(spec(sim, faults, **extra))
    cf = sim._run(spec(sim, [], **extra))
    return res, cf


def win(a):
    return a[T0:T1]


def services_through(sim, link):
    return [sim.services.index(s) for s in sim.kg.services_on(link)]


def toy(nodes, edges, demands=None):
    data = {"directed": False, "multigraph": False, "graph": {"demands": demands or {}},
            "nodes": [{"id": n, "pos": [i * 100.0, 0.0]} for i, n in enumerate(nodes)],
            "edges": [{"source": u, "target": v, "dist": d} for u, v, d in edges]}
    return from_node_link(data, "toy", geo=False)


# ------------------------------------------------------------------ physics


def test_queue_and_loss_shapes():
    rho = np.array([0.1, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0, 1.2, 2.0])
    q = queue_delay_ms(rho, 0.05, 25.0)
    assert np.all(np.diff(q) >= 0) and q[0] < 0.01 and q[-1] == 25.0
    p = congestion_loss(rho, 0.9, 0.2)
    assert np.all(np.diff(p) >= 0) and p[:4].max() == 0
    assert p[6] == pytest.approx(0.002)
    assert p[-1] == pytest.approx(1 - (1 - 0.002) * 0.5)


def test_dropped_traffic_is_not_counted_downstream(cfg):
    """Audit M3: a chain whose first hop runs at rho = 2 must show about half the load on the last hop."""
    sim = Simulator(toy("ABCD", [("A", "B", 100), ("B", "C", 100), ("C", "D", 100)]), cfg)
    r = sim.routing(((), (), ()))
    dem = np.zeros((1, len(sim.pairs)))
    dem[0, sim.pair_idx[("A", "D")]] = 1.0
    sim.capacity = np.array([10.0 if l != "link:A--B" else 0.5 for l in sim.links])
    sim.capacity2 = np.repeat(sim.capacity, 2)
    loads = sim._carried(r, dem, np.zeros((1, 2 * len(sim.links))))
    first, last = 2 * sim.link_idx["link:A--B"], 2 * sim.link_idx["link:C--D"]
    assert loads[0, first] == pytest.approx(1.0)
    assert loads[0, last] == pytest.approx(0.5 * (1 - 0.002), rel=1e-3)   # the aggregate model gave 1.0 here


def test_real_demand_matrix_branch(cfg):
    """Audit m9: the real-matrix branch, on a toy matrix (never on a test network)."""
    sim = Simulator(toy("ABC", [("A", "B", 50), ("B", "C", 50)], demands={"A": {"C": 3.0}, "B": {"C": 1.0}}), cfg)
    d = dict(zip(sim.pairs, sim.base_demand))
    assert d[("A", "C")] == pytest.approx(d[("C", "A")]) == pytest.approx(0.75 * cfg["simulator"]["demand"]["total"] / 2)
    assert d[("A", "B")] == 0
    with pytest.raises(ValueError, match="not in topology"):
        Simulator(toy("ABC", [("A", "B", 50), ("B", "C", 50)], demands={"A": {"Z": 1.0}}), cfg)


def test_deterministic_from_seed(sim):
    s = sample_scenario(sim, "d", seed=8103)
    a, b = sim.simulate(s), sim.simulate(copy.deepcopy(s))
    for k in a.obs:
        np.testing.assert_array_equal(a.obs[k], b.obs[k])
    np.testing.assert_array_equal(a.labels, b.labels)
    assert a.observed_events == b.observed_events and a.impacts == b.impacts
    c = sim.simulate({**s, "seed": 8104})
    assert not np.allclose(np.nan_to_num(a.obs["util"]), np.nan_to_num(c.obs["util"]))


def test_baseline_band_and_diurnal(sim):
    res = sim.simulate(spec(sim))
    peaks = res.truth["util"].max(0)
    normal = [i for i, l in enumerate(sim.links) if l not in sim.hot_links]
    assert 0.3 < np.median(peaks) < 0.75 and peaks[normal].max() < 0.85
    u = res.truth["util"].reshape(-1, sim.steps_per_day, len(sim.links))
    assert u[:, 36:60].mean() < 0.6 * u[:, 228:252].mean()


def test_background_probe_loss_is_realistic(sim):
    """Audit C2: healthy services lose the odd probe, so one lost probe is not a fault signature."""
    res = sim.simulate(spec(sim))
    lost = np.nanmean(res.obs["svc_loss_pct"] > 0)
    assert 0.01 <= lost <= 0.10
    assert np.nanmean(res.obs["svc_loss_pct"]) < 0.05          # healthy mean loss stays tiny (percent)


# ------------------------------------------------------------------ fault propagation


def test_link_down_reroutes_with_lag_and_convergence_loss(sim, busy):
    res, cf = run(sim, fault(sim, "link_down", busy))
    li = sim.link_idx[busy]
    u = win(res.obs["util"])[:, li]
    assert np.all(u[~np.isnan(u)] == 0) and np.all(np.isnan(win(res.obs["latency_ms"])[:, li]))
    assert np.nanmax(win(res.obs["loss_pct"])[:, li]) == 0        # counters still readable, nothing to discard
    gain = win(res.truth["util"]) - win(cf.truth["util"])
    gain[:, li] = 0
    assert gain.max() > 0.02
    on = services_through(sim, busy)
    assert (res.truth["svc_loss_pct"][T0, on] - cf.truth["svc_loss_pct"][T0, on]).max() > 0   # convergence loss at onset
    assert np.allclose(res.truth["util"][T1 + 3:], cf.truth["util"][T1 + 3:])
    st = [e for e in res.observed_events if e.get("element") == busy]
    assert [e["status"] for e in st] == ["down", "up"]
    assert T0 <= st[0]["t"] <= T0 + 3 and T1 <= st[1]["t"] <= T1 + 3            # audit M6: polling lag
    twin = observed_kg(sim, res)
    assert twin.paths_through(busy, t=T0 + 4) == [] and twin.paths_through(busy, t=T1 + 4) != []
    im = res.impacts["F0"]
    assert im["observable"] and im["observability"]["obs_status"]


def test_node_down_looks_like_its_links_going_down(sim):
    hub = max(sim.pops, key=lambda p: (len([s for s in sim.services if p in (sim.kg.G.nodes[s]["src"], sim.kg.G.nodes[s]["dst"])]), p))
    pop = pop_node(hub)
    res, cf = run(sim, fault(sim, "node_down", pop))
    inc = sim._root_links(pop)
    assert np.all(np.nan_to_num(win(res.obs["util"])[:, inc]) == 0)
    at = [i for i, s in enumerate(sim.services) if hub in (sim.kg.G.nodes[s]["src"], sim.kg.G.nodes[s]["dst"])]
    assert np.all(np.isnan(win(res.truth["svc_latency_ms"])[:, at]))
    assert np.nanmin(win(res.obs["svc_loss_pct"])[:, at]) == 100.0
    downs = {e["element"] for e in res.observed_events if e["type"] == "link_status" and e["status"] == "down"}
    assert downs == {sim.links[i] for i in inc}                  # neighbours report each incident link
    assert observed_kg(sim, res).paths_through(pop, t=T0 + 4) == []


@pytest.mark.parametrize("silent", [True, False])
def test_gray_failure_below_goodput_knee(sim, busy, silent):
    # burst_sigma 0 keeps every step below the 0.5% goodput knee (bursts above it legitimately cut goodput)
    res, cf = run(sim, fault(sim, "link_degradation", busy, loss_pct=0.3, silent=silent, ramp_steps=0, ramp_down_steps=0, burst_sigma=0.0))
    li = sim.link_idx[busy]
    np.testing.assert_allclose(res.truth["util"], cf.truth["util"], rtol=0.004, atol=1e-6)   # normal, bar the 0.3% dropped before downstream links
    on = services_through(sim, busy)
    off = [i for i in range(len(sim.services)) if i not in on]
    dloss = win(res.truth["svc_loss_pct"]) - win(cf.truth["svc_loss_pct"])
    assert dloss[:, on].mean() > 0.1 and np.abs(dloss[:, off]).max() < 1e-9
    counter = win(res.truth["counter_loss_pct"])[:, li] - win(cf.truth["counter_loss_pct"])[:, li]
    assert (np.abs(counter).max() < 1e-12) if silent else (counter.mean() > 0.1)
    assert [e for e in res.observed_events if e["type"] != "pop_status"] == []


def test_heavy_gray_failure_cuts_goodput(sim, busy):
    res, cf = run(sim, fault(sim, "link_degradation", busy, loss_pct=2.0, silent=True, ramp_steps=0, ramp_down_steps=0))
    li = sim.link_idx[busy]
    assert win(res.truth["util"])[:, li].mean() < 0.9 * win(cf.truth["util"])[:, li].mean()


def test_flapping_partial_down_and_convergence_loss(sim, busy):
    f = fault(sim, "link_flapping", busy, active_prob=0.9, max_down_frac=0.15, flaps_per_step=4, convergence_s=0.5)
    res, cf = run(sim, f)
    li = sim.link_idx[busy]
    d = win(res.truth["down_frac"])[:, li]
    assert ((d > 0) & (d < 1)).mean() > 0.6
    assert win(res.truth["util"])[:, li].mean() < win(cf.truth["util"])[:, li].mean() - 0.02
    on = services_through(sim, busy)
    assert (win(res.truth["svc_loss_pct"]) - win(cf.truth["svc_loss_pct"]))[:, on].mean() > 0.5
    seen = [e for e in res.observed_events if e.get("element") == busy]
    assert 0 < len(seen) < 2 * (T1 - T0)


def test_demand_surge_congests_without_reroute(sim):
    hub = max(sim.pops, key=lambda p: (sim.topo.degree(p), p))
    res, cf = run(sim, fault(sim, "demand_surge", pop_node(hub), factor=2.5, ramp_steps=1, ramp_down_steps=0))
    inc = sim._root_links(pop_node(hub))
    far = [i for i in range(len(sim.links)) if i not in inc]
    gain = win(res.truth["util"]) - win(cf.truth["util"])
    # traffic dropped at a congested link no longer loads links further on (audit M3), so a few links may fall slightly
    assert gain[:, inc].mean() > 2 * gain[:, far].mean() and gain.min() > -0.05
    assert res.truth["keys"] == cf.truth["keys"]
    assert [e for e in res.observed_events if e["type"] != "pop_status"] == []


def test_misconfig_shifts_traffic_but_twin_does_not_see_it(sim, busy):
    f = fault(sim, "routing_misconfig", busy, seed=1)
    res, cf = run(sim, f)
    assert res.truth["keys"][T0] != cf.truth["keys"][T0]
    assert np.abs(win(res.truth["util"]) - win(cf.truth["util"])).max() > 0.009
    twin = observed_kg(sim, res)
    assert twin.paths_through(busy, t=T0 + 1) == twin.paths_through(busy, t=0)
    assert [e for e in twin.changes_between(0, sim.n_steps) if e["type"] != "pop_status"] == []


# ------------------------------------------------------------------ shortcuts removed (audit M1, M2)


def test_down_link_signature_is_not_unique(sim):
    """'util present, latency missing' also happens on healthy links (probe timeouts and probe-responder gaps)."""
    res = sim.simulate(spec(sim, gaps=[{"id": "P0", "scope": "probe", "target": sim.links[0], "t_start": 100, "t_end": 110}]))
    pattern = ~np.isnan(res.obs["util"]) & np.isnan(res.obs["latency_ms"])
    assert pattern.sum() > 10 and not res.truth["fully_down"].any()


def test_status_blind_variant_hides_down_links(sim, busy):
    res = sim.simulate(spec(sim, [fault(sim, "link_down", busy)], status_blind=True))
    li = sim.link_idx[busy]
    for k in ("util", "latency_ms", "loss_pct"):
        assert np.all(np.isnan(win(res.obs[k])[:, li]))
    assert [e for e in res.observed_events if e["type"] != "maintenance"] == []


def test_pop_gap_with_dead_responder_mimics_unreachable(sim):
    pop = max(sim.pops, key=lambda p: (len([s for s in sim.services if p in (sim.kg.G.nodes[s]["src"], sim.kg.G.nodes[s]["dst"])]), p))
    g = [{"id": "G0", "scope": "pop", "target": pop_node(pop), "t_start": 100, "t_end": 130, "responder_down": True}]
    res = sim.simulate(spec(sim, gaps=g))
    at = [i for i, s in enumerate(sim.services) if pop in (sim.kg.G.nodes[s]["src"], sim.kg.G.nodes[s]["dst"])]
    loss = res.obs["svc_loss_pct"][100:130, at]
    assert np.all(loss[~np.isnan(loss)] == 100) and np.isfinite(loss).mean() > 0.9   # random drop-outs aside
    assert np.all(np.isnan(res.obs["svc_latency_ms"][100:130, at]))
    assert not np.isnan(res.truth["svc_latency_ms"][100:130, at]).any()     # the network itself is fine
    assert all(e["type"] == "pop_status" for e in res.observed_events)


def test_sampler_overlaps_and_rules(sim, cfg):
    placements, n = [], 0
    for seed in range(8200, 8260):
        s = sample_scenario(sim, f"s{seed}", seed)
        for f in s["faults"]:
            placements.append(f["placement"])
            assert f["t_start"] >= sim.warmup_steps and f["type"] in FAULT_TYPES
            if f["caused_by_maintenance"]:
                m = next(m for m in s["maintenance"] if m["id"] == f["linked_maintenance"])
                assert f["root_cause"] == m["target"] or f["root_cause"][4:] in m["target"]
        flaps = [(f["t_start"], f["t_end"]) for f in s["faults"] if f["type"] == "link_flapping"]
        assert all(b <= c or d <= a for i, (a, b) in enumerate(flaps) for c, d in flaps[i + 1:])
        for m in s["maintenance"]:
            if m["action"] != "noop":
                key = ((m["target"],), (), ()) if m["action"] == "down" else ((), (), ((m["target"], cfg["simulator"]["drain_metric"]),))
                assert sim.planned_peak_util(key, m["t_start"], m["t_end"], s["event_days"]) <= cfg["distractors"]["maintenance_max_util"]
        n += 1
    c = {p: placements.count(p) for p in set(placements)}
    assert c.get("maintenance_linked", 0) > 0 and c.get("concurrent", 0) > 0 and c.get("isolated", 0) > len(placements) / 2
    assert sample_scenario(sim, "x", 8299) == sample_scenario(sim, "x", 8299)


def test_maintenance_overlap_simulates(sim):
    for seed in range(8300, 8340):
        s = sample_scenario(sim, "o", seed)
        if any(f["placement"] != "isolated" for f in s["faults"]):
            res = sim.simulate(s)
            assert set(res.impacts) == {f["id"] for f in s["faults"]}
            return
    pytest.fail("no overlapping fault sampled")


# ------------------------------------------------------------------ distractors


def test_maintenance_drain_is_logged_and_benign(sim, busy):
    m = [{"id": "M0", "target": busy, "t_start": 300, "t_end": 336, "action": "drain", "description": "planned traffic drain"}]
    res = sim.simulate(spec(sim, maintenance=m))
    assert res.truth["util"][300:336, sim.link_idx[busy]].max() < 0.01
    assert [e["id"] for e in res.observed_events if e["type"] == "maintenance"] == ["M0"]
    twin = observed_kg(sim, res)
    assert twin.paths_through(busy, t=301) == [] and twin.paths_through(busy, t=336) != []


def test_gaps_blank_telemetry(sim, busy):
    g = [{"id": "G1", "scope": "link", "target": busy, "t_start": 200, "t_end": 205}]
    res = sim.simulate(spec(sim, gaps=g))
    for k in ("util", "latency_ms", "loss_pct"):
        assert np.all(np.isnan(res.obs[k][200:205, sim.link_idx[busy]]))
    assert res.observed_events == []


def test_event_day_ramps(sim):
    t = np.arange(sim.n_steps)
    f = sim.event_factor(t, [{"day": 1, "factor": 1.2}])
    edge = f[sim.steps_per_day - 12: sim.steps_per_day + 12]
    assert np.max(np.diff(edge)) < 0.05 and f[sim.steps_per_day + 50] == pytest.approx(1.2)


# ------------------------------------------------------------------ ground truth: observability and labels


def test_observability_flags(sim):
    idle = [l for l in sim.links if not sim.kg.services_on(l)]
    if not idle:
        pytest.skip("fixture has no service-free link")
    # no bursts, so loss never crosses the goodput knee (bursts above it would show up in utilisation, correctly)
    res, _ = run(sim, fault(sim, "link_degradation", idle[0], loss_pct=0.3, silent=True, ramp_steps=0, ramp_down_steps=0, burst_sigma=0.0))
    im = res.impacts["F0"]
    assert not im["observable"] and not any(im["observability"].values())   # physically undetectable


def test_labels_root_and_symptomatic(sim, busy):
    res, _ = run(sim, fault(sim, "link_down", busy))
    li = sim.link_idx[busy]
    assert np.all(res.labels[T0:T1, li] == 2)
    assert (res.labels[T0:T1] == 1).any()
    assert np.all(res.labels[:T0 - 1] == 0)


def test_observed_events_never_leak_ground_truth(sim):
    for seed in range(8400, 8410):
        res = sim.simulate(sample_scenario(sim, "s", seed), counterfactual=False)
        for e in res.observed_events:
            assert e["type"] in ("maintenance", "link_status", "pop_status")
            assert not set(e) & {"root_cause", "params", "fault", "impact", "placement"}


def test_write_scenario_separates_truth(sim, tmp_path):
    res = sim.simulate(sample_scenario(sim, "rt", 8501))
    entry = write_scenario(res, tmp_path)
    write_manifest([entry], tmp_path)
    links = pd.read_parquet(tmp_path / "observable" / "rt" / "links.parquet")
    pd.testing.assert_frame_equal(links, to_frames(res)[0])
    assert sorted(p.name for p in (tmp_path / "observable" / "rt").iterdir()) == ["links.parquet", "meta.json", "observed_events.json", "services.parquet"]
    meta = json.loads((tmp_path / "observable" / "rt" / "meta.json").read_text())
    assert set(meta) == {"scenario_id", "topology", "n_steps", "step_minutes", "status_blind"}   # no seed: it would let one re-run the sampler
    labels = pd.read_parquet(tmp_path / "truth" / "rt" / "labels.parquet")
    assert len(labels) == sim.n_steps * len(sim.links)
    manifest = json.loads((tmp_path / "truth" / "scenarios.json").read_text())
    assert "observable" in manifest[0]["faults"][0]["impact"]


def test_reserved_seeds_are_refused(cfg):
    with pytest.raises(ValueError, match="reserved"):
        check_fresh_seed(cfg, 2026)
    check_fresh_seed(cfg, 123456)


# ------------------------------------------------------------------ audit re-verify fixes


def test_concurrent_fault_does_not_make_undetectable_one_observable(sim, busy):
    """N1, the auditor's case: a silent gray failure on a service-free link stays unobservable and subtle when a
    link_down elsewhere overlaps it."""
    idle = [l for l in sim.links if not sim.kg.services_on(l) and l != busy]
    if not idle:
        pytest.skip("fixture has no service-free link")
    gray = fault(sim, "link_degradation", idle[0], loss_pct=0.3, silent=True, ramp_steps=0, ramp_down_steps=0, burst_sigma=0.0)
    gray["id"] = "G"
    down = fault(sim, "link_down", busy, t0=T0 + 6, t1=T1 - 6)
    down["id"] = "D"
    alone = sim.simulate(spec(sim, [gray])).impacts["G"]
    both = sim.simulate(spec(sim, [gray, down])).impacts
    assert not alone["observable"] and alone["impact_class"] == "subtle"
    assert not both["G"]["observable"] and not any(both["G"]["observability"].values())
    assert both["G"]["impact_class"] == "subtle" and both["G"]["services_affected"] == 0
    assert both["D"]["observable"] and both["D"]["observability"]["obs_status"]


def test_planned_outage_status_is_not_credited_to_a_fault(sim, busy):
    """N1: a status event from a planned outage on the same link is not the fault's doing."""
    m = [{"id": "M0", "target": busy, "t_start": T0, "t_end": T1, "action": "down", "description": "planned link outage"}]
    g = fault(sim, "link_degradation", busy, loss_pct=0.3, silent=True, ramp_steps=0, ramp_down_steps=0, burst_sigma=0.0)
    im = sim.simulate(spec(sim, [g], maintenance=m)).impacts["F0"]
    assert not im["observability"]["obs_status"] and not im["status_change_observed"]


def test_gap_covered_channels_are_unobservable(sim, busy):
    """N8: a fault whose only symptom sits under a telemetry gap is not observable on that channel."""
    idle = [l for l in sim.links if not sim.kg.services_on(l)]
    if not idle:
        pytest.skip("fixture has no service-free link")
    g = fault(sim, "link_degradation", idle[0], loss_pct=0.8, silent=False, ramp_steps=0, ramp_down_steps=0, burst_sigma=0.0)
    open_ = sim.simulate(spec(sim, [g])).impacts["F0"]
    gap = [{"id": "G0", "scope": "link", "target": idle[0], "t_start": T0 - 2, "t_end": T1 + 5}]
    covered = sim.simulate(spec(sim, [g], gaps=gap)).impacts["F0"]
    assert open_["observability"]["obs_link_counters"]
    assert not covered["observability"]["obs_link_counters"] and not covered["observability"]["obs_link_util"]


def test_no_fault_free_buffer_around_maintenance(sim):
    """N6: isolated faults may sit right next to (or on) a change window."""
    near = 0
    for seed in range(8600, 8700):
        s = sample_scenario(sim, "b", seed)
        for f in s["faults"]:
            if f["placement"] != "isolated":
                continue
            for m in s["maintenance"]:
                if -36 < f["t_start"] - m["t_end"] < 36 or (f["t_start"] < m["t_end"] and f["t_end"] > m["t_start"]):
                    near += 1
    assert near > 0


def test_observable_paths_are_neutral(sim, tmp_path):
    """N7: no observable path or filename carries a fault type or severity word."""
    from nettwin.simulator import LEAKY_WORDS

    with pytest.raises(ValueError, match="leaks ground truth"):
        write_scenario(sim.simulate({**spec(sim), "scenario_id": "aarnet_link_down_severe"}), tmp_path)
    for k, seed in enumerate(range(8710, 8714)):
        res = sim.simulate(sample_scenario(sim, f"demo_{k:04d}", seed))
        write_scenario(res, tmp_path)
    paths = [p.relative_to(tmp_path).as_posix().lower() for p in (tmp_path / "observable").rglob("*")]
    words = set(LEAKY_WORDS) | {ft.lower() for ft in FAULT_TYPES} | {"mild", "severe", "subtle", "moderate", "obvious"}
    assert paths and not [p for p in paths for w in words if w in p]
    for ev_file in (tmp_path / "observable").rglob("observed_events.json"):
        text = ev_file.read_text().lower()
        assert not [ft for ft in FAULT_TYPES if ft in text]


def test_blind_view_equals_a_status_blind_run(sim, busy):
    """The dataset derives the blind variant from the default run; it must match simulating with status_blind."""
    f = fault(sim, "link_down", busy)
    g = fault(sim, "node_down", pop_node(sim.pops[0]), t0=T1 + 40, t1=T1 + 60)
    g["id"] = "F1"
    sp = spec(sim, [f, g], gaps=[{"id": "G0", "scope": "pop", "target": pop_node(sim.pops[1]), "t_start": 50, "t_end": 80, "responder_down": True}])
    default, blind = sim.simulate(sp, with_blind=True)
    real = sim.simulate({**sp, "status_blind": True})
    for k in real.obs:
        np.testing.assert_array_equal(blind.obs[k], real.obs[k])
    assert blind.observed_events == real.observed_events
    assert blind.impacts == real.impacts and default.impacts != blind.impacts
    np.testing.assert_array_equal(blind.labels, real.labels)
