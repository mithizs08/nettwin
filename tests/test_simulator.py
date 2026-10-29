"""Simulator tests: determinism, baseline sanity, and that every fault type actually propagates."""
import copy
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from nettwin.faults import make_fault, sample_scenario
from nettwin.kg import FAULT_TYPES, pop_node
from nettwin.simulator import Simulator, congestion_loss, observed_kg, queue_delay_ms, to_frames, write_manifest, write_scenario
from nettwin.topology import load

BUSY = "link:Canberra1--Melbourne1"   # busiest AARNet link, not a bridge
T0, T1 = 516, 564                     # day 2, about 19:00 to 23:00 local: peak traffic


@pytest.fixture(scope="module")
def cfg():
    with open("config.yaml") as f:
        c = yaml.safe_load(f)
    c = copy.deepcopy(c)
    c["simulator"]["days"] = 3        # shorter weeks keep the suite fast
    return c


@pytest.fixture(scope="module")
def sim(cfg):
    return Simulator(load("topozoo/Aarnet"), cfg)


def spec(sim, faults=(), seed=7, **extra):
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


# ------------------------------------------------------------------ physics


def test_queue_and_loss_shapes():
    rho = np.array([0.1, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0, 1.2, 2.0])
    q = queue_delay_ms(rho, 0.05, 25.0)
    assert np.all(np.diff(q) >= 0) and q[0] < 0.01 and q[-1] == 25.0
    p = congestion_loss(rho, 0.9, 0.2)
    assert np.all(np.diff(p) >= 0)
    assert p[3] == 0 and p[:4].max() == 0                      # nothing below the knee
    assert p[6] == pytest.approx(0.002)                         # 0.2% at capacity
    assert p[-1] == pytest.approx(1 - (1 - 0.002) * 0.5)        # half the traffic dropped at 2x


def test_deterministic_from_seed(sim):
    s = sample_scenario(sim, "d", seed=3)
    a, b = sim.simulate(s), sim.simulate(copy.deepcopy(s))
    for k in a.obs:
        np.testing.assert_array_equal(a.obs[k], b.obs[k])
    assert a.observed_events == b.observed_events and a.impacts == b.impacts
    c = sim.simulate({**s, "seed": 4})
    assert not np.allclose(np.nan_to_num(a.obs["util"]), np.nan_to_num(c.obs["util"]))


def test_baseline_band_and_diurnal(sim):
    res = sim.simulate(spec(sim))
    peaks = res.truth["util"].max(0)
    assert 0.3 < np.median(peaks) < 0.7 and peaks.max() < 1.0
    normal = [i for i, l in enumerate(sim.links) if l not in sim.hot_links]
    assert peaks[normal].max() < 0.85                           # only the designated hot links run near capacity
    u = res.truth["util"].reshape(-1, sim.steps_per_day, len(sim.links))
    assert u[:, 36:60].mean() < 0.6 * u[:, 228:252].mean()     # 03-05h well below 19-21h
    assert np.isnan(res.obs["util"]).mean() < 0.01 and res.observed_events == []


def test_noise_is_realistic_not_zero(sim):
    res = sim.simulate(spec(sim))
    resid = res.obs["util"] - res.truth["util"]
    assert 0.001 < np.nanstd(resid) < 0.01
    svc_loss = res.obs["svc_loss_pct"]
    assert np.nanmax(svc_loss) < 1.0                            # healthy services: no big loss


# ------------------------------------------------------------------ fault propagation


def test_link_down_reroutes_and_congests(sim):
    res, cf = run(sim, fault(sim, "link_down", BUSY))
    li = sim.link_idx[BUSY]
    assert np.all(win(res.obs["util"])[:, li][~np.isnan(win(res.obs["util"])[:, li])] == 0)
    assert np.all(np.isnan(win(res.obs["latency_ms"])[:, li]))
    gain = win(res.truth["util"]) - win(cf.truth["util"])
    gain[:, li] = 0
    assert gain.max() > 0.1                                     # traffic lands on other links
    assert (gain.max(0) > 0.05).sum() >= 2
    dlat = np.nanmax(win(res.truth["svc_latency_ms"]) - win(cf.truth["svc_latency_ms"]), axis=0)
    assert dlat[services_through(sim, BUSY)].max() > 0.5       # rerouted services take longer paths
    assert np.allclose(res.truth["util"][T1 + 3:], cf.truth["util"][T1 + 3:])  # recovery after the window
    kinds = [(e["t"], e["status"]) for e in res.observed_events if e.get("element") == BUSY]
    assert kinds == [(T0, "down"), (T1, "up")]
    twin = observed_kg(sim, res)
    assert twin.paths_through(BUSY, t=T0 + 1) == [] and twin.paths_through(BUSY, t=T1) != []
    assert res.impacts["F0"]["impact_class"] == "obvious"


def test_node_down_isolates_pop(sim):
    pop = pop_node("Hobart")
    res, cf = run(sim, fault(sim, "node_down", pop))
    incident = [sim.link_idx[l] for l in sim.links if "Hobart" in l]
    assert np.all(np.nan_to_num(win(res.obs["util"])[:, incident]) == 0)
    hob = [i for i, s in enumerate(sim.services) if "Hobart" in (sim.kg.G.nodes[s]["src"], sim.kg.G.nodes[s]["dst"])]
    assert hob, "fixture expects at least one service at Hobart"
    assert np.all(np.isnan(win(res.truth["svc_latency_ms"])[:, hob]))
    loss = win(res.obs["svc_loss_pct"])[:, hob]
    assert np.nanmin(loss) == 100.0
    assert res.impacts["F0"]["services_unreachable"] == len(hob)
    assert observed_kg(sim, res).paths_through(pop, t=T0) == []


@pytest.mark.parametrize("silent", [True, False])
def test_gray_failure_hidden_in_util_visible_end_to_end(sim, silent):
    f = fault(sim, "link_degradation", BUSY, loss_pct=0.5, silent=silent, ramp_steps=0)
    res, cf = run(sim, f)
    li = sim.link_idx[BUSY]
    np.testing.assert_allclose(res.truth["util"], cf.truth["util"])          # utilisation looks normal
    np.testing.assert_allclose(res.truth["svc_latency_ms"], cf.truth["svc_latency_ms"])
    on = services_through(sim, BUSY)
    off = [i for i in range(len(sim.services)) if i not in on]
    dloss = win(res.truth["svc_loss_pct"]) - win(cf.truth["svc_loss_pct"])
    assert dloss[:, on].mean() > 0.2 and np.abs(dloss[:, off]).max() < 1e-9  # only services on the link see it
    counter = win(res.truth["counter_loss_pct"])[:, li] - win(cf.truth["counter_loss_pct"])[:, li]
    if silent:
        assert np.abs(counter).max() < 1e-12                    # silent drops never show in link counters
    else:
        assert counter.mean() > 0.2
    assert res.observed_events == []                            # never appears as an event


def test_subtle_gray_failure_is_buried_in_probe_noise(sim):
    f = fault(sim, "link_degradation", BUSY, loss_pct=0.02, silent=True, ramp_steps=0)
    res, cf = run(sim, f)
    on = services_through(sim, BUSY)
    obs_in = win(res.obs["svc_loss_pct"])[:, on]
    assert np.nanmean(obs_in == 0) > 0.8                        # most 5-min samples show zero loss
    assert res.impacts["F0"]["impact_class"] == "subtle"


def test_flapping_partial_down_and_convergence_loss(sim):
    f = fault(sim, "link_flapping", BUSY, active_prob=0.9, max_down_frac=0.5, flaps_per_step=6, convergence_s=2.0)
    res, cf = run(sim, f)
    li = sim.link_idx[BUSY]
    d = win(res.truth["down_frac"])[:, li]
    assert ((d > 0) & (d < 1)).mean() > 0.6
    assert win(res.truth["util"])[:, li].mean() < win(cf.truth["util"])[:, li].mean() - 0.05
    on = services_through(sim, BUSY)
    assert (win(res.truth["svc_loss_pct"]) - win(cf.truth["svc_loss_pct"]))[:, on].mean() > 1.0
    seen = [e for e in res.observed_events if e.get("element") == BUSY]
    assert 0 < len(seen) < 2 * (T1 - T0)                        # polling catches some flaps, not all


def test_demand_surge_congests_without_reroute(sim):
    pop = pop_node("Melbourne1")
    f = fault(sim, "demand_surge", pop, factor=2.5, ramp_steps=1)
    res, cf = run(sim, f)
    incident = [sim.link_idx[l] for l in sim.links if "Melbourne1" in l.split(":")[1].split("--")]
    far = [sim.link_idx[l] for l in sim.links if sim.link_idx[l] not in incident]
    gain = win(res.truth["util"]) - win(cf.truth["util"])
    assert gain[:, incident].mean() > 0.03                      # transit traffic dominates, so the rise is modest
    assert gain[:, incident].mean() > 2 * gain[:, far].mean()   # strongest next to the surging PoP
    assert gain.min() > -1e-9                                   # nothing gets less traffic
    touch = (sim.pair_src == sim.pop_idx["Melbourne1"]) | (sim.pair_dst == sim.pop_idx["Melbourne1"])
    assert touch.sum() == 2 * (len(sim.pops) - 1)
    assert res.truth["keys"] == cf.truth["keys"]                # no routing change at all
    assert res.observed_events == []


def test_misconfig_shifts_traffic_but_twin_does_not_see_it(sim):
    f = fault(sim, "routing_misconfig", BUSY, seed=1)
    assert f["params"]["new_metric"] != f["params"]["old_metric"]
    res, cf = run(sim, f)
    assert res.truth["keys"][T0] != cf.truth["keys"][T0]
    shift = np.abs(win(res.truth["util"]) - win(cf.truth["util"])).max()
    assert shift > 0.05
    assert res.observed_events == []
    twin = observed_kg(sim, res)
    assert twin.paths_through(BUSY, t=T0 + 1) == twin.paths_through(BUSY, t=0)  # twin keeps intended paths
    assert twin.changes_between(0, sim.n_steps) == []


def test_misconfig_sampler_always_changes_routing(sim):
    rng = np.random.default_rng(5)
    for link in sim.links[:8]:
        try:
            p = make_fault(sim, "routing_misconfig", link, T0, 10, rng)["params"]
        except ValueError:
            continue                                            # some links (bridges) cannot shift traffic
        assert sim.routing_changes({link: p["new_metric"]})


# ------------------------------------------------------------------ distractors


def test_maintenance_is_logged_and_benign(sim):
    m = [{"id": "M0", "target": BUSY, "t_start": T0, "t_end": T1, "action": "drain", "description": "planned traffic drain"},
         {"id": "M1", "target": "link:Cairns--Townsville", "t_start": T1 + 50, "t_end": T1 + 80, "action": "noop", "description": "planned config check"}]
    res = sim.simulate(spec(sim, maintenance=m))
    cf = sim.simulate(spec(sim))
    li = sim.link_idx[BUSY]
    assert win(res.truth["util"])[:, li].max() < 0.01           # drained
    assert [e["id"] for e in res.observed_events if e["type"] == "maintenance"] == ["M0", "M1"]
    np.testing.assert_allclose(res.truth["util"][T1 + 50:T1 + 80], cf.truth["util"][T1 + 50:T1 + 80])  # noop
    twin = observed_kg(sim, res)
    assert twin.paths_through(BUSY, t=T0 + 1) == [] and twin.paths_through(BUSY, t=T1) != []
    assert [e["type"] for e in twin.changes_between(T0, T0)][:2] == ["maintenance_start", "metric_change"]


def test_gaps_blank_telemetry(sim):
    g = [{"id": "G0", "scope": "pop", "target": pop_node("Darwin"), "t_start": 100, "t_end": 110},
         {"id": "G1", "scope": "link", "target": BUSY, "t_start": 200, "t_end": 205}]
    res = sim.simulate(spec(sim, gaps=g))
    darwin = [sim.link_idx[l] for l in sim.links if "Darwin" in l]
    for k in ("util", "latency_ms", "loss_pct"):
        assert np.all(np.isnan(res.obs[k][100:110, darwin]))
        assert np.all(np.isnan(res.obs[k][200:205, sim.link_idx[BUSY]]))
    assert res.observed_events == []                            # a gap is not a status change


def test_event_day_raises_traffic_benignly(sim):
    res = sim.simulate(spec(sim, event_days=[{"day": 1, "factor": 1.2}]))
    cf = sim.simulate(spec(sim))
    day = slice(sim.steps_per_day, 2 * sim.steps_per_day)
    ratio = res.truth["util"][day].mean() / cf.truth["util"][day].mean()
    assert 1.15 < ratio < 1.25 and res.observed_events == []


# ------------------------------------------------------------------ sampler, observability, output


def test_sampler_places_faults_cleanly(sim, cfg):
    for seed in range(15):
        s = sample_scenario(sim, f"s{seed}", seed)
        wins = sorted([(f["t_start"], f["t_end"]) for f in s["faults"]] + [(m["t_start"], m["t_end"]) for m in s["maintenance"]])
        assert all(a >= sim.warmup_steps for a, _ in wins)
        assert all(b + cfg["faults"]["min_gap_steps"] <= a2 for (_, b), (a2, _) in zip(wins, wins[1:]))
        assert all(f["type"] in FAULT_TYPES for f in s["faults"])
        for m in s["maintenance"]:
            assert (m["t_start"] % sim.steps_per_day) * 5 / 60 < cfg["distractors"]["maintenance_start_hours"][1]
            if m["action"] != "noop":
                key = ((m["target"],), (), ()) if m["action"] == "down" else ((), (), ((m["target"], cfg["simulator"]["drain_metric"]),))
                assert sim.planned_peak_util(key, m["t_start"], m["t_end"]) <= cfg["distractors"]["maintenance_max_util"]
    assert sample_scenario(sim, "x", 9) == sample_scenario(sim, "x", 9)


def test_observed_events_never_leak_ground_truth(sim):
    for seed in range(10):
        res = sim.simulate(sample_scenario(sim, "s", seed), counterfactual=False)
        for e in res.observed_events:
            assert e["type"] in ("maintenance", "link_status", "pop_status")
            assert not set(e) & {"root_cause", "params", "fault", "impact"}


def test_write_scenario_roundtrip(sim, tmp_path):
    res = sim.simulate(sample_scenario(sim, "rt", 2))
    entry = write_scenario(res, tmp_path)
    write_manifest([entry], tmp_path)
    links = pd.read_parquet(tmp_path / "rt" / "links.parquet")
    assert len(links) == sim.n_steps * len(sim.links) and list(links.columns) == ["t", "link", "util", "latency_ms", "loss_pct"]
    lf, _ = to_frames(res)
    pd.testing.assert_frame_equal(links, lf)
    manifest = json.loads((tmp_path / "scenarios.json").read_text())
    assert manifest[0]["faults"][0]["impact"]["impact_class"] in ("subtle", "moderate", "obvious")
    assert json.loads((tmp_path / "rt" / "observed_events.json").read_text()) == json.loads(json.dumps(res.observed_events))
