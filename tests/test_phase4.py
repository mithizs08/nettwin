"""Phase 4: alerts from observable data only, baselines respect the banned-feature rules, and the evaluation harness
computes what the decision log says. Runs on topohub's shipped `gabriel/20/1` (in no split) and toy arrays."""
import copy
import dataclasses
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.metrics import average_precision_score

from nettwin.alerts import alerts_at, generate_alerts
from nettwin.baselines import FEATURES, B1EwmaZ, B2IsolationForest, B3Plus, B3Tomography, Index, index_for, pop_scores
from nettwin.dataset import _truth_entry, blanked_cells
from nettwin.evaluation import (WARMUP, ap_from_hists, labels_matrix, make_edges, related_links, scenario_record,
                                threshold_for, tie_credit, to_bins, window_classes)
from nettwin.faults import make_fault
from nettwin.observed import Observed, freeze_cfg, inventory, load_observed
from nettwin.simulator import Simulator, write_scenario
from nettwin.topology import load

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = "gabriel/20/1"
TIME_WORDS = ("hour", "time", "clock", "day_of", "weekday", "since_start", "file", "size")


@pytest.fixture(scope="module")
def cfg():
    c = copy.deepcopy(yaml.safe_load((ROOT / "config.yaml").read_text()))
    c["simulator"]["days"] = 3
    return c


@pytest.fixture(scope="module")
def scen(cfg, tmp_path_factory):
    """One 3-day scenario with a link_down and an overlapping node_down elsewhere, written like the v1 dataset."""
    sim = Simulator(load(FIXTURE), cfg)
    link = max(sim.links, key=lambda l: (len(sim.kg.paths_through(l)), l))
    ends = {sim.kg.G.nodes[link]["u"], sim.kg.G.nodes[link]["v"]}
    pop = next(p for p in sim.pops if p not in ends and all(n not in ends for n in sim.topo.neighbors(p)))
    rng = np.random.default_rng(0)
    faults = [make_fault(sim, "link_down", link, 400, 30, rng, fault_id="F0"),
              make_fault(sim, "node_down", f"pop:{pop}", 410, 10, rng, fault_id="F1")]
    for f in faults:
        f.update(placement="concurrent", linked_maintenance=None, caused_by_maintenance=False, overlaps_maintenance=False)
    spec = {"scenario_id": "va99999", "topology": sim.name, "seed": 8901, "n_steps": sim.n_steps, "step_minutes": 5,
            "faults": faults, "maintenance": [], "gaps": [], "event_days": []}
    res, blind = sim.simulate(spec, with_blind=True)
    blanked = blanked_cells(res, blind)
    root = tmp_path_factory.mktemp("p4")
    out = {}
    for variant, r in (("with_status", res), ("status_blind", blind)):
        e = write_scenario(r, obs_dir=root / "observable" / "val" / variant, truth_dir=root / "truth" / "val" / variant)
        out[variant] = _truth_entry(e, r, sim, root, "val", variant, {"topology": sim.name, "k": 0},
                                    blanked if variant == "status_blind" else None)
    return root, out, cfg


def _obs(root, variant):
    return load_observed(root / "observable" / "val" / variant / "va99999")


def test_alerts_come_from_observable_data_only(scen, tmp_path):
    root, entries, cfg = scen
    kg = inventory(FIXTURE, freeze_cfg(cfg))
    copy_root = tmp_path / "only_observable"
    shutil.copytree(root / "observable", copy_root / "observable")       # no truth folder at all
    a = generate_alerts(load_observed(copy_root / "observable" / "val" / "with_status" / "va99999"), kg)
    pd.testing.assert_frame_equal(a, generate_alerts(_obs(root, "with_status"), kg))
    with pytest.raises(ValueError, match="observable"):
        load_observed(root / "truth" / "val" / "with_status" / "va99999")
    f0 = entries["with_status"]["faults"][0]
    down = a[(a.rule == "link_status_down") & (a.target == f0["root_cause"])]
    assert len(down) and down.t_start.min() >= f0["t_start"] and down.t_start.min() <= f0["t_start"] + 3
    blind = generate_alerts(_obs(root, "status_blind"), kg)
    assert not blind.rule.isin(["link_status_down", "pop_unreachable"]).any()
    assert (blind.rule == "link_no_data").any()


def test_no_clock_and_no_banned_features():
    fields = {f.name for f in dataclasses.fields(Observed)}
    assert fields == {"scenario_id", "topology", "status_blind", "links", "services", "util", "latency", "discards",
                      "svc_latency", "svc_loss", "events"}
    assert not [f for f in FEATURES if any(w in f for w in TIME_WORDS)]


def test_scores_are_causal(scen):
    """Scores at step t use data up to t only: cutting the scenario at step 500 changes nothing before it."""
    root, _, cfg = scen
    obs = _obs(root, "with_status")
    kg = inventory(FIXTURE, freeze_cfg(cfg))
    ix = index_for(obs, kg)
    cut = dataclasses.replace(obs, util=obs.util[:500], latency=obs.latency[:500], discards=obs.discards[:500],
                              svc_latency=obs.svc_latency[:500], svc_loss=obs.svc_loss[:500],
                              events=[e for e in obs.events if e["t"] < 500])
    b2 = B2IsolationForest(seed=1, n_cells=2000, n_estimators=20).fit([obs])
    for m in (B1EwmaZ(), b2, B3Tomography()):
        full, part = m.score(obs, kg, ix)[0], m.score(cut, kg, ix)[0]
        np.testing.assert_allclose(full[:500], part, err_msg=m.name)


def test_binned_ap_matches_exact():
    rng = np.random.default_rng(0)
    y = rng.random(200_000) < 0.002
    s = rng.normal(size=y.size) + 2.5 * y
    edges = make_edges(s[::10])                       # label-free bins from a sample of all scores (audit Q10)
    b = to_bins(s, edges)
    nb = len(edges) + 1
    binned = ap_from_hists(np.bincount(b[y], minlength=nb), np.bincount(b[~y], minlength=nb))
    assert abs(binned - average_precision_score(y, s)) < 0.005
    ties = np.r_[np.ones(10), np.zeros(90)]
    assert np.isclose(ap_from_hists(np.array([0, 10]), np.array([0, 90])), average_precision_score(ties > 0, np.ones(100)))


class Fixed:
    """A stub localiser: fixed score per element."""
    def __init__(self, scores):
        self.scores = scores

    def localise(self, state, t0, t1, agg="q90"):
        return self.scores


def test_localisation_is_set_based_with_expected_ties(scen):
    root, entries, cfg = scen
    entry = entries["with_status"]
    obs = _obs(root, "with_status")
    ix = index_for(obs, inventory(FIXTURE, freeze_cfg(cfg)))
    lab, pres = labels_matrix(pd.read_parquet(root / "truth" / "val" / "with_status" / entry["files"]["labels"]), ix.links)
    roots = [ix.elements.index(f["root_cause"]) for f in entry["faults"]]
    link = np.zeros(lab.shape)
    edges = np.array([0.0, 1.0])
    s = np.zeros(len(ix.elements))
    s[roots] = 1.0                                         # both concurrent roots tie at the top
    rec = scenario_record(Fixed(s), (link,), ix, entry, lab, pres, edges)
    for f in rec["faults"]:                                # the other true root is taken out, not counted against
        assert f["loc"]["q90"]["top1"] == 1 and f["loc"]["q90"]["rank"] == 1
    rec = scenario_record(Fixed(np.zeros(len(ix.elements))), (link,), ix, entry, lab, pres, edges)
    n = len(ix.elements) - 1                               # candidates left after removing the other root
    for f in rec["faults"]:                                # all tied: random tie-breaking in expectation (audit Q2)
        lc = f["loc"]["mean"]
        assert np.isclose(lc["top1"], 1 / n) and np.isclose(lc["top3"], 3 / n)
        assert lc["rank"] == 1 + (n - 1) / 2 and lc["rank_pessimistic"] == n


def toy_index():
    """PoPs A (hub), B, C, D (leaf on C). Links A-B, A-C, B-C, C-D (C-D is a spur)."""
    pops = ["pop:A", "pop:B", "pop:C", "pop:D"]
    links = ["link:A--B", "link:A--C", "link:B--C", "link:C--D"]
    ends = np.array([(0, 1), (0, 2), (1, 2), (2, 3)])
    inc = np.zeros((4, 4))
    inc[np.arange(4), ends[:, 0]] = 1
    inc[np.arange(4), ends[:, 1]] = 1
    return Index(links, pops, ends, inc)


def test_a_pop_or_a_spur_link_can_rank_first():
    """Audit Q2: with the PoP score a mean of link scores and pessimistic ties, a PoP root or a spur-link root could
    never be ranked first by a per-link method. Now it can."""
    ix = toy_index()
    keep_all = lambda true: np.arange(8) != true   # noqa: E731
    link = np.array([[5.0, 5.0, 0.0, 0.0]])        # both links of PoP A are anomalous (A is down)
    for agg in ("max", "second_max"):
        sc = np.hstack([link, pop_scores(link, ix, agg)])[0]
        true = ix.elements.index("pop:A")
        assert tie_credit(sc, true, keep_all(true))["top1"] > 0
    sc = np.hstack([link, pop_scores(link, ix, "mean")])[0]
    assert tie_credit(sc, 4, keep_all(4))["rank_pessimistic"] > 1          # the old rule: never first
    spur = np.array([[0.0, 0.0, 0.0, 5.0]])                                  # only the spur link C-D is anomalous
    sc = np.hstack([spur, pop_scores(spur, ix, "second_max")])[0]
    true = ix.links.index("link:C--D")
    assert tie_credit(sc, true, keep_all(true))["top1"] == 1                 # the leaf PoP D no longer ties with it
    sc = np.hstack([spur, pop_scores(spur, ix, "mean")])[0]
    assert tie_credit(sc, true, keep_all(true))["top1"] == 0.5               # mean: a coin flip with D, no longer zero


def test_b3_is_preregistered_and_b3plus_is_the_variant(scen):
    """Audit Q1: B3's own link path is the link alone; B3+ adds the link's two end PoPs."""
    root, _, cfg = scen
    obs = _obs(root, "with_status")
    kg = inventory(FIXTURE, freeze_cfg(cfg))
    ix = index_for(obs, kg)
    S, L = len(obs.services), len(ix.links)
    _, M = B3Tomography()._matrices(obs, kg, ix)
    _, Mp = B3Plus()._matrices(obs, kg, ix)
    assert (M[S:S + L, L:] == 0).all() and (M[S:S + L, :L] == np.eye(L)).all()
    assert (Mp[S:S + L, L:].sum(1) == 2).all()
    assert (M[:S] == Mp[:S]).all() and (M[S + L:] == Mp[S + L:]).all()


def test_alert_fields_never_use_future_steps(scen):
    """Audit Q9: cutting the data at step c changes no field of an alert raised before c, except that an alert still
    open at c is open (t_end = c) in the cut data; `alerts_at` hides the clear time of alerts open at t."""
    root, _, cfg = scen
    obs = _obs(root, "with_status")
    kg = inventory(FIXTURE, freeze_cfg(cfg))
    full = generate_alerts(obs, kg)
    for c in (300, 405, 412, 431, 600):
        cut = dataclasses.replace(obs, util=obs.util[:c], latency=obs.latency[:c], discards=obs.discards[:c],
                                  svc_latency=obs.svc_latency[:c], svc_loss=obs.svc_loss[:c],
                                  events=[e for e in obs.events if e["t"] < c])
        part = generate_alerts(cut, kg).drop(columns="alert_id")
        expect = full[full.t_start < c].drop(columns="alert_id").copy()
        expect["t_end"] = expect["t_end"].clip(upper=c)
        key = ["t_start", "rule", "target"]
        pd.testing.assert_frame_equal(part.sort_values(key).reset_index(drop=True), expect.sort_values(key).reset_index(drop=True))
        known = alerts_at(full, c)
        assert (known.t_start <= c).all() and known.t_end.isna().sum() == (full[full.t_start <= c].t_end > c).sum()


def test_symptoms_of_concurrent_faults_go_to_the_related_fault(scen):
    """Audit Q11: a symptomatic cell inside two overlapping fault windows goes to the fault whose KG neighbourhood
    contains the link."""
    root, entries, cfg = scen
    entry = entries["with_status"]
    obs = _obs(root, "with_status")
    kg = inventory(FIXTURE, freeze_cfg(cfg))
    ix = index_for(obs, kg)
    lab, pres = labels_matrix(pd.read_parquet(root / "truth" / "val" / "with_status" / entry["files"]["labels"]), ix.links)
    related = [related_links(kg, ix, f["root_cause"]) for f in entry["faults"]]
    rec = scenario_record(Fixed(np.zeros(len(ix.elements))), (np.zeros(lab.shape),), ix, entry, lab, pres, np.array([0.5]), related)
    L = len(ix.links)
    sym = rec["pos_task"] == 1
    cells, faults = rec["pos_cell"][sym], rec["pos_fault"][sym]
    t, j = cells // L, cells % L
    f1 = entry["faults"][1]
    both = (t >= f1["t_start"]) & (t < f1["t_end"] + 3)        # inside both windows (F1 lies inside F0)
    for k in (0, 1):
        sel = both & (faults == k)
        owned_elsewhere = [jj for jj in j[sel] if jj not in related[k]]
        assert not owned_elsewhere or all(jj not in related[1 - k] for jj in owned_elsewhere)
    assert rec["sym_reassigned"] + rec["sym_ambiguous"] <= both.sum()


def test_telemetry_absence_channel(scen):
    """Phase 4 audit follow-up: in status_blind a down link that is blanked counts as observable through the
    telemetry-absence channel; in with_status nothing is blanked, so the channel is always false."""
    _, entries, _ = scen
    f0_ws, f0_sb = entries["with_status"]["faults"][0], entries["status_blind"]["faults"][0]
    assert f0_ws["type"] == "link_down"
    assert f0_sb["impact"]["observability"]["obs_telemetry_absence"] and f0_sb["impact"]["observable"]
    assert not any(f["impact"]["observability"]["obs_telemetry_absence"] for f in entries["with_status"]["faults"])


def test_record_counts_cells_as_defined(scen):
    root, entries, cfg = scen
    entry = entries["status_blind"]
    obs = _obs(root, "status_blind")
    ix = index_for(obs, inventory(FIXTURE, freeze_cfg(cfg)))
    lab, pres = labels_matrix(pd.read_parquet(root / "truth" / "val" / "status_blind" / entry["files"]["labels"]), ix.links)
    rec = scenario_record(Fixed(np.zeros(len(ix.elements))), (np.zeros(lab.shape),), ix, entry, lab, pres, np.array([0.5]))
    assert rec["sym_unowned"] == 0
    assert rec["neg"].sum() == np.sum(lab[WARMUP:] == 0)               # every label-0 cell after warmup, with or without telemetry
    assert rec["neg_present"].sum() == np.sum((lab[WARMUP:] == 0) & pres[WARMUP:])
    assert np.sum(rec["pos_task"] == 2) == np.sum(lab[WARMUP:] == 2)   # every root cell belongs to a fault
    assert not rec["pos_present"][rec["pos_task"] == 2].all()          # status_blind: down root links have no telemetry


def test_window_classes_and_threshold():
    entry = {"faults": [{"t_start": 300, "t_end": 310}], "maintenance": [{"t_start": 400, "t_end": 410}]}
    starts, ends, cls = window_classes(entry, 600)
    assert starts[0] == WARMUP and cls[0] == 0 and cls[1] == 2         # 288-299 is clear, 300-311 meets the fault
    assert cls[list(starts).index(408)] == 1 and cls[list(starts).index(420)] == 1   # change window + 12-step tail
    assert cls[list(starts).index(432)] == 0
    x = np.arange(1000.0)
    thr = threshold_for(x, 0.05)
    assert np.mean(x >= thr) <= 0.05 and np.mean(x >= thr - 1) > 0.05


def test_b3_localises_a_failed_link(scen):
    root, entries, cfg = scen
    obs = _obs(root, "with_status")
    kg = inventory(FIXTURE, freeze_cfg(cfg))
    ix = index_for(obs, kg)
    m = B3Tomography()
    state = m.score(obs, kg, ix)
    f0, f1 = entries["with_status"]["faults"]
    top = np.argsort(-m.localise(state, f0["t_start"], f0["t_end"]))[:3]
    assert ix.elements.index(f0["root_cause"]) in top or ix.elements.index(f1["root_cause"]) in top


def test_phase4_script_reads_train_and_val_only():
    import re
    src = (ROOT / "scripts" / "phase4_baselines.py").read_text()
    splits = set(re.findall(r'/ "(\w+)" /', src)) | set(re.findall(r'"(\w+)", variant', src))
    assert {"train", "val"} <= splits <= {"train", "val", "truth", "observable", "data", "datasets", "results"}
