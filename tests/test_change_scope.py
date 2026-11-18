"""Phase 13: the change-scope tool, the A3 tool list and prompt, the fresh sample and the test refusal."""
import importlib.util
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from nettwin import change_scope as CS
from nettwin.agent.loop import SYSTEM, TOOLS_NOTE
from nettwin.agent.tools import tools_for
from nettwin.observed import freeze_cfg, inventory
from nettwin.scope_agent import A3_TOOLS, SCOPE_TOOL, sample_phase13, system_a3, tools_a3


@pytest.fixture(scope="module")
def kg():
    cfg = yaml.safe_load(open("config.yaml"))
    return inventory("topozoo/Aarnet", freeze_cfg(cfg))


def test_footprint_of_a_drain_moves_traffic_and_leaves_the_inventory_unchanged(kg):
    link = next(l for l in kg.nodes_of_kind("Link") if kg.paths_through(l, t=0))
    n_events = len(kg.events)
    fp = CS.footprint(kg, {"id": "M0", "action": "drain", "target": link, "t_start": 300, "t_end": 320})
    assert link in fp["links_losing_traffic"] and link not in fp["links_gaining_traffic"]
    assert fp["pop_pairs_rerouted"] > 0 and fp["services_rerouted"]
    assert fp["pop_pairs_cut_off"] == 0                   # a drained link still carries traffic if it must
    assert len(kg.events) == n_events                      # the shared inventory is never changed


def _sym(kind, el, start):
    return {"kind": kind, "element": el, "start": start, "end": None}


def test_explains_only_what_the_change_can_cause(kg):
    link = next(l for l in kg.nodes_of_kind("Link") if kg.paths_through(l, t=0))
    ch = {"id": "M0", "action": "drain", "target": link, "t_start": 300, "t_end": 320}
    fp = CS.footprint(kg, ch)
    gain = fp["links_gaining_traffic"][0]
    other = next(l for l in kg.nodes_of_kind("Link") if l not in fp["links_gaining_traffic"] and l not in fp["links_losing_traffic"])
    assert CS._explains(fp, ch, _sym("util_up", gain, 305), 0)
    assert CS._explains(fp, ch, _sym("util_down", link, 301), 0)
    assert not CS._explains(fp, ch, _sym("util_up", other, 305), 0)            # a link the change does not reroute
    assert not CS._explains(fp, ch, _sym("link_status_down", link, 305), 0)    # a drain takes nothing down
    assert not CS._explains(fp, ch, _sym("link_discards", gain, 305), 0)
    assert not CS._explains(fp, ch, _sym("pop_unreachable", fp["target_pops"][0], 305), 0)
    assert not CS._explains(fp, ch, _sym("util_up", gain, 320 + CS.MAINT_TAIL), 0)   # after the change and its tail
    assert not CS._explains(fp, ch, _sym("util_up", gain, 290), 0)             # before the change
    down = dict(ch, action="down")
    assert CS._explains(CS.footprint(kg, down), down, _sym("link_status_down", link, 305), 0)


class FakeObs:
    def __init__(self, links, T=700):
        self.links, self.services, self.events = links, [], []
        self.util = np.full((T, len(links)), 0.4)
        self.discards = np.zeros((T, len(links)))
        self.latency = np.full((T, len(links)), 5.0)


class FakeInc:
    def __init__(self, obs, kg, t0, t1):
        self.obs, self.kg, self.t0, self.now, self.floor = obs, kg, t0, t1, t0 - 288
        self.links = obs.links
        self.alerts = pd.DataFrame(columns=["alert_id", "rule", "target", "severity", "t_start", "t_end", "value"])

    def rel(self, t):
        return int(t) - self.t0


def test_unexplained_traffic_shift_is_listed_and_a_network_wide_shift_is_not(kg):
    links = kg.nodes_of_kind("Link")
    obs = FakeObs(links)
    inc = FakeInc(obs, kg, 400, 420)
    assert CS.change_scope(inc)["n_unexplained"] == 0
    obs.util[400:420] *= 1.5                               # a busier day everywhere: no symptom
    assert CS.change_scope(inc)["n_symptoms"] == 0
    obs.util[400:420, 3] = 0.95                            # one link jumps: a symptom no change explains
    out = CS.change_scope(inc)
    assert out["n_unexplained"] == 1 and out["unexplained_symptoms"][0]["element"] == links[3]
    assert "no logged drain" in out["verdict"]


def test_a3_is_a2_plus_the_scope_tool():
    a2, a3 = tools_for("A2"), tools_a3()
    assert [t["name"] for t in a3] == [t["name"] for t in a2[:-1]] + [SCOPE_TOOL, "submit_diagnosis"]
    assert a3[-1] == a2[-1] and a3[:len(a2) - 1] == a2[:-1]
    assert set(A3_TOOLS) == {t["name"] for t in a3} - {"submit_diagnosis"}
    s = system_a3(15)
    assert s.startswith(SYSTEM) and s.endswith(TOOLS_NOTE.format(n=15)) and SCOPE_TOOL in s


def test_fresh_sample_never_overlaps_a_phase6_incident():
    cfg = yaml.safe_load(open("config.yaml"))
    from nettwin.splits import guard
    from pathlib import Path
    root = Path(cfg["dataset"]["root"])
    entries = json.loads(guard(root / "truth" / "val" / "with_status" / "manifest.json", {"val"}).read_text())
    p6 = json.load(open("results/phase6/val_haiku_v2/incidents.json"))
    inc = sample_phase13(entries, p6, 1313, n_change_overlap=30, n_other_per_type=3, n_distractor=10)
    prior = {}
    for i in p6:
        prior.setdefault(i["scenario_id"], []).append((i["t0"], i["t1"]))
    for i in inc:
        assert not any(i["t0"] < b and a < i["t1"] for a, b in prior.get(i["scenario_id"], []))
    assert sum(i["stratum"] == "change_overlap" for i in inc) == 30
    assert all(i["overlaps_logged_change"] for i in inc if i["stratum"] == "change_overlap")


def test_test_split_refuses_without_preregistration_and_freeze(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("p13", "scripts/phase13_scope.py")
    p13 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p13)
    monkeypatch.setattr(p13, "FROZEN", tmp_path / "missing.json")
    args = type("A", (), {"split": "test", "limit": 0, "resume": False, "name": "x", "threads": 1})()
    from nettwin.agent import load_agent_config
    with pytest.raises(SystemExit) as e:
        p13.run(args, yaml.safe_load(open("config.yaml")), load_agent_config(p13.ROOT))
    assert "refusing the test run" in str(e.value)
