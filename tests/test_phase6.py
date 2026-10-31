"""Phase 6: the agent tools see observable data only, nothing at or after the incident's end, times relative to the
window start; the change log includes drains; scoring uses the harness's tie credit; groundedness catches invented ids
and numbers; the loop keeps its budgets. No API calls (a fake client). Runs on topohub's `gabriel/20/1` (in no split)."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from nettwin.agent.eval import localisation_credit, ranking_scores, sample_incidents, score_agent
from nettwin.agent.loop import groundedness, run_incident
from nettwin.agent.rag import RunbookIndex
from nettwin.agent.tools import CONDITIONS, Incident, call_tool, tools_for
from nettwin.faults import make_fault
from nettwin.gnn.models import NetTwinGNN
from nettwin.observed import freeze_cfg, inventory, load_observed
from nettwin.simulator import Simulator, write_scenario
from nettwin.topology import load

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = "gabriel/20/1"


@pytest.fixture(scope="module")
def inc(tmp_path_factory):
    cfg = copy.deepcopy(yaml.safe_load((ROOT / "config.yaml").read_text()))
    cfg["simulator"]["days"] = 3
    sim = Simulator(load(FIXTURE), cfg)
    links = sorted(sim.links, key=lambda l: (-len(sim.kg.paths_through(l)), l))
    rng = np.random.default_rng(0)
    f = make_fault(sim, "link_down", links[0], 400, 30, rng, fault_id="F0")
    f.update(placement="isolated", linked_maintenance=None, caused_by_maintenance=False, overlaps_maintenance=False)
    maint = [{"id": "M0", "target": links[1], "t_start": 380, "t_end": 420, "action": "drain", "description": "planned traffic drain"},
             {"id": "M1", "target": links[2], "t_start": 440, "t_end": 470, "action": "down", "description": "planned outage"}]
    spec = {"scenario_id": "va99997", "topology": sim.name, "seed": 8903, "n_steps": sim.n_steps, "step_minutes": 5,
            "faults": [f], "maintenance": maint, "gaps": [], "event_days": []}
    res = sim.simulate(spec)
    root = tmp_path_factory.mktemp("p6")
    write_scenario(res, obs_dir=root / "observable" / "val" / "with_status", truth_dir=root / "truth" / "val" / "with_status")
    obs = load_observed(root / "observable" / "val" / "with_status" / "va99997")
    cs = freeze_cfg(cfg)
    kg = inventory(FIXTURE, cs)
    torch.manual_seed(0)
    model = NetTwinGNN(conv="gat", hidden=16).eval()
    return Incident(obs, kg, 400, 430, cs, gnn=model, runbooks=RunbookIndex()), links


def test_tools_stop_at_the_window_end(inc):
    I, links = inc
    s = json.loads(call_tool(I, "get_metric_series", {"element": links[3], "metric": "util", "t0": -20, "t1": 500})[0])
    assert s["range"] == [-20, 29]                               # clipped to now - 1, relative to the window start
    out, err = call_tool(I, "get_metric_series", {"element": links[3], "metric": "util", "t0": 30, "t1": 60})
    assert err and "data ends" in out
    a = json.loads(call_tool(I, "get_active_alerts", {})[0])
    assert all(x["start"] < 30 and (x["end"] is None or x["end"] < 30) for x in a["alerts"])
    # the GNN is fed nothing at or after now
    g, feat, det, _ = I._gnn_scores()
    assert feat.shape[0] == det.shape[0] == I.now


def test_change_log_includes_drains_and_only_started_changes(inc):
    I, links = inc
    log = json.loads(call_tool(I, "get_change_log", {"t0": -100, "t1": 29})[0])
    acts = {(c["action"], c["target"]) for c in log["changes"]}
    assert ("drain", links[1]) in acts
    assert all(c["action"] != "down" for c in log["changes"])   # M1 starts at 440, after now (430): not shown
    assert [c["start"] for c in log["changes"]] == [-20]


def test_unknown_elements_and_tools_come_back_as_errors(inc):
    I, _ = inc
    out, err = call_tool(I, "kg_neighbours", {"element": "link:nope--nope"})
    assert err and "unknown element" in out
    assert [t["name"] for t in tools_for("A0")] == ["submit_diagnosis"]
    assert "get_gnn_scores" not in CONDITIONS["A1"] and "get_change_log" in CONDITIONS["A1"]
    assert {"get_gnn_scores", "explain_gnn", "search_runbooks"} <= set(CONDITIONS["A2"])


def test_runbook_search_finds_drains():
    rb = RunbookIndex()
    assert len(rb.docs) == 9
    assert rb.search("planned drain maintenance change log", k=1)[0][0] == "RB08_maintenance_and_drains"
    assert rb.search("interface flapping down up repeatedly", k=1)[0][0] == "RB04_link_flapping"


def test_ranking_uses_harness_tie_credit():
    els = ["link:a--b", "link:b--c", "pop:a", "pop:b"]
    sc = ranking_scores({"root_cause": "link:b--c", "alternatives": ["pop:a", "bogus"]}, els)
    assert list(sc) == [0, 3, 2, 0]
    assert localisation_credit(sc, els, "link:b--c") == {"top1": 1.0, "top3": 1.0}
    assert localisation_credit(sc, els, "pop:a") == {"top1": 0.0, "top3": 1.0}
    none = localisation_credit(ranking_scores({"root_cause": "none", "alternatives": []}, els), els, "pop:a")
    assert none["top1"] == pytest.approx(0.25) and none["top3"] == pytest.approx(0.75)   # expected rank among 4 ties
    s = score_agent({"kind": "distractor", "type": "distractor_drain"}, {"fault_type": "planned_maintenance", "root_cause": "none"}, els)
    assert s["claims_fault"] is False


def test_groundedness_flags_invented_ids_and_numbers():
    corpus = '{"link":"link:R1--R2","mean_now":0.665,"change":0.125}'
    ok = groundedness({"evidence": ["link:R1--R2 utilisation 0.67 now (was +0.125)"]}, corpus)
    assert ok["n_ungrounded_items"] == 0
    bad = groundedness({"evidence": ["link:R9--R10 at 0.91", "link:R1--R2 mean 0.665"]}, corpus)
    assert bad["n_ungrounded_items"] == 1 and set(bad["ungrounded"][0]["not_found"]) == {"link:R9--R10", "0.91"}


class FakeClient:
    """Calls one tool per turn from a script, then submits."""
    def __init__(self, script, allow_any=False):
        self.script, self.k, self.allow_any = script, 0, allow_any
        self.messages = self

    def create(self, **kw):
        assert kw["output_config"]["effort"] and kw["cache_control"] == {"type": "ephemeral"}
        names = [t["name"] for t in kw["tools"]]
        name, args = self.script[min(self.k, len(self.script) - 1)]
        if name not in names and not self.allow_any:
            name, args = "submit_diagnosis", {"root_cause": "none", "fault_type": "no_fault", "confidence": 0.5,
                                              "alternatives": [], "evidence": [], "recommended_action": "none"}
        self.k += 1
        block = SimpleNamespace(type="tool_use", id=f"tu{self.k}", name=name, input=args)
        usage = SimpleNamespace(input_tokens=1000, cache_creation_input_tokens=0, cache_read_input_tokens=0, output_tokens=100)
        return SimpleNamespace(content=[block], usage=usage, stop_reason="tool_use")


def test_loop_budget_forces_a_diagnosis(inc):
    I, _ = inc
    acfg = yaml.safe_load((ROOT / "config.yaml").read_text())["agent"] | {"max_tool_calls": 3}
    client = FakeClient([("get_active_alerts", {})] * 10)
    out = run_incident(client, I, "A2", acfg["dev_model"], acfg, status_polling=True)
    assert out["n_tool_calls"] == 3 and out["forced_submit"] and out["stop"] == "submitted"
    assert out["diagnosis"]["fault_type"] == "no_fault"
    assert out["cost_usd"] == pytest.approx(4 * (1000 * 0.10 + 100 * 0.50) / 1e6)
    # a tool outside the condition is refused
    out = run_incident(FakeClient([("get_gnn_scores", {}), ("submit_diagnosis", {"root_cause": "none", "fault_type": "no_fault", "confidence": 1,
                                                                                 "alternatives": [], "evidence": [], "recommended_action": ""})], allow_any=True),
                       I, "A1", acfg["dev_model"], acfg, status_polling=True)
    assert out["calls"][0]["error"]


def test_sampling_is_isolated_and_stratified():
    e = [{"scenario_id": f"va{i:05d}", "faults": [{"type": t, "root_cause": "link:a--b", "t_start": 400, "t_end": 430, "placement": "isolated",
                                                  "impact": {"observable": True}}],
          "maintenance": [{"action": "drain", "target": "link:c--d", "t_start": 800, "t_end": 830}]}
         for i, t in enumerate(["link_down", "demand_surge"] * 4)]
    inc = sample_incidents(e, 2, 1, 0)
    assert sum(i["kind"] == "fault" for i in inc) == 4 and len({i["scenario_id"] for i in inc}) == len(inc)
    assert any(i["type"] == "distractor_drain" for i in inc)
