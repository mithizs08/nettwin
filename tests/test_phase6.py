"""Phase 6: the agent tools see observable data only, nothing at or after the incident's end, times relative to the
window start; the change log includes drains; scoring uses the harness's tie credit; groundedness catches invented ids
and numbers; the loop keeps its budgets. No API calls (a fake client). Runs on topohub's `gabriel/20/1` (in no split)."""
import copy
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from nettwin.agent.eval import GROUPS, localisation_credit, ranking_scores, sample_incidents, score_agent
from nettwin.agent import freeze, load_agent_config
from nettwin.agent.loop import groundedness, run_incident
from nettwin.agent.rag import RunbookIndex
from nettwin.agent.tools import CONDITIONS, HISTORY, Incident, call_tool, tools_for
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


def _shifted(I, k):
    """The same incident with k extra steps of telemetry before the scenario start (copies of the first k steps)."""
    o = I.obs
    pad = {f: np.concatenate([getattr(o, f)[:k], getattr(o, f)]) for f in ("util", "latency", "discards", "svc_latency", "svc_loss")}
    ev = [e | {key: e[key] + k for key in ("t", "t_start", "t_end") if key in e} for e in o.events]
    o2 = dataclasses.replace(o, events=ev, **pad)
    return Incident(o2, I.kg, I.t0 + k, I.now + k, I.cfg_s, gnn=I.gnn, runbooks=I.runbooks)


def test_tools_do_not_reveal_the_window_position(inc):
    """Phase 6 audit S3: history is cut HISTORY steps before the window start, so a request for everything gives the
    same output wherever the window sits in the scenario, and nothing reaches back to the scenario start."""
    I, links = inc
    J = _shifted(I, 100)
    calls = [("get_metric_series", {"element": links[3], "metric": "util", "t0": -10**7, "t1": 10**7}),
             ("get_metric_series", {"element": links[0], "metric": "latency", "t0": -400, "t1": -250}),
             ("list_links_by_change", {"metric": "util", "t0": -10**7, "t1": 10**7}),
             ("get_change_log", {"t0": -10**7, "t1": 10**7})]
    for name, args in calls:
        assert call_tool(I, name, args) == call_tool(J, name, args), name
    assert json.loads(call_tool(I, "get_metric_series", calls[0][1])[0])["range"] == [-HISTORY, 29]
    a = json.loads(call_tool(I, "get_active_alerts", {"t0": -10**7, "t1": 10**7})[0])
    assert all(x["start"] >= -HISTORY for x in a["alerts"])
    assert json.loads(call_tool(I, "get_gnn_scores", {"t": -HISTORY, "over_window": False})[0])["range"] == [-HISTORY, -HISTORY]
    assert call_tool(I, "get_gnn_scores", {"t": -HISTORY - 1, "over_window": False}) == call_tool(J, "get_gnn_scores", {"t": -HISTORY - 1, "over_window": False})
    with pytest.raises(ValueError):
        Incident(I.obs, I.kg, HISTORY - 1, HISTORY + 5, I.cfg_s)


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
    assert list(ranking_scores({"root_cause": "link:c--b", "alternatives": []}, els)) == [0, 3, 0, 0]   # reversed link name
    none = localisation_credit(ranking_scores({"root_cause": "none", "alternatives": []}, els), els, "pop:a")
    assert none["top1"] == pytest.approx(0.25) and none["top3"] == pytest.approx(0.75)   # expected rank among 4 ties
    s = score_agent({"kind": "distractor", "type": "distractor_drain"}, {"fault_type": "planned_maintenance", "root_cause": "none"}, els)
    assert s["claims_fault"] is False


def test_groundedness_flags_invented_ids_and_numbers():
    known = ["link:R1--R2", "link:New Orleans--Tampa", "pop:New Orleans", "pop:R9"]
    corpus = '{"link":"link:R1--R2","mean_now":0.665,"change":0.125},{"element":"link:New Orleans--Tampa","mean":0.0}'
    ok = groundedness({"evidence": ["link:R1--R2 utilisation 0.67 now (was +0.125).", "link:New Orleans--Tampa util 0.0",
                                    "no link:... or pop:... alerts"]}, corpus, known)
    assert ok["n_ungrounded_items"] == 0
    bad = groundedness({"evidence": ["link:R9--R10 at 0.91", "link:R1--R2 mean 0.665", "pop:New Orleans is down"]}, corpus, known)
    assert bad["n_ungrounded_items"] == 2
    assert set(bad["ungrounded"][0]["not_found"]) == {"link:R9--R10", "0.91"}
    assert bad["ungrounded"][1]["not_found"] == ["pop:New Orleans"]       # a real element the tools never returned
    # strict (Phase 6 audit S5): numbers must appear verbatim, so the rounded 0.67 no longer counts
    assert ok["n_ungrounded_items_strict"] == 1 and ok["ungrounded_strict"][0]["not_found"] == ["0.67"]
    assert bad["n_ungrounded_items_strict"] == 2


class FakeClient:
    """Calls one tool per turn from a script, then submits."""
    def __init__(self, script, allow_any=False, parallel=1):
        self.script, self.k, self.allow_any, self.names, self.parallel = script, 0, allow_any, {}, parallel
        self.messages = self

    def create(self, **kw):
        assert kw["output_config"]["effort"] and kw["cache_control"] == {"type": "ephemeral"}
        assert kw["system"][-1]["cache_control"] == {"type": "ephemeral"}      # tools + system prefix cached
        names = [t["name"] for t in kw["tools"]]
        name, args = self.script[min(self.k, len(self.script) - 1)]
        forced = any(m["role"] == "user" and isinstance(m["content"], str) and m["content"].startswith("Budget reached") for m in kw["messages"])
        assert names == self.names.setdefault("tools", names)        # the tool list never changes in a conversation
        if (name not in names and not self.allow_any) or forced:
            name, args = "submit_diagnosis", {"root_cause": "none", "fault_type": "no_fault", "confidence": 0.5,
                                              "alternatives": [], "evidence": [], "recommended_action": "none"}
        self.k += 1
        n = 1 if name == "submit_diagnosis" else self.parallel
        blocks = [SimpleNamespace(type="tool_use", id=f"tu{self.k}_{i}", name=name, input=args) for i in range(n)]
        usage = SimpleNamespace(input_tokens=1000, cache_creation_input_tokens=0, cache_read_input_tokens=0, output_tokens=100)
        return SimpleNamespace(content=blocks, usage=usage, stop_reason="tool_use")


def test_loop_budget_forces_a_diagnosis(inc):
    I, _ = inc
    acfg = load_agent_config(ROOT) | {"max_tool_calls": 3}
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
    # Phase 6 audit S6: parallel calls in the last turn beyond the budget are refused, not run
    out = run_incident(FakeClient([("get_active_alerts", {})] * 10, parallel=2), I, "A2", acfg["dev_model"], acfg, status_polling=True)
    assert out["n_tool_calls"] == 3 and out["n_refused_calls"] == 1 and out["stop"] == "submitted"
    assert [c["refused"] for c in out["calls"]] == [False, False, False, True]


def test_sampling_is_isolated_and_stratified():
    e = [{"scenario_id": f"va{i:05d}", "faults": [{"type": t, "root_cause": "link:a--b", "t_start": 400, "t_end": 430, "placement": "isolated",
                                                  "impact": {"observable": True}}],
          "maintenance": [{"action": "drain", "target": "link:c--d", "t_start": 800, "t_end": 830}]}
         for i, t in enumerate(["link_down", "demand_surge"] * 4)]
    inc = sample_incidents(e, 2, 1, 0)
    assert sum(i["kind"] == "fault" for i in inc) == 4 and len({i["scenario_id"] for i in inc}) == len(inc)
    assert any(i["type"] == "distractor_drain" for i in inc)
    # the change-overlap stratum (Phase 6 audit S2) adds faults that overlap a drain, leaving the main sample alone
    for k in (1, 3):
        e[k]["maintenance"] = [{"action": "drain", "target": "link:c--d", "t_start": 410, "t_end": 420}]
    more = sample_incidents(e, 2, 1, 0, n_change_overlap=3)
    assert more[:len(inc)] == [i for i in more if i["stratum"] == "main"]
    assert [(i["scenario_id"], i["type"]) for i in more[:len(inc)]] == [(i["scenario_id"], i["type"]) for i in inc]
    ov = [i for i in more if GROUPS["change_overlap_faults"](i)]
    assert 1 <= len(ov) <= 3 and all(i["overlaps_logged_change"] for i in more if i["stratum"] == "change_overlap")
    assert not any(GROUPS["faults"](i) for i in more if i["stratum"] == "change_overlap")


def test_test_runs_are_frozen_and_run_once(tmp_path):
    """Phase 6 audit S1: a test run needs the frozen hashes, the full design, and a model with no earlier test run."""
    acfg = load_agent_config(ROOT)
    h = freeze.agent_hashes(ROOT, acfg)
    assert "src/nettwin/agent/runbooks/RB08_maintenance_and_drains.md" in h["files"] and "scripts/phase6_agent.py" in h["files"]
    assert freeze.agent_hashes(ROOT, acfg | {"max_tool_calls": 16})["code_sha256"] != h["code_sha256"]
    m, full = "claude-haiku-5-5", ["A0", "A1", "A2"]
    ok = dict(root=tmp_path, frozen=h, current=h, model=m, model_key="eval", limit=None, conditions=full,
              variants=["with_status", "status_blind"], required_conditions=full, resume=False)
    assert freeze.test_run_refusal(**ok) is None
    assert "missing" in freeze.test_run_refusal(**ok | {"frozen": None})
    assert "differs" in freeze.test_run_refusal(**ok | {"frozen": h | {"code_sha256": "x", "files": {}}})
    assert "--limit" in freeze.test_run_refusal(**ok | {"limit": 10})
    assert "dev model" in freeze.test_run_refusal(**ok | {"model_key": "dev"})
    assert "exactly" in freeze.test_run_refusal(**ok | {"conditions": ["A2"]})
    out = freeze.test_out_dir(tmp_path, m)
    out.mkdir(parents=True)
    (out / "LOCK.json").write_text(json.dumps({"model": m, "code_sha256": h["code_sha256"]}))
    assert "run once" in freeze.test_run_refusal(**ok)                                    # a second run is refused
    assert freeze.test_run_refusal(**ok | {"resume": True}) is None                       # an unfinished one may resume
    (out / "summary.json").write_text("{}")
    assert "finished" in freeze.test_run_refusal(**ok | {"resume": True})
    other = tmp_path / "results" / "phase6" / "test_renamed"
    other.mkdir()
    (other / "LOCK.json").write_text(json.dumps({"model": "claude-opus-5-5", "code_sha256": h["code_sha256"]}))
    assert "already exists in test_renamed" in freeze.test_run_refusal(**ok | {"model": "claude-opus-5-5", "model_key": "secondary"})
