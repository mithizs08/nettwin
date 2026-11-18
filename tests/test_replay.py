"""Phase 12 replay: episode merging, the live KG takes only status events that have arrived, and the agent's view."""
import json

import numpy as np
import pytest
import yaml

from nettwin import replay as R
from nettwin.observed import freeze_cfg, inventory


def test_episodes_merge_repeated_alarms_on_one_element():
    T, L = 400, 3
    elements = ["link:a", "link:b", "link:c", "pop:x"]
    det = np.zeros((T, L))
    elem = np.zeros((T, L + 1))
    elem[:, 0] = 1.0                                     # top root candidate is link:a throughout
    for t in (300, 305, 316, 340):                       # 316 is within 12 steps of 305; 340 is not within 12 of 316
        det[t, 1] = 0.9
    eps = R.episodes(det, elem, elements, threshold=0.5)
    assert [(e["element"], e["open"], e["last"], e["n_alarms"]) for e in eps] == [("link:a", 300, 316, 3), ("link:a", 340, 340, 1)]
    assert eps[0]["alarm_links"] == ["link:b"]


def test_alarms_before_the_warmup_are_ignored():
    det = np.full((300, 1), 0.9)
    elem = np.ones((300, 2))
    eps = R.episodes(det, elem, ["link:a", "pop:x"], threshold=0.5)
    assert eps[0]["open"] == R.START


class FakeObs:
    def __init__(self, events):
        self.events = events


def test_live_kg_applies_only_arrived_status_events():
    cfg = yaml.safe_load(open("config.yaml"))
    inv = inventory("topozoo/Aarnet", freeze_cfg(cfg))
    link = inv.nodes_of_kind("Link")[0]
    obs = FakeObs([{"t": 300, "type": "maintenance", "action": "drain", "t_start": 300, "t_end": 320},
                   {"t": 310, "type": "link_status", "element": link, "status": "down"},
                   {"t": 330, "type": "link_status", "element": link, "status": "up"}])
    assert link not in R.kg_at(inv, obs, 309)._down_links
    assert link in R.kg_at(inv, obs, 310)._down_links
    assert link not in R.kg_at(inv, obs, 335)._down_links
    assert link not in inv._down_links                    # the shared inventory is never changed
    assert R.change_active(obs, 300) and R.change_active(obs, 331) and not R.change_active(obs, 332)


def test_page_scoring_counts_detection_and_false_pages():
    entry = {"faults": [{"id": "F0", "type": "link_down", "root_cause": "link:a", "t_start": 400, "t_end": 420,
                         "impact": {"observable": True}}], "maintenance": []}
    elements = ["link:a", "link:b", "pop:x"]
    pages = [{"episode_id": "E0", "t": 350, "kind": "ids", "ranking": ["link:b"]},
             {"episode_id": "E1", "t": 410, "kind": "ids", "ranking": ["link:b", "link:a"]}]
    s = R.score_scenario(entry, pages, elements, T=R.START + 2 * R.DAY)
    assert (s["pages"], s["false_pages"], s["detected"], s["delay_sum"], s["top1_sum"], s["top3_sum"]) == (2, 1, 1, 10, 0.0, 1.0)
    assert s["days"] == pytest.approx(2.0)


def test_test_split_refuses_without_preregistration_and_freeze(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("p12", "scripts/phase12_replay.py")
    p12 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p12)
    monkeypatch.setattr(p12, "FROZEN", tmp_path / "missing.json")
    args = type("A", (), {"split": "test", "limit": 0, "dry": False, "resume": False, "name": "x", "threads": 1})()
    with pytest.raises(SystemExit) as e:
        p12.run(args, json.loads(json.dumps(yaml.safe_load(open("config.yaml")))))
    assert "refusing the test run" in str(e.value)
