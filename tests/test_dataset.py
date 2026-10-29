"""Phase 3 dataset: split hygiene, seed hygiene, ground truth kept out of observable data, and the simulator freeze."""
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml

from nettwin.dataset import FROZEN_FILES, SimulatorNotFrozen, build, check_frozen, plan
from nettwin.faults import check_fresh_seed
from nettwin.kg import FAULT_TYPES
from nettwin.simulator import LEAKY_WORDS, OBSERVABLE_META_KEYS
from nettwin.synth import generated_split

ROOT = Path(__file__).resolve().parents[1]
EVENT_KEYS = {"t", "type", "element", "status", "id", "target", "t_start", "t_end", "action", "description"}
COLUMNS = {"links.parquet": ["t", "link", "util", "latency_ms", "loss_pct"], "services.parquet": ["t", "service", "latency_ms", "loss_pct"]}
TRUTH_WORDS = ("seed", "root_cause", "params", "impact", "observab", "placement", "label", "fault", "caused_by", "linked_maintenance")


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load((ROOT / "config.yaml").read_text())


def split_topologies(cfg):
    out = {}
    for t in plan(cfg):
        out.setdefault(t["split"], set()).add(t["topology"])
    return out


def test_no_test_or_dev_topology_in_train_or_val(cfg):
    sp = split_topologies(cfg)
    held_out = set(cfg["topologies"]["test"]) | set(cfg["topologies"]["dev"])
    assert sp["test"] == set(cfg["topologies"]["test"]) and sp["dev"] == set(cfg["topologies"]["dev"])
    assert not (sp["train"] | sp["val"]) & held_out
    assert not sp["train"] & sp["val"]
    assert all(x.startswith("gabaug/") for x in sp["train"])                    # train is generated graphs only
    assert sp["train"] == set(generated_split(cfg)["train"])
    assert {x for x in sp["val"] if x.startswith("gabaug/")} == set(generated_split(cfg)["val"])
    assert not (sp["train"] | sp["val"] | sp["test"]) & set(cfg["topologies"]["skip"])


def test_no_family_of_test_or_dev_in_train_or_val(cfg):
    fam = json.loads((ROOT / cfg["topologies"]["reference_pool_file"]).read_text())["family_of"]
    sp = split_topologies(cfg)
    held = {fam[x] for x in set(cfg["topologies"]["test"]) | set(cfg["topologies"]["dev"]) if x in fam}
    real_trainval = [x for x in sp["train"] | sp["val"] if x in fam]
    assert real_trainval and not {fam[x] for x in real_trainval} & held


def test_seeds_are_fresh_and_disjoint_across_splits(cfg):
    ranges = []
    for t in plan(cfg):
        for s in (t["first_seed"], t["first_seed"] + t["n"] - 1):
            check_fresh_seed(cfg, s)
        ranges.append((t["first_seed"], t["first_seed"] + t["n"], t["split"]))
    ranges.sort()
    assert all(b <= c for (_, b, _), (c, _, _) in zip(ranges, ranges[1:]))     # no seed is used twice anywhere
    assert all(n <= 1000 for n in cfg["dataset"]["scenarios_per_topology"].values())


@pytest.fixture(scope="module")
def tiny(cfg, tmp_path_factory):
    """A real build of two small train / val topologies (never test or dev), 2 scenarios each."""
    root = tmp_path_factory.mktemp("ds")
    tasks = [dict(t, n=2) for t in plan(cfg) if t["topology"] in (generated_split(cfg)["train"][0], "topozoo/Nextgen")]
    assert len(tasks) == 2
    manifests = build(cfg, root, tasks=tasks)
    return root, manifests


def test_observable_tree_has_no_ground_truth(tiny):
    root, _ = tiny
    files = [p for p in (root / "observable").rglob("*") if p.is_file()]
    assert files
    for p in files:
        rel = p.relative_to(root / "observable").as_posix().lower()   # path below the observable root
        assert not [w for w in LEAKY_WORDS if w in rel] and not [w for w in TRUTH_WORDS if w in rel]
        if p.name in COLUMNS:
            assert list(pd.read_parquet(p).columns) == COLUMNS[p.name]
        elif p.name == "observed_events.json":
            events = json.loads(p.read_text())
            assert all(set(e) <= EVENT_KEYS for e in events)
            assert not [ft for ft in FAULT_TYPES if ft in p.read_text()]
        elif p.name == "meta.json":
            assert set(json.loads(p.read_text())) == set(OBSERVABLE_META_KEYS)
        else:
            pytest.fail(f"unexpected observable file {rel}")
        if p.suffix == ".json":
            text = p.read_text().lower()
            assert not [w for w in TRUTH_WORDS if f'"{w}' in text]


def test_ground_truth_lives_only_under_truth(tiny):
    root, manifests = tiny
    for split, by_variant in manifests.items():
        for variant, entries in by_variant.items():
            for e in entries:
                assert (root / "truth" / split / variant / e["files"]["labels"]).exists()
                assert not (root / "observable" / split / variant / e["files"]["labels"]).exists()
                assert "seed" in e and all("impact" in f for f in e["faults"])
            assert (root / "truth" / split / variant / "manifest.json").exists()
    assert not list((root / "observable").rglob("manifest.json")) and not list((root / "observable").rglob("labels.parquet"))


def test_status_blind_variant_has_no_status_events(tiny):
    root, manifests = tiny
    for split in manifests:
        for p in (root / "observable" / split / "status_blind").rglob("observed_events.json"):
            assert all(e["type"] == "maintenance" for e in json.loads(p.read_text()))
        for p in (root / "observable" / split / "status_blind").rglob("meta.json"):
            assert json.loads(p.read_text())["status_blind"] is True


def test_manifest_topologies_match_their_split(tiny, cfg):
    _, manifests = tiny
    sp = split_topologies(cfg)
    for split, by_variant in manifests.items():
        for e in by_variant["with_status"]:
            assert e["topology"] in sp[split]


def test_simulator_is_frozen():
    """Fails if the simulator, fault code, topology code or config changed after the freeze. Changing any of them
    needs a logged reason in docs/decision_log.md and an updated results/simulator_freeze.json."""
    assert check_frozen(ROOT) is None


def test_freeze_guard_refuses_changes(tmp_path):
    for f in FROZEN_FILES + ("results/simulator_freeze.json",):
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / f, tmp_path / f)
    (tmp_path / "config.yaml").write_text((tmp_path / "config.yaml").read_text() + "\n# tweak\n")
    with pytest.raises(SimulatorNotFrozen, match="config.yaml"):
        check_frozen(tmp_path)
    assert check_frozen(tmp_path, reason="testing")["changed_files"] == ["config.yaml"]


def test_built_dataset_split_hygiene_if_present(cfg):
    root = ROOT / cfg["dataset"]["root"] / "truth"
    if not root.exists():
        pytest.skip("full dataset not built here")
    held = set(cfg["topologies"]["test"]) | set(cfg["topologies"]["dev"])
    for split in ("train", "val"):
        topos = {e["topology"] for e in json.loads((root / split / "with_status" / "manifest.json").read_text())}
        assert not topos & held
