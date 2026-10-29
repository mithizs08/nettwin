"""Phase 3 dataset: split hygiene, secret seeds, ground truth kept out of observable data, and the simulator freeze."""
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml

from nettwin.dataset import FROZEN_FILES, SimulatorNotFrozen, build, check_frozen, load_key, plan, scenario_seed
from nettwin.faults import check_fresh_seed, sample_scenario
from nettwin.kg import FAULT_TYPES
from nettwin.simulator import LEAKY_WORDS, OBSERVABLE_META_KEYS, Simulator
from nettwin.synth import generated_split
from nettwin.topology import load

ROOT = Path(__file__).resolve().parents[1]
EVENT_KEYS = {"t", "type", "element", "status", "id", "target", "t_start", "t_end", "action", "description"}
COLUMNS = {"links.parquet": ["t", "link", "util", "latency_ms", "loss_pct"], "services.parquet": ["t", "service", "latency_ms", "loss_pct"]}
TRUTH_WORDS = ("seed", "root_cause", "params", "impact", "observab", "placement", "label", "fault", "caused_by", "linked_maintenance", "slot")
TEST_KEY = "a5" * 32  # a fixed key for the unit tests only; the real key is local and never in the repo


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load((ROOT / "config.yaml").read_text())


def split_topologies(cfg):
    out = {}
    for t in plan(cfg):
        out.setdefault(t["split"], set()).add(t["topology"])
    return out


def public_seeds(cfg, split, topology):
    """Every seed a repo-only attacker can enumerate for one topology: the old public seed block for its plan position
    (seed_base + index * 1000 + k, as the first v1 build used) and every reserved range."""
    index = [t["topology"] for t in plan(cfg) if t["split"] == split].index(topology)
    base = cfg["dataset"]["seed_base"][split] + index * 1000
    out = set(range(base, base + 1000))
    for lo, hi in cfg["seeds"]["reserved_ranges"]:
        out |= set(range(lo, hi + 1))
    return out


def test_no_test_or_dev_topology_in_train_or_val(cfg):
    sp = split_topologies(cfg)
    held_out = set(cfg["topologies"]["test"]) | set(cfg["topologies"]["dev"])
    assert sp["test"] == set(cfg["topologies"]["test"]) and sp["dev"] == set(cfg["topologies"]["dev"])
    assert not (sp["train"] | sp["val"]) & held_out
    assert not sp["train"] & sp["val"]
    assert all(x.startswith("gabaug/") for x in sp["train"])
    assert sp["train"] == set(generated_split(cfg)["train"])
    assert {x for x in sp["val"] if x.startswith("gabaug/")} == set(generated_split(cfg)["val"])
    assert not (sp["train"] | sp["val"] | sp["test"]) & set(cfg["topologies"]["skip"])


def test_no_family_of_test_or_dev_in_train_or_val(cfg):
    fam = json.loads((ROOT / cfg["topologies"]["reference_pool_file"]).read_text())["family_of"]
    sp = split_topologies(cfg)
    held = {fam[x] for x in set(cfg["topologies"]["test"]) | set(cfg["topologies"]["dev"]) if x in fam}
    real_trainval = [x for x in sp["train"] | sp["val"] if x in fam]
    assert real_trainval and not {fam[x] for x in real_trainval} & held


def test_keyed_seeds_and_ids(cfg):
    a, b = plan(cfg, TEST_KEY), plan(cfg, "5a" * 32)
    seeds = [s for t in a for s in t["seeds"]]
    sids = [s for t in a for s in t["sids"]]
    assert len(set(seeds)) == len(seeds) and len(set(sids)) == len(sids)       # no seed or id used twice anywhere
    for s in seeds[::50]:
        check_fresh_seed(cfg, s)
    assert plan(cfg, TEST_KEY) == a                                            # deterministic given the key
    assert [s for t in b for s in t["seeds"]] != seeds and [s for t in b for s in t["sids"]] != sids
    by_split = {}
    for t in a:
        by_split.setdefault(t["split"], []).extend(int(x[2:]) for x in t["sids"])
    assert all(sorted(v) == list(range(len(v))) for v in by_split.values())    # a permutation of 0..n-1 per split
    first = [t for t in a if t["split"] == "train"][0]
    assert first["sids"] != sorted(first["sids"]) or int(first["sids"][0][2:]) != 0   # ids do not follow plan order


def test_key_is_required():
    with pytest.raises(FileNotFoundError, match="dataset key"):
        load_key(ROOT / "tests")
    assert ".nettwin_dataset_key" in (ROOT / ".gitignore").read_text().split()


@pytest.fixture(scope="module")
def tiny(cfg, tmp_path_factory):
    """A real keyed build of two small train / val topologies (never test or dev), 2 scenarios each."""
    root = tmp_path_factory.mktemp("ds")
    keep = (generated_split(cfg)["train"][0], "topozoo/Nextgen")
    tasks = [dict(t, n=2, seeds=t["seeds"][:2], sids=t["sids"][:2]) for t in plan(cfg, TEST_KEY) if t["topology"] in keep]
    assert len(tasks) == 2
    return root, build(cfg, root, TEST_KEY, tasks=tasks)


def test_no_key_cannot_recover_faults(tiny, cfg):
    """Audit P1: with the repo alone (no key), the sampler cannot reproduce faults from observable files. The attacker
    reads the topology from meta.json, enumerates every seed the repo makes public for it (the old seed block, the id
    read as a number, the reserved ranges) and matches the observed maintenance log, which fingerprints a seed. No
    candidate matches. With the key, the slot in ground truth gives the seed back."""
    root, manifests = tiny
    sims = {}
    for split, by_variant in manifests.items():
        truth = {e["scenario_id"]: e for e in by_variant["with_status"]}
        for d in sorted((root / "observable" / split / "with_status").iterdir()):
            meta = json.loads((d / "meta.json").read_text())
            seen = sorted((e["target"], e["t_start"], e["t_end"]) for e in json.loads((d / "observed_events.json").read_text()) if e["type"] == "maintenance")
            topo = meta["topology"]
            cands = public_seeds(cfg, split, topo) | {int(meta["scenario_id"][2:])}
            true = truth[meta["scenario_id"]]
            assert true["seed"] not in cands
            sim = sims.setdefault(topo, Simulator(load(topo, cfg=cfg), cfg))
            for seed in sorted(cands)[:300]:   # the old block's first 300 seeds cover every slot the v1 build used
                s = sample_scenario(sim, "x", seed)
                assert not (s["maintenance"] and sorted((m["target"], m["t_start"], m["t_end"]) for m in s["maintenance"]) == seen)
            assert scenario_seed(TEST_KEY, split, true["slot"]["topology"], true["slot"]["k"]) == true["seed"]


def test_observable_tree_has_no_ground_truth(tiny):
    root, _ = tiny
    files = [p for p in (root / "observable").rglob("*") if p.is_file()]
    assert files
    for p in files:
        rel = p.relative_to(root / "observable").as_posix().lower()
        assert not [w for w in LEAKY_WORDS if w in rel] and not [w for w in TRUTH_WORDS if w in rel]
        if p.name in COLUMNS:
            assert list(pd.read_parquet(p).columns) == COLUMNS[p.name]
        elif p.name == "observed_events.json":
            assert all(set(e) <= EVENT_KEYS for e in json.loads(p.read_text()))
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
                assert "seed" in e and "slot" in e and all("impact" in f for f in e["faults"])
            assert (root / "truth" / split / variant / "manifest.json").exists()
    assert not list((root / "observable").rglob("manifest.json")) and not list((root / "observable").rglob("labels.parquet"))


def test_unobservable_class_and_per_variant_labels(tiny):
    """P5: unobservable faults get the class 'unobservable'. P6: labels carry telemetry_present per variant."""
    root, manifests = tiny
    for split, by_variant in manifests.items():
        for variant, entries in by_variant.items():
            for e in entries:
                for f in e["faults"]:
                    im = f["impact"]
                    assert (im["impact_class"] == "unobservable") == (not im["observable"])
                    assert im["symptom_class"] in ("subtle", "moderate", "obvious")
                lab = pd.read_parquet(root / "truth" / split / variant / e["files"]["labels"])
                assert list(lab.columns) == ["t", "link", "label", "telemetry_present"]
                links = pd.read_parquet(root / "observable" / split / variant / e["files"]["links"])
                present = links[["util", "latency_ms", "loss_pct"]].notna().any(axis=1).to_numpy()
                assert (lab["telemetry_present"].to_numpy() == present).all()


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
        entries = json.loads((root / split / "with_status" / "manifest.json").read_text())
        assert not {e["topology"] for e in entries} & held
        assert all("slot" in e for e in entries)   # built with keyed seeds
