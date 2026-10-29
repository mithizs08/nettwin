"""Phase 3 dataset builder (PLAN.md section 3 splits).

Layout under `dataset.root` (default data/datasets/v1):

  observable/<split>/<variant>/<sid>/{links.parquet, services.parquet, observed_events.json, meta.json}
  truth/<split>/<variant>/<sid>/labels.parquet
  truth/<split>/<variant>/manifest.json         ground truth per scenario: faults, impact, observability, topology, seed

Detectors and the agent are only ever given `observable/`. Seeds and ids are derived from a secret local key
(`.nettwin_dataset_key`, gitignored), so the repo alone cannot re-run the sampler to recover faults (audit P1). Variants: `with_status` (status events, SNMP-style down links)
and `status_blind` (down links fully missing, no status or reachability events). Scenario ids are neutral
(`tr00000`, `va00000`, ...: a keyed permutation within the split) and say nothing about topology, order, seed or
faults.

The builder refuses to run when the simulator code or config differs from the frozen hashes in
`results/simulator_freeze.json`, unless a reason is passed (and then recorded in the summary).
"""
import hashlib
import hmac
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from nettwin.faults import check_fresh_seed, sample_scenario
from nettwin.simulator import Simulator, write_scenario
from nettwin.synth import generated_split
from nettwin.topology import load

ROOT = Path(__file__).resolve().parents[2]
FROZEN_FILES = (
    "config.yaml",
    "src/nettwin/simulator.py",
    "src/nettwin/faults.py",
    "src/nettwin/synth.py",
    "src/nettwin/topology.py",
    "src/nettwin/kg.py",
)
SPLIT_CODES = {"train": "tr", "val": "va", "test": "te", "dev": "dv"}
VARIANTS = ("with_status", "status_blind")


def freeze_hashes(root=ROOT):
    parts = {f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in FROZEN_FILES}
    combined = hashlib.sha256("".join(parts[f] for f in FROZEN_FILES).encode()).hexdigest()
    return combined, parts


class SimulatorNotFrozen(RuntimeError):
    pass


def check_frozen(root=ROOT, reason=None):
    """Raise unless the simulator matches its frozen hashes; with a reason, return it so the caller records it."""
    frozen = json.loads((root / "results" / "simulator_freeze.json").read_text())
    combined, parts = freeze_hashes(root)
    if combined == frozen["combined"]:
        return None
    changed = sorted(f for f in FROZEN_FILES if parts[f] != frozen["files"].get(f))
    if not reason:
        raise SimulatorNotFrozen(f"simulator changed since the freeze ({changed}); log a reason in docs/decision_log.md and pass it")
    return {"changed_files": changed, "reason": reason}


KEY_FILE = ".nettwin_dataset_key"   # gitignored; holds the secret that seeds and ids are derived from (audit P1)


def load_key(root=ROOT):
    """The local dataset key. Without it nobody, including the repo itself, can rebuild or re-identify scenarios."""
    path = Path(root) / KEY_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: the dataset key is local only (see README, 'Reproducing the datasets')")
    key = path.read_text().strip()
    if len(bytes.fromhex(key)) < 32:
        raise ValueError("dataset key must be at least 32 bytes of hex")
    return key


def _keyed(key, msg):
    return hmac.new(bytes.fromhex(key), msg.encode(), hashlib.sha256).digest()


def scenario_seed(key, split, topology, k):
    """Secret seed for one scenario slot: HMAC-SHA256(key, slot), 60 bits. Without the key the seed space cannot be
    enumerated, so the sampler cannot be re-run to recover faults (audit P1)."""
    return int.from_bytes(_keyed(key, f"seed|{split}|{topology}|{k}")[:8], "big") >> 4


def plan(cfg, key=None):
    """Every (split, topology, n_scenarios). Split by topology first, then by scenario seed. With the key, each task
    also gets its secret per-scenario seeds and its opaque scenario ids: a keyed permutation of all slots within the
    split, so an id says nothing about topology, position or seed."""
    t = cfg["topologies"]
    gen = generated_split(cfg)
    n = cfg["dataset"]["scenarios_per_topology"]
    groups = [
        ("train", gen["train"], n["train"]),
        ("val", gen["val"], n["val_generated"]),
        ("val", t["validation_real"], n["validation_real"]),
        ("test", t["test"], n["test"]),
        ("dev", t["dev"], n["dev"]),
    ]
    tasks = [{"split": split, "topology": name, "n": k} for split, names, k in groups for name in names]
    if key is None:
        return tasks
    for split in SPLIT_CODES:
        slots = [(task, k) for task in tasks if task["split"] == split for k in range(task["n"])]
        order = sorted(range(len(slots)), key=lambda i: _keyed(key, f"id|{split}|{slots[i][0]['topology']}|{slots[i][1]}"))
        for rank, i in enumerate(order):
            task, k = slots[i]
            task.setdefault("sids", [None] * task["n"])[k] = f"{SPLIT_CODES[split]}{rank:05d}"
    for task in tasks:
        task["seeds"] = [scenario_seed(key, task["split"], task["topology"], k) for k in range(task["n"])]
    return tasks


def lean_subset(cfg, key=None):
    """Every real network (validation_real, test, dev) plus a fixed sample of generated graphs. Ids and seeds are those
    of the full plan, so files compare byte for byte with a full build."""
    real = set(cfg["topologies"]["validation_real"]) | set(cfg["topologies"]["test"]) | set(cfg["topologies"]["dev"])
    tasks = plan(cfg, key)
    gen = {s: [t for t in tasks if t["split"] == s and t["topology"].startswith("gabaug/")] for s in ("train", "val")}
    return [t for t in tasks if t["topology"] in real] + gen["train"][::40] + gen["val"][::15]


def _truth_entry(entry, res, sim, root, split, variant, slot):
    """Ground-truth extras the simulator does not write (no frozen file touched):
    - P5: an unobservable fault's impact class becomes "unobservable" (its symptom-size class stays as `symptom_class`)
    - P6: labels.parquet gets `telemetry_present`, which differs between variants (status_blind blanks down links)
    - the slot (topology, scenario number) that the secret seed was derived from"""
    for f in entry["faults"]:
        im = f["impact"]
        im["symptom_class"] = im["impact_class"]
        if not im["observable"]:
            im["impact_class"] = "unobservable"
    entry["n_links"], entry["n_services"] = len(sim.links), len(sim.services)
    entry["slot"] = slot
    present = ~(np.isnan(res.obs["util"]) & np.isnan(res.obs["latency_ms"]) & np.isnan(res.obs["loss_pct"]))
    path = root / "truth" / split / variant / entry["files"]["labels"]
    labels = pd.read_parquet(path)
    labels["telemetry_present"] = present.ravel()
    labels.to_parquet(path, index=False)
    return entry


def _build_topology(args):
    task, cfg, root = args
    sim = Simulator(load(task["topology"], cfg=cfg), cfg)
    out = {v: [] for v in VARIANTS}
    for k in range(task["n"]):
        seed, sid = task["seeds"][k], task["sids"][k]
        check_fresh_seed(cfg, seed)
        spec = sample_scenario(sim, sid, seed)
        res, blind = sim.simulate(spec, with_blind=True)
        for variant, r in (("with_status", res), ("status_blind", blind)):
            entry = write_scenario(r, obs_dir=root / "observable" / task["split"] / variant, truth_dir=root / "truth" / task["split"] / variant)
            out[variant].append(_truth_entry(entry, r, sim, root, task["split"], variant, {"topology": task["topology"], "k": k}))
    return task, out


def build(cfg, root, key, tasks=None, workers=1):
    """Build the scenarios for `tasks` (default: the whole plan) with the secret key. Returns
    {split: {variant: [manifest entries]}}. Each task carries its keyed ids and seeds, so a subset build produces the
    same files as the full build."""
    tasks = plan(cfg, key) if tasks is None else tasks
    if any("seeds" not in t for t in tasks):
        raise ValueError("tasks must come from plan(cfg, key)")
    root = Path(root)
    manifests = {}
    jobs = [(t, cfg, root) for t in tasks]
    results = Pool(workers).imap(_build_topology, jobs) if workers > 1 else map(_build_topology, jobs)
    for task, out in results:
        for variant, entries in out.items():
            manifests.setdefault(task["split"], {}).setdefault(variant, []).extend(entries)
    for split, by_variant in manifests.items():
        for variant, entries in by_variant.items():
            entries.sort(key=lambda e: e["scenario_id"])
            d = root / "truth" / split / variant
            d.mkdir(parents=True, exist_ok=True)
            (d / "manifest.json").write_text(json.dumps(entries, indent=1) + "\n")
    return manifests
