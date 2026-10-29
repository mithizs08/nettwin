"""Phase 3 dataset builder (PLAN.md section 3 splits).

Layout under `dataset.root` (default data/datasets/v1):

  observable/<split>/<variant>/<sid>/{links.parquet, services.parquet, observed_events.json, meta.json}
  truth/<split>/<variant>/<sid>/labels.parquet
  truth/<split>/<variant>/manifest.json         ground truth per scenario: faults, impact, observability, topology, seed

Detectors and the agent are only ever given `observable/`. Variants: `with_status` (status events, SNMP-style down links)
and `status_blind` (down links fully missing, no status or reachability events). Scenario ids are neutral
(`tr00001`, `va00001`, `te00001`, `dv00001`) and say nothing about topology or faults.

The builder refuses to run when the simulator code or config differs from the frozen hashes in
`results/simulator_freeze.json`, unless a reason is passed (and then recorded in the summary).
"""
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path

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


def plan(cfg):
    """Every (split, topology, n_scenarios, first_seed). Split by topology first, then by scenario seed."""
    dc, t = cfg["dataset"], cfg["topologies"]
    gen = generated_split(cfg)
    n = dc["scenarios_per_topology"]
    groups = [
        ("train", gen["train"], n["train"]),
        ("val", gen["val"], n["val_generated"]),
        ("val", t["validation_real"], n["validation_real"]),
        ("test", t["test"], n["test"]),
        ("dev", t["dev"], n["dev"]),
    ]
    tasks, per_split = [], {}
    for split, names, k in groups:
        for name in names:
            i = per_split.get(split, 0)
            per_split[split] = i + 1
            tasks.append({"split": split, "topology": name, "n": k, "first_seed": dc["seed_base"][split] + i * 1000, "index": i})
    return tasks


def lean_subset(cfg):
    """Tasks the lean fresh-clone check rebuilds: every real network (validation_real, test, dev) plus a fixed sample of
    generated graphs. Ids and seeds are those of the full plan, so files compare byte for byte."""
    real = set(cfg["topologies"]["validation_real"]) | set(cfg["topologies"]["test"]) | set(cfg["topologies"]["dev"])
    tasks = plan(cfg)
    gen = {s: [t for t in tasks if t["split"] == s and t["topology"].startswith("gabaug/")] for s in ("train", "val")}
    sample = gen["train"][::40] + gen["val"][::15]
    return [t for t in tasks if t["topology"] in real] + sample


def _build_topology(args):
    task, cfg, root = args
    sim = Simulator(load(task["topology"], cfg=cfg), cfg)
    out = {v: [] for v in VARIANTS}
    for k in range(task["n"]):
        seed = task["first_seed"] + k
        check_fresh_seed(cfg, seed)
        sid = f"{SPLIT_CODES[task['split']]}{task['index'] * 1000 + k:07d}"
        spec = sample_scenario(sim, sid, seed)
        res, blind = sim.simulate(spec, with_blind=True)
        for variant, r in (("with_status", res), ("status_blind", blind)):
            entry = write_scenario(
                r,
                obs_dir=root / "observable" / task["split"] / variant,
                truth_dir=root / "truth" / task["split"] / variant,
            )
            entry["n_links"], entry["n_services"] = len(sim.links), len(sim.services)
            out[variant].append(entry)
    return task, out


def build(cfg, root, tasks=None, workers=1):
    """Build the scenarios for `tasks` (default: the whole plan). Returns {split: {variant: [manifest entries]}}."""
    tasks = plan(cfg) if tasks is None else tasks   # each task carries its plan index, so ids never depend on the subset
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
