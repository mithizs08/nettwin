"""Phase 3: build the train / val / test / dev scenario datasets and the dataset card.

  python scripts/phase3_build_dataset.py --estimate      sample a few train graphs, extrapolate disk and time
  python scripts/phase3_build_dataset.py --workers 6     full build

Writes data/datasets/v1/{observable,truth}/... (gitignored), results/phase3_dataset_summary.json and
docs/dataset_card.md (every number in the card comes from the summary).
"""
import argparse
import json
import os
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from nettwin.dataset import VARIANTS, build, check_frozen, freeze_hashes, lean_subset, load_key, plan
from nettwin.kg import FAULT_TYPES
from nettwin.topology import load

ROOT = Path(__file__).resolve().parents[1]
CHANNELS = ("obs_status", "obs_link_util", "obs_link_counters", "obs_e2e", "obs_telemetry_absence")
CLASSES = ("subtle", "moderate", "obvious")
HARD = ("link_degradation", "demand_surge", "routing_misconfig", "link_flapping")


def dir_bytes(path):
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())


def size_features(cfg, name):
    g = load(name, cfg=cfg)
    return g.number_of_edges(), cfg["kg"]["services_per_pop"] * g.number_of_nodes()


def estimate(cfg, scratch, key):
    """Build one scenario on six train graphs of different sizes; fit bytes and seconds per scenario on links and
    services; extrapolate to the whole plan. Train graphs only, so nothing is generated on test before the build."""
    tasks = [t for t in plan(cfg, key) if t["split"] == "train"]
    by_n = sorted(tasks, key=lambda t: int(t["topology"].split("/")[1]))
    sample = [dict(by_n[i], n=1, seeds=by_n[i]["seeds"][:1], sids=by_n[i]["sids"][:1]) for i in np.linspace(0, len(by_n) - 1, 6).astype(int)]
    if scratch.exists():
        shutil.rmtree(scratch)
    rows = []
    for i, t in enumerate(sample):
        t0 = time.time()
        build(cfg, scratch / f"s{i}", key, tasks=[dict(t)], workers=1)
        secs = time.time() - t0
        L, S = size_features(cfg, t["topology"])
        rows.append((L, S, dir_bytes(scratch / f"s{i}"), secs))
    X = np.array([[r[0], r[1]] for r in rows], dtype=float)
    b = np.linalg.lstsq(X, np.array([r[2] for r in rows], dtype=float), rcond=None)[0]
    tfit = np.polyfit([r[0] for r in rows], [r[3] for r in rows], 1)
    total_bytes, total_secs, per_split = 0.0, 0.0, Counter()
    for t in plan(cfg):
        L, S = size_features(cfg, t["topology"])
        by = max(0.0, b[0] * L + b[1] * S) * t["n"]
        total_bytes += by
        per_split[t["split"]] += by
        total_secs += max(0.5, np.polyval(tfit, L)) * t["n"]
    shutil.rmtree(scratch)
    free = shutil.disk_usage(ROOT).free
    return {
        "sample": [{"links": r[0], "services": r[1], "bytes_both_variants": r[2], "seconds": round(r[3], 2)} for r in rows],
        "estimated_total_gb": round(total_bytes / 1e9, 2),
        "estimated_gb_by_split": {k: round(v / 1e9, 2) for k, v in per_split.items()},
        "estimated_cpu_hours_single_process": round(total_secs / 3600, 2),
        "free_disk_gb": round(free / 1e9, 1),
    }


def summarise(cfg, manifests, root, build_seconds, freeze_note):
    out = {"freeze_hash": freeze_hashes(ROOT)[0], "unfrozen_change": freeze_note, "build_minutes": round(build_seconds / 60, 1),
           "root": cfg["dataset"]["root"], "splits": {}}
    for split in ("train", "val", "test", "dev"):
        sp = {"disk_gb": round(dir_bytes(root / "observable" / split) / 1e9 + dir_bytes(root / "truth" / split) / 1e9, 3), "variants": {}}
        for variant in VARIANTS:
            entries = manifests[split][variant]
            faults = [f for e in entries for f in e["faults"]]
            topo = defaultdict(lambda: {"scenarios": 0, "faults": 0})
            for e in entries:
                key = e["topology"] if not e["topology"].startswith("gabaug/") else f"generated ({split})"
                topo[key]["scenarios"] += 1
                topo[key]["faults"] += len(e["faults"])
            src_of = {id(f): ("generated" if e["topology"].startswith("gabaug/") else "real") for e in entries for f in e["faults"]}
            by_type = {}
            for ft in FAULT_TYPES:
                fs = [f for f in faults if f["type"] == ft]
                by_type[ft] = {
                    "n": len(fs),
                    # P2 / P5: classes are counted over observable faults only (the pre-registered T2 denominator);
                    # unobservable faults carry the class "unobservable" and are counted on their own
                    "impact_class": {c: sum(f["impact"]["impact_class"] == c for f in fs) for c in CLASSES},
                    "unobservable": sum(f["impact"]["impact_class"] == "unobservable" for f in fs),
                    "by_source": {
                        src: {"observable": sum(f["impact"]["observable"] for f in fs if src_of[id(f)] == src),
                              **{c: sum(f["impact"]["impact_class"] == c for f in fs if src_of[id(f)] == src) for c in CLASSES}}
                        for src in ("generated", "real")
                    },
                    "observable": sum(f["impact"]["observable"] for f in fs),
                    "channels": {c: sum(f["impact"]["observability"][c] for f in fs) for c in CHANNELS},
                    "overlaps_maintenance": sum(f["overlaps_maintenance"] for f in fs),
                    "placement": dict(Counter(f["placement"] for f in fs)),
                }
            sp["variants"][variant] = {
                "scenarios": len(entries),
                "faults": len(faults),
                "scenarios_with_concurrent_faults": sum(any(f["placement"] == "concurrent" for f in e["faults"]) for e in entries),
                "scenarios_with_maintenance": sum(bool(e["maintenance"]) for e in entries),
                "maintenance_windows": sum(len(e["maintenance"]) for e in entries),
                "faults_overlapping_maintenance": sum(f["overlaps_maintenance"] for f in faults),
                "unobservable_faults": sum(not f["impact"]["observable"] for f in faults),
                "fault_start_hour_bins_3h": [sum(1 for f in faults if (f["t_start"] % 288) // 36 == b) for b in range(8)],
                "by_topology": dict(sorted(topo.items())),
                "by_fault_type": by_type,
            }
        out["splits"][split] = sp
    gen_sizes = defaultdict(list)
    for split in ("train", "val"):
        for e in manifests[split]["with_status"]:
            if e["topology"].startswith("gabaug/"):
                gen_sizes[split].append(int(e["topology"].split("/")[1]))
    out["generated_graphs"] = {s: {"graphs": len(set(e["topology"] for e in manifests[s]["with_status"] if e["topology"].startswith("gabaug/"))),
                                   "nodes_min": min(v), "nodes_max": max(v)} for s, v in gen_sizes.items()}
    return out


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "n/a"


def write_card(summary, cfg, path):
    t = cfg["topologies"]
    L = []
    w = L.append
    w("# Dataset card: NetTwin scenarios v1")
    w("")
    w("Generated by `scripts/phase3_build_dataset.py`; every number below is copied from `results/phase3_dataset_summary.json`. "
      "All telemetry is synthetic. Topologies are public (Internet Topology Zoo, SNDlib) or generated. None of this is Telstra's network.")
    w("")
    w("## What a scenario is")
    w("")
    w(f"One simulated week ({cfg['simulator']['days']} days, {cfg['simulator']['step_minutes']}-minute steps, "
      f"{cfg['simulator']['days'] * 24 * 60 // cfg['simulator']['step_minutes']} steps) on one topology: link utilisation, one-hop latency and "
      "discard counters per link; end-to-end latency and probe loss per synthetic service; observed events (lagged link status, PoP "
      "reachability, planned-maintenance log). Up to three faults per scenario after a one-day warmup, plus distractors (maintenance, "
      "telemetry gaps, probe gaps, event days, hot links). The simulator is described in `PLAN.md` section 4 and `src/nettwin/simulator.py`.")
    w("")
    w(f"Simulator frozen at hash `{summary['freeze_hash'][:16]}` (sha256 over `config.yaml`, `simulator.py`, `faults.py`, `synth.py`, "
      "`topology.py`, `kg.py`; full value in `results/simulator_freeze.json`).")
    w("")
    w("## Splits")
    w("")
    w("Split by topology first, then by scenario seed (`PLAN.md` section 3). Every scenario has its own secret seed (keyed hash of its "
      "split, topology and slot), checked against every reserved (previously viewed) range; `config.dataset.seed_base` is no longer used.")
    w("")
    w("| Split | Topologies | Scenarios | Faults | Disk (both variants) |")
    w("|---|---|---|---|---|")
    gg = summary["generated_graphs"]
    desc = {
        "train": f"{gg['train']['graphs']} generated graphs ({gg['train']['nodes_min']} to {gg['train']['nodes_max']} nodes)",
        "val": f"{gg['val']['graphs']} generated graphs ({gg['val']['nodes_min']} to {gg['val']['nodes_max']} nodes) + validation_real: " + ", ".join(x.split('/')[1] for x in t["validation_real"]),
        "test": ", ".join(x.split("/")[1] for x in t["test"]),
        "dev": "AARNet (seen during development; report separately, never pool with test)",
    }
    for split, sp in summary["splits"].items():
        v = sp["variants"]["with_status"]
        w(f"| {split} | {desc[split]} | {v['scenarios']} | {v['faults']} | {sp['disk_gb']} GB |")
    w("")
    w("## Variants")
    w("")
    w("Every scenario is stored twice, from the same simulation run:")
    w("- `with_status`: status events and SNMP-style down links (util exactly 0, discards 0, one-hop latency missing). "
      "\"util exactly 0 with latency missing\" is a near-unique, realistic down-link signature (audit re-verify N2).")
    w("- `status_blind`: down links fully missing (like a telemetry gap) and no status or reachability events; the maintenance log stays. "
      "Observability flags are recomputed for this view, including a fifth channel, telemetry absence: a root link fully "
      "missing for 2+ steps inside the fault window and outside planned changes. Missing telemetry is itself visible, so such "
      "faults count as observable (Phase 4 audit follow-up; the channel is always false in with_status, where nothing is blanked).")
    w("Results for link_down and node_down must be reported on both variants.")
    w("")
    w("## Faults by split and type (with_status variant)")
    w("")
    w("Severity class is the impact class from noise-free deltas against a counterfactual with only that fault removed. "
      "As in the pre-registered target T2, subtle / moderate / obvious are percentages of the **observable** faults of that type "
      "(audit P2). Unobservable faults have the class `unobservable` (audit P5) and are shown on their own, as a count and as a "
      "percentage of all faults. Channel and maintenance columns are percentages of all faults.")
    w("")
    for split, sp in summary["splits"].items():
        v = sp["variants"]["with_status"]
        w(f"### {split}")
        w("")
        w("| Fault type | n | observable | subtle | moderate | obvious | unobservable | status | link util | link counters | end-to-end | overlaps maintenance |")
        w("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for ft, d in v["by_fault_type"].items():
            n, no = d["n"], d["observable"]
            w(f"| {ft} | {n} | {no} | {pct(d['impact_class']['subtle'], no)} | {pct(d['impact_class']['moderate'], no)} | {pct(d['impact_class']['obvious'], no)} "
              f"| {d['unobservable']} ({pct(d['unobservable'], n)}) | {pct(d['channels']['obs_status'], n)} | {pct(d['channels']['obs_link_util'], n)} "
              f"| {pct(d['channels']['obs_link_counters'], n)} | {pct(d['channels']['obs_e2e'], n)} | {pct(d['overlaps_maintenance'], n)} |")
        w("")
        w(f"Scenarios with concurrent faults: {v['scenarios_with_concurrent_faults']} of {v['scenarios']} "
          f"({pct(v['scenarios_with_concurrent_faults'], v['scenarios'])}). Faults overlapping or right after a maintenance window: "
          f"{v['faults_overlapping_maintenance']} of {v['faults']} ({pct(v['faults_overlapping_maintenance'], v['faults'])}). "
          f"Maintenance windows: {v['maintenance_windows']} in {v['scenarios_with_maintenance']} scenarios. Unobservable faults: "
          f"{v['unobservable_faults']} ({pct(v['unobservable_faults'], v['faults'])}).")
        b = sp["variants"]["status_blind"]
        w(f"Status-blind variant: unobservable faults {b['unobservable_faults']} ({pct(b['unobservable_faults'], b['faults'])}); "
          f"link_down observable {b['by_fault_type']['link_down']['observable']} of {b['by_fault_type']['link_down']['n']}, "
          f"node_down observable {b['by_fault_type']['node_down']['observable']} of {b['by_fault_type']['node_down']['n']}. "
          f"Faults with the telemetry-absence channel (a root link blanked for 2+ steps, Phase 4 audit): "
          f"{sum(d['channels']['obs_telemetry_absence'] for d in b['by_fault_type'].values())}.")
        w("")
    w("## Difficulty mix against T2, by source (with_status variant)")
    w("")
    w("T2 (pre-registered, checked on validation only): at least 20% subtle and at most 50% obvious among observable faults. "
      "Shown for every split so the mix can be compared, generated graphs and real networks separately; only the val rows are "
      "a T2 check.")
    w("")
    w("| Split | Source | Fault type | observable | subtle | obvious | T2 |")
    w("|---|---|---|---|---|---|---|")
    for split, sp in summary["splits"].items():
        for src in ("generated", "real"):
            for ft in HARD:
                d = sp["variants"]["with_status"]["by_fault_type"][ft]["by_source"][src]
                if not d["observable"]:
                    continue
                sub, obv = d["subtle"] / d["observable"], d["obvious"] / d["observable"]
                verdict = ("hit" if sub >= 0.2 and obv <= 0.5 else "miss") if split == "val" else "n/a"
                w(f"| {split} | {src} | {ft} | {d['observable']} | {pct(d['subtle'], d['observable'])} | {pct(d['obvious'], d['observable'])} | {verdict} |")
    w("")
    w("## Scenarios and faults by topology (with_status variant)")
    w("")
    w("| Split | Topology | Scenarios | Faults |")
    w("|---|---|---|---|")
    for split, sp in summary["splits"].items():
        for topo, d in sp["variants"]["with_status"]["by_topology"].items():
            w(f"| {split} | {topo} | {d['scenarios']} | {d['faults']} |")
    w("")
    w("## Layout and use")
    w("")
    w(f"Root `{summary['root']}` (gitignored; rebuild with the script):")
    w("- `observable/<split>/<variant>/<sid>/`: `links.parquet` (t, link, util, latency_ms, loss_pct), `services.parquet` "
      "(t, service, latency_ms, loss_pct), `observed_events.json`, `meta.json` (scenario id, topology, step count, step length, variant). "
      "This is the only tree detectors and the agent may read.")
    w("- `truth/<split>/<variant>/manifest.json`: per scenario the seed, faults with params, placement, impact, observability flags, "
      "maintenance, gaps and event days; `truth/<split>/<variant>/<sid>/labels.parquet`: per-link per-step labels "
      "(2 root, 1 symptomatic, 0 unaffected).")
    w("- Scenario ids are neutral (`tr`, `va`, `te`, `dv` plus a number). The observable tree holds no seed, since the seed plus "
      "the config would let anyone re-run the sampler and recover the faults. Seeds are secret: each is derived from a local key "
      "(`.nettwin_dataset_key`, gitignored) and the scenario's slot, so the repo alone cannot enumerate them (audit P1). The id to "
      "seed mapping is only in `truth/`.")
    w("")
    w("## Known limitations (see `docs/decision_log.md`)")
    w("")
    w("- Telemetry is synthetic; traffic follows a gravity model with a fixed daily shape. Services are synthetic (2 per PoP).")
    w("- Pre-registered difficulty target T2 was missed for routing_misconfig and link_flapping (too few subtle, too many obvious on "
      "validation), and T6 overlap with maintenance is above its 20% ceiling. Difficulty was not retuned to fix these.")
    w("- link_down, node_down and most flapping are easy in the with_status variant (status events and the down-link signature). "
      "Report them per type, never averaged with the hard types, and on both variants.")
    w("- Impact classes use relative latency thresholds tuned on validation; the mix differs by topology (see the per-topology "
      "breakdown in the summary JSON). Sparse, tree-like test networks (York, Peer1, Belnet2010) produce more unreachable services.")
    w("- AARNet (dev) was used to develop the simulator; its numbers are illustrative only.")
    w("- The variants differ in two ways, not one (audit P3): status_blind drops status and reachability events, and it also "
      "reports down links as fully missing instead of util 0 / discards 0 / latency missing. Services and maintenance are identical.")
    bins = [sum(sp["variants"]["with_status"]["fault_start_hour_bins_3h"][b] for sp in summary["splits"].values()) for b in range(8)]
    w("- Fault start times lean toward night (audit P4). Fault starts per 3-hour bin of sim time, 00:00 onward, all splits: "
      + ", ".join(str(x) for x in bins) + ". The excess comes from maintenance-linked placement (most change windows start "
      "00:00 to 04:00). So time of day carries a weak fault prior. Models must not be given absolute time of day in a way that "
      "lets them learn this maintenance-timing pattern; diurnal-normalised traffic features are fine. Faults never start on day 0 "
      "(warmup), so time since the scenario start must not be a feature either.")
    w("- File sizes carry a weak signal (audit P7: within-topology size of `links.parquet` separates scenarios with a gray failure at "
      "AUC 0.62, from `docs/audit_phase3.md`). Parquet output is a deterministic function of the content, so sizes leak nothing beyond "
      "the telemetry, but file metadata must never be a model input.")
    w("- Labels per variant (audit P6): `labels.parquet` has `label` (2 root, 1 symptomatic, 0 unaffected; identical in both variants) "
      "and `telemetry_present` (some link metric observed in that cell; differs by variant, since status_blind blanks down links). "
      "RQ1 scoring must state whether cells without telemetry are scored and treat them the same way for every method.")
    Path(path).write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--unfrozen-reason", default=None, help="required if the simulator changed since the freeze")
    ap.add_argument("--lean", metavar="ROOT", default=None, help="rebuild only the lean fresh-clone subset into ROOT (no summary or card)")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    note = check_frozen(ROOT, args.unfrozen_reason)
    key = load_key(ROOT)
    root = ROOT / cfg["dataset"]["root"]
    if args.estimate:
        est = estimate(cfg, ROOT / "data" / "datasets" / "_estimate", key)
        (ROOT / "results" / "phase3_estimate.json").write_text(json.dumps(est, indent=2) + "\n")
        print(json.dumps(est, indent=2))
        return
    if args.lean:
        tasks = lean_subset(cfg, key)
        build(cfg, Path(args.lean), key, tasks=tasks, workers=args.workers)
        print(f"lean subset: {len(tasks)} topologies, {sum(x['n'] for x in tasks)} scenarios")
        return
    if root.exists():
        shutil.rmtree(root)
    t0 = time.time()
    manifests = build(cfg, root, key, workers=args.workers)
    summary = summarise(cfg, manifests, root, time.time() - t0, note)
    summary.pop("build_minutes")  # wall-clock time is not reproducible; keep outputs byte-identical across runs
    (ROOT / "results" / "phase3_dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_card(summary, cfg, ROOT / "docs" / "dataset_card.md")
    print(json.dumps({s: {"scenarios": v["variants"]["with_status"]["scenarios"], "faults": v["variants"]["with_status"]["faults"], "disk_gb": v["disk_gb"]}
                      for s, v in summary["splits"].items()}, indent=1))
    print(f"build took {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
