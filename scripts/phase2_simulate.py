"""Phase 2: simulator sanity plots and shortcut / difficulty diagnostics.

Writes:
  results/figures/phase2_link_down_aarnet.png     one AARNet link-down incident end to end
  results/figures/phase2_fault_<type>.png         one figure per fault type, a mild and a severe instance
  results/figures/phase2_distractors_aarnet.png   maintenance drain, telemetry gap, event day
  results/phase2_summary.json                     demo details, and the pre-registered difficulty targets T1 to T8
  data/scenarios/phase2_demo/{observable,truth}/  telemetry and events apart from ground truth (gitignored)
  results/tuning_log.jsonl                        appended only with --log-tuning (audit M8)

AARNet is the dev / demo network (seen during development) and is used only for plots. Difficulty and shortcut
diagnostics run on validation networks only: held-out generated graphs plus validation_real. Never on test.
"""
import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from nettwin.faults import make_fault, sample_scenario  # noqa: E402
from nettwin.kg import FAULT_TYPES, pop_node  # noqa: E402
from nettwin.simulator import Simulator, observed_kg, write_manifest, write_scenario  # noqa: E402
from nettwin.synth import generated_split  # noqa: E402
from nettwin.topology import load  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SAVE = {"dpi": 130, "bbox_inches": "tight", "metadata": {"Software": None}}
H = 5 / 60  # hours per step


def base_spec(sim, sid, seed, faults=(), **extra):
    s = {"scenario_id": sid, "topology": sim.name, "seed": seed, "n_steps": sim.n_steps, "step_minutes": sim.step_minutes,
         "faults": list(faults), "maintenance": [], "gaps": [], "event_days": []}
    s.update(extra)
    return s


def peak_start(sim, day, hour):
    return day * sim.steps_per_day + int(hour * 60 / sim.step_minutes)


def incident_links(sim, pop):
    name = pop.split(":", 1)[1]
    return [i for i, l in enumerate(sim.links) if name in l.split(":", 1)[1].split("--")]


def shade(ax, f):
    ax.axvspan(f["t_start"] * H, f["t_end"] * H, color="#d7191c", alpha=0.08, lw=0)


def label(el):
    return el.split(":", 1)[1]


def map_positions(sim, offset_deg=0.6):
    """PoP lon/lat with co-located PoPs nudged apart (NW / SE), as in the Phase 1 map."""
    by_pos = defaultdict(list)
    for p in sim.pops:
        by_pos[tuple(sim.topo.nodes[p]["pos"])].append(p)
    out = {}
    for (x, y), names in by_pos.items():
        for i, n in enumerate(sorted(names)):
            ang = 3 * np.pi / 4 + 2 * np.pi * i / len(names)
            out[n] = (x, y) if len(names) == 1 else (x + offset_deg * np.cos(ang), y + offset_deg * np.sin(ang))
    return out


# ---------------------------------------------------------------------- AARNet link-down demo


def choose_demo_link(sim, t0, t1):
    """Non-bridge link whose failure at peak pushes another link closest to, but not over, capacity
    (shows congestion onset). Falls back to the highest if every failure overloads something."""
    demand = sim.base_demand * sim._pair_profile(np.arange(t0, t1)).mean(0)
    base = sim.routing(((), (), ()))
    best = None
    for li, link in enumerate(sim.links):
        r = sim.routing(((link,), (), ()))
        if not r.reachable.all() or any(not p for p in r.pair_paths.values()):
            continue
        util = np.maximum(demand @ r.rf.T, demand @ r.rb.T) / sim.capacity
        util[li] = 0
        score = util.max()
        rank = (score < 1.0, score if score < 1.0 else -score)
        if best is None or rank > best[0]:
            best = (rank, link)
    assert base is not None
    return best[1]


def plot_link_down_demo(sim, cfg, out, scen_dir, entries):
    t0 = peak_start(sim, 2, 17.5)
    t1 = t0 + 48
    link = choose_demo_link(sim, t0, t1)
    f = make_fault(sim, "link_down", link, t0, t1 - t0, np.random.default_rng(0), fault_id="F0")
    spec = base_spec(sim, "aarnet_link_down_demo", 2026, [f])
    res = sim.simulate(spec)
    cf = sim._run({**spec, "faults": []})
    entries.append(write_scenario(res, scen_dir))
    li = sim.link_idx[link]
    win = slice(t0, t1)
    gain = (res.truth["util"][win] - cf.truth["util"][win]).max(0)
    gain[li] = -1
    top = list(np.argsort(-gain)[:4])
    dlat = np.nan_to_num(res.truth["svc_latency_ms"][win] - cf.truth["svc_latency_ms"][win]).max(0)
    svc_top = list(np.argsort(-dlat)[:3])
    twin = observed_kg(sim, res)
    rerouted = [e for e in twin.changes_between(t0, t0) if e["type"] == "reroute"]
    zoom = (max(0, t0 - 288), min(sim.n_steps, t1 + 144))
    hrs = np.arange(*zoom) * H
    zs = slice(*zoom)

    fig = plt.figure(figsize=(17, 10.5), layout="constrained")
    gs = fig.add_gridspec(3, 3, width_ratios=[1.1, 1, 1])
    ax_map = fig.add_subplot(gs[:, 0])
    pos = map_positions(sim)
    cmap = plt.get_cmap("Reds")
    for i, l in enumerate(sim.links):
        u, v = sim.kg.G.nodes[l]["u"], sim.kg.G.nodes[l]["v"]
        (x0, y0), (x1, y1) = pos[u], pos[v]
        if i == li:
            ax_map.plot([x0, x1], [y0, y1], color="k", ls="--", lw=2.5, label=f"failed: {label(l)}", zorder=4)
        else:
            g = max(gain[i], 0)
            ax_map.plot([x0, x1], [y0, y1], color=cmap(0.25 + 0.75 * min(1, g / max(gain.max(), 1e-9))), lw=1 + 6 * g)
    for p, (x, y) in pos.items():
        ax_map.scatter([x], [y], s=20, color="#1f78b4", zorder=3)
    for i in top:
        u, v = sim.kg.G.nodes[sim.links[i]]["u"], sim.kg.G.nodes[sim.links[i]]["v"]
        mx, my = (pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2
        ax_map.annotate(f"+{gain[i]:.2f}", (mx, my), fontsize=8, color="#a50f15")
    for p, (x, y) in pos.items():
        ax_map.annotate(p, (x, y), xytext=(3, 3), textcoords="offset points", fontsize=6, color="#333")
    ax_map.set_aspect(1.2)
    ax_map.legend(loc="lower left", fontsize=8)
    ax_map.set_title("Peak utilisation increase during the outage\n(noise-free, vs the same week without the fault;\nco-located PoPs offset 0.6 deg)", fontsize=9)
    ax_map.set_xlabel("lon")
    ax_map.set_ylabel("lat")

    colours = plt.get_cmap("tab10")
    axes = [fig.add_subplot(gs[r, c]) for r in range(3) for c in (1, 2)]
    ax = axes[0]
    ax.plot(hrs, res.obs["util"][zs, li], color="k", lw=1, label=f"{label(link)} (failed)")
    for k, i in enumerate(top):
        ax.plot(hrs, res.obs["util"][zs, i], color=colours(k), lw=0.9, label=label(sim.links[i]))
        ax.plot(hrs, cf.truth["util"][zs, i], color=colours(k), lw=0.8, ls=":")
    ax.set_title("Link utilisation (observed); dotted = same links without the fault", fontsize=9)
    ax.legend(fontsize=6.5, loc="upper left")
    ax = axes[1]
    for k, i in enumerate(top):
        ax.plot(hrs, res.obs["latency_ms"][zs, i], color=colours(k), lw=0.9)
    ax.set_title("Link latency, one-hop probe (ms); queueing on the busiest", fontsize=9)
    ax = axes[2]
    for k, i in enumerate(top):
        ax.plot(hrs, res.obs["loss_pct"][zs, i], color=colours(k), lw=0.9)
    ax.set_title("Link discard counters (%): congestion loss above 90% util", fontsize=9)
    ax = axes[3]
    n_paths = [len(twin.paths_through(link, t=int(tt))) for tt in np.arange(*zoom)]
    ax.plot(hrs, n_paths, color="k")
    ax.set_title(f"Twin: paths through {label(link)} ({len(rerouted)} PoP pairs rerouted at onset)", fontsize=9)
    ax = axes[4]
    for k, si in enumerate(svc_top):
        s = sim.services[si]
        d = sim.kg.G.nodes[s]
        ax.plot(hrs, res.obs["svc_latency_ms"][zs, si], color=colours(k + 5), lw=0.9, label=f"{label(s)} {d['src']} to {d['dst']}")
        ax.plot(hrs, cf.truth["svc_latency_ms"][zs, si], color=colours(k + 5), lw=0.8, ls=":")
    ax.set_title("Service latency (ms): rerouted services take longer paths", fontsize=9)
    ax.legend(fontsize=6.5, loc="upper left")
    ax = axes[5]
    for k, si in enumerate(svc_top):
        ax.plot(hrs, res.obs["svc_loss_pct"][zs, si], color=colours(k + 5), lw=0.9)
    ax.set_title("Service loss (%, 600 probes per 5 min)", fontsize=9)
    for ax in axes:
        shade(ax, f)
        ax.set_xlabel("hours since scenario start", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"AARNet link-down end to end: {label(link)} down for {(t1 - t0) * H:.0f} h at evening peak (synthetic telemetry)", fontsize=12)
    fig.savefig(out, **SAVE)
    plt.close(fig)
    return {
        "link": link,
        "t_start": t0,
        "t_end": t1,
        "impact": res.impacts["F0"],
        "twin_pairs_rerouted_at_onset": len(rerouted),
        "top_gaining_links": [{"link": sim.links[i], "max_util_gain": round(float(gain[i]), 4),
                               "max_observed_util": round(float(np.nanmax(res.obs["util"][win, i])), 4)} for i in top],
        "services_with_largest_latency_increase": [{"service": sim.services[i], "max_latency_delta_ms": round(float(dlat[i]), 3)} for i in svc_top],
    }


# ---------------------------------------------------------------------- one figure per fault type


def fault_instances(sim):
    """A mild and a severe hand-picked instance per type (AARNet), for illustration only."""
    rng = np.random.default_rng(42)
    peak = peak_start(sim, 2, 17.5)
    night = peak_start(sim, 3, 1.0)
    base = sim.routing(((), (), ()))
    load_by_link = (sim.base_demand @ base.rf.T) + (sim.base_demand @ base.rb.T)
    nonbridge = [l for l in sim.links if sim.routing(((l,), (), ())).reachable.all()]
    quiet = min(nonbridge, key=lambda l: (load_by_link[sim.link_idx[l]], l))
    busy = max(nonbridge, key=lambda l: (load_by_link[sim.link_idx[l]], l))
    small_pop = min(sim.pops, key=lambda p: (sim.base_demand[(sim.pair_src == sim.pop_idx[p])].sum(), p))
    big_pop = max(sim.pops, key=lambda p: (sim.base_demand[(sim.pair_src == sim.pop_idx[p])].sum(), p))
    hub = max(sim.pops, key=lambda p: (sim.topo.degree(p), p))
    leaf = min(sim.pops, key=lambda p: (sim.topo.degree(p), p))
    mk = lambda ft, root, t0, dur, **kw: make_fault(sim, ft, root, t0, dur, rng, fault_id="F0", **kw)
    inst = {
        "link_down": [mk("link_down", quiet, night, 24), mk("link_down", busy, peak, 48)],
        "node_down": [mk("node_down", pop_node(leaf), night, 6), mk("node_down", pop_node(hub), peak, 12)],
        "link_degradation": [
            mk("link_degradation", busy, peak - 96, 240, loss_pct=0.03, silent=True, ramp_steps=24),
            mk("link_degradation", busy, peak - 96, 240, loss_pct=1.0, silent=False, ramp_steps=36),
        ],
        "link_flapping": [
            mk("link_flapping", busy, peak, 36, active_prob=0.3, max_down_frac=0.05, flaps_per_step=1.0, convergence_s=0.1),
            mk("link_flapping", busy, peak, 36, active_prob=0.9, max_down_frac=0.15, flaps_per_step=4, convergence_s=0.5)  # audit m6: inside config ranges,
        ],
        "demand_surge": [
            mk("demand_surge", pop_node(small_pop), peak, 36, factor=1.4, ramp_steps=6),
            mk("demand_surge", pop_node(big_pop), peak, 36, factor=3.0, ramp_steps=2),
        ],
    }
    cands = []
    for i, l in enumerate(sim.links):
        try:
            cands.append(mk("routing_misconfig", l, peak - 48, 144))
        except ValueError:
            pass
    scored = []
    for c in cands:
        r = sim.simulate(base_spec(sim, "probe", 99, [c]))
        im = r.impacts["F0"]
        scored.append((im["max_other_util_delta"] + im["max_root_util_delta"], c["root_cause"], c))
    scored.sort(key=lambda x: (x[0], x[1]))
    inst["routing_misconfig"] = [scored[0][2], scored[-1][2]]  # lowest- and highest-impact candidates
    return inst


def plot_fault(sim, ftype, faults, out, scen_dir, entries):
    fig, axes = plt.subplots(4, 2, figsize=(14, 11.5), sharex="col", layout="constrained")
    info = []
    for col, f in enumerate(faults):
        sid = f"aarnet_{ftype}_{'mild' if col == 0 else 'severe'}"
        spec = base_spec(sim, sid, 3000 + col)
        spec["faults"] = [f]
        res = sim.simulate(spec)
        cf = sim._run({**spec, "faults": []})
        entries.append(write_scenario(res, scen_dir))
        s, e = f["t_start"], f["t_end"]
        root = f["root_cause"]
        root_links = [sim.link_idx[root]] if root in sim.link_idx else incident_links(sim, root)
        du = np.abs(res.truth["util"][s:e] - cf.truth["util"][s:e]).max(0)
        du[root_links] = -1
        other = int(np.argmax(du))
        dl = np.nan_to_num(res.truth["svc_latency_ms"][s:e] - cf.truth["svc_latency_ms"][s:e]).max(0)
        dp = (res.truth["svc_loss_pct"][s:e] - cf.truth["svc_loss_pct"][s:e]).max(0)
        svc = int(np.argmax(dp + dl)) if (dp + dl).max() > 0 else 0
        z = slice(max(0, s - 144), min(sim.n_steps, e + 144))
        hrs = np.arange(z.start, z.stop) * H
        ax = axes[0, col]
        for k, i in enumerate(root_links[:3]):
            ax.plot(hrs, res.obs["util"][z, i], lw=0.9, color=f"C{k}", label=f"root {label(sim.links[i])}" if root in sim.link_idx else label(sim.links[i]))
            ax.plot(hrs, cf.truth["util"][z, i], lw=0.8, ls=":", color=f"C{k}")
        ax.plot(hrs, res.obs["util"][z, other], lw=0.9, color="C4", label=f"most shifted other: {label(sim.links[other])}")
        ax.plot(hrs, cf.truth["util"][z, other], lw=0.8, ls=":", color="C4")
        ax.legend(fontsize=6.5, loc="upper left")
        ax.set_ylabel("link util", fontsize=8)
        ax = axes[1, col]
        for k, i in enumerate(root_links[:3]):
            ax.plot(hrs, res.obs["loss_pct"][z, i], lw=0.9, color=f"C{k}")
        ax.set_ylabel("root link\ndiscard counters %", fontsize=8)
        sname = sim.services[svc]
        d = sim.kg.G.nodes[sname]
        ax = axes[2, col]
        ax.plot(hrs, res.obs["svc_latency_ms"][z, svc], lw=0.9, color="C5", label=f"{label(sname)} {d['src']} to {d['dst']}")
        ax.plot(hrs, cf.truth["svc_latency_ms"][z, svc], lw=0.8, ls=":", color="k", label="without fault (noise-free)")
        ax.legend(fontsize=6.5, loc="upper left")
        ax.set_ylabel("most affected\nservice latency ms", fontsize=8)
        ax = axes[3, col]
        ax.plot(hrs, res.obs["svc_loss_pct"][z, svc], lw=0.9, color="C5", label="observed (probes)")
        ax.plot(hrs, res.truth["svc_loss_pct"][z, svc], lw=0.8, color="C3", label="true loss")
        ax.legend(fontsize=6.5, loc="upper left")
        ax.set_ylabel("service loss %", fontsize=8)
        ax.set_xlabel("hours since scenario start", fontsize=8)
        for ax in axes[:, col]:
            shade(ax, f)
            ax.tick_params(labelsize=7)
        im = res.impacts["F0"]
        pstr = ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in f["params"].items() if k != "burst_sigma")
        axes[0, col].set_title(
            f"{'Mild' if col == 0 else 'Severe'}: {label(root)}, {(e - s) * H:.1f} h  [{im['impact_class']}]\n{pstr}\n"
            f"max svc dlat {im['max_service_latency_delta_ms']:.2f} ms, max svc dloss {im['max_service_loss_delta_pct']:.2f} pp, "
            f"{im['services_affected']} svc affected, status seen: {im['status_change_observed']}",
            fontsize=8,
        )
        info.append({"scenario": sid, "fault": f, "impact": im})
    fig.suptitle(f"Fault type: {ftype} (AARNet, synthetic telemetry; shaded = fault window, dotted = no-fault counterfactual)", fontsize=11)
    fig.savefig(out, **SAVE)
    plt.close(fig)
    return info


# ---------------------------------------------------------------------- distractors


def plot_distractors(sim, out, scen_dir, entries):
    busy = "link:Canberra1--Melbourne1"
    t_m = peak_start(sim, 2, 1.0)  # overnight, as planned work is
    spec = base_spec(
        sim, "aarnet_distractors", 4040,
        maintenance=[{"id": "M0", "target": busy, "t_start": t_m, "t_end": t_m + 36, "action": "drain", "description": "planned traffic drain"}],
        gaps=[{"id": "G0", "scope": "pop", "target": pop_node("Perth1"), "t_start": peak_start(sim, 4, 10), "t_end": peak_start(sim, 4, 10) + 18}],
        event_days=[{"day": 3, "factor": 1.2}],
    )
    res = sim.simulate(spec)
    cf = sim._run({**spec, "maintenance": [], "gaps": [], "event_days": []})
    entries.append(write_scenario(res, scen_dir))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), layout="constrained")
    li = sim.link_idx[busy]
    z = slice(t_m - 72, t_m + 108)
    hrs = np.arange(z.start, z.stop) * H
    gain = (res.truth["util"][t_m:t_m + 36] - cf.truth["util"][t_m:t_m + 36]).max(0)
    gain[li] = -1
    nb = int(np.argmax(gain))
    ax = axes[0]
    ax.plot(hrs, res.obs["util"][z, li], label=f"{label(busy)} (drained)")
    ax.plot(hrs, res.obs["util"][z, nb], label=f"{label(sim.links[nb])} (takes the traffic)")
    ax.axvspan(t_m * H, (t_m + 36) * H, color="#2c7bb6", alpha=0.1)
    ax.set_title("Planned overnight drain: logged in the change log, so benign", fontsize=9)
    ax.legend(fontsize=7)
    g = spec["gaps"][0]
    z2 = slice(g["t_start"] - 36, g["t_end"] + 36)
    perth = incident_links(sim, g["target"])
    ax = axes[1]
    for i in perth:
        ax.plot(np.arange(z2.start, z2.stop) * H, res.obs["util"][z2, i], marker=".", ms=2, lw=0.8, label=label(sim.links[i]))
    ax.axvspan(g["t_start"] * H, g["t_end"] * H, color="#999", alpha=0.15)
    ax.set_title("Telemetry gap at Perth1 collector: data missing, not a fault", fontsize=9)
    ax.legend(fontsize=7)
    ax = axes[2]
    daily = res.truth["util"].reshape(-1, sim.steps_per_day, len(sim.links)).mean(2)
    for d in range(daily.shape[0]):
        weekend = d % 7 >= 5
        ax.plot(np.arange(sim.steps_per_day) * H, daily[d], lw=1.8 if d == 3 else 0.8, color="#d7191c" if d == 3 else ("#bbb" if weekend else "#555"),
                ls="--" if weekend else "-", label="day 3: event day x1.2" if d == 3 else ("weekdays" if d == 0 else ("weekend" if d == 5 else None)))
    ax.set_title("Normal peaks: network mean utilisation by day", fontsize=9)
    ax.set_xlabel("hour of day (sim time)")
    ax.legend(fontsize=7)
    for ax in axes[:2]:
        ax.set_xlabel("hours since scenario start")
    fig.suptitle("Distractors (AARNet, synthetic telemetry)", fontsize=11)
    fig.savefig(out, **SAVE)
    plt.close(fig)


# ---------------------------------------------------------------------- pre-registered targets (validation only)

HARD = ("link_degradation", "demand_surge", "routing_misconfig")
RULES = ("any_lost_probe", "lost_probe_on_clean_service", "any_link_util_ge_0.9", "any_link_counter_loss_ge_0.05pct", "any_status_event")


def wilson(k, n, z=1.96):
    """Rate with a 95% Wilson interval (audit m10)."""
    if n == 0:
        return {"k": 0, "n": 0, "rate": None, "ci95": [None, None]}
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"k": int(k), "n": int(n), "rate": round(p, 4), "ci95": [round(max(0, c - h), 4), round(min(1, c + h), 4)]}


def naive_rules(res, s, e, clean_services):
    o = res.obs
    lost = np.nan_to_num(o["svc_loss_pct"][s:e]) > 0
    return {
        "any_lost_probe": bool(lost.any()),
        "lost_probe_on_clean_service": bool(lost[:, clean_services].any()) if len(clean_services) else False,
        "any_link_util_ge_0.9": bool(np.nanmax(o["util"][s:e], initial=0) >= 0.9),
        "any_link_counter_loss_ge_0.05pct": bool(np.nanmax(o["loss_pct"][s:e], initial=0) >= 0.05),
        "any_status_event": any(ev["type"] in ("link_status", "pop_status") and s <= ev["t"] < e for ev in res.observed_events),
    }


def diagnostics(cfg):
    val_gen = generated_split(cfg)["val"][::2]
    groups = [(n, "generated_val", 3) for n in val_gen] + [(n, n, 10) for n in cfg["topologies"]["validation_real"]]
    faults, ff_windows, cells = [], [], {"pattern": 0, "pattern_down": 0}
    lags, n_scen, n_concurrent = [], 0, 0
    seed = 8600
    for ti, (name, group, k_scen) in enumerate(groups):
        sim = Simulator(load(name, cfg=cfg), cfg)
        for k in range(k_scen):
            seed += 1
            spec = sample_scenario(sim, f"diag_{ti}_{k}", seed)
            res = sim.simulate(spec)
            n_scen += 1
            n_concurrent += any(f["placement"] == "concurrent" for f in spec["faults"])
            warm = slice(0, sim.warmup_steps)
            clean = np.flatnonzero(~(np.nan_to_num(res.obs["svc_loss_pct"][warm]) > 0).any(0))
            pat = ~np.isnan(res.obs["util"]) & np.isnan(res.obs["latency_ms"])
            cells["pattern"] += int(pat.sum())
            cells["pattern_down"] += int((pat & res.truth["fully_down"]).sum())
            for f in spec["faults"]:
                im = res.impacts[f["id"]]
                faults.append({"group": group, "type": f["type"], "placement": f["placement"], "impact_class": im["impact_class"],
                               "observable": im["observable"], **im["observability"], **naive_rules(res, f["t_start"], f["t_end"], clean)})
                if f["type"] == "link_down":
                    st = [ev["t"] for ev in res.observed_events if ev["type"] == "link_status" and ev.get("element") == f["root_cause"] and ev["t"] >= f["t_start"]]
                    if st:
                        lags.append(st[0] - f["t_start"])
            busy = [(f["t_start"] - 36, f["t_end"] + 36) for f in spec["faults"]]
            mwin = [(m["t_start"], m["t_end"]) for m in spec["maintenance"]]
            rng = np.random.default_rng(seed)
            for _ in range(6):
                dur = int(rng.integers(12, 72))
                s0 = int(rng.integers(sim.warmup_steps, sim.n_steps - dur))
                if any(not (s0 + dur <= a or s0 >= b) for a, b in busy):
                    continue
                in_maint = any(not (s0 + dur <= a or s0 >= b) for a, b in mwin)
                ff_windows.append({"maintenance": in_maint, **naive_rules(res, s0, s0 + dur, clean)})
    rules = RULES

    def rate(rows, key):
        return wilson(sum(bool(r[key]) for r in rows), len(rows))

    per_type = {}
    for ft in FAULT_TYPES:
        rows = [f for f in faults if f["type"] == ft]
        obs_rows = [f for f in rows if f["observable"]]
        per_type[ft] = {
            "n": len(rows),
            "unobservable": wilson(sum(not f["observable"] for f in rows), len(rows)),
            "channels": {c: rate(rows, c) for c in ("obs_status", "obs_link_util", "obs_link_counters", "obs_e2e")},
            "impact_class_observable": {c: wilson(sum(f["impact_class"] == c for f in obs_rows), len(obs_rows)) for c in ("subtle", "moderate", "obvious")},
            "rule_hit": {r: rate(rows, r) for r in rules},
            "by_group": {g: {c: sum(f["impact_class"] == c and f["group"] == g for f in rows) for c in ("subtle", "moderate", "obvious")}
                         for g in sorted({f["group"] for f in faults})},
        }
    clean_ff = [w for w in ff_windows if not w["maintenance"]]
    maint_ff = [w for w in ff_windows if w["maintenance"]]
    ff = {"non_maintenance": {r: rate(clean_ff, r) for r in rules}, "maintenance": {r: rate(maint_ff, r) for r in rules}}

    # T7 / T8 from fault-free weeks with distractors off
    base = {}
    pooled_peaks, lost, tot = [], 0, 0
    for name in cfg["topologies"]["validation_real"] + val_gen[::4]:
        sim = Simulator(load(name, cfg=cfg), cfg)
        r = sim.simulate(base_spec(sim, "base", 8999))
        peaks = r.truth["util"].max(0)
        lp = np.nan_to_num(r.obs["svc_loss_pct"]) > 0
        lost += int(lp.sum())
        tot += lp.size
        if name in cfg["topologies"]["validation_real"]:
            base[name] = round(float(np.median(peaks)), 4)
        else:
            pooled_peaks.extend(peaks.tolist())
    base["generated_val_pooled"] = round(float(np.median(pooled_peaks)), 4)

    placements = [f["placement"] for f in faults]
    t = {}
    t["T1_unobservable_le_25pct"] = {ft: {"value": per_type[ft]["unobservable"], "pass": (per_type[ft]["unobservable"]["rate"] or 0) <= 0.25} for ft in HARD}
    t["T2_subtle_ge_20_obvious_le_50"] = {
        ft: {"subtle": per_type[ft]["impact_class_observable"]["subtle"]["rate"], "obvious": per_type[ft]["impact_class_observable"]["obvious"]["rate"],
             "pass": (per_type[ft]["impact_class_observable"]["subtle"]["rate"] or 0) >= 0.2 and (per_type[ft]["impact_class_observable"]["obvious"]["rate"] or 1) <= 0.5}
        for ft in HARD + ("link_flapping",)
    }
    t["T3_youden_j_le_0.5"] = {
        ft: {r: {"j": round(per_type[ft]["rule_hit"][r]["rate"] - ff["non_maintenance"][r]["rate"], 4), "pass": per_type[ft]["rule_hit"][r]["rate"] - ff["non_maintenance"][r]["rate"] <= 0.5} for r in rules}
        for ft in HARD
    }
    frac_down = cells["pattern_down"] / max(cells["pattern"], 1)
    t["T4_down_share_of_pattern_le_80pct"] = {"value": round(frac_down, 4), **cells, "pass": frac_down <= 0.8}
    zero = sum(x == 0 for x in lags) / max(len(lags), 1)
    t["T5_status_lag"] = {"n": len(lags), "zero_lag_frac": round(zero, 4), "median_lag": float(np.median(lags)) if lags else None,
                          "pass": 0.3 <= zero <= 0.7 and (float(np.median(lags)) >= 1 if lags else False)}
    linked = sum(p == "maintenance_linked" for p in placements) / max(len(placements), 1)
    t["T6_overlap"] = {"maintenance_linked_frac": round(linked, 4), "scenarios_with_concurrent_frac": round(n_concurrent / n_scen, 4),
                       "pass": 0.10 <= linked <= 0.20 and n_concurrent / n_scen >= 0.05}
    t["T7_baseline_median_peak"] = {"value": base, "pass": all(0.4 <= v <= 0.7 for v in base.values())}
    t["T8_lost_probe_service_steps"] = {"value": round(lost / tot, 4), "pass": 0.01 <= lost / tot <= 0.10}
    return {
        "networks": {"generated_val": len(val_gen), "validation_real": cfg["topologies"]["validation_real"]},
        "scenarios": n_scen,
        "faults": len(faults),
        "placements": {p: placements.count(p) for p in sorted(set(placements))},
        "per_fault_type": per_type,
        "fault_free_windows": {"n_non_maintenance": len(clean_ff), "n_maintenance": len(maint_ff), **ff},
        "targets": t,
    }


def config_hash(root):
    return hashlib.sha256((root / "config.yaml").read_bytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-tuning", default=None, help="append the target results to results/tuning_log.jsonl with this label")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    figures = ROOT / cfg["paths"]["figures"]
    figures.mkdir(parents=True, exist_ok=True)
    scen_dir = ROOT / cfg["paths"]["scenarios_demo"]
    scen_dir.mkdir(parents=True, exist_ok=True)
    entries = []

    sim = Simulator(load("topozoo/Aarnet"), cfg)
    summary = {"note": "AARNet is the dev / demo network, seen during development; its numbers are illustrative only.",
               "config_hash": config_hash(ROOT),
               "aarnet_link_down_demo": plot_link_down_demo(sim, cfg, figures / "phase2_link_down_aarnet.png", scen_dir, entries)}
    summary["aarnet_fault_examples"] = {}
    for ftype, faults in fault_instances(sim).items():
        summary["aarnet_fault_examples"][ftype] = plot_fault(sim, ftype, faults, figures / f"phase2_fault_{ftype}.png", scen_dir, entries)
    plot_distractors(sim, figures / "phase2_distractors_aarnet.png", scen_dir, entries)
    for k in range(3):
        entries.append(write_scenario(sim.simulate(sample_scenario(sim, f"aarnet_sampled_{k}", 5000 + k)), scen_dir))
    write_manifest(entries, scen_dir)

    summary["diagnostics_validation"] = diagnostics(cfg)
    (ROOT / cfg["paths"]["results"] / "phase2_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    if args.log_tuning:
        with open(ROOT / cfg["paths"]["results"] / "tuning_log.jsonl", "a") as fh:
            fh.write(json.dumps({"label": args.log_tuning, "config_hash": summary["config_hash"], "targets": summary["diagnostics_validation"]["targets"]}, default=float) + "\n")
    print(json.dumps(summary["diagnostics_validation"]["targets"], indent=1, default=float))


if __name__ == "__main__":
    main()
