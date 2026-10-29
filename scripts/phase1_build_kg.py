"""Phase 1: build the KG for the dev and validation topologies, export Neo4j CSV, draw AARNet figures.

Writes:
  results/phase1_kg_summary.json      counts per kind / rel, plus an AARNet single-link-failure check
  results/figures/aarnet_kg.png       KG schema with counts, and the multi-layer slice for three services
  results/figures/aarnet_map.png      AARNet on lon/lat from the Topology Zoo PoP coordinates
  data/kg/<topology>/*.csv            Neo4j-style CSV export (gitignored)
"""
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

from nettwin.kg import SCHEMA, NetTwinKG  # noqa: E402
from nettwin.topology import load  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
KIND_COLOURS = {
    "Service": "#7b3294",
    "RoutingPath": "#c2a5cf",
    "Link": "#008837",
    "PoP": "#1f78b4",
    "FaultType": "#bababa",
    "Alert": "#d7191c",
    "Maintenance": "#fdae61",
    "Runbook": "#a6611a",
}
SAVE = {"dpi": 150, "bbox_inches": "tight", "metadata": {"Software": None}}


def failure_check(kg):
    """Cut the busiest link at t=1, restore at t=2; report what the KG says changed."""
    busiest = max(kg.nodes_of_kind("Link"), key=lambda l: (len(kg.paths_through(l)), l))
    paths_before, services_before = kg.paths_through(busiest), kg.services_on(busiest)
    kg.apply_change(1, links_down=[busiest], reason="phase1 check")
    reroutes = [e for e in kg.changes_between(1, 1) if e["type"] == "reroute"]
    stranded = [s for s in services_before if not kg.service_paths(s)]
    kg.apply_change(2, links_up=[busiest], reason="phase1 check")
    restored = kg.paths_through(busiest) == paths_before
    return {
        "link": busiest,
        "paths_through_before": len(paths_before),
        "services_on_before": len(services_before),
        "pairs_rerouted": len(reroutes),
        "pairs_unreachable": sum(e["unreachable"] for e in reroutes),
        "services_without_path": len(stranded),
        "paths_restored_after_link_up": restored,
    }


def draw_kg(kg, out):
    fig, (ax_s, ax_l) = plt.subplots(1, 2, figsize=(16, 7.5), gridspec_kw={"width_ratios": [1, 1.6]})
    s = kg.summary()

    # Left: schema with counts
    pos = {
        "Service": (0.5, 3.6),
        "RoutingPath": (0.5, 2.4),
        "Link": (0.5, 1.2),
        "PoP": (0.5, 0.0),
        "Alert": (1.7, 2.4),
        "Maintenance": (1.7, 0.6),
        "Runbook": (-0.7, 0.6),
        "FaultType": (-0.7, -0.6),
    }
    for rel, (srcs, dsts) in SCHEMA.items():
        for a in srcs:
            for b in dsts:
                (x0, y0), (x1, y1) = pos[a], pos[b]
                ax_s.annotate("", (x1, y1), (x0, y0), arrowprops={"arrowstyle": "-|>", "color": "#555", "lw": 1.2, "shrinkA": 28, "shrinkB": 28})
                label = f"{rel} ({s['edges_by_rel'][rel]})"
                if x0 == x1:
                    ax_s.text(x0 - 0.06, (y0 + y1) / 2, label, fontsize=7.5, color="#333", ha="right", va="center")
                else:
                    ax_s.text(x0 + 0.6 * (x1 - x0) + 0.05, y0 + 0.6 * (y1 - y0), label, fontsize=7.5, color="#333", va="center")
    for kind, (x, y) in pos.items():
        n = s["nodes_by_kind"][kind]
        stub = kind in ("Alert", "Maintenance", "Runbook")
        ax_s.scatter([x], [y], s=2600, color=KIND_COLOURS[kind], alpha=0.35 if stub else 0.9, edgecolors="k", linewidths=0.6, zorder=3)
        ax_s.text(x, y, f"{kind}\n{n}", ha="center", va="center", fontsize=8.5, zorder=4, weight="bold")
    ax_s.set_xlim(-1.4, 2.4)
    ax_s.set_ylim(-1.2, 4.2)
    ax_s.axis("off")
    ax_s.set_title("Schema with node / edge counts\n(faded = schema only, populated in later phases)", fontsize=10)

    # Right: layered slice for three services
    all_svc = kg.nodes_of_kind("Service")
    ecmp = [sv for sv in all_svc if len(kg.service_paths(sv)) > 1]
    first = min(ecmp, key=lambda sv: (kg.G.nodes[kg.service_paths(sv)[0]]["hops"], sv))
    services = [first] + [sv for sv in all_svc if sv != first][:2]
    path_labels = {}
    layer = {"Service": 3, "RoutingPath": 2, "Link": 1, "PoP": 0}
    members = defaultdict(set)
    edges = []
    for svc in services:
        members["Service"].add(svc)
        for p in kg.service_paths(svc):
            members["RoutingPath"].add(p)
            path_labels.setdefault(p, f"P{len(path_labels) + 1}")
            edges.append((svc, p))
            for lid in kg.links_of_path(p):
                members["Link"].add(lid)
                edges.append((p, lid))
                for pop in (f"pop:{kg.G.nodes[lid]['u']}", f"pop:{kg.G.nodes[lid]['v']}"):
                    members["PoP"].add(pop)
                    edges.append((lid, pop))
    xy = {}
    for kind, nodes in members.items():
        nodes = sorted(nodes, key=lambda n: (kg.G.nodes[n]["x"], n) if kind == "PoP" else (0, n))
        for i, n in enumerate(nodes):
            xy[n] = ((i + 0.5) / len(nodes), layer[kind])
    for a, b in sorted(set(edges)):
        ax_l.plot(*zip(xy[a], xy[b]), color="#999", lw=0.7, zorder=1)
    for kind, nodes in members.items():
        for n in sorted(nodes):
            x, y = xy[n]
            ax_l.scatter([x], [y], s=90, color=KIND_COLOURS[kind], edgecolors="k", linewidths=0.4, zorder=2)
            label = n.split(":", 1)[1]
            if kind == "RoutingPath":
                label = path_labels[n]
            elif kind == "Service":
                d = kg.G.nodes[n]
                label = f"{label} ({d['cls']})\n{d['src']} to {d['dst']}"
            rot = 0 if kind in ("Service", "RoutingPath") else 60
            ax_l.text(x, y + (0.12 if kind != "PoP" else -0.12), label, fontsize=5.8, ha="center",
                      va="bottom" if kind != "PoP" else "top", rotation=rot)
    for kind, y in layer.items():
        ax_l.text(-0.06, y, kind, ha="right", va="center", fontsize=9, weight="bold", color=KIND_COLOURS[kind])
    legend = "\n".join(f"{lab}: {' > '.join(kg.G.nodes[p]['pops'])}" for p, lab in sorted(path_labels.items(), key=lambda kv: int(kv[1][1:])))
    ax_l.text(0.0, -1.55, legend, fontsize=7, va="top", family="monospace")
    ax_l.set_xlim(-0.2, 1.05)
    ax_l.set_ylim(-1.6 - 0.22 * len(path_labels), 4.3)
    ax_l.axis("off")
    ax_l.set_title(f"Multi-layer slice: three synthetic services ({first.split(':')[1]} has ECMP), their active paths, links and PoPs", fontsize=10)

    fig.suptitle("AARNet knowledge graph (public Topology Zoo topology; services are synthetic)", fontsize=12)
    fig.savefig(out, **SAVE)
    plt.close(fig)


def draw_map(topo, out, offset_deg=0.6):
    """Plain lon/lat plot, no basemap. Co-located PoPs (same coordinates) are nudged apart so both show."""
    by_pos = defaultdict(list)
    for n, d in topo.nodes(data=True):
        by_pos[d["pos"]].append(n)
    xy, label_dir = {}, {}
    for (lon, lat), names in by_pos.items():
        for i, n in enumerate(sorted(names)):
            if len(names) == 1:
                xy[n], label_dir[n] = (lon, lat), (1, 0.6)
            else:
                ang = 3 * math.pi / 4 + 2 * math.pi * i / len(names)  # NW / SE, across the NE-running east-coast corridor
                xy[n] = (lon + offset_deg * math.cos(ang), lat + offset_deg * math.sin(ang))
                label_dir[n] = (math.cos(ang), math.sin(ang))

    def in_box(p):
        return zoom[0] <= p[0] <= zoom[1] and zoom[2] <= p[1] <= zoom[3]

    def draw(a, labelled, fontsize, size):
        for u, v, d in topo.edges(data=True):
            colo = d["dist_km"] == 0
            a.plot(*zip(xy[u], xy[v]), color="#e66101" if colo else "#4d4d4d", lw=2.0 if colo else 1.3, zorder=1)
        for n, (x, y) in xy.items():
            a.scatter([x], [y], s=size, color=KIND_COLOURS["PoP"], edgecolors="k", linewidths=0.5, zorder=2)
            if labelled(n):
                dx, dy = label_dir[n]
                a.annotate(n, (x, y), xytext=(7 * dx, 7 * dy), textcoords="offset points", fontsize=fontsize,
                           ha="left" if dx >= 0 else "right", va="bottom" if dy >= 0 else "top")

    # south-east corridor is dense; label it in a zoomed inset instead of the main panel
    zoom = (143.0, 155.0, -39.5, -26.0)
    mean_lat = sum(lat for _, lat in xy.values()) / len(xy)
    aspect = 1 / math.cos(math.radians(mean_lat))
    fig, ax = plt.subplots(figsize=(10, 8.5))
    draw(ax, lambda n: not in_box(xy[n]), 8, 55)
    ins = ax.inset_axes([0.02, 0.43, 0.38, 0.55])
    draw(ins, lambda n: in_box(xy[n]), 7.5, 45)
    ins.set_xlim(zoom[0], zoom[1])
    ins.set_ylim(zoom[2], zoom[3])
    ins.set_aspect(aspect)
    ins.set_xticks([])
    ins.set_yticks([])
    ins.set_facecolor("#fafafa")
    ins.text(0.03, 0.97, "South-east (zoom)", transform=ins.transAxes, fontsize=8, va="top")
    ax.indicate_inset_zoom(ins, edgecolor="#888")
    ax.set_aspect(aspect)
    ax.set_xlabel("Longitude (deg E)")
    ax.set_ylabel("Latitude (deg)")
    ax.grid(alpha=0.3)
    ax.plot([], [], color="#4d4d4d", label="inter-city link")
    ax.plot([], [], color="#e66101", lw=2, label="intra-city link between co-located PoPs (0 km)")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title(
        f"AARNet, Internet Topology Zoo: {topo.number_of_nodes()} PoPs, {topo.number_of_edges()} links\n"
        f"PoP coordinates from the Zoo; co-located PoPs offset by {offset_deg} deg for visibility; no basemap",
        fontsize=10,
    )
    fig.savefig(out, **SAVE)
    plt.close(fig)


def main():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    results = ROOT / cfg["paths"]["results"]
    figures = ROOT / cfg["paths"]["figures"]
    figures.mkdir(parents=True, exist_ok=True)

    summary = {"topologies": {}}
    for name in cfg["topologies"]["dev"] + cfg["topologies"]["validation_real"]:  # never on test (audit C1, L11)
        topo = load(name, use_names=cfg["topologies"]["use_names"])
        kg = NetTwinKG.from_config(topo, cfg)
        kg.validate()
        summary["topologies"][name] = kg.summary()
        kg.export_csv(ROOT / cfg["paths"]["kg"] / name.replace("/", "_"))
        if name == "topozoo/Aarnet":
            draw_kg(kg, figures / "aarnet_kg.png")
            draw_map(topo, figures / "aarnet_map.png")
            summary["aarnet_single_link_failure"] = failure_check(kg)
            kg.validate()

    (results / "phase1_kg_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
