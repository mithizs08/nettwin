"""Phase 10: the report's key figures, drawn only from committed test results (no data folder, no API key).

    python scripts/phase10_figures.py

Writes results/figures/test_localisation_phase5.png, test_agent_phase6.png and test_forecast_phase8.png. Each is a
dot-and-interval plot of a pre-registered test metric with its 95% bootstrap CI, one colour per variant.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = RES / "figures"
COLORS = {"with_status": "#2a78d6", "status_blind": "#eb6834"}        # categorical slots 1 and 2 (validated default)
LABELS = {"with_status": "with status events", "status_blind": "status-blind"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e4dd"


def dotplot(ax, rows, title, xlabel, xlim=(0, 1)):
    """rows: [(label, {variant: (value, lo, hi)})], top to bottom."""
    n = len(rows)
    for i, (label, vals) in enumerate(rows):
        y = n - 1 - i
        for k, (v, (val, lo, hi)) in enumerate(vals.items()):
            off = 0 if len(vals) == 1 else 0.14 if k == 0 else -0.14
            ax.plot([lo, hi], [y + off, y + off], color=COLORS[v], lw=2, solid_capstyle="round")
            ax.plot(val, y + off, "o", ms=8, color=COLORS[v], mec="white", mew=2, label=LABELS[v] if i == 0 else None)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in rows][::-1], color=INK)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)


def phase5():
    t = json.loads((RES / "phase5_test.json").read_text())["splits"]["test"]["variants"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, (group, metric, title) in zip(axes, (("hard", "top3", "Hard fault types: top-3"), ("easy", "top1", "Easy fault types: top-1"))):
        rows = []
        for m, label in (("B1", "B1 EWMA z-score"), ("B2", "B2 Isolation Forest"), ("B3", "B3 tomography"), ("B3+", "B3+ tomography"),
                         ("gat-h32-mlp", "MLP ablation (3 seeds)"), ("gat-h32", "GNN GAT-h32 (3 seeds)")):
            vals = {}
            for v in COLORS:
                if m.startswith("gat"):
                    s = t[v]["seed_summary"][m][f"{group}:{metric}@q90"]
                    vals[v] = (s["mean"], *s["seed_mean_ci"])
                else:
                    x = t[v]["methods"][m]["groups"][group]["localisation"]["q90"][metric]
                    vals[v] = (x["value"], *x["ci"])
            rows.append((label, vals))
        dotplot(ax, rows, title, "root-cause accuracy (oracle window, q90)")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, loc="upper right", ncol=2, fontsize=9)
    fig.suptitle("Phase 5 test: localisation on 7 unseen public topologies (210 scenarios, 95% CI)", x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "test_localisation_phase5.png", dpi=160)


def phase6():
    s = {m: json.loads((RES / "phase6" / f"test_claude-{m}-5-5" / "summary.json").read_text())["variants"] for m in ("haiku", "opus")}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
    panels = (("faults", "top3", "Faults: top-3 localisation"), ("distractors", "claims_fault", "Distractors: fault claims (lower is better)"),
              ("change_overlap_faults", "claims_fault", "Change-overlapping faults: recall"))
    methods = (("haiku", "B3+", "B3+ tomography"), ("haiku", "GNN", "GNN (3 seeds)"), ("haiku", "GNN+CL", "GNN + change-log rule"),
               ("haiku", "A0", "Haiku A0 (alerts only)"), ("haiku", "A1", "Haiku A1 (no GNN)"), ("haiku", "A2", "Haiku A2 (full agent)"),
               ("opus", "A0", "Opus A0 (alerts only)"), ("opus", "A2", "Opus A2 (full agent)"))
    for ax, (g, metric, title) in zip(axes, panels):
        rows = [(label, {v: (s[mod][v]["metrics"][m][g][metric]["value"], *s[mod][v]["metrics"][m][g][metric]["ci"]) for v in COLORS})
                for mod, m, label in methods]
        dotplot(ax, rows, title, "share of incidents")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, loc="upper right", ncol=2, fontsize=9)
    fig.suptitle("Phase 6 test: agent conditions against the non-LLM references (167 incidents, 95% CI)", x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "test_agent_phase6.png", dpi=160)


def phase8():
    t = json.loads((RES / "phase8_test.json").read_text())["splits"]["test"]["metrics"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.9), sharey=True)
    names = (("persistence", "Persistence"), ("seasonal_naive", "Seasonal-naive"), ("lightgbm", "LightGBM, lag features"),
             ("gnn_head", "Frozen GNN encoder + head"))
    for ax, metric in zip(axes, ("precision", "recall")):
        rows = [(label, {"with_status": (t[m]["all"][metric]["value"], *t[m]["all"][metric]["ci"])}) for m, label in names]
        dotplot(ax, rows, f"Breach {metric}", metric, xlim=(0.5, 1))
    fig.suptitle("Phase 8 test: utilisation above 80% within 60 minutes (210 scenarios, with status events, 95% CI)", x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "test_forecast_phase8.png", dpi=160)


if __name__ == "__main__":
    for f in (phase5, phase6, phase8):
        f()
    print("wrote results/figures/test_localisation_phase5.png, test_agent_phase6.png, test_forecast_phase8.png")
