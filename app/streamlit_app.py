"""NetTwin demo dashboard (Phase 7). Runs offline from committed results; no data folder and no API key.

    streamlit run app/streamlit_app.py

Incident explorer: AARNet (dev split, seen during development) incidents with the alert timeline, the change log, the
GNN's top-ranked elements and the saved tool traces of Haiku 5.5 A2 and Opus 5.5 A2 next to the ground truth.
Test results: the Phase 6 test table from `results/phase6/`. Telemetry is synthetic, topologies are public, and none of
this is Telstra's network.
"""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
STEP_MIN = 5
HONESTY = ("Telemetry is synthetic (from the NetTwin simulator). Topologies are public (Internet Topology Zoo, SNDlib). "
           "None of this is Telstra's network or Telstra data.")
COL = {"truth": "#d62728", "gnn": "#ff7f0e", "haiku": "#1f77b4", "opus": "#9467bd", "change": "#7f7f7f", "link": "#c7c7c7"}
MODELS = {"haiku": "Haiku 5.5 A2", "opus": "Opus 5.5 A2"}


@st.cache_data
def load_json(rel):
    return json.loads((RES / rel).read_text())


def hours(step):
    return step * STEP_MIN / 60


def link_ends(link):
    u, v = link[5:].split("--", 1)
    return u, v


# ---------------------------------------------------------------- map
def network_map(topo, marks):
    pos = {p["name"]: (p["x"], p["y"]) for p in topo["pops"]}
    fig = go.Figure()
    for l in topo["links"]:
        (x0, y0), (x1, y1) = pos[l["u"]], pos[l["v"]]
        fig.add_trace(go.Scattergeo(lon=[x0, x1], lat=[y0, y1], mode="lines", line=dict(width=1.5, color=COL["link"]),
                                    hoverinfo="text", text=l["id"], showlegend=False))
    for label, (els, color, width) in marks.items():
        first = True
        for el in els:
            if not el:
                continue
            if el.startswith("link:"):
                u, v = link_ends(el)
                if u not in pos or v not in pos:
                    continue
                (x0, y0), (x1, y1) = pos[u], pos[v]
                fig.add_trace(go.Scattergeo(lon=[x0, x1], lat=[y0, y1], mode="lines", line=dict(width=width, color=color),
                                            name=label, legendgroup=label, showlegend=first, hoverinfo="text", text=f"{label}: {el}"))
            elif el.startswith("pop:") and el[4:] in pos:
                x, y = pos[el[4:]]
                fig.add_trace(go.Scattergeo(lon=[x], lat=[y], mode="markers", marker=dict(size=16, color=color, symbol="circle-open", line=dict(width=3)),
                                            name=label, legendgroup=label, showlegend=first, hoverinfo="text", text=f"{label}: {el}"))
            else:
                continue
            first = False
    fig.add_trace(go.Scattergeo(lon=[p["x"] for p in topo["pops"]], lat=[p["y"] for p in topo["pops"]], mode="markers",
                                marker=dict(size=6, color="#333"), text=[p["name"] for p in topo["pops"]], hoverinfo="text", showlegend=False))
    fig.update_geos(lonaxis_range=[110, 156], lataxis_range=[-45, -9], showcountries=True, showland=True, landcolor="#f4f1ea",
                    showocean=True, oceancolor="#e8f0f7", resolution=50, projection_type="mercator")
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h", y=-0.02), dragmode=False)
    return fig


# ---------------------------------------------------------------- timeline
def timeline(view, n):
    rows = []
    sev_color = {"critical": "#d62728", "major": "#ff7f0e", "minor": "#e7ba52", "info": "#9edae5"}
    for a in view["alerts"]["alerts"]:
        end = n if a["end"] is None else a["end"]
        rows.append({"lane": f"{a['rule']} | {a['target']}", "start": a["start"], "end": max(end, a["start"] + 1), "color": sev_color.get(a["severity"], "#999"),
                     "text": f"{a['rule']} on {a['target']} ({a['severity']}), value {a['value']}"})
    for c in view["changes"]:
        rows.append({"lane": f"change {c['action']} | {c['target']}", "start": c["start"], "end": max(c["end"], c["start"] + 1), "color": COL["change"],
                     "text": f"logged change {c['id']}: {c['description']} on {c['target']}"})
    fig = go.Figure()
    fig.add_vrect(x0=0, x1=hours(n), fillcolor="#d62728", opacity=0.07, line_width=0, annotation_text="incident window", annotation_position="top left")
    lanes = list(dict.fromkeys(r["lane"] for r in rows))
    for r in rows:
        fig.add_trace(go.Bar(base=[hours(r["start"])], x=[hours(r["end"] - r["start"])], y=[r["lane"]], orientation="h",
                             marker_color=r["color"], hovertext=r["text"], hoverinfo="text", showlegend=False))
    fig.update_layout(height=max(220, 26 * len(lanes) + 80), barmode="overlay", margin=dict(l=0, r=0, t=20, b=0),
                      xaxis_title="hours relative to the window start", yaxis=dict(categoryorder="array", categoryarray=lanes[::-1]))
    return fig


def util_chart(view):
    fig = go.Figure()
    t0 = view["util"]["t_start"]
    for link, s in view["util"]["series"].items():
        fig.add_trace(go.Scatter(x=[hours(t0 + i) for i in range(len(s))], y=s, mode="lines", name=link))
    fig.add_vrect(x0=0, x1=hours(t0 + max((len(s) for s in view["util"]["series"].values()), default=0)), fillcolor="#d62728", opacity=0.07, line_width=0)
    fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="hours relative to the window start", yaxis_title="utilisation",
                      legend=dict(orientation="h", y=-0.3))
    return fig


def pretty(s, limit=4000):
    try:
        s = json.dumps(json.loads(s), indent=1)
    except (TypeError, ValueError):
        s = str(s)
    return s if len(s) <= limit else s[:limit] + f"\n... ({len(s) - limit} more characters)"


def show_trace(tr):
    if tr is None:
        st.info("No saved trace for this incident and variant.")
        return
    g = tr["groundedness"]
    st.caption(f"{tr['n_tool_calls']} tool calls, {tr['n_refused_calls']} refused, {'forced to submit at the budget, ' if tr['forced_submit'] else ''}"
               f"${tr['cost_usd']:.4f}, {tr['seconds']} s, ungrounded evidence items {g['n_ungrounded_items']} of {g['n_items']} "
               f"(strict {g['n_ungrounded_items_strict']})")
    for k, c in enumerate(tr["calls"], 1):
        status = " (refused: over the 15-call budget)" if c["refused"] else " (error)" if c["error"] else ""
        args = ", ".join(f"{a}={json.dumps(v)}" for a, v in c["input"].items())
        with st.expander(f"{k}. {c['tool']}({args}){status}"):
            st.code(pretty(c["output"]), language="json")
    d = tr["diagnosis"] or {}
    st.markdown("**Final diagnosis**")
    st.json(d, expanded=True)


def explorer():
    b = load_json("phase7/demo_bundle.json")
    st.subheader("Incident explorer: AARNet (dev)")
    st.caption("AARNet is the demo network. It was seen during development, so these incidents are dev, not held-out evidence. "
               "Picked by a fixed rule (Phase 6 sampler, seed 6006, one per fault type plus one drain distractor). " + HONESTY)
    labels = {f"{e['incident']['incident_id']}: {e['incident']['type']} ({e['incident']['scenario_id']})": e for e in b["incidents"]}
    c1, c2 = st.columns([3, 2])
    e = labels[c1.selectbox("Incident", list(labels), key="incident")]
    v = c2.radio("Variant", ["with_status", "status_blind"], horizontal=True, key="variant",
                 help="status_blind removes link up/down status events; the agent then works from telemetry alone")
    inc, view = e["incident"], e["variants"][v]
    n = inc["t1"] - inc["t0"]
    agents = view["agents"]
    diag = {m: ((agents.get(m) or {}).get("diagnosis") or {}) for m in MODELS}
    marks = {"ground truth": ([inc["root"]] if inc["root"] != "none" else [], COL["truth"], 6),
             "logged change": ([inc.get("change_target")], COL["change"], 4),
             "GNN top-3": ([x["element"] for x in view["gnn_top"][:3]], COL["gnn"], 3),
             **{MODELS[m]: ([diag[m].get("root_cause")], COL[m], 2) for m in MODELS}}

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.plotly_chart(network_map(b["topology"], marks), width="stretch", config={"scrollZoom": False})
    with right:
        st.markdown("**Ground truth**")
        truth = {"type": inc["type"], "root cause": inc["root"], "window": f"{n} steps ({hours(n):.1f} h)"}
        if inc.get("change_target"):
            truth["logged change"] = inc["change_target"]
        if inc["kind"] == "fault":
            truth["overlaps a logged change"] = inc.get("overlaps_logged_change")
        st.markdown("\n".join(f"- **{k}**: `{v}`" for k, v in truth.items()))
        for m in MODELS:
            st.markdown(f"**{MODELS[m]}** says: `{diag[m].get('root_cause')}` ({diag[m].get('fault_type')}, "
                        f"confidence {diag[m].get('confidence')})")
        st.markdown(f"**GNN** top element: `{view['gnn_top'][0]['element']}`")

    rows = []
    for m in MODELS:
        sc = view["scores"].get(f"{m}-A2", {})
        rows.append({"method": MODELS[m], "claims fault": None if sc.get("claims_fault") is None else float(sc["claims_fault"]), "root cause": diag[m].get("root_cause"),
                     "fault type": diag[m].get("fault_type"), "top-1": sc.get("top1"), "top-3": sc.get("top3")})
    for ref in ("GNN", "GNN+CL", "B3+"):
        sc = view["scores"].get(ref, {})
        rows.append({"method": ref, "claims fault": sc.get("claims_fault"), "root cause": view["gnn_top"][0]["element"] if ref != "B3+" else "",
                     "fault type": "", "top-1": sc.get("top1"), "top-3": sc.get("top3")})
    st.markdown("**Diagnoses next to ground truth**")
    st.dataframe(pd.DataFrame(rows).set_index("method"), width="stretch")
    st.caption("Top-1 / top-3 use tie-aware credit. For the references, 'claims fault' is the share of the 3 GNN seeds (or "
               "B3+) whose frozen 5% fault-free threshold was crossed in the window. GNN+CL is the GNN silenced when the "
               "change log shows a drain or outage overlapping the window.")

    st.markdown("**Alert timeline and change log** (what an operator sees when the incident is opened at the window end; one day of history)")
    st.caption(f"{view['alerts']['n_alerts']} alerts in this range, {view['alerts']['shown']} shown.")
    st.plotly_chart(timeline(view, n), width="stretch")
    if view["util"]["series"]:
        st.markdown("**Utilisation of the links in question** (ground truth root, logged change, GNN top links)")
        st.plotly_chart(util_chart(view), width="stretch")

    g1, g2 = st.columns([2, 3])
    with g1:
        st.markdown("**GNN top-ranked elements** (mean of 3 seeds, q90 over the window)")
        df = pd.DataFrame(view["gnn_top"])
        df.index = range(1, len(df) + 1)
        df["ground truth"] = df.element == inc["root"]
        st.dataframe(df, width="stretch")
        st.caption(f"Rank of the true root: {view['gnn_true_rank'] or 'n/a (no fault)'}. A2's GNN tool uses seed 0 only.")
    with g2:
        st.markdown("**Agent tool traces** (step by step, as logged)")
        tabs = st.tabs([MODELS[m] for m in MODELS])
        for t, m in zip(tabs, MODELS):
            with t:
                show_trace(agents.get(m))


# ---------------------------------------------------------------- results
def cell(metrics, method, group, metric):
    x = metrics.get(method, {}).get(group, {}).get(metric)
    if not x:
        return ""
    lo, hi = x["ci"]
    return f"{x['value']:.3f} [{lo:.2f}, {hi:.2f}]"


def results():
    st.subheader("Phase 6 test results (run once, as pre-registered)")
    st.caption(HONESTY + " 167 test incidents on seven held-out public topologies: 120 faults, 30 maintenance distractors, "
               "and 40 faults whose window overlaps a logged drain or outage. 95% CIs from 1000 incident bootstrap resamples.")
    haiku = load_json("phase6/test_claude-haiku-5-5/summary.json")
    opus = load_json("phase6/test_claude-opus-5-5/summary.json")
    v = st.radio("Variant", ["with_status", "status_blind"], horizontal=True, key="results_variant")
    mh, mo = haiku["variants"][v]["metrics"], opus["variants"][v]["metrics"]
    rows = []
    cols = [("faults top-1", "faults", "top1"), ("faults top-3", "faults", "top3"), ("hard top-1", "hard", "top1"),
            ("fault recall", "faults", "claims_fault"), ("distractor fault claims (lower is better)", "distractors", "claims_fault"),
            ("recall on change-overlapping faults", "change_overlap_faults", "claims_fault")]
    for label, mets, m in [("B3+ (topology heuristic)", mh, "B3+"), ("GNN (3 seeds)", mh, "GNN"), ("GNN seed 0", mh, "GNN-s0"),
                           ("GNN + change-log rule", mh, "GNN+CL"), ("Haiku 5.5 A0 (alerts only)", mh, "A0"),
                           ("Haiku 5.5 A1 (no GNN)", mh, "A1"), ("Haiku 5.5 A2 (full agent)", mh, "A2"),
                           ("Opus 5.5 A0 (alerts only)", mo, "A0"), ("Opus 5.5 A2 (full agent)", mo, "A2")]:
        rows.append({"method": label, **{c: cell(mets, m, g, k) for c, g, k in cols}})
    st.dataframe(pd.DataFrame(rows).set_index("method"), width="stretch")

    st.markdown("**Agent against the GNN + change-log rule, and other paired comparisons** (difference, 95% CI, verdict)")
    keys = [("Haiku", haiku, "A2 vs GNN+CL"), ("Haiku", haiku, "A1 vs GNN+CL"), ("Opus", opus, "A2 vs GNN+CL"),
            ("Haiku", haiku, "A2 vs GNN"), ("Opus", opus, "A2 vs GNN"), ("Haiku", haiku, "A2 vs A0"), ("Opus", opus, "A2 vs A0"),
            ("Haiku", haiku, "A2 vs A1"), ("Haiku", haiku, "GNN+CL vs GNN")]
    rows = []
    for model, s, pair in keys:
        comps = s["variants"][v]["comparisons"]
        for g, m in (("faults", "top1"), ("faults", "top3"), ("hard", "top1"), ("distractors", "claims_fault"),
                     ("change_overlap_faults", "claims_fault")):
            c = comps.get(f"{g}|{m}|{pair}")
            if c:
                rows.append({"model": model, "pair": pair, "group": g, "metric": m, "difference": c["point"],
                             "95% CI": f"[{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]", "verdict": c["verdict"]})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption("On distractors the sign is flipped so that 'supported' means fewer false fault claims. "
               "'GNN+CL vs GNN' compares the GNN with and without the change-log rule; the agent pairs are model-specific.")

    st.markdown("**Haiku 5.5 against Opus 5.5** (descriptive, not pre-registered; paired over the same test incidents)")
    cmp = load_json("phase6/test_haiku_vs_opus.json")["variants"][v]["comparisons"]
    st.dataframe(pd.DataFrame([{"comparison": k, "difference": c["point"], "95% CI": f"[{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]",
                                "verdict": c["verdict"]} for k, c in cmp.items()]), width="stretch", hide_index=True)
    cost = {f"Haiku {c}": x for c, x in haiku["variants"][v]["agent_cost_latency_groundedness"].items()}
    cost.update({f"Opus {c}": x for c, x in opus["variants"][v]["agent_cost_latency_groundedness"].items()})
    st.markdown("**Cost, latency and groundedness per run**")
    st.dataframe(pd.DataFrame(cost).T[["mean_usd", "mean_seconds", "ungrounded_item_rate", "ungrounded_item_rate_strict", "forced_submissions"]],
                 width="stretch")
    st.caption("Groundedness checks that cited ids and numbers appear in that run's tool outputs (strict: numbers verbatim). "
               "It does not check the reasoning that links them.")

    dev_path = RES / "phase7" / "aarnet_dev_summary.json"
    if dev_path.exists():
        st.markdown("**AARNet dev traces** (reported separately; seen during development, not held-out evidence)")
        d = load_json("phase7/aarnet_dev_summary.json")
        st.dataframe(pd.DataFrame(d["by_variant"][v]).T, width="stretch")
        st.caption(f"{d['n_incidents']} incidents, ${d['cost_usd']:.2f} for both models. Too few to compare methods.")


def limitations():
    st.subheader("What this is, and what it is not")
    st.markdown(f"""
- **{HONESTY}**
- **Synthetic telemetry.** Utilisation, latency, loss, alerts and the change log come from our simulator, with faults
  and maintenance we inject. Real networks have messier telemetry, correlated failures and incomplete inventories.
- **Small public topologies.** AARNet (19 PoPs, 24 links) is the demo network; the test networks are seven other
  public topologies. AARNet was used during development, so its numbers are dev, not held-out.
- **What the results say (test).** The GNN carries most of the localisation. Haiku 5.5 with tools matches it
  (inconclusive), and Opus 5.5 with tools beats it on faults top-3. The agents' low false-alarm rate on maintenance
  windows comes from reading the change log; a one-line rule that silences the GNN on logged changes does as well.
  The agents also miss more real faults that coincide with logged changes than the plain GNN does.
- **Expectations that failed on test.** A1 beating A0 on hard faults held only without status events, and A2
  beating A1 on hard faults did not hold in either variant, although both held on validation.
- **Groundedness is a narrow check.** It confirms cited ids and numbers exist in the tool outputs, not that the
  explanation is right.
- **Runbooks favour the agent.** We wrote them ourselves, with knowledge of the simulator's fault types.
- Full details, pre-registration and every number's source: `docs/decision_log.md` and `results/`.
""")


st.set_page_config(page_title="NetTwin demo", layout="wide")
st.title("NetTwin")
st.caption("A knowledge-graph digital twin of a public backbone, with GNN localisation and an LLM root-cause agent. "
           "Offline demo from saved results.")
page = st.sidebar.radio("Page", ["Incident explorer", "Test results", "Limitations"], key="page")
st.sidebar.warning(HONESTY)
{"Incident explorer": explorer, "Test results": results, "Limitations": limitations}[page]()
