"""NetTwin demo dashboard (Phase 7). Runs offline from committed results; no data folder and no API key.

    streamlit run app/streamlit_app.py

Incident explorer: AARNet (dev split, seen during development) incidents with the alert timeline, the change log, the
GNN's top-ranked elements and the saved tool traces of Haiku 5.5 A2 and Opus 5.5 A2 next to the ground truth.
Test results: the Phase 6 test table from `results/phase6/`. All telemetry is simulated on public network topologies;
no real operator data is used.
"""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
STEP_MIN = 5
HONESTY = ("All telemetry is simulated (by the NetTwin simulator) on public network topologies (Internet Topology Zoo, "
           "SNDlib). No real operator data is used, so results show behaviour in this simulator, not on a production network.")
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
        silenced = ref == "GNN+CL" and not sc.get("claims_fault")
        top = "" if ref == "B3+" else "(silenced by the change-log rule)" if silenced else view["gnn_top"][0]["element"]
        rows.append({"method": ref, "claims fault": sc.get("claims_fault"), "root cause": top,
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
    st.caption(HONESTY + " 167 test incidents on seven held-out public topologies: 120 main-sample faults, 30 maintenance distractors and "
               "17 extra faults from the change-overlap stratum. The 40 change-overlapping faults are those 17 plus 23 of the 120. 95% CIs from 1000 incident bootstrap resamples.")
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
                prereg = model == "Haiku" or pair == "A2 vs A0"
                rows.append({"model": model, "pair": pair, "pre-registered": "yes" if prereg else "no (descriptive)",
                             "group": g, "metric": m, "difference": c["point"],
                             "95% CI": f"[{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]", "verdict": c["verdict"]})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption("On distractors the sign is flipped so that 'supported' means fewer false fault claims. "
               "'GNN+CL vs GNN' compares the GNN with and without the change-log rule; the agent pairs are model-specific. "
               "For Opus 5.5 only A2 vs A0 was pre-registered; its other pairs are descriptive (audit M1).")

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


def forecasting():
    st.subheader("Capacity forecasting (RQ5): will a link exceed 80% utilisation in the next 60 minutes?")
    st.caption(HONESTY + " Predictions every 30 minutes for every link, from observed utilisation only (no clock). "
               "Test scored once with models and thresholds frozen on validation. 95% CIs from 1000 scenario bootstrap resamples.")
    path = RES / "phase8_test.json"
    if not path.exists():
        st.info("results/phase8_test.json is not there yet: the test set has not been scored.")
        return
    t = load_json("phase8_test.json")
    names = {"persistence": "Persistence (above 0.8 now)", "seasonal_naive": "Seasonal-naive (same hour yesterday)",
             "lightgbm": "LightGBM on lag features", "gnn_head": "Frozen GNN encoder + small head"}
    classes = {"all": "all windows", "fault_free": "fault-free", "fault": "fault", "maintenance": "maintenance"}
    split = st.radio("Split", ["test", "dev"], horizontal=True, key="fc_split",
                     format_func=lambda s: "test (210 scenarios, 7 held-out topologies)" if s == "test" else "AARNet dev (30 scenarios, seen during development, reported separately)")
    x = t["splits"][split]

    def fmt(m):
        if m["value"] is None:
            return "n/a"
        lo, hi = m["ci"]
        return f"{m['value']:.3f} [{lo:.2f}, {hi:.2f}]"

    rows = []
    for m, label in names.items():
        for c, cl in classes.items():
            r = x["metrics"][m][c]
            rows.append({"method": label, "window": cl, "precision": fmt(r["precision"]), "recall": fmt(r["recall"]),
                         "F1": fmt(r["f1"]), "breaches": r["n_breaches"], "predicted": r["n_pairs_predicted_breach"]})
    df = pd.DataFrame(rows)
    pick = st.multiselect("Windows", list(classes.values()), default=list(classes.values()), key="fc_windows")
    st.dataframe(df[df.window.isin(pick)], width="stretch", hide_index=True)
    st.markdown("**Paired comparisons** (difference, 95% CI, verdict)")
    st.dataframe(pd.DataFrame([{"comparison": k, "difference": c["point"], "95% CI": f"[{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]",
                                "verdict": c["verdict"]} for k, c in x["comparisons"].items() if c["point"] is not None]), width="stretch", hide_index=True)
    st.caption("Window classes use ground truth only for scoring: 'fault' if a fault window overlaps the hour before or "
               "after the prediction time, 'maintenance' if only logged maintenance does, 'fault-free' otherwise. "
               "The maintenance class has few breaches (114 on test, none on dev) and carries no conclusion; its CIs drop "
               "resamples that draw no maintenance breach.")


def kgqa():
    st.subheader("KG question answering (Phase 11): can a small model answer exact questions about the inventory graph?")
    st.caption(HONESTY + " 252 template test questions on seven held-out public topologies with PoPs renamed P01, P02 and so on, "
               "exact answers from the KG code. All conditions on Haiku 5.5. Test scored once, as pre-registered. "
               "95% CIs from 1000 question bootstrap resamples. The renaming hides PoP names, not the real link lengths "
               "and metrics, and no run with real names was made.")
    path = RES / "phase11" / "test" / "summary.json"
    if not path.exists():
        st.info("results/phase11/test/summary.json is not there yet: the test set has not been scored.")
        return
    s = load_json("phase11/test/summary.json")
    names = {"Q0": "Q0 closed book (no graph)", "Q1": "Q1 BM25 top-50 graph facts", "Q2": "Q2 KG query tools"}
    metric = st.radio("Metric", ["f1", "em"], horizontal=True, key="qa_metric",
                      format_func=lambda m: "set F1" if m == "f1" else "exact match")
    groups = [("all", "all questions")] + [(g, g.split("=")[1].replace("_", "-")) for g in s["metrics"] if g.startswith("type=")] \
        + [(g, g.split("=")[1] + " graphs") for g in s["metrics"] if g.startswith("size=")]
    rows = []
    for g, label in groups:
        m = s["metrics"][g]
        row = {"group": label, "n": m["Q0"][metric]["n"]}
        for c in s["conditions"]:
            x = m[c][metric]
            row[names[c]] = f"{x['value']:.3f} [{x['ci'][0]:.2f}, {x['ci'][1]:.2f}]"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows).set_index("group"), width="stretch")
    st.markdown("**Cost and tool calls per question**")
    st.dataframe(pd.DataFrame({names[c]: x for c, x in s["cost"].items()}).T, width="stretch")
    st.markdown("**Pre-registered expectations**")
    st.dataframe(pd.DataFrame([{"id": k, "expectation": e["expectation"], "met on test": "yes" if e["met"] else "no"}
                               for k, e in s["expectations"].items()]), width="stretch", hide_index=True)
    st.markdown("**Paired comparisons** (difference, 95% CI, verdict)")
    st.dataframe(pd.DataFrame([{"comparison": k.replace("|", " "), "difference": c["point"],
                                "95% CI": f"[{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]", "verdict": c["verdict"]}
                               for k, c in s["comparisons"].items() if "template=" not in k and f"|{metric}|" in k]),
                 width="stretch", hide_index=True)
    if (RES / "phase11" / "test" / "clustered.json").exists():
        cl = load_json("phase11/test/clustered.json")
        st.markdown("**Network-level check** (descriptive, not pre-registered): the CIs above treat questions as independent, "
                    "but questions on one network share its graph. Here whole networks are resampled instead (set F1).")
        st.dataframe(pd.DataFrame([{"group": g, "networks": r["n_networks"], "questions": r["n_questions"],
                                    "comparison": k.split("|")[1], "difference": c["point"],
                                    "95% CI": f"[{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]" if c["ci"][0] is not None else "n/a",
                                    "verdict": c["verdict"]} for g, r in cl["groups"].items() for k, c in r["comparisons"].items()]),
                     width="stretch", hide_index=True)
        st.caption("With 7 networks these CIs are rough. The medium group is a single network, so it has no network-level CI.")

    st.markdown("**Browse the test questions**")
    qs = [json.loads(l) for l in (RES / "phase11" / "questions_test.jsonl").read_text().splitlines()]
    log = {}
    for l in (RES / "phase11" / "test" / "log.jsonl").read_text().splitlines():
        r = json.loads(l)
        log[(r["qid"], r["condition"])] = r
    templates = sorted({q["template"] for q in qs})
    t = st.selectbox("Template", templates, key="qa_template")
    pick = [q for q in qs if q["template"] == t]
    qid = st.selectbox("Question", [q["qid"] for q in pick], key="qa_qid",
                       format_func=lambda i: f"{i} ({next(q['network'] for q in pick if q['qid'] == i)})")
    q = next(q for q in pick if q["qid"] == qid)
    st.markdown(f"> {q['question']}")
    st.markdown(f"**Gold answer:** `{q['answer']}`")
    for c in s["conditions"]:
        r = log.get((qid, c))
        if not r:
            continue
        ans = r["answer"] or {}
        shown = ans.get("number") if q["answer_kind"] == "number" and ans.get("number") is not None else ans.get("items")
        st.markdown(f"**{names[c]}:** `{shown}` (exact {r['em']:.0f}, F1 {r['f1']:.2f}, {r['n_tool_calls']} tool calls, ${r['cost_usd']:.4f})")
        if c == "Q2" and r["calls"]:
            with st.expander("Q2 tool calls"):
                for k in r["calls"]:
                    st.code(f"{k['tool']}({json.dumps(k['input'])})" + ("  -> error" if k["error"] else ""), language=None)


REPLAY_NAMES = {"GNN": "GNN (every alarm episode pages)", "GNN+CL": "GNN + change-log rule", "Agent": "Haiku 5.5 triage agent"}
REPLAY_METRICS = [("pages_per_day", "pages per scenario-day", 2), ("false_pages_per_day", "false pages per scenario-day", 2),
                  ("detection_rate", "fault detection rate", 3), ("time_to_detect_min", "time to detect (min)", 1),
                  ("top1_at_escalation", "top-1 at escalation (own detected faults)", 3),
                  ("top3_at_escalation", "top-3 at escalation (own detected faults)", 3),
                  ("top3_all_faults", "top-3 over all faults", 3), ("missed_during_maintenance", "missed faults during maintenance", 3),
                  ("cost_per_day_usd", "cost per scenario-day (USD)", 4)]
HIGHER_BETTER = {"detection_rate", "top1_at_escalation", "top3_at_escalation", "top3_all_faults"}


def favours(key, verdict):
    """Which policy a paired comparison 'metric|A vs B' favours, given which direction is better for the metric."""
    name, pair = key.split("|")
    a, b = pair.split(" vs ")
    if verdict not in ("supported", "contradicted"):
        return "neither (inconclusive)"
    return a if (verdict == "supported") == (name in HIGHER_BETTER) else b


def replay_timeline(sc):
    """One scenario: true faults and logged changes as bands, GNN alarm steps, episodes, agent decisions and pages."""
    fig = go.Figure()
    lanes = ["true faults", "logged changes", "GNN alarms", "agent decisions", "pages: Agent", "pages: GNN+CL", "pages: GNN"]
    for f in sc["faults"]:
        fig.add_trace(go.Bar(base=[hours(f["t_start"])], x=[max(hours(f["t_end"] - f["t_start"]), 0.1)], y=["true faults"], orientation="h",
                             marker_color=COL["truth"], showlegend=False, hoverinfo="text",
                             hovertext=f"{f['id']} {f['type']} on {f['root_cause']}, hours {hours(f['t_start']):.1f} to {hours(f['t_end']):.1f}"))
    for c in sc["changes"]:
        fig.add_trace(go.Bar(base=[hours(c["t_start"])], x=[max(hours(c["t_end"] - c["t_start"]), 0.1)], y=["logged changes"], orientation="h",
                             marker_color=COL["change"], showlegend=False, hoverinfo="text",
                             hovertext=f"{c['id']} {c['action']} on {c['target']}"))
    fig.add_trace(go.Scatter(x=[hours(t) for t in sc["alarm_steps"]], y=["GNN alarms"] * len(sc["alarm_steps"]), mode="markers",
                             marker=dict(color=COL["gnn"], size=5, symbol="line-ns-open"), showlegend=False, hoverinfo="x"))
    sym = {"escalate": ("triangle-up", COL["truth"]), "dismiss_maintenance": ("square", COL["change"]),
           "dismiss_noise": ("x", "#555555"), "wait": ("circle-open", COL["haiku"])}
    for d, (symbol, color) in sym.items():
        rows = [r for r in sc["decisions"] if r["decision"] == d]
        if rows:
            fig.add_trace(go.Scatter(x=[hours(r["t"]) for r in rows], y=["agent decisions"] * len(rows), mode="markers", name=d,
                                     marker=dict(symbol=symbol, size=10, color=color), hoverinfo="text",
                                     hovertext=[f"{r['episode_id']} round {r['round']}: {d} ({r['root_cause']}). {r['reason']}" for r in rows]))
    for m in ("GNN", "GNN+CL", "Agent"):
        ps = sc["pages"][m]
        fig.add_trace(go.Scatter(x=[hours(p["t"]) for p in ps], y=[f"pages: {m}"] * len(ps), mode="markers", showlegend=False,
                                 marker=dict(symbol="diamond", size=9, color={"GNN": COL["gnn"], "GNN+CL": "#8c564b", "Agent": COL["haiku"]}[m]),
                                 hoverinfo="text", hovertext=[f"{m} page, episode {p['episode_id']}" for p in ps]))
    fig.update_layout(height=360, barmode="overlay", margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=-0.25),
                      xaxis_title="hours since the scenario start (replay starts at hour 24)",
                      yaxis=dict(categoryorder="array", categoryarray=lanes[::-1]))
    return fig


def replay():
    st.subheader("Real-time replay (Phase 12): alarms, triage and pages without an oracle incident window")
    st.caption(HONESTY + " Each test scenario is replayed step by step from hour 24. The frozen GNN raises alarms, repeated "
               "alarms on one element merge into an episode, and a policy decides which episodes page an operator. The "
               "agent sees only data up to the step it is woken. Test scored once, as pre-registered. 95% CIs from 1000 "
               "scenario bootstrap resamples.")
    path = RES / "phase12" / "test" / "summary.json"
    if not path.exists():
        st.info("results/phase12/test/summary.json is not there yet: the test set has not been scored.")
        return
    s = load_json("phase12/test/summary.json")
    st.markdown(f"**{s['n_scenarios']} test scenarios, {s['scenario_days']:.0f} scenario-days, {s['n_faults']} observable faults, "
                f"{s['n_episodes']} GNN alarm episodes**")
    rows = []
    for key, label, nd in REPLAY_METRICS:
        row = {"metric": label}
        for m, name in REPLAY_NAMES.items():
            x = s["metrics"][m][key]
            row[name] = "n/a" if x["value"] is None else f"{x['value']:.{nd}f} [{x['ci'][0]:.{nd}f}, {x['ci'][1]:.{nd}f}]"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows).set_index("metric"), width="stretch")
    st.caption("Detection counts any page during a fault, whatever element it names. The two 'at escalation' rows average "
               "each policy over its own detected faults, which differ (see the same-faults comparison below).")
    st.markdown("**Paired comparisons** (A minus B, 95% CI; the verdict is about the sign: supported means the CI is above 0, "
                "contradicted below 0. Whether that is good depends on the metric, so the last two columns say which way is "
                "better and which policy the result favours.)")
    st.dataframe(pd.DataFrame([{"comparison": k.replace("|", ": "), "difference": c["point"],
                                "95% CI": f"[{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]" if c["ci"][0] is not None else "n/a",
                                "verdict": c["verdict"], "better": "higher" if k.split("|")[0] in HIGHER_BETTER else "lower",
                                "favours": favours(k, c["verdict"])} for k, c in s["comparisons"].items()]),
                 width="stretch", hide_index=True)
    if (RES / "phase12" / "test" / "descriptive.json").exists():
        d = load_json("phase12/test/descriptive.json")
        st.markdown("**Descriptive, not pre-registered** (from the saved test run, no API call)")
        cr = d["correct_root_detection"]
        st.dataframe(pd.DataFrame([{"metric": "detected with the root in the page's top 3", **{
            REPLAY_NAMES[m]: f"{cr[m]['value']:.3f} [{cr[m]['ci'][0]:.3f}, {cr[m]['ci'][1]:.3f}]" for m in REPLAY_NAMES}}]).set_index("metric"),
            width="stretch")
        st.caption("A fault counts only if some page during it names the true root among its top 3 (higher is better). "
                   + "; ".join(f"{k}: {v['point']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}], {v['verdict']}"
                               for k, v in cr.items() if " minus " in k) + ".")
        mrows = []
        for pair, r in d["matched_faults"].items():
            a, b = pair.split(" vs ")
            for c in ("top1", "top3"):
                x = r[f"{c}|{a} minus {b}"]
                mrows.append({"pair": pair, "faults both detect": r["n_both_detect"], "metric": c.replace("top", "top-"),
                              a: r[f"{c}|{a}"]["value"], b: r[f"{c}|{b}"]["value"], "difference": x["point"],
                              "95% CI": f"[{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]", "verdict": x["verdict"]})
        st.markdown("Root cause on the same faults: top-1 and top-3 at the first detecting page, on the faults both policies detect (higher is better).")
        st.dataframe(pd.DataFrame(mrows), width="stretch", hide_index=True)
    st.markdown("**Pre-registered expectations**")
    st.dataframe(pd.DataFrame([{"id": k, "expectation": e["expectation"], "met on test": "yes" if e["met"] else "no"}
                               for k, e in s["expectations"].items()]), width="stretch", hide_index=True)
    a = s["agent"]
    st.caption(f"Agent: {a['wakes']} wakes ({a['rechecks']} re-checks after a wait), first decisions {a['first_decisions']}, "
               f"{a['mean_tool_calls']} tool calls per wake on average, ${a['cost_usd']:.2f} in all.")

    st.markdown("**Scenario timeline**")
    tl = load_json("phase12/test/timeline.json")
    topo = st.selectbox("Network", sorted({x["topology"] for x in tl}), key="rp_topo")
    pick = [x for x in tl if x["topology"] == topo]
    sid = st.selectbox("Scenario", [x["scenario_id"] for x in pick], key="rp_scen",
                       format_func=lambda i: f"{i} ({len(next(x for x in pick if x['scenario_id'] == i)['faults'])} faults)")
    sc = next(x for x in pick if x["scenario_id"] == sid)
    st.plotly_chart(replay_timeline(sc), width="stretch")
    st.caption("Red bands are true faults (ground truth, shown only for scoring); grey bands are logged changes. Hover a "
               "marker for the agent's reason or a page's episode.")


SCOPE_NAMES = {"A2": "A2 (Phase 6 agent)", "A3": "A3 (+ change-scope tool)", "GNN": "GNN", "GNN+CL": "GNN + change-log rule",
               "GNN+CL+scope": "GNN + rule + scope check", "SCOPE": "scope tool alone"}
SCOPE_ROWS = [("change_overlap", "claims_fault", "recall, faults overlapping a logged change"),
              ("change_overlap", "top3", "top-3, faults overlapping a logged change"),
              ("other_fault", "claims_fault", "recall, other faults"), ("other_fault", "top3", "top-3, other faults"),
              ("distractors", "claims_fault", "fault claims on change-only windows (lower is better)")]


def scope():
    st.subheader("Change-scope tool (Phase 13): a pre-registered follow-up on faults hidden by planned work")
    st.caption(HONESTY + " Phases 6 and 12 found that the agent misses real faults that overlap a logged drain or planned "
               "outage. Phase 13 gives it one more tool: a knowledge-graph check of what each logged change can explain, "
               "listing the observed symptoms outside it. A3 is the frozen Phase 6 A2 (Haiku 5.5) plus that tool; both ran "
               "once on fresh test incidents never drawn in Phase 6. Scored on oracle incident windows, like Phase 6, and "
               "not re-tested in the Phase 12 live replay, for budget reasons. 95% CIs from 1000 scenario resamples.")
    path = RES / "phase13" / "test" / "summary.json"
    if not path.exists():
        st.info("results/phase13/test/summary.json is not there yet: the test set has not been scored.")
        return
    s = load_json("phase13/test/summary.json")
    st.markdown(f"**{s['n_incidents']} test incidents from {s['n_scenarios']} scenarios: {s['strata']['change_overlap']} faults "
                f"overlapping a logged change, {s['strata']['other_fault']} other faults, {s['strata']['distractor']} change-only windows**")
    rows = []
    for g, metric, label in SCOPE_ROWS:
        row = {"metric": label}
        for m, name in SCOPE_NAMES.items():
            x = s["metrics"].get(m, {}).get(g, {}).get(metric)
            row[name] = "" if x is None else f"{x['value']:.3f} [{x['ci'][0]:.3f}, {x['ci'][1]:.3f}]"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows).set_index("metric"), width="stretch")
    st.caption("For 'scope tool alone' the value is the share of windows where it listed an unexplained symptom.")
    keep = ("A3 vs A2", "A3 vs GNN", "A3 vs GNN+CL", "GNN+CL+scope vs GNN+CL")
    st.markdown("**Paired comparisons** (A minus B, 95% CI; on change-only windows the sign is flipped so that supported "
                "always favours A)")
    st.dataframe(pd.DataFrame([{"comparison": k.replace("|", ": "), "difference": c["point"],
                                "95% CI": f"[{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]" if c["ci"][0] is not None else "n/a",
                                "verdict": c["verdict"]}
                               for k, c in s["comparisons"].items()
                               if k.split("|")[0] in ("change_overlap", "other_fault", "distractors") and k.split("|")[1] in ("claims_fault", "top3")
                               and k.split("|")[2] in keep]), width="stretch", hide_index=True)
    st.markdown("**Pre-registered expectations**")
    st.dataframe(pd.DataFrame([{"expectation": k, "met on test": "yes" if v else "no"} for k, v in s["expectations"].items()]),
                 width="stretch", hide_index=True)
    if (RES / "phase13" / "test" / "descriptive.json").exists():
        d = load_json("phase13/test/descriptive.json")["change_overlap"]
        st.markdown("**Descriptive, not pre-registered:** why the tool did not help on test")
        st.dataframe(pd.DataFrame([{"fault type": t, "faults": v["n"], "tool flagged": v["tool_flagged"]} for t, v in d["by_type"].items() if v["n"]]),
                     width="stretch", hide_index=True)
        f, n = d["tool_flagged"], d["tool_flagged_nothing"]
        st.caption(f"On the {f['n']} windows the tool flagged, A2 and A3 claimed {f['A2_fault_claims']} and {f['A3_fault_claims']} "
                   f"faults; on the {n['n']} it did not, {n['A2_fault_claims']} and {n['A3_fault_claims']}. A3 answered planned "
                   f"maintenance for {d['A3_said_planned_maintenance']} of these faults, A2 for {d['A2_said_planned_maintenance']}.")
    a = s["agents"]
    st.caption(f"Phase 13 spend ${s['phase13_spend_usd']:.2f} of a $2.00 guard. A3 called the tool in "
               f"{a['A3']['called_scope_share_change_windows']:.0%} of its runs on windows with a logged change.")


def limitations():
    st.subheader("What this is, and what it is not")
    st.markdown(f"""
- **{HONESTY}**
- **Synthetic telemetry.** Utilisation, latency, loss, alerts and the change log come from our simulator, with faults
  and maintenance we inject. Real networks have messier telemetry, correlated failures and incomplete inventories.
- **Small public topologies.** AARNet (19 PoPs, 24 links) is the demo network; the test networks are seven other
  public topologies. AARNet was used during development, so its numbers are dev, not held-out.
- **What the results say (test, pre-registered).** The GNN carries most of the localisation. Haiku 5.5 with tools
  shows no clear difference from it at this sample size (faults top-3 +0.044 [-0.006, +0.095] with status events,
  -0.022 [-0.079, +0.034] without). The agents' low false-alarm rate on maintenance windows comes from reading the
  change log: a one-line rule that silences the GNN on logged changes also claims a fault in 0 of 30 such windows
  (a description, not a tested equivalence). The agents also miss more real faults that coincide with logged changes
  than the plain GNN does.
- **Descriptive, not pre-registered.** Opus 5.5 with tools is above the 3-seed GNN on faults top-3. Against GNN seed 0,
  the checkpoint its tool uses, this holds only for faults top-3 with status events (+0.078 [+0.021, +0.142]).
  The Haiku against Opus file makes 68 comparisons with no multiple-comparison correction.
- **Expectations that failed on test.** A1 beating A0 on hard faults held only without status events, and A2
  beating A1 on hard faults did not hold in either variant, although both held on validation.
- **Oracle incident window (Phases 4 to 6 only).** Their localisation is scored on a window that starts and ends with
  the true fault, so no method had to find it. The real-time replay page (Phase 12) removes it: the GNN alarms step by
  step and the agent decides which alarms page an operator.
- **Real-time replay caveats (Phase 12).** Detection counts any page during a fault, whatever element it names; the
  replay page also shows detection with the true root in the page's top 3. The change log shows each started
  change's planned end, which the simulator always keeps and real changes often overrun, so this flatters the
  change-log rule and the agent a little.
- **Change-scope tool (Phase 13).** A KG check of what a logged change can explain did not raise the agent's recall
  of faults that overlap a logged change on 15 fresh test faults (inconclusive, and lower as a point estimate). The
  tool missed most of the subtle fault types there. It was scored on oracle windows and not re-tested in the replay.
- **Groundedness is a narrow check.** It confirms cited ids and numbers exist in the tool outputs, not that the
  explanation is right.
- **Runbooks favour the agent.** We wrote them ourselves, with knowledge of the simulator's fault types.
- Full details, pre-registration and every number's source: `docs/decision_log.md` and `results/`.
""")


st.set_page_config(page_title="NetTwin demo", layout="wide")
st.title("NetTwin")
st.caption("A knowledge-graph digital twin of a public backbone, with GNN localisation and an LLM root-cause agent. "
           "Offline demo from saved results.")
page = st.sidebar.radio("Page", ["Incident explorer", "Test results", "Capacity forecasting", "KG question answering (Phase 11)", "Real-time replay (Phase 12)", "Change-scope tool (Phase 13)", "Limitations"], key="page")
st.sidebar.warning(HONESTY)
{"Incident explorer": explorer, "Test results": results, "Capacity forecasting": forecasting, "KG question answering (Phase 11)": kgqa, "Real-time replay (Phase 12)": replay, "Change-scope tool (Phase 13)": scope, "Limitations": limitations}[page]()
