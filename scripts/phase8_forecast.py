"""Phase 8: capacity forecasting (RQ5). Breach = observed link utilisation above 0.8 within the next 60 minutes.

    python scripts/phase8_forecast.py            # train on train, tune on the validation subset, freeze
    python scripts/phase8_forecast.py --test     # score test once (and AARNet dev, separately)

Methods: persistence, seasonal-naive (same horizon yesterday), LightGBM on lag features, and the frozen Phase 5 GNN
encoder (gat-h32-s0, no retraining) with a small MLP head. The plan, the tuning budget and the expectations are in the
Phase 8 pre-registration in docs/decision_log.md. `--test` refuses if that entry is missing, if any hash in
results/phase8_frozen.json differs, or if results/phase8_test.json exists (the test set is scored once).
"""
import lightgbm as lgb  # noqa: I001  (before torch: torch's libomp crashes multithreaded LightGBM otherwise)

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score

from nettwin.evaluation import boot_weights, ci, verdict
from nettwin.forecast import (CLASSES, LAG_FEATURES, Head, counts, gnn_embedding, grid, head_scalars, lag_features, persistence,
                              seasonal_naive, targets, window_class)
from nettwin.gnn.data import graph_for
from nettwin.gnn.method import gnn_method, sha256
from nettwin.observed import freeze_cfg, load_observed
from nettwin.splits import guard

ROOT = Path(__file__).resolve().parents[1]
VARIANT = "with_status"
ENCODER = "gat-h32-s0"
MDIR = ROOT / "data" / "models" / "phase8"
FROZEN = ROOT / "results" / "phase8_frozen.json"
VAL_OUT = ROOT / "results" / "phase8_val.json"
TEST_OUT = ROOT / "results" / "phase8_test.json"
PREREG = "### 2026-10-31: Phase 8 pre-registration"
N_TRAIN, N_VAL = 300, 120
LGB_GRID = [{"num_leaves": nl, "learning_rate": lr} for nl in (31, 63) for lr in (0.05, 0.1)]
HEAD_LRS = (1e-3, 3e-3)
METHODS = ("persistence", "seasonal_naive", "lightgbm", "gnn_head")
CODE = ("src/nettwin/forecast.py", "scripts/phase8_forecast.py")


def cfg_and_encoder():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    cfg_s = freeze_cfg(cfg)
    p5 = json.loads((ROOT / "results" / "phase5_frozen.json").read_text())
    rel = f"data/models/phase5/{ENCODER}.pt"
    if sha256(ROOT / rel) != p5["files"][rel]:
        sys.exit(f"refusing: {rel} differs from results/phase5_frozen.json")
    torch.set_num_threads(1)          # with lightgbm loaded, multithreaded torch segfaults too (libomp clash)
    enc = gnn_method(ENCODER, (ROOT / rel).read_bytes(), cfg_s).model
    for p in enc.parameters():
        p.requires_grad_(False)
    return cfg, cfg_s, enc


def scenario_ids(cfg, split, n=None, seed=None):
    root = ROOT / cfg["dataset"]["root"]
    entries = json.loads(guard(root / "truth" / split / VARIANT / "manifest.json", {split}).read_text())
    if n is not None:
        idx = np.random.default_rng(seed).choice(len(entries), n, replace=False)
        entries = [entries[i] for i in sorted(idx)]
    return root, entries


def scenario_data(root, split, entry, cfg_s, enc):
    """Everything one scenario contributes: inputs and labels per (prediction time, link), ground-truth classes."""
    obs = load_observed(guard(root / "observable" / split / VARIANT / entry["scenario_id"], {split}))
    times = grid(obs.util.shape[0])
    y, ok = targets(obs.util, times)
    g = graph_for(obs.topology, cfg_s)
    assert [l for l in g.elements[:g.n_links]] == obs.links, "link order differs between the graph and the telemetry"
    emb = gnn_embedding(enc, obs, g, times)
    return {"y": y, "ok": ok, "lag": lag_features(obs, times), "head": np.concatenate([emb, head_scalars(obs, times)], -1),
            "persistence": persistence(obs, times), "seasonal_naive": seasonal_naive(obs, times),
            "cls": window_class(entry, times)}


def stack(items, key):
    return np.concatenate([d[key][d["ok"]] for d in items])


def train_head(Xtr, ytr, Xva, yva, lr, seed=0, epochs=4, batch=4096):
    torch.manual_seed(seed)
    m = Head(Xtr.shape[1])
    m.mean.copy_(torch.from_numpy(Xtr.mean(0))); m.std.copy_(torch.from_numpy(Xtr.std(0) + 1e-6))
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    pw = torch.tensor((len(ytr) - ytr.sum()) / max(1, ytr.sum()), dtype=torch.float32)
    X, Y = torch.from_numpy(Xtr), torch.from_numpy(ytr.astype(np.float32))
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        m.train()
        for idx in np.array_split(rng.permutation(len(X)), max(1, len(X) // batch)):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(m(X[idx]), Y[idx], pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
    return m, head_prob(m, Xva)


@torch.no_grad()
def head_prob(m, X):
    m.eval()
    return torch.sigmoid(m(torch.from_numpy(X))).numpy()


def best_threshold(y, p):
    order = np.argsort(-p, kind="stable")
    ys = y[order]
    tp = np.cumsum(ys); fp = np.cumsum(~ys)
    f1 = 2 * tp / (tp + fp + ys.sum())
    k = int(np.argmax(f1))
    return float(p[order][k]), float(f1[k])


def prf(tp, fp, fn):
    with np.errstate(invalid="ignore", divide="ignore"):
        return tp / (tp + fp), tp / (tp + fn), 2 * tp / (2 * tp + fp + fn)


def tune():
    cfg, cfg_s, enc = cfg_and_encoder()
    t0 = time.time()
    data = {}
    for split, n, seed in (("train", N_TRAIN, 8008), ("val", N_VAL, 8009)):
        root, entries = scenario_ids(cfg, split, n, seed)
        data[split] = [scenario_data(root, split, e, cfg_s, enc) for e in entries]
        print(f"{split}: {len(entries)} scenarios loaded ({time.time() - t0:.0f}s)", flush=True)
    tr, va = data["train"], data["val"]
    ytr, yva = stack(tr, "y"), stack(va, "y")
    print(f"pairs: train {len(ytr)} ({ytr.mean():.4f} breach), val {len(yva)} ({yva.mean():.4f})", flush=True)
    out = {"note": "Phase 8 tuning on the validation subset only. Telemetry is synthetic; topologies are public.",
           "variant": VARIANT, "n_train_scenarios": N_TRAIN, "n_val_scenarios": N_VAL, "train_pairs": int(len(ytr)),
           "val_pairs": int(len(yva)), "train_breach_rate": round(float(ytr.mean()), 5), "val_breach_rate": round(float(yva.mean()), 5)}

    Ltr, Lva = stack(tr, "lag"), stack(va, "lag")
    runs = []
    for c in LGB_GRID:
        m = lgb.LGBMClassifier(n_estimators=400, subsample=1.0, colsample_bytree=1.0, min_child_samples=50, random_state=0,
                               verbose=-1, **c)
        m.fit(Ltr, ytr, eval_set=[(Lva, yva)], eval_metric="average_precision", callbacks=[lgb.early_stopping(30, verbose=False)])
        p = m.predict_proba(Lva)[:, 1]
        runs.append({"config": c, "best_iteration": int(m.best_iteration_), "val_ap": round(float(average_precision_score(yva, p)), 4), "_m": m, "_p": p})
        print(f"lightgbm {c}: val AP {runs[-1]['val_ap']} at {m.best_iteration_} rounds", flush=True)
    best = max(runs, key=lambda r: r["val_ap"])
    thr, f1 = best_threshold(yva, best["_p"])
    MDIR.mkdir(parents=True, exist_ok=True)
    best["_m"].booster_.save_model(str(MDIR / "lightgbm.txt"), num_iteration=best["best_iteration"])
    out["lightgbm"] = {"runs": [{k: v for k, v in r.items() if not k.startswith("_")} for r in runs], "selected": best["config"],
                       "threshold": thr, "val_f1_at_threshold": round(f1, 4), "features": list(LAG_FEATURES)}

    Htr, Hva = stack(tr, "head"), stack(va, "head")
    runs = []
    for lr in HEAD_LRS:
        m, p = train_head(Htr, ytr, Hva, yva, lr)
        runs.append({"lr": lr, "val_ap": round(float(average_precision_score(yva, p)), 4), "_m": m, "_p": p})
        print(f"gnn head lr {lr}: val AP {runs[-1]['val_ap']}", flush=True)
    best = max(runs, key=lambda r: r["val_ap"])
    thr_h, f1 = best_threshold(yva, best["_p"])
    torch.save(best["_m"].state_dict(), MDIR / "gnn_head.pt")
    out["gnn_head"] = {"runs": [{k: v for k, v in r.items() if not k.startswith("_")} for r in runs], "selected_lr": best["lr"],
                       "threshold": thr_h, "val_f1_at_threshold": round(f1, 4), "encoder": ENCODER, "n_inputs": int(Htr.shape[1])}

    # validation precision / recall of every method at its frozen rule (for reference; selection used AP and F1 above)
    preds = {"persistence": stack(va, "persistence"), "seasonal_naive": stack(va, "seasonal_naive"),
             "lightgbm": lgb.Booster(model_file=str(MDIR / "lightgbm.txt")).predict(Lva) >= thr, "gnn_head": best["_p"] >= thr_h}
    out["val_rules"] = {}
    for k, p in preds.items():
        tp, fp, fn = int((p & yva).sum()), int((p & ~yva).sum()), int((~p & yva).sum())
        pr, rc, f = prf(tp, fp, fn)
        out["val_rules"][k] = {"precision": round(float(pr), 4), "recall": round(float(rc), 4), "f1": round(float(f), 4)}
    out["seconds"] = round(time.time() - t0, 1)
    VAL_OUT.write_text(json.dumps(out, indent=1) + "\n")
    frozen = {"note": "sha256 of what scripts/phase8_forecast.py --test loads; it refuses on any difference",
              "thresholds": {"lightgbm": thr, "gnn_head": thr_h}, "encoder": ENCODER, "head_inputs": int(Htr.shape[1]),
              "files": {rel: sha256(ROOT / rel) for rel in ["data/models/phase8/lightgbm.txt", "data/models/phase8/gnn_head.pt",
                                                             f"data/models/phase5/{ENCODER}.pt", *CODE]}}
    FROZEN.write_text(json.dumps(frozen, indent=1) + "\n")
    print(json.dumps(out["val_rules"], indent=1))
    print(f"wrote {VAL_OUT.relative_to(ROOT)} and {FROZEN.relative_to(ROOT)} ({out['seconds']}s)")


def score_split(cfg, cfg_s, enc, split, booster, head, thr):
    root, entries = scenario_ids(cfg, split)
    rows = []
    for k, e in enumerate(entries):
        d = scenario_data(root, split, e, cfg_s, enc)
        shp = d["y"].shape
        pred = {"persistence": d["persistence"], "seasonal_naive": d["seasonal_naive"],
                "lightgbm": booster.predict(d["lag"].reshape(-1, len(LAG_FEATURES))).reshape(shp) >= thr["lightgbm"],
                "gnn_head": head_prob(head, d["head"].reshape(-1, d["head"].shape[-1])).reshape(shp) >= thr["gnn_head"]}
        rows.append({"scenario_id": e["scenario_id"], "topology": e["topology"],
                     **{m: counts(p, d["y"], d["ok"], d["cls"]) for m, p in pred.items()}})
        if (k + 1) % 50 == 0:
            print(f"{split}: {k + 1}/{len(entries)}", flush=True)
    return rows


def summarise(rows, pairs):
    """Precision, recall, F1 with scenario-bootstrap CIs per method and class, and paired verdicts."""
    S = len(rows)
    Wt = boot_weights(S, 1000, 42)
    out, boots = {}, {}
    for m in METHODS:
        for c in CLASSES:
            tp, fp, fn = (np.array([r[m][c][i] for r in rows], float) for i in range(3))
            pr, rc, f = prf(Wt @ tp, Wt @ fp, Wt @ fn)
            boots[(m, c)] = {"precision": pr, "recall": rc, "f1": f}
            out.setdefault(m, {})[c] = {"n_breaches": int(tp.sum() + fn.sum()), "n_pairs_predicted_breach": int(tp.sum() + fp.sum()),
                                        **{k: {"value": None if not np.isfinite(v[0]) else round(float(v[0]), 4), "ci": ci(v[1:])}
                                           for k, v in boots[(m, c)].items()}}
    comps = {}
    for a, b, c, metric in pairs:
        comps[f"{c}|{metric}|{a} vs {b}"] = verdict(boots[(a, c)][metric] - boots[(b, c)][metric])
    return out, comps


PAIRS = [("lightgbm", "persistence", "all", "f1"), ("lightgbm", "seasonal_naive", "all", "f1"), ("gnn_head", "lightgbm", "all", "f1"),
         ("gnn_head", "persistence", "all", "f1"), ("lightgbm", "seasonal_naive", "maintenance", "recall"),
         ("lightgbm", "persistence", "all", "recall"), ("lightgbm", "persistence", "all", "precision"),
         ("lightgbm", "persistence", "fault", "f1"), ("gnn_head", "lightgbm", "fault", "f1")]


def test():
    if PREREG not in (ROOT / "docs" / "decision_log.md").read_text():
        sys.exit("refusing: the Phase 8 pre-registration is not in docs/decision_log.md")
    if TEST_OUT.exists():
        sys.exit(f"refusing: {TEST_OUT.relative_to(ROOT)} exists; the test set is scored once")
    frozen = json.loads(FROZEN.read_text())
    bad = [rel for rel, h in frozen["files"].items() if not (ROOT / rel).exists() or sha256(ROOT / rel) != h]
    if bad:
        sys.exit(f"refusing: sha256 mismatch or missing file: {bad}")
    cfg, cfg_s, enc = cfg_and_encoder()
    booster = lgb.Booster(model_file=str(MDIR / "lightgbm.txt"))
    head = Head(frozen["head_inputs"])
    head.load_state_dict(torch.load(MDIR / "gnn_head.pt"))
    t0 = time.time()
    out = {"note": "Phase 8 test scoring, run once with the frozen models and validation thresholds. Telemetry is synthetic; "
                   "topologies are public; not Telstra's network. AARNet (dev, seen during development) is reported separately.",
           "variant": VARIANT, "frozen": frozen, "splits": {}}
    for split in ("test", "dev"):
        rows = score_split(cfg, cfg_s, enc, split, booster, head, frozen["thresholds"])
        metrics, comps = summarise(rows, PAIRS)
        out["splits"][split] = {"n_scenarios": len(rows), "metrics": metrics, "comparisons": comps}
    out["seconds"] = round(time.time() - t0, 1)
    TEST_OUT.write_text(json.dumps(out, indent=1) + "\n")
    for split, x in out["splits"].items():
        print(split, x["n_scenarios"])
        for m in METHODS:
            print(f"  {m:15}", {c: (x["metrics"][m][c]["precision"]["value"], x["metrics"][m][c]["recall"]["value"]) for c in CLASSES})
        for k, c in x["comparisons"].items():
            print(f"  {k:45} {c['verdict']:13} {c['point']} {c['ci']}")
    print(f"wrote {TEST_OUT.relative_to(ROOT)} ({out['seconds']}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    test() if ap.parse_args().test else tune()
