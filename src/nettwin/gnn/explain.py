"""Gradient x input attribution for one element's localisation probability at one step (Phase 6 `explain_gnn`)."""
import numpy as np
import torch

from nettwin.baselines import FEATURES
from nettwin.gnn.data import W, windows
from nettwin.gnn.models import element_log_probs, same_graph_batch


def explain(model, feat, graph, element_idx, t, top=6):
    """Which (link, feature) inputs over the last W steps push the element's log-probability at step t up or down.
    Returns the element's probability and the `top` inputs by absolute attribution (gradient x normalised input,
    summed over the window)."""
    model.eval()
    x = torch.from_numpy(np.ascontiguousarray(windows(feat, [t])[0])).float()           # (L, W, F)
    xn = ((x - model.f_mean) / model.f_std).requires_grad_(True)
    b = same_graph_batch(graph, 1, graph.edge_index)
    b["enc"] = model.enc(xn)
    _, link_logit, pop_logit = model(b)
    lp = element_log_probs(b, link_logit, pop_logit)[element_idx]
    lp.backward()
    attr = (xn.grad * xn).sum(1).detach().numpy()                                       # (L, F)
    order = np.argsort(-np.abs(attr), axis=None)[:top]
    rows = []
    for k in order:
        j, f = np.unravel_index(k, attr.shape)
        rows.append({"link": graph.elements[j], "feature": FEATURES[f], "attribution": round(float(attr[j, f]), 4),
                     "value_now": None if np.isnan(feat[t, j, f]) else round(float(feat[t, j, f]), 4)})
    return {"probability": round(float(np.exp(lp.item())), 4), "window_steps": W, "top_inputs": rows}
