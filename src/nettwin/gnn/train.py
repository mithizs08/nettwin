"""Training loop for the Phase 5 GNN (CPU, deterministic for a given seed and thread count)."""
import time
import zlib

import numpy as np
import torch

from nettwin.gnn.models import NetTwinGNN, collate, loss_fn


def norm_stats(samples, graphs):
    """Per-feature mean / std over train windows, and per-static-feature mean / std over train topologies."""
    rng = np.random.default_rng(0)
    pick = rng.choice(len(samples), size=min(2000, len(samples)), replace=False)
    x = np.concatenate([np.asarray(samples[i][1], np.float32).reshape(-1, samples[i][1].shape[-1]) for i in pick])
    st = np.concatenate([graphs[t].static for t in sorted({s[0] for s in samples})])
    return x.mean(0), x.std(0) + 1e-3, st.mean(0), st.std(0) + 1e-3


def shuffle_perm(topology, n_links, seed):
    """Shuffled-edges ablation: a fixed random relabelling of the links inside the line-graph edges of one topology
    (same degree sequence, wrong neighbours), the same at training and at evaluation."""
    return np.random.default_rng([seed, zlib.crc32(topology.encode())]).permutation(n_links)


def shuffle_perms(graphs, seed):
    return {t: shuffle_perm(t, g.n_links, seed) for t, g in sorted(graphs.items())}


def train(samples, graphs, conv, hidden, seed, epochs, batch=32, lr=2e-3, path_static=True, shuffle=False, log=print):
    """samples: list of (topology, x (L, W, F) float16, det (L,), roots). Returns the model and a training log."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = NetTwinGNN(conv=conv, hidden=hidden, path_static=path_static)
    model.set_norm(*norm_stats(samples, graphs))
    perm = shuffle_perms(graphs, seed) if shuffle else None
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history, t0 = [], time.time()
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(samples))
        tot = np.zeros(3)
        for s in range(0, len(order), batch):
            b = collate([samples[i] for i in order[s:s + batch]], graphs, perm)
            loss, dl, ll = loss_fn(b, model(b))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += (loss.item(), dl.item(), ll.item())
        n = int(np.ceil(len(order) / batch))
        history.append({"epoch": ep + 1, "loss": round(float(tot[0]) / n, 4), "det_loss": round(float(tot[1]) / n, 4),
                        "loc_loss": round(float(tot[2]) / n, 4)})
        log(f"  {conv} h{hidden} seed{seed}{' noPath' if not path_static else ''}{' shuffled' if shuffle else ''} "
            f"epoch {ep + 1}: {history[-1]}")
    return model, {"history": history, "train_seconds": round(time.time() - t0, 1),
                   "n_params": sum(p.numel() for p in model.parameters())}
