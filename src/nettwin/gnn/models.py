"""GraphSAGE, GAT and GINE on the line graph with a temporal encoder, plus the MLP ablation (Phase 5).

Per link: a causal 1D-conv encoder over the last W steps of its features, mixed with static features, then message
passing on the line graph (links adjacent when they share a PoP). Two heads:
- detection: one logit per link per step (trained on label >= 1);
- localisation: one logit per element (links and PoPs). A PoP's input is the mean of its links' embeddings plus its
  log degree; elements of one graph are normalised with a softmax.
The MLP ablation replaces message passing by per-link layers of the same size. PoP pooling stays in every variant,
since it is how elements are defined, not message passing between links.
"""
import numpy as np
import torch
from torch import nn
from torch_geometric.nn import GATConv, GINEConv, SAGEConv
from torch_geometric.utils import scatter, softmax

from nettwin.gnn.data import N_FEATURES, PATH_STATIC, STATIC, W


class Temporal(nn.Module):
    def __init__(self, n_in, h):
        super().__init__()
        self.c1 = nn.Conv1d(n_in, h, 3)
        self.c2 = nn.Conv1d(h, h, 3)
        self.out = nn.Linear(2 * h, h)

    def forward(self, x):                       # x: (N, W, F)
        z = torch.relu(self.c2(torch.relu(self.c1(x.transpose(1, 2)))))
        return torch.relu(self.out(torch.cat([z[:, :, -1], z.mean(2)], 1)))

    def sequence(self, x):
        """Same output as `forward` on every W-step window, computed once over a whole sequence.
        x: (L, T + W - 1, F), the sequence with its first row repeated W - 1 times in front (as `windows` pads).
        Returns (T, L, h)."""
        z = torch.relu(self.c2(torch.relu(self.c1(x.transpose(1, 2)))))   # (L, h, T + W - 5)
        k = W - 4                                                           # conv outputs per window
        c = torch.cumsum(torch.nn.functional.pad(z, (1, 0)), 2)
        mean = (c[:, :, k:] - c[:, :, :-k]) / k                             # (L, h, T)
        last = z[:, :, k - 1:]
        return torch.relu(self.out(torch.cat([last, mean], 1).permute(2, 0, 1)))


class NetTwinGNN(nn.Module):
    def __init__(self, conv="sage", hidden=48, layers=2, path_static=True):
        super().__init__()
        self.conv_type, self.path_static = conv, path_static
        self.register_buffer("f_mean", torch.zeros(N_FEATURES))
        self.register_buffer("f_std", torch.ones(N_FEATURES))
        self.register_buffer("s_mean", torch.zeros(len(STATIC)))
        self.register_buffer("s_std", torch.ones(len(STATIC)))
        self.static_mask = torch.tensor([0.0 if (s in PATH_STATIC and not path_static) else 1.0 for s in STATIC])
        self.enc = Temporal(N_FEATURES, hidden)
        self.static = nn.Linear(len(STATIC), hidden)
        self.mix = nn.Linear(2 * hidden, hidden)
        convs = []
        for _ in range(layers):
            if conv == "sage":
                convs.append(SAGEConv(hidden, hidden))
            elif conv == "gat":
                convs.append(GATConv(hidden, hidden // 4, heads=4, edge_dim=1))
            elif conv == "gine":
                convs.append(GINEConv(nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)), edge_dim=1))
            elif conv == "mlp":
                convs.append(nn.Linear(hidden, hidden))
            else:
                raise ValueError(conv)
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.det = nn.Linear(hidden, 1)
        self.loc_link = nn.Linear(hidden, 1)
        self.loc_pop = nn.Sequential(nn.Linear(hidden + 1, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def set_norm(self, f_mean, f_std, s_mean, s_std):
        self.f_mean.copy_(torch.as_tensor(f_mean)); self.f_std.copy_(torch.as_tensor(f_std))
        self.s_mean.copy_(torch.as_tensor(s_mean)); self.s_std.copy_(torch.as_tensor(s_std))

    def forward(self, b):
        """`b["enc"]`, if given, is the temporal encoding precomputed by `Temporal.sequence` (fast full-scenario path)."""
        s = (b["static"] - self.s_mean) / self.s_std * self.static_mask
        enc = b["enc"] if "enc" in b else self.enc((b["x"] - self.f_mean) / self.f_std)
        h = torch.relu(self.mix(torch.cat([enc, torch.relu(self.static(s))], 1)))
        for conv, norm in zip(self.convs, self.norms):
            if self.conv_type == "mlp":
                m = conv(h)
            elif self.conv_type == "sage":
                m = conv(h, b["edge_index"])
            else:
                m = conv(h, b["edge_index"], b["edge_attr"])
            h = norm(h + torch.relu(m))
        pop_h = scatter(h[b["link_pop"][0]], b["link_pop"][1], dim=0, dim_size=b["n_pops"], reduce="mean")
        pop_logit = self.loc_pop(torch.cat([pop_h, b["pop_static"]], 1)).squeeze(1)
        return self.det(h).squeeze(1), self.loc_link(h).squeeze(1), pop_logit


def collate(items, graphs, shuffle_perm=None):
    """items: (topology, x (L, W, F), det (L,) or None, roots or None). Returns one batched dict."""
    xs, st, ei, ea, lp, ps, det, link_graph, pop_graph = [], [], [], [], [], [], [], [], []
    roots, lo, po = [], 0, 0
    for gi, (topo, x, d, r) in enumerate(items):
        g = graphs[topo]
        e = g.edge_index if shuffle_perm is None else shuffle_perm[topo][g.edge_index]
        xs.append(np.asarray(x, np.float32)); st.append(g.static); ei.append(e + lo); ea.append(g.edge_attr)
        lp.append(g.link_pop + np.array([[lo], [po]])); ps.append(g.pop_static)
        link_graph.append(np.full(g.n_links, gi)); pop_graph.append(np.full(g.n_pops, gi))
        if d is not None:
            det.append(d)
        if r is not None:
            roots.append([(lo + i) if i < g.n_links else ("pop", po + i - g.n_links) for i in r])
        lo += g.n_links
        po += g.n_pops
    out = {"x": torch.from_numpy(np.concatenate(xs)), "static": torch.from_numpy(np.concatenate(st)),
           "edge_index": torch.from_numpy(np.concatenate(ei, 1)), "edge_attr": torch.from_numpy(np.concatenate(ea)),
           "link_pop": torch.from_numpy(np.concatenate(lp, 1)), "pop_static": torch.from_numpy(np.concatenate(ps)),
           "n_links": lo, "n_pops": po, "link_graph": torch.from_numpy(np.concatenate(link_graph)),
           "pop_graph": torch.from_numpy(np.concatenate(pop_graph))}
    if det:
        out["det"] = torch.from_numpy(np.concatenate(det))
    if roots:
        out["roots"] = roots
    return out


def element_log_probs(b, link_logit, pop_logit):
    """Log softmax over the elements (links then PoPs) of each graph in the batch."""
    logits = torch.cat([link_logit, pop_logit])
    index = torch.cat([b["link_graph"], b["pop_graph"]])
    return torch.log(softmax(logits, index) + 1e-12)


def loss_fn(b, out):
    det_logit, link_logit, pop_logit = out
    det_loss = nn.functional.binary_cross_entropy_with_logits(det_logit, b["det"])
    lp = element_log_probs(b, link_logit, pop_logit)
    L = b["n_links"]
    terms = []
    for r in b["roots"]:
        if r:
            idx = torch.tensor([i if not isinstance(i, tuple) else L + i[1] for i in r])
            terms.append(-lp[idx].mean())
    loc_loss = torch.stack(terms).mean() if terms else torch.zeros(())
    return det_loss + loc_loss, det_loss.detach(), loc_loss.detach()


def same_graph_batch(graph, n, edge_index):
    """Batch of n copies of one topology, built with array arithmetic (no Python loop over copies)."""
    L, P = graph.n_links, graph.n_pops
    off_l = (np.arange(n) * L)[:, None, None]
    off_p = (np.arange(n) * P)[:, None]
    ei = (edge_index[None] + off_l).transpose(1, 0, 2).reshape(2, -1)
    lp = np.stack([(graph.link_pop[0][None] + off_l[:, 0]).ravel(), (graph.link_pop[1][None] + off_p).ravel()])
    return {"static": torch.from_numpy(np.tile(graph.static, (n, 1))), "edge_index": torch.from_numpy(ei),
            "edge_attr": torch.from_numpy(np.tile(graph.edge_attr, (n, 1))), "link_pop": torch.from_numpy(lp),
            "pop_static": torch.from_numpy(np.tile(graph.pop_static, (n, 1))), "n_links": n * L, "n_pops": n * P,
            "link_graph": torch.from_numpy(np.repeat(np.arange(n), L)), "pop_graph": torch.from_numpy(np.repeat(np.arange(n), P))}


@torch.no_grad()
def predict(model, feat, graph, perm=None, chunk=252):
    """Full scenario: detection probability (T, L) and element probabilities (T, L + P) at every step.
    `perm` is the shuffled-edges relabelling for this topology (ablation only)."""
    T = feat.shape[0]
    L, P = graph.n_links, graph.n_pops
    ei = graph.edge_index if perm is None else perm[graph.edge_index]
    det, elem = np.zeros((T, L), np.float32), np.zeros((T, L + P), np.float32)
    model.eval()
    cache = {}
    seq = np.concatenate([np.repeat(feat[:1], W - 1, 0), feat])                      # pad like `windows`
    x = (torch.from_numpy(np.ascontiguousarray(seq.transpose(1, 0, 2))) - model.f_mean) / model.f_std
    enc = model.enc.sequence(x)                                                         # (T, L, h)
    for s in range(0, T, chunk):
        steps = np.arange(s, min(s + chunk, T))
        n = len(steps)
        if n not in cache:
            cache[n] = same_graph_batch(graph, n, ei)
        b = dict(cache[n])
        b["enc"] = enc[steps].reshape(n * L, -1)
        d, ll, pl = model(b)
        det[steps] = torch.sigmoid(d).numpy().reshape(n, L)
        p = torch.exp(element_log_probs(b, ll, pl)).numpy()
        elem[steps, :L] = p[:n * L].reshape(n, L)
        elem[steps, L:] = p[n * L:].reshape(n, P)
    return det, elem


def window_len():
    return W
