"""Runbook retrieval for the agent: BM25 over the short runbooks in `runbooks/` (no embedding model, no new dependency)."""
import math
import re
from collections import Counter
from pathlib import Path

RUNBOOK_DIR = Path(__file__).resolve().parent / "runbooks"
STOP = set("a an and are as at be by do does for from has if in is it its not of on or so the that their them then there "
           "this to was when which while with".split())


def tokens(text):
    return [w for w in re.findall(r"[a-z0-9_]+", text.lower()) if w not in STOP]


class RunbookIndex:
    def __init__(self, folder=RUNBOOK_DIR, k1=1.5, b=0.75):
        self.docs = [(p.stem, p.read_text()) for p in sorted(folder.glob("*.md"))]
        self.tf = [Counter(tokens(t)) for _, t in self.docs]
        self.len = [sum(c.values()) for c in self.tf]
        self.avg = sum(self.len) / len(self.len)
        df = Counter(w for c in self.tf for w in c)
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query):
        q = tokens(query)
        out = []
        for c, dl in zip(self.tf, self.len):
            s = 0.0
            for w in q:
                f = c.get(w, 0)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg))
            out.append(s)
        return out

    def search(self, query, k=2):
        """The k best runbooks as (id, score, text); ties keep file order."""
        sc = self.scores(query)
        order = sorted(range(len(sc)), key=lambda i: (-sc[i], i))[:k]
        return [(self.docs[i][0], round(sc[i], 3), self.docs[i][1]) for i in order if sc[i] > 0]
