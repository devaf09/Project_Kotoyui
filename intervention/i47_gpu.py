"""I47 の道具の重い所を GPU（torch）で回すための小さな部品。torch が無い、GPU が無い時は numpy で同じ答えを出す。

  Scenes(K).top(q, k)  : しまってある場面の波 K（場面 × D）と問いの波 q の共鳴 |K q*| の強い順に k 個（番号, 強さ）。
  flow_sums(seq, ix, W): 語の波を流れから学ぶ時の足し上げ（同じ行・隣の行に出た語の波の和と、その数）。

環境変数:
  KOTOYUI_DEVICE = cuda / cuda:1 / cpu   （既定: GPU があれば cuda、無ければ numpy）
"""
from __future__ import annotations

import os

import numpy as np

try:
    import torch
except Exception:                                           # torch が無ければ numpy だけで動く
    torch = None


def device():
    d = os.environ.get("KOTOYUI_DEVICE", "")
    if torch is None or d == "numpy":
        return None
    if d:
        return torch.device(d)
    return torch.device("cuda") if torch.cuda.is_available() else None


class Scenes:
    def __init__(self, K):
        self.K = np.ascontiguousarray(K)
        self.dev = device()
        self.t = None
        if self.dev is not None:
            try:
                self.t = torch.from_numpy(self.K).to(self.dev)
            except RuntimeError:                            # GPU のメモリに載らない時は numpy で
                self.t = None

    def __len__(self):
        return len(self.K)

    def top(self, q, k):
        k = min(k, len(self.K))
        if self.t is not None:
            r = (self.t @ torch.from_numpy(np.conj(q)).to(self.dev)).abs()
            v, i = torch.topk(r, k)
            return i.cpu().numpy(), v.cpu().numpy()
        r = np.abs(self.K @ np.conj(q))
        i = np.argsort(-r)[:k]
        return i, r[i]


def flow_sums(seq, ix, W, chunk=20000):
    """acc[j] = Σ（語 j の出た行 k）（行 k と前後の行の語の波の和 − 自分の波）、n[j] = その相手の数。"""
    dev = device()
    V, D = W.shape
    if dev is None:
        acc = np.zeros_like(W)
        n = np.zeros(V)
        for k, ws in enumerate(seq):
            ctx = set(ws) | set(seq[k - 1] if k else []) | set(seq[k + 1] if k + 1 < len(seq) else [])
            ids = [ix[w] for w in ctx if w in ix]
            if len(ids) < 2:
                continue
            s = W[ids].sum(0)
            for w in set(ws):
                j = ix.get(w)
                if j is not None:
                    acc[j] += s - W[j]
                    n[j] += len(ids) - 1
        return acc, n
    Wt = torch.from_numpy(np.ascontiguousarray(W)).to(dev)
    acc = torch.zeros_like(Wt)
    n = torch.zeros(V, dtype=torch.float64, device=dev)
    own = torch.zeros(V, dtype=torch.float64, device=dev)
    for s0 in range(0, len(seq), chunk):
        cl, ci, wl, wi, sz = [], [], [], [], []
        r = 0
        for k in range(s0, min(len(seq), s0 + chunk)):
            ws = seq[k]
            ctx = set(ws) | set(seq[k - 1] if k else []) | set(seq[k + 1] if k + 1 < len(seq) else [])
            ids = [ix[w] for w in ctx if w in ix]
            if len(ids) < 2:
                continue
            mine = [ix[w] for w in set(ws) if w in ix]
            ci += ids
            cl += [r] * len(ids)
            wi += mine
            wl += [r] * len(mine)
            sz.append(len(ids) - 1)
            r += 1
        if not r:
            continue
        ci, cl, wi, wl = (torch.tensor(x, dtype=torch.long, device=dev) for x in (ci, cl, wi, wl))
        S = torch.zeros((r, D), dtype=Wt.dtype, device=dev).index_add_(0, cl, Wt[ci])
        acc.index_add_(0, wi, S[wl])
        szt = torch.tensor(sz, dtype=torch.float64, device=dev)
        n.index_add_(0, wi, szt[wl])
        own.index_add_(0, wi, torch.ones(len(wi), dtype=torch.float64, device=dev))
    acc -= own[:, None].to(Wt.dtype) * Wt
    return acc.cpu().numpy(), n.cpu().numpy()
