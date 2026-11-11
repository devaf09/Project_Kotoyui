"""板の 4 つの操作の、行列の版（I27u〜）: numpy でも torch（GPU）でも同じコードで動く

board_ops.py（辞書で 1 鍵ずつ）と同じ答えを、行列でまとめて出す。
  鍵に番号をつけ、行ごとの板を「行 × 鍵」の行列 R にする。
  束ねる bundle_many(R, Wt)        = Wt @ R（Wt: か所 × 行 の坂の重み）。2,000 か所の板を 1 回の掛け算で作る。
  照らす light_many(B, cols, sat)  = B[:, cols] を読み、1 - exp(-v / sat) で飽和。minus があれば先に引く。
  差分   react_many(e, o)          = o - e（+ は発見、- は違和感）。大きさは abs。
  勝者   winner_many(S)            = 行ごとの argmax。
backend("torch") で GPU（あれば）に載る。CUDA_VISIBLE_DEVICES= のときは torch でも CPU。
"""
from __future__ import annotations


class XP:
    def __init__(self, name="numpy", device=None):
        self.name = name
        if name == "torch":
            import torch
            self.t = torch
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        else:
            import numpy as np
            self.np = np
            self.device = "cpu"

    def asarray(self, x, dtype="float32"):
        if self.name == "torch":
            return self.t.as_tensor(x, dtype=getattr(self.t, dtype), device=self.device)
        return self.np.asarray(x, dtype=dtype)

    def zeros(self, shape):
        if self.name == "torch":
            return self.t.zeros(shape, device=self.device)
        return self.np.zeros(shape, dtype="float32")

    def exp(self, x):
        return self.t.exp(x) if self.name == "torch" else self.np.exp(x)

    def maximum0(self, x):
        return self.t.clamp(x, min=0) if self.name == "torch" else self.np.maximum(x, 0)

    def abs(self, x):
        return self.t.abs(x) if self.name == "torch" else self.np.abs(x)

    def argmax(self, x, axis=-1):
        return x.argmax(dim=axis) if self.name == "torch" else x.argmax(axis=axis)

    def tolist(self, x):
        return x.cpu().tolist() if self.name == "torch" else x.tolist()


def backend(name="numpy", device=None):
    return XP(name, device)


class KeyIndex(dict):
    """鍵 → 列の番号。"""

    def id(self, k):
        if k not in self:
            self[k] = len(self)
        return self[k]


def rows_matrix(xp, rows, index):
    """rows: [Board]（行ごとの板）→ 行 × 鍵 の行列（index に無い鍵は足す）。"""
    ii, jj, vv = [], [], []
    for i, b in enumerate(rows):
        for k, v in b.items():
            ii.append(i)
            jj.append(index.id(k))
            vv.append(v)
    R = xp.zeros((len(rows), len(index)))
    if xp.name == "torch":
        R.index_put_((xp.asarray(ii, "long"), xp.asarray(jj, "long")), xp.asarray(vv), accumulate=True)
    else:
        xp.np.add.at(R, (xp.np.asarray(ii), xp.np.asarray(jj)), xp.np.asarray(vv, dtype="float32"))
    return R


def line_slope(xp, sites, n):
    """直線の坂の重み（か所 × 行）。自分の行は 0（I27q の slope("直線") と同じ）。"""
    li = xp.asarray(sites)[:, None]
    lj = xp.asarray(list(range(n)))[None, :]
    d = xp.abs(lj - li)
    W = xp.maximum0(1.0 - d / n)
    return W * (d > 0)


def bundle_many(R, Wt):
    """束ねる: か所ごとの板 = 坂の重み @ 行ごとの板。"""
    return Wt @ R


def light_many(xp, B, sat=None, minus=None):
    v = B if minus is None else B - minus
    v = xp.maximum0(v)
    if sat is None:
        return v
    return 1.0 - xp.exp(-v / sat)


def react_many(expected, observed):
    return observed - expected


def winner_many(xp, S):
    return xp.argmax(S, axis=-1)
