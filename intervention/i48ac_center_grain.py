"""I48ac: 中心に置くグレイン。板（機能語・品詞・詰め物）ごとに 1 枚。焦点の語を波の中心（ずれ 0）に置き、文の残りを焦点から見た相対の位置
（左は負、右は正）のずれで周りに重ねる（足すだけ）。周りの語は名前でなく語板の属性の波（拡散の符号: 属性の周波数ごとに D 全体の決まった乱数の位相）。

学び: 板 c のグレイン G_c = Σ（その板が焦点の所） Σ_o roll(語の波_o, o × SHIFT)。相対の位置 o ごとに属性の重みを足しておき（A[c, o, 周波数]）、
最後に波にする（足し算の順を替えただけで同じ物）。グレインは、ふだんの形（全部の板の重みつきの平均）を引いて板ごとの違いだけを残す。
当て: 見ていない所のまわりの波 C と各板のグレインの近さ cos(G_c, C) × 温度 ＋ 板の出やすさの対数 で、いちばん強い板を選ぶ。温度は取りおきの文で選ぶ。

手で置く物: 窓 W（左右に何語まで見るか）、温度の候補の並び、拡散の符号の乱数の作り方（周波数の番号から決める）。
"""
from __future__ import annotations

from collections import Counter

import numpy as np


def codes(D):
    """周波数の番号 f ごとの拡散の符号（行 f、長さ D、複素、長さ 1）。"""
    r = np.random.default_rng(20261116)
    return (np.exp(2j * np.pi * r.random((D, D))) / np.sqrt(D)).astype(np.complex64)


class CenterGrain:
    def __init__(self, S, W, classes, C=None):
        self.S, self.W, self.D = S, W, S.D
        self.cls = list(classes)
        self.ci = {c: i for i, c in enumerate(self.cls)}
        self.A = np.zeros((len(self.cls), 2 * W + 1, self.D), dtype=np.float32)
        self.n = np.zeros(len(self.cls))
        self.C = codes(self.D) if C is None else C
        self._buf = []

    def add(self, cls, ctx):
        """ctx: [(相対の位置 o, (周波数の配列, 重みの配列))]"""
        i = self.ci.get(cls)
        if i is None:
            return
        self.n[i] += 1
        for o, (f, v) in ctx:
            if 0 < abs(o) <= self.W:
                self._buf.append((np.full(len(f), i), np.full(len(f), o + self.W), f, v))
        if len(self._buf) > 20000:
            self.flush()

    def flush(self):
        if self._buf:
            a, b, f, v = (np.concatenate(x) for x in zip(*self._buf))
            np.add.at(self.A, (a, b, f), v)
            self._buf = []

    def _wave(self, A_o):
        """A_o: (2W+1, D) の属性の重み → 波（長さ D）。"""
        X = np.zeros(self.D, dtype=np.complex64)
        for k in range(2 * self.W + 1):
            o = k - self.W
            if o == 0 or not A_o[k].any():
                continue
            X += np.roll(A_o[k] @ self.C, o * self.S.shift)
        return X

    def build(self):
        self.flush()
        keep = self.n > 0
        G = np.stack([self._wave(self.A[i] / max(1, self.n[i])) for i in range(len(self.cls))])
        mean = (G * self.n[:, None]).sum(0) / self.n.sum()
        G = G - mean[None, :]
        self.Gn = G / np.maximum(1e-9, np.linalg.norm(G, axis=1, keepdims=True))
        self.prior = np.where(keep, np.log(np.maximum(self.n, 1) / self.n.sum()), -1e9)
        self.temp = 10.0
        return self

    def ctx_wave(self, ctx):
        X = np.zeros(self.D, dtype=np.complex64)
        for o, (f, v) in ctx:
            if 0 < abs(o) <= self.W:
                X += np.roll(v.astype(np.complex64) @ self.C[f], o * self.S.shift)
        return X

    def scores(self, ctx):
        X = self.ctx_wave(ctx)
        return np.real(self.Gn @ np.conj(X)) / max(1e-9, float(np.linalg.norm(X)))

    def predict(self, ctx):
        return self.cls[int(np.argmax(self.scores(ctx) * self.temp + self.prior))]

    def fit_temp(self, items, grid=(1, 3, 10, 30, 100, 300)):
        """items: [(ctx, 答え)]。取りおきの文で、当たりがいちばん良い温度を選ぶ。"""
        S_ = [(self.scores(c), y) for c, y in items]
        best = max((sum(self.cls[int(np.argmax(s * t + self.prior))] == y for s, y in S_), t) for t in grid)
        self.temp = best[1]
        return self.temp


DARK_W = {"千分の 1": 1e-3, "万分の 1": 1e-4}               # 持ち主の決め（2026-11-16）: だめだった並びは、OK の時のこの重さで暗い側へ


class Remember:
    """（途中。先読みと正誤の判断の段で使う）OK だった並びは関わったグレインに重さ 1 で足し、だめだった並びは暗い側のグレインに DARK_W の重さで足す
    （消さずに薄く。言い回しには正解がいくつもあるので、だめを強く覚えない）。近さは 明るい側 − 暗い側。"""

    def __init__(self, cg, dark_w):
        self.cg, self.w = cg, float(dark_w)
        self.dark = np.zeros_like(cg.A)
        self.nd = np.zeros_like(cg.n)

    def ok(self, cls, ctx):
        self.cg.add(cls, ctx)

    def bad(self, cls, ctx):
        i = self.cg.ci.get(cls)
        if i is None:
            return
        self.nd[i] += self.w
        for o, (f, v) in ctx:
            if 0 < abs(o) <= self.cg.W:
                np.add.at(self.dark[i, o + self.cg.W], f, v * self.w)

    def dark_scores(self, ctx):
        X = self.cg.ctx_wave(ctx)
        Dk = np.stack([self.cg._wave(self.dark[i]) for i in range(len(self.cg.cls))])
        return np.real(Dk @ np.conj(X)) / max(1e-9, float(np.linalg.norm(X)))


class NeighborTable:
    """基準: 左右の隣（語そのもの）を鍵にした数え。(左, 右) → 片方 → いちばん多い物。"""

    def __init__(self):
        self.t2, self.tl, self.tr, self.t0 = {}, {}, {}, Counter()

    def add(self, l, r, y):
        for t, k in ((self.t2, (l, r)), (self.tl, l), (self.tr, r)):
            t.setdefault(k, Counter())[y] += 1
        self.t0[y] += 1

    def predict(self, l, r):
        for t, k in ((self.t2, (l, r)), (self.tl, l), (self.tr, r)):
            c = t.get(k)
            if c:
                return c.most_common(1)[0][0]
        return self.t0.most_common(1)[0][0]
