"""I49zzt: 言葉の 3 つ組（誰が・何を・どうする）を、掛け合わせた波で持つ。外れた時だけ位相を動かす（I49zzs を、語が増えていく形にした物。未確定の試し）。

席ごとに、語の位相（次元ごとに 1 つの角度）を持つ。語は、初めて出た時に席ごとの番号をもらう（位相は i49zzs_bind と同じく、素数の平方根の周波数。乱数は使わない）。
  TripleCount: 問う席ごとに、残り 2 語の位相を足した波（掛け合わせ）を鍵にして、答えの語ごとに鍵を足して数える。位相は動かさない。
               同じ 2 語の組を前に聞いていなければ、点は 0 に近い（聞いていない組には届かない）。
  TripleNudge: 3 つ組の点は Σ cos(θ誰が + θ何を + θどうする)。聞いた 3 つ組について、席ごとに候補の中の 1 位を出し、外れた時だけ、
               正しい 3 つ組は点が上がる向きへ、間違って選んだ語との 3 つ組は点が下がる向きへ、位相を少し動かす。
どちらも ask(si, tri, cands) → dict(答え, 点) と learn(tri, cands) を持つ（本体 i49_foundation.body の ask と同じ問い方）。
手で置いた物: 次元、動かす大きさ（i49zzs_bind.ETA）、周波数の決め方。
"""
from __future__ import annotations

import numpy as np

import i49zzs_bind as B

GROW = 256   # 語の表を広げる時の刻み（速さのためだけ）


class _Tables:
    def __init__(self, dims):
        self.dims = dims
        self.ix = [{}, {}, {}]
        self.th = [np.zeros((0, dims), dtype=np.float32) for _ in range(3)]
        self._pr = []

    def _prime(self, k):
        if k >= len(self._pr):
            self._pr = B.primes(max(2 * k, 3 * GROW))
        return self._pr[k]

    def row(self, si, w):
        """席 si の語 w の番号。初めての語には、新しい位相を付ける。"""
        ix = self.ix[si]
        if w not in ix:
            i = len(ix)
            ix[w] = i
            if i >= len(self.th[si]):
                self.th[si] = np.concatenate([self.th[si], np.zeros((GROW, self.dims), dtype=np.float32)])
            d = np.arange(self.dims, dtype=np.float64) + 1.0
            self.th[si][i] = (2.0 * np.pi * ((np.sqrt(float(self._prime(3 * i + si))) * d) % 1.0)).astype(np.float32)
        return ix[w]


def _pick(cands, scores):
    best = max(range(len(cands)), key=lambda i: (float(scores[i]), -i))
    return cands[best]


class TripleCount(_Tables):
    def __init__(self, dims):
        super().__init__(dims)
        self.m = [{}, {}, {}]

    def _key(self, si, tri):
        a, b = [j for j in range(3) if j != si]
        ra, rb = self.row(a, tri[a]), self.row(b, tri[b])
        return np.exp(1j * (self.th[a][ra] + self.th[b][rb])).astype(np.complex64)

    def ask(self, si, tri, cands):
        cands = list(cands)
        k = np.conj(self._key(si, tri))
        s = [float(np.real(np.vdot(np.conj(self.m[si][c]), k))) / self.dims if c in self.m[si] else 0.0 for c in cands]
        return dict(答え=_pick(cands, s), 点=dict(zip(cands, s)))

    def learn(self, tri, cands=None):
        for si in range(3):
            k = self._key(si, tri)
            if tri[si] in self.m[si]:
                self.m[si][tri[si]] += k
            else:
                self.m[si][tri[si]] = k.copy()
        return 0


class TripleNudge(_Tables):
    def __init__(self, dims, eta=None):
        super().__init__(dims)
        self.eta = np.float32(B.ETA if eta is None else eta)

    def _base(self, si, tri):
        a, b = [j for j in range(3) if j != si]
        ra, rb = self.row(a, tri[a]), self.row(b, tri[b])
        return self.th[a][ra] + self.th[b][rb]

    def ask(self, si, tri, cands):
        cands = list(cands)
        rows = [self.row(si, c) for c in cands]
        base = self._base(si, tri)
        s = np.cos(base[None, :] + self.th[si][rows]).sum(axis=1) / self.dims
        return dict(答え=_pick(cands, s), 点=dict(zip(cands, (float(x) for x in s))))

    def learn(self, tri, cands):
        """席ごとに、候補の中の 1 位が正しい語でなければ位相を動かす。→ 外れた席の数。"""
        miss = 0
        for si in range(3):
            cs = [c for c in cands if c not in {tri[j] for j in range(3) if j != si}]
            if tri[si] not in cs:
                cs.append(tri[si])
            w = self.ask(si, tri, cs)["答え"]
            if w == tri[si]:
                continue
            miss += 1
            a, b = [j for j in range(3) if j != si]
            ra, rb, rc, rw = self.row(a, tri[a]), self.row(b, tri[b]), self.row(si, tri[si]), self.row(si, w)
            base = self.th[a][ra] + self.th[b][rb]
            g = np.sin(base + self.th[si][rc])     # 正しい 3 つ組の点を上げる向き
            gw = np.sin(base + self.th[si][rw])    # 間違って選んだ語との点を下げる向き
            self.th[a][ra] += self.eta * (gw - g)
            self.th[b][rb] += self.eta * (gw - g)
            self.th[si][rc] -= self.eta * g
            self.th[si][rw] += self.eta * gw
        return miss
