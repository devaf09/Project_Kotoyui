"""I49zzs: 2 つの語を掛け合わせた波で、3 つ組の関係を持てるか。外れた時だけ位相を動かすと、見ていない組まで当たるか（持ち主の案、未確定の試し）。

持ち主の案:「周辺頻度が均一な3項関係、「猫が・魚を・食べる」と「犬が・骨を・かじる」が混ざると、2語ペアの線形和では
  「猫が・骨を・食べる」を分けられない問題について、外れた時だけ位相 θ を誤差の方向へ少し動かすを4096次元位相空間に適用すれば、
  ペア統計の限界をあげれるかも？」
ここに有る物:
  sym_phases: 記号ごとの位相（次元ごとに 1 つの角度）。記号の番号に比例させない（比例させると、足し算の答えが初めから入ってしまう）。
    記号ごとに違う素数の平方根を周波数にする。乱数は使わない。
  PairSum:   比べ役。2 つずつの数え（1 つ目の語 → 答え、2 つ目の語 → 答え）の足し算。
  BindCount: 2 つの語の位相を足した波（掛け合わせ）を鍵にして、答えごとに鍵を足して数える。位相は動かさない。
  BindNudge: 点は Σ cos(θ1 + θ2 − θ答え)。外れた時だけ、正しい答えへ近付け、間違って選んだ答えから遠ざける向きへ、位相を少し動かす。
    shared=True は、1 つ目・2 つ目・答えが同じ記号の表を使う（算数の数字のように、同じ物が 3 つの席に出る時）。
    shared=False は、席ごとに別の表（「誰が・何を・どうする」のように、席で語が違う時）。
手で置いた物: 動かす大きさ ETA、回す上限 EPOCHS、周波数の決め方（素数の平方根）。
"""
from __future__ import annotations

import numpy as np

import i49_situation as SIT

ETA = 0.1       # 外れた時に位相を動かす大きさ（手で置いた値）
EPOCHS = 200    # 回す上限（外れが 0 になったら止める。手で置いた値）


def primes(n):
    out, k = [], 2
    while len(out) < n:
        if all(k % p for p in out if p * p <= k):
            out.append(k)
        k += 1
    return out


def sym_phases(n, dims, skip=0):
    """記号 n 個の位相（n × dims、ラジアン）。記号 i の周波数は (skip + i) 番目の素数の平方根。"""
    r = np.sqrt(np.array(primes(n + skip)[skip:], dtype=np.float64))[:, None]
    d = np.arange(dims, dtype=np.float64)[None, :] + 1.0
    return (2.0 * np.pi * ((r * d) % 1.0)).astype(np.float32)


class PairSum:
    def __init__(self, n, triples):
        self.c1 = np.zeros((n, n), dtype=np.float32)
        self.c2 = np.zeros((n, n), dtype=np.float32)
        for a, b, c in triples:
            self.c1[a, c] += 1
            self.c2[b, c] += 1

    def scores(self, a, b):
        return self.c1[a] / max(float(self.c1[a].sum()), 1.0) + self.c2[b] / max(float(self.c2[b].sum()), 1.0)


class BindCount:
    def __init__(self, n, triples, dims):
        self.t1, self.t2 = sym_phases(n, dims), sym_phases(n, dims, n)
        self.m = np.zeros((n, dims), dtype=np.complex64)
        for a, b, c in triples:
            self.m[c] += self.key(a, b)

    def key(self, a, b):
        return np.exp(1j * (self.t1[a] + self.t2[b])).astype(np.complex64)

    def scores(self, a, b):
        return np.real(self.m @ np.conj(self.key(a, b)))


class BindNudge:
    def __init__(self, n, dims, shared=True):
        self.a = sym_phases(n, dims)
        self.b = self.a if shared else sym_phases(n, dims, n)
        self.c = self.a if shared else sym_phases(n, dims, 2 * n)
        self.shared = shared

    def scores(self, a, b):
        return np.cos(self.a[a] + self.b[b] - self.c).sum(axis=1)

    def hear(self, a, b, c, eta=None):
        """1 つ聞く。当たっていれば何もしない。外れていれば位相を少し動かす。→ 外れたか。"""
        eta = ETA if eta is None else eta
        w = int(np.argmax(self.scores(a, b)))
        if w == c:
            return False
        g = np.sin(self.c[c] - self.a[a] - self.b[b])       # 正しい答えへ近付ける向き
        gw = np.sin(self.c[w] - self.a[a] - self.b[b])      # 間違って選んだ答えから遠ざける向き
        d = np.float32(eta) * (g - gw)
        self.a[a] += d
        self.b[b] += d
        self.c[c] -= np.float32(eta) * g
        self.c[w] += np.float32(eta) * gw
        return True

    def learn(self, triples, eta=None, epochs=None):
        """黄金比の並びで回す。→ 回ごとの外れの数。"""
        epochs = EPOCHS if epochs is None else epochs
        order = SIT.golden_order(len(triples))
        miss = []
        for _ in range(epochs):
            miss.append(sum(self.hear(*triples[i], eta=eta) for i in order))
            if miss[-1] == 0:
                break
        return miss


def accuracy(model, triples):
    return sum(int(np.argmax(model.scores(a, b))) == c for a, b, c in triples) / max(len(triples), 1)
