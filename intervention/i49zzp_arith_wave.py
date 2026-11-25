"""I49zzp: 算数の文字（0〜9 と ＋ − ＝）の並びだけから、文字の波と位相の関係が出来るかを見る（持ち主の案、未確定の試し）。

持ち主の案:「０～９と＋、－、＝の文字の組み合わせで、位相の関係性が作られるか試験してもらいたい。」
作り方（I49zzk・I49zzm と同じ。数えて足すだけ。数の並びも、＋と−の意味も置かない）:
  式は文字の並び（"3+4=7"）。文字ごとに、同じ式の中の他の文字を席にして数える。
  席は (どれだけ離れているか, 文字)。offsets=False なら、離れ方を見ない（文字だけ）。
  文字の波 = その文字の席ごとの割合 − 全部の文字を合わせた席ごとの割合（平均を引く）。
読む物:
  near: 2 つの文字の波の近さ（cos）。
  plane: 数字の波の、一番強い 2 つの向き。その面の上の角度を、数字の位相として読む（読むための物。学習には使わない）。
  answer: "a+b=" の後に来る物を、波の合い方（点の足し算）で選ぶ。2 文字ずつの結び付きしか見ない。
手で置いた物: 席に離れ方を入れる事、候補の点を文字の数で割る事、平均を引く事、近さを cos で測る事、位相を上の 2 つの向きの角度で読む事。
乱数は使わない。float32。
"""
from __future__ import annotations

import numpy as np

DIGITS = "0123456789"


class CharWave:
    def __init__(self, offsets=True):
        self.offsets = offsets
        self.seat = {}
        self.count = {}

    def learn(self, line):
        for i, u in enumerate(line):
            c = self.count.setdefault(u, {})
            for j, x in enumerate(line):
                if i != j:
                    s = self.seat.setdefault((j - i if self.offsets else 0, x), len(self.seat))
                    c[s] = c.get(s, 0) + 1

    def waves(self):
        """→ (文字の並び, 波の行列)。行が文字、列が席。"""
        units = sorted(self.count)
        m = np.zeros((len(units), len(self.seat)), dtype=np.float32)
        for a, u in enumerate(units):
            for s, n in self.count[u].items():
                m[a, s] = n
        tot = m.sum(axis=0)
        m = m / np.maximum(m.sum(axis=1, keepdims=True), 1) - (tot / max(float(tot.sum()), 1.0))[None, :]
        return units, m.astype(np.float32)

    def near(self, a, b):
        units, m = self.waves()
        if a not in units or b not in units:
            return 0.0
        x, y = m[units.index(a)], m[units.index(b)]
        d = float(np.linalg.norm(x) * np.linalg.norm(y))
        return float(x @ y) / d if d > 0 else 0.0

    def plane(self, chars=DIGITS):
        """chars の波の、一番強い 2 つの向きの上の座標と、向きごとの取り分。"""
        units, m = self.waves()
        have = [c for c in chars if c in units]
        x = m[[units.index(c) for c in have]]
        x = x - x.mean(axis=0, keepdims=True)
        u, s, _ = np.linalg.svd(x.astype(np.float64), full_matrices=False)
        p = (u[:, :2] * s[:2]).astype(np.float32)
        share = (s ** 2 / max(float((s ** 2).sum()), 1e-12)).astype(np.float32)
        return have, p, share

    def phase(self, chars=DIGITS):
        """上の 2 つの向きの面で、初めの文字から続けて測った角度（ラジアン。隣どうしの差を足していく）。"""
        have, p, _ = self.plane(chars)
        ang = np.arctan2(p[:, 1], p[:, 0]).astype(np.float64)
        out = [0.0]
        for k in range(1, len(have)):
            d = (ang[k] - ang[k - 1] + np.pi) % (2 * np.pi) - np.pi
            out.append(out[-1] + float(d))
        return have, np.array(out, dtype=np.float32)

    def answer(self, head, cands=DIGITS):
        """head（"3+4="）の後に来る物を、候補（1 文字でも、"12" や "-3" のような並びでも）から選ぶ。→ 点の高い順の (候補, 点)。
        点は、head の文字と候補の文字の 2 つずつの結び付きの足し算を、候補の文字の数で割った物。"""
        units, m = self.waves()
        ix = {u: i for i, u in enumerate(units)}
        n = len(head)
        out = []
        for cand in cands:
            sc = 0.0
            for k, c in enumerate(cand):
                for i, x in enumerate(head):
                    off = (i - n - k) if self.offsets else 0
                    s1, s2 = self.seat.get((off, x)), self.seat.get((-off, c))
                    if c in ix and s1 is not None:
                        sc += float(m[ix[c], s1])
                    if x in ix and s2 is not None:
                        sc += float(m[ix[x], s2])
            out.append((cand, sc / max(len(cand), 1)))
        return sorted(out, key=lambda t: (-t[1], t[0]))
