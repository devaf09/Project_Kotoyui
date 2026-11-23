"""I49k: 語の向きを、座り方（どの席で、どの語と一緒に聞かれたか）から足し算で作り直す骨組み（持ち主の案「語板を再設計する必要があるかも」から、未確定）。

今の語板の向きは、差に意味が乗るように作られていない（I49j: ずれを足すと、かえって外れる）。
ここでは、語の向きを「一緒に座った語の元の向きを、席の組ごとに決まった回し方で回して、足した物」にする。数えて足すだけで、少しずつ寄せる学習はしない。
同じ席で同じ相手と聞かれる語どうしは、向きが近くなる。元の向きを持たない語も、相手に向きが有れば向きを持てる。

  SeatVec(base_of, weight="偏り")
    base_of(語) → 元の向き か None      今の語板の向き。足す材料としてだけ使う
    hear(3 つ組)                         座り方を数える（足すだけ）
    of(語) → 作った向き か None          その時までに数えた座り方から作る。相手に向きが 1 つも無ければ None
    weight="数"    聞いた数をそのまま重さにする
    weight="偏り"  たまたまより多く一緒に聞いた分だけを重さにする: max(0, log(一緒の数 × 全体 ÷ (語の数 × 相手の数)))
  回し方: 席の組（自分の席, 相手の席）ごとに、向きの並びを決まった数だけずらす（np.roll）。席の違いを向きに残すため。
手で置く物: 回し方（ずらす数 = (席の組の番号 + 1) × 次元 ÷ 7）、重さの 2 通り、元の向きを長さ 1 にそろえる事。数は薄れさせていない。乱数は置かない。
"""
from __future__ import annotations

import math

import numpy as np

PAIRS = [(i, j) for i in range(3) for j in range(3) if i != j]


class SeatVec:
    def __init__(self, base_of, weight="偏り"):
        assert weight in ("数", "偏り")
        self.base_of, self.weight = base_of, weight
        self.ctx = {}                                     # 語 → {(自分の席, 相手の席, 相手の語): 数}
        self.n_w = {}                                     # (自分の席, 語) → 数
        self.n_c = {}                                     # (自分の席, 相手の席, 相手の語) → 数
        self.n_p = {}                                     # (自分の席, 相手の席) → 数
        self._b = {}                                      # 語 → 長さ 1 の元の向き（無ければ None）
        self._v = {}                                      # 語 → 作った向き（hear のたびに空にする）

    def hear(self, tri):
        t = tuple(tri)
        for si, w in enumerate(t):
            d = self.ctx.setdefault(w, {})
            self.n_w[(si, w)] = self.n_w.get((si, w), 0) + 1
            for sj, v in enumerate(t):
                if sj == si:
                    continue
                k = (si, sj, v)
                d[k] = d.get(k, 0) + 1
                self.n_c[k] = self.n_c.get(k, 0) + 1
                self.n_p[(si, sj)] = self.n_p.get((si, sj), 0) + 1
        self._v.clear()

    def _base(self, w):
        if w not in self._b:
            v = self.base_of(w)
            if v is not None:
                v = np.asarray(v, dtype=np.complex64)
                n = float(np.linalg.norm(v))
                v = v / n if n > 0 else None
            self._b[w] = v
        return self._b[w]

    def _w(self, w, k, c):
        if self.weight == "数":
            return float(c)
        si, sj, _ = k
        return max(0.0, math.log(c * self.n_p[(si, sj)] / (self.n_w[(si, w)] * self.n_c[k])))

    def of(self, w):
        if w in self._v:
            return self._v[w]
        out = None
        for k, c in self.ctx.get(w, {}).items():
            b = self._base(k[2])
            if b is None:
                continue
            g = self._w(w, k, c)
            if g <= 0:
                continue
            step = (PAIRS.index((k[0], k[1])) + 1) * (len(b) // 7)
            x = g * np.roll(b, step)
            out = x if out is None else out + x
        self._v[w] = out
        return out
