"""I49u: 並んだ上位の候補から、「話のそれまでの流れ」で選び直す（持ち主の言葉「内面の波はかなりいいかも」から。未確定の試し）。

I49t で、解く形は正解を上位に入れているが、点の並んだ候補から選ぶ所で落としていると分かった。
ここでは元の答え方（rel）は変えずに、上位の候補だけを、流れの波で並べ直す。乱数は使わない。
流れの波 = この話でそれまでに聞いた 3 つ組の語の、元の向き（base_of）を足した物。話が終わると空に戻す（tick）。
内面の波そのものではなく、その代わりに置いた物。

  FlowPick(rel, base_of, pick, top=3, minus_self=False)
    pick:
      "そのまま"        rel の答えのまま
      "上位"            点の上位 top 個の中で、流れの波に一番近い語
      "九割以内"        一位の 9 割以上の点の候補の中で、流れの波に一番近い語
      "順位を足す"      全候補で、点の順位と流れへの近さの順位を足して、一番小さい語
      "出た語"          点の上位 top 個の中で、この話で一番最近に出た語。どれも出ていなければ一位のまま（比べ役）
    minus_self: 候補自身がこの話で出た分を、流れの波から引いてから比べる（「もう出た語を選ぶだけ」になっていないかを見る）
    ask / learn / tick は rel と同じ形。picked は選び直しで答えが変わった数。
手で置く物: 上位の数 top、九割、流れを足す重さ 1（薄めない）、話ごとに空に戻す事。
"""
from __future__ import annotations

import numpy as np


class FlowPick:
    def __init__(self, rel, base_of, pick="上位", top=3, minus_self=False):
        self.rel, self.base_of, self.pick, self.top, self.minus_self = rel, base_of, pick, top, minus_self
        self.picked = 0
        self._reset()

    def _reset(self):
        self.ctx, self.said, self.t = None, {}, 0

    def near(self, c):
        b = self.base_of(c)
        if b is None or self.ctx is None:
            return None
        x = self.ctx
        if self.minus_self and c in self.said:
            x = x - self.said[c][0] * b
        nx, nb = float(np.linalg.norm(x)), float(np.linalg.norm(b))
        if nx <= 1e-6 or nb <= 0:
            return None
        return float(np.real(np.vdot(x, b))) / (nx * nb)

    def ask(self, si, tri, cands):
        o = self.rel.ask(si, tri, cands)
        pts = o["点"]
        if self.pick == "そのまま" or o["答え"] is None or not pts:
            return o
        cands = list(cands)
        order = sorted(pts, key=lambda c: (-pts[c], cands.index(c)))
        if self.pick == "九割以内":
            pool = [c for c in order if pts[c] >= 0.9 * pts[order[0]]]
        elif self.pick == "順位を足す":
            pool = order
        else:
            pool = order[:self.top]
        if self.pick == "出た語":
            best = max(pool, key=lambda c: (self.said.get(c, (0, -1))[1], -order.index(c)))
        else:
            nr = {c: self.near(c) for c in pool}
            if all(v is None for v in nr.values()):
                return o
            lo = min(v for v in nr.values() if v is not None) - 1.0
            nr = {c: (lo if v is None else v) for c, v in nr.items()}
            if self.pick == "順位を足す":
                by = sorted(pool, key=lambda c: (-nr[c], order.index(c)))
                best = min(pool, key=lambda c: (order.index(c) + by.index(c), order.index(c)))
            else:
                best = max(pool, key=lambda c: (nr[c], -order.index(c)))
        if best != o["答え"]:
            self.picked += 1
            o = dict(o, 答え=best)
        return o

    def learn(self, tri):
        self.rel.learn(tri)
        self.t += 1
        for w in tri:
            b = self.base_of(w)
            if b is not None:
                self.ctx = b.copy() if self.ctx is None else self.ctx + b
            n = self.said.get(w, (0, 0))[0]
            self.said[w] = (n + 1, self.t)

    def tick(self):
        self.rel.tick()
        self._reset()
