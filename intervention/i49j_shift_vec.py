"""I49j: 語を鍵にした数の表を持たず、向きの足し算だけで「2 つから残りの 1 つ」を答える骨組み（持ち主の案・未確定）。

持ち主の案: 「語板どうしの繋がりを値の表無しで、調整だけで紐づけられないか」（王 − 女王 ＝ 亭主 − 女房 のように、向きの差が中身を決める形）。
算数の「ずれを運ぶ」（I49g）を向きでやる。述語側の語ごとに、向きを足して持つだけ。3 つ組そのものは覚えない。
  席と述語側は関係の表（I49e）と同じ: だれが・何を を隠す時の述語側は どうした、どうした を隠す時は 何を。

  ShiftVec(vec_of, mode="ずれ")
    mode="ずれ"    述語側の語ごとに「答えの向き − もう片方の語の向き」を足して平均し、問いのもう片方の語の向きに足す。そこに近い候補を答える
    mode="真ん中"  述語側の語ごとに「答えの向き」を足して平均する。もう片方の語は見ない（ずれが効いているのかを分ける比べ役）
    mode="近さ"    何も覚えない。もう片方の語の向きに近い候補を答える（今の連想と同じ形の比べ役）
    learn(3 つ組)               足すだけ。向きの無い語を含む席は足さない
    ask(隠す席, 3 つ組, 候補)    → dict(段, 点, 答え)。関係の表と同じ形。向きが足りなければ答えない（白紙）
  近さは cos の実部（向きの差に意味を持たせるので、符号を捨てない）。
手で置く物: 述語側の決め方（関係の表と同じ）、平均を取る事、近さを cos の実部にした事。数は薄れさせていない。切る線・乱数は置かない。
"""
from __future__ import annotations

import numpy as np

from i49e_relation import NEAR, _unit

MODES = ("ずれ", "真ん中", "近さ")


class ShiftVec:
    def __init__(self, vec_of, mode="ずれ"):
        assert mode in MODES
        self.vec_of, self.mode = vec_of, mode
        self.sum = {}                                     # (隠す席, 述語側の語) → [足した向き, 足した数]
        self._u = {}
        self.rec = []

    def tick(self, n=1):
        pass

    def _unit_of(self, w):
        if w not in self._u:
            v = self.vec_of(w)
            self._u[w] = None if v is None else _unit(v)
        return self._u[w]

    def learn(self, tri):
        if self.mode == "近さ":
            return
        for si in range(3):
            near = NEAR[si]
            other = next(j for j in range(3) if j not in (si, near))
            ua, uo = self._unit_of(tri[si]), self._unit_of(tri[other])
            if ua is None or (self.mode == "ずれ" and uo is None):
                continue
            d = ua - uo if self.mode == "ずれ" else ua
            k = (si, tri[near])
            if k in self.sum:
                self.sum[k][0] = self.sum[k][0] + d
                self.sum[k][1] += 1
            else:
                self.sum[k] = [d.copy(), 1]

    def ask(self, si, tri, cands):
        cands = list(cands)
        near = NEAR[si]
        other = next(j for j in range(3) if j not in (si, near))
        uo = self._unit_of(tri[other])
        hit = self.sum.get((si, tri[near]))
        aim = None
        if self.mode == "近さ":
            aim = uo
        elif hit is not None:
            mean = hit[0] / hit[1]
            aim = mean if self.mode == "真ん中" else (None if uo is None else uo + mean)
        aim = None if aim is None else _unit(aim)
        out = dict(段="答えられない", 点={}, 答え=None)
        if aim is not None:
            s = {}
            for c in cands:
                uc = self._unit_of(c)
                if uc is not None:
                    s[c] = float(np.real(np.vdot(aim, uc)))
            if s:
                best = max(s, key=lambda c: (s[c], -cands.index(c)))
                out = dict(段=self.mode, 点=s, 答え=best)
        self.rec.append((0, si, (tri[near], tri[other]), out["段"], out["答え"]))
        return out
