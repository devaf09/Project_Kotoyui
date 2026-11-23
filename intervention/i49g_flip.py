"""I49g: もう知っている事実の「席の入れ替え」で、新しい種類の問いに答える（引き算を足し算から。持ち主の案の試し、未確定）。

ねらい: 足し算の事実 (A, B, C) を知っている。引き算 (C, B, A) の事実は 1 つも教わっていない。
引き算の問いを探り当てるたびに、その事実の席を入れ替えた物が、知っている事実の中に有るかを数える。入れ替え方は 6 通り全部を数える（有った・無かった）。
問われたら、候補ごとに「この答えなら、入れ替えた物が知っている事実に有るか」を見て、入れ替え方の当たりの割合で票を入れる。
手で置く物: 席の入れ替えを見る事そのもの、票 = 当たりの割合 × 知っている事実の数。どの入れ替えが正しいかは置かない（数える）。
記録: 入れ替え方ごとの 有った・無かった、問いごとの票。
"""
from __future__ import annotations

from collections import Counter
from itertools import permutations

PERMS = list(permutations(range(3)))


class Flip:
    def __init__(self, known):
        self.known = known                                # もう知っている事実を持つ関係の表（I49e の Relation）
        self.hit, self.miss = Counter(), Counter()
        self.rec = []

    @staticmethod
    def _turn(tri, p):
        return (tri[p[0]], tri[p[1]], tri[p[2]])

    def learn(self, tri):
        """探り当てた（または聞いた）新しい種類の事実 1 つ。入れ替え方ごとに、知っている事実に有ったかを数える。"""
        for p in PERMS:
            if self._turn(tri, p) in self.known.tri:
                self.hit[p] += 1
            else:
                self.miss[p] += 1

    def ask(self, a, b, cands):
        votes = Counter()
        for p in PERMS:
            if not self.hit[p]:
                continue
            share = self.hit[p] / (self.hit[p] + self.miss[p])
            for c in cands:
                t = self._turn((a, b, c), p)
                if t in self.known.tri:
                    votes[c] += share * self.known._count(t)
        cl = list(cands)
        order = sorted(votes, key=lambda c: (-votes[c], cl.index(c)))
        self.rec.append((a, b, dict(votes)))
        return dict(答え=order[0] if order else None, 点=dict(votes), 順=order)
