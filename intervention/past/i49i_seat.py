"""I49i: 向きを持たない語を、同じ席に座る仲間から「空席に座らせる」。関係の表（I49e）の上に足す骨組み（算数の知見を文に当てる 2 つ目、未確定）。

算数の試しでは、知らない数を「聞いた事実の空いている所」に 1 回で座らせられた（I49g の fill）。
文では、向き（語板のベクトル）を持たない語は、同じ語としてしか使えず、似た語の数を借りられない。
ここでは、語の「座り方」を数える: その語が、どの席で、どの語と一緒に聞かれたか。
座り方が似ている 2 語は、向きが無くても近いとみなす。新しい表は作らず、聞いた 3 つ組を数えるだけ。

  SeatRelation(vec_of, half=1024, where="無い語")
    where="無い語"  どちらかの語に向きが無い時だけ、座り方の近さを使う（向きが有る語どうしは今まで通り |cos|）
    where="全部"    向きが有る語どうしでも、|cos| と座り方の近さの大きい方を使う
    seat_near(a, b)  座り方の近さ。2 語の「（自分の席, 相手の席, 相手の語）→ 聞いた数」の cos。0〜1。同じ席で同じ相手と聞かれていなければ 0
    ほかは Relation と同じ（learn, tick, ask, tally）
手で置く物: 座り方を「自分の席・相手の席・相手の語」で数える事、近さを cos にした事、where の 2 通り。
            座り方の数は薄れさせていない（消してはいない。関係の表の数は今まで通り薄れる）。切る線・上位何個・乱数は置かない。
"""
from __future__ import annotations
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする

import math

from i49e_relation import Relation


class SeatRelation(Relation):
    def __init__(self, vec_of, half=1024.0, where="無い語"):
        super().__init__(vec_of, half)
        assert where in ("無い語", "全部")
        self.where = where
        self.ctx = {}                                     # 語 → {(自分の席, 相手の席, 相手の語): 聞いた数}
        self.sq = {}                                      # 語 → 数の 2 乗の和（cos の分母。足すたびに直す）
        self.ver = {}                                     # 語 → 座り方が変わった回数（近さの覚え書きを古くするため）
        self._sn = {}                                     # (a, b) → (a の回数, b の回数, 近さ)
        self.used = 0                                     # 座り方の近さが 0 より大きく使われた回数

    def learn(self, tri):
        t = tuple(tri)
        for si, w in enumerate(t):
            d = self.ctx.setdefault(w, {})
            for sj, v in enumerate(t):
                if sj == si:
                    continue
                k = (si, sj, v)
                c = d.get(k, 0)
                d[k] = c + 1
                self.sq[w] = self.sq.get(w, 0) + 2 * c + 1
            self.ver[w] = self.ver.get(w, 0) + 1
        super().learn(t)

    def seat_near(self, a, b):
        if a == b:
            return 1.0
        da, db = self.ctx.get(a), self.ctx.get(b)
        if not da or not db:
            return 0.0
        key = (a, b) if (str(a), id(a)) <= (str(b), id(b)) else (b, a)
        hit = self._sn.get(key)
        va, vb = self.ver[a], self.ver[b]
        if hit is not None and hit[0] == va + vb:
            return hit[1]
        if len(da) > len(db):
            da, db = db, da
        dot = sum(c * db[k] for k, c in da.items() if k in db)
        s = dot / math.sqrt(self.sq[a] * self.sq[b]) if dot else 0.0
        self._sn[key] = (va + vb, s)
        return s

    def near(self, a, b):
        if a == b:
            return 1.0
        ua, ub = self._unit_of(a), self._unit_of(b)
        if ua is None or ub is None:
            s = self.seat_near(a, b)
        elif self.where == "全部":
            s = max(super().near(a, b), self.seat_near(a, b))
        else:
            return super().near(a, b)
        if s > 0:
            self.used += 1
        return s
