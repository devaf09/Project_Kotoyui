"""I49e: 数えて覚える「関係の表」。ことゆいの部品の骨組み（持ち主の案「A＋B＝？みたいな関係性を聞く」「学習を増やせないなぁと」から、未確定）。

3 つ組（だれが・何を・どうした）を聞くたびに数える。問われたら、3 つのうち 2 つから残りの 1 つを選ぶ。
語の向き（似ている）と数えた表（関係）をつなぐ: まったく同じ組を見ていなくても、似た語で覚えた数を借りる。
下がり方は、数えの試し（tools/i49e_triple_count.py の七歩目）でいちばん良かった順。上の段に 1 つでも数があれば、それを使う。
  1. 2 語        見せた 2 語とまったく同じ組の数
  2. 片方だけ借りる  述語側の語は同じ物に限り、もう片方の語だけ、似た語の数を近さ（|cos|）を掛けて足す
  3. 述語側      述語側の 1 語だけの数
  4. もう片方    もう片方の 1 語だけの数
  5. 借りる      2 語とも、似た語の数を近さを掛けて足す
  どの段にも数が無ければ、答えない（白紙。無理には答えさせない）。
  述語側 = だれが・何を を隠す時は どうした、どうした を隠す時は 何を。

  Relation(vec_of, half=1024)
    vec_of(語) → ベクトル か None   語の向き。None の語は、同じ語としてしか使えない（借りられない）
    learn(3 つ組)                    数える（足すだけ）
    tick()                           時間を 1 つ進める。記憶は消さずに薄れる: 数に 0.5^(1/half) を掛ける（half は器のいちばん遅い波と同じ 1,024）
    ask(隠す席, 3 つ組, 候補)        → dict(段, 点 = 候補 → 数, 答え = 点がいちばん大きい候補（同じ点なら候補の並びで先の物）か None)
                                     3 つ組の隠す席の中身は見ない。問いと答えは全部 rec に残す
手で置く物: 下がる順、近さ |cos|、薄れる速さ（器と同じ 1,024）。切る線・上位何個・乱数は置かない。
"""
from __future__ import annotations

import numpy as np

SLOTS = ("だれが", "何を", "どうした")
NEAR = {0: 2, 1: 2, 2: 1}                                 # 隠す席 → 述語側の席
LEVELS = ("2語", "片方だけ借りる", "述語側", "もう片方", "借りる")


def _unit(v):
    v = np.asarray(v, dtype=np.complex64)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else None


class Relation:
    def __init__(self, vec_of, half=1024.0):
        self.vec_of = vec_of
        self.keep = float(0.5 ** (1.0 / half))
        self.now = 0
        self.tri = {}                                     # 3 つ組 → [数, 最後に触った時]（薄れるのは、触る時にまとめて掛ける）
        self.by = [dict() for _ in SLOTS]                 # 席 → 語 → その語を持つ 3 つ組の集まり
        self._u = {}                                      # 語 → 長さ 1 の向き（無ければ None）
        self.rec = []                                     # (時, 隠す席, 見せた 2 語, 段, 答え)

    # ---- 覚える ----
    def tick(self, n=1):
        self.now += int(n)

    def _count(self, t):
        c, at = self.tri[t]
        return c * self.keep ** (self.now - at)

    def learn(self, tri):
        t = tuple(tri)
        self.tri[t] = [self._count(t) + 1.0 if t in self.tri else 1.0, self.now]
        for si, w in enumerate(t):
            self.by[si].setdefault(w, set()).add(t)

    # ---- 問う ----
    def _unit_of(self, w):
        if w not in self._u:
            v = self.vec_of(w)
            self._u[w] = None if v is None else _unit(v)
        return self._u[w]

    def near(self, a, b):
        """語どうしの近さ |cos|。同じ語は 1。向きの無い語は、同じ語でなければ 0。"""
        if a == b:
            return 1.0
        ua, ub = self._unit_of(a), self._unit_of(b)
        if ua is None or ub is None:
            return 0.0
        return min(1.0, float(np.abs(np.vdot(ua, ub))))

    def _level(self, lv, si, tri, cands):
        near = NEAR[si]
        other = next(j for j in range(3) if j not in (si, near))
        s = {c: 0.0 for c in cands}
        if lv == "借りる":
            pool = self.tri
        else:
            key = near if lv != "もう片方" else other
            pool = self.by[key].get(tri[key], ())
        for t in pool:
            if t[si] not in s:
                continue
            if lv == "2語":
                w = float(t[other] == tri[other])
            elif lv == "片方だけ借りる":
                w = self.near(tri[other], t[other])
            elif lv == "借りる":
                w = self.near(tri[near], t[near]) * self.near(tri[other], t[other])
            else:
                w = 1.0
            if w > 0:
                s[t[si]] += w * self._count(t)
        return s

    def ask(self, si, tri, cands):
        cands = list(cands)
        near = NEAR[si]
        other = next(j for j in range(3) if j not in (si, near))
        out = dict(段="答えられない", 点={}, 答え=None)
        for lv in LEVELS:
            s = self._level(lv, si, tri, cands)
            if any(v > 0 for v in s.values()):
                best = max(cands, key=lambda c: (s[c], -cands.index(c)))
                out = dict(段=lv, 点=s, 答え=best)
                break
        self.rec.append((self.now, si, (tri[near], tri[other]), out["段"], out["答え"]))
        return out

    def tally(self):
        """どの段で答えたかの内訳。"""
        n = {lv: 0 for lv in LEVELS + ("答えられない",)}
        for r in self.rec:
            n[r[3]] += 1
        return n
