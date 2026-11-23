"""I49g: 借りる時に、ずれも一緒に運ぶ（気づいた後の直し方の案、未確定）。関係の表（I49e の Relation、変えない）の上に乗せる小さな部品。

ねらい: 関係の表は、聞いた事の無い問い（4＋3）に、隣の問いの答え（3＋3＝6）をそのまま借りて外す。
「4 は 3 の次」と「片方が次にずれると、答えも次にずれる」を数えて覚えていれば、「6 の次で 7」と運べる。

覚える物は 2 つ。どちらも数えるだけ。
  並び     数え歌（0 1 2 …）を聞いて、(n, 次, n+1) の 3 つ組として関係の表に入れる。「次」「前」はこの表を引く。
  ずれの決まり  新しい事実 (A, B, C) を聞くたびに、もう聞いた事実のうち片方の席だけが違う物と比べる。
              その席のずれ（次・前）に対して、答えのずれ（次・前・同じ・ほか）が何だったかを数える。席ごとに別に数える。
答え方:
  聞いた事のある組なら、関係の表の答えをそのまま（2 語の段）。
  無ければ、片方の席が「次」か「前」だけずれた聞いた事のある事実を集め、ずれの決まりの数の割合で答えをずらして票を入れる。いちばん票の多い物。
  ずらせる事実が無ければ、関係の表の下がり方に任せる。
手で置く物: ずれを 1 歩（次・前）だけ見る事、票 = 決まりの割合 × 事実の数。切る線・上位何個・乱数は置かない。
記録: 問いごとに、どの段で答えたか、使った事実とずれ。
"""
from __future__ import annotations

from collections import Counter

STEP = "次"
BACK = {"次": "前", "前": "次", "同じ": "同じ"}


class Shifter:
    def __init__(self, rel):
        self.rel = rel
        self.rule = {}                                    # (席, その席のずれ) → Counter(答えのずれ)
        self.rec = []

    # ---- 並び ----
    def hear_song(self, seq):
        for a, b in zip(seq, seq[1:]):
            self.rel.learn((a, STEP, b))

    def _move(self, w, d):
        """w の 次（前）。並びを知らなければ None。数のいちばん多い物。"""
        if d == "同じ":
            return w
        si, so = (0, 2) if d == "次" else (2, 0)
        best = None
        for t in self.rel.by[si].get(w, ()):
            if t[1] == STEP:
                c = self.rel._count(t)
                if best is None or c > best[0]:
                    best = (c, t[so])
        return None if best is None else best[1]

    def between(self, a, b):
        """a から見た b のずれ: 同じ・次・前。どれでもなければ None。"""
        if a == b:
            return "同じ"
        if self._move(a, "次") == b:
            return "次"
        if self._move(a, "前") == b:
            return "前"
        return None

    def _facts(self, si, w):
        return [t for t in self.rel.by[si].get(w, ()) if t[1] != STEP]

    # ---- 覚える ----
    def learn(self, tri):
        tri = tuple(tri)
        for si in (0, 1):
            other = 1 - si
            for f in self._facts(other, tri[other]):
                for x, y in ((f, tri), (tri, f)):         # 両方の向きで数える（聞いた順に寄らない）
                    d = self.between(x[si], y[si])
                    if d in ("次", "前"):
                        self.rule.setdefault((si, d), Counter())[self.between(x[2], y[2]) or "ほか"] += 1
        self.rel.learn(tri)

    # ---- 問う ----
    def ask(self, a, b, cands):
        o = self.rel.ask(2, (a, b, None), cands)
        if o["段"] == "2語":
            self.rec.append((a, b, "2語", o["答え"], []))
            return o
        q, votes, used = (a, b), Counter(), []
        for si in (0, 1):
            other = 1 - si
            for d in ("次", "前"):
                rule = self.rule.get((si, d))
                src = self._move(q[si], BACK[d])           # src から見て、問いの語が d だけずれている
                if not rule or src is None:
                    continue
                total = sum(rule.values())
                for f in self._facts(other, q[other]):
                    if f[si] != src:
                        continue
                    for dc, n in rule.items():
                        c = self._move(f[2], dc) if dc != "ほか" else None
                        if c is not None and c in cands:
                            votes[c] += (n / total) * self.rel._count(f)
                            used.append((f, d, dc))
        if not votes:
            self.rec.append((a, b, o["段"], o["答え"], []))
            return o
        cl = list(cands)
        best = max(votes, key=lambda c: (votes[c], -cl.index(c)))
        out = dict(段="ずれを運ぶ", 点=dict(votes), 答え=best)
        self.rec.append((a, b, out["段"], best, used))
        return out
