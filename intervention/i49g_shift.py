"""I49g: 借りる時に、ずれも一緒に運ぶ（気づいた後の直し方の案、未確定）。関係の表（I49e の Relation、変えない）の上に乗せる小さな部品。

ねらい: 関係の表は、聞いた事の無い問い（4＋3）に、隣の問いの答え（3＋3＝6）をそのまま借りて外す。
「4 は 3 の次」と「片方が次にずれると、答えも次にずれる」を数えて覚えていれば、「6 の次で 7」と運べる。

覚える物は 2 つ。どちらも数えるだけ。
  並び     数え歌（0 1 2 …）を聞いて、(n, 次, n+1) の 3 つ組として、事実とは別の関係の表に入れる。「次」「前」はこの表を引く。
  ずれの決まり  新しい事実 (A, B, C) を聞くたびに、もう聞いた事実のうち片方の席だけが違う物と比べる。
              その席のずれ（次・前）に対して、答えのずれ（次・前・同じ・ほか）が何だったかを数える。席ごとに別に数える。
答え方:
  聞いた事のある組なら、関係の表の答えをそのまま（2 語の段）。
  無ければ、片方の席が「次」か「前」だけずれた聞いた事のある事実を集め、ずれの決まりの数の割合で答えをずらして票を入れる。いちばん票の多い物。
  1 歩で届かなければ、聞いた事のある事実に着くまで片方ずつ 1 歩ずつ戻り、戻った歩の分だけ答えを運ぶ（数え足し）。いちばん近い事実だけを使う。
  ずらせる事実が無ければ、関係の表の下がり方に任せる。
並びを埋める（fill=True の時だけ。持ち主の案「一度でも 9 がどこかで出てくれば、その後の 9 を理解する」の試し）:
  新しい事実の答えが並びに居ない語の時、その問いを自分で運んでみる。最後の 1 歩だけが「x の次（前）を知らない」で止まり、その席が空いていれば、聞いた答えをそこに座らせる。
  例: 0〜8 しか知らない時に 4＋5＝9 を聞く。3＋5＝8 から運ぶと「8 の次」で止まる。8 の次は空いている。9 を 8 の次に置く。
  いちばん数の多い決まりだけを使う。埋めた並びは、決まりの数には入れない。
手で置く物: ずれを 1 歩（次・前）だけ見る事、票 = 決まりの割合 × 事実の数。切る線・上位何個・乱数は置かない。
記録: 問いごとに、どの段で答えたか、使った事実とずれ。
"""
from __future__ import annotations

from collections import Counter

from i49e_relation import Relation

STEP = "次"
BACK = {"次": "前", "前": "次", "同じ": "同じ"}


class Shifter:
    def __init__(self, rel, fill=False, back=True):
        self.rel = rel                                    # 事実の表
        self.ord = Relation(lambda w: None)               # 並びの表（数え歌）。事実の表とは別に持つ（並びが事実の答えに混ざらないように）
        self.back = bool(back)                            # 並びを逆向きに問えるか（「x の前」= 次が x になる語）。比べのために切れる
        self.fill = bool(fill)                            # 並びを自分で埋めるか
        self._mv = {}                                     # 次・前の引きの控え（並びが増えたら捨てる）
        self.filled = []                                  # 埋めた並び: (前の語, 後の語)
        self.rule = {}                                    # (席, その席のずれ) → Counter(答えのずれ)
        self.rec = []

    # ---- 並び ----
    def hear_song(self, seq):
        for a, b in zip(seq, seq[1:]):
            self.ord.learn((a, STEP, b))
        self._mv.clear()

    def _move(self, w, d):
        """w の 次（前）。並びを知らなければ None。数のいちばん多い物。"""
        if d == "同じ":
            return w
        if d == "前" and not self.back:
            return None
        if (w, d) in self._mv:
            return self._mv[(w, d)]
        si, so = (0, 2) if d == "次" else (2, 0)
        best = None
        for t in self.ord.by[si].get(w, ()):
            c = self.ord._count(t)
            if best is None or c > best[0]:
                best = (c, t[so])
        self._mv[(w, d)] = None if best is None else best[1]
        return self._mv[(w, d)]

    def between(self, a, b):
        """a から見た b のずれ: 同じ・次・前。どれでもなければ None。"""
        if a == b:
            return "同じ"
        if self._move(a, "次") == b:
            return "次"
        if self._move(a, "前") == b:
            return "前"
        return None

    def _fill(self, tri):
        """並びを埋める: 自分で運んだ答えが「x の次（前）」で止まり、その席が空いていて、聞いた答えも並びに居ない時、そこに座らせる。"""
        c = tri[2]
        if self._move(c, "次") is not None or self._move(c, "前") is not None:
            return None
        _, ends, _ = self._carry(tri[0], tri[1], None, True)
        if not ends:
            return None
        x, dc = max(ends, key=ends.get)
        lo, hi = (x, c) if dc == "次" else (c, x)
        if self._move(lo, "次") is not None or self._move(hi, "前") is not None:
            return None
        self.ord.learn((lo, STEP, hi))
        self._mv.clear()
        self.filled.append((lo, hi))
        return {lo, hi}

    def _facts(self, si, w):
        return list(self.rel.by[si].get(w, ()))

    # ---- 覚える ----
    def learn(self, tri):
        tri = tuple(tri)
        mine = self._fill(tri) if self.fill else None
        for si in (0, 1):
            other = 1 - si
            for f in self._facts(other, tri[other]):
                for x, y in ((f, tri), (tri, f)):         # 両方の向きで数える（聞いた順に寄らない）
                    d = self.between(x[si], y[si])
                    if d not in ("次", "前"):
                        continue
                    if mine is not None and {x[2], y[2]} == mine:
                        continue                          # 自分で埋めた並びは、決まりの数には入れない（自分で自分を確かめない）
                    self.rule.setdefault((si, d), Counter())[self.between(x[2], y[2]) or "ほか"] += 1
        self.rel.learn(tri)

    # ---- 問う ----
    def _carry(self, a, b, cset, far):
        """聞いた事のある事実に着くまで片方ずつ 1 歩ずつ戻り、戻った歩の分だけ答えを運ぶ。
        返す: (答えの票, 最後の 1 歩だけ運べなかった所 (語, ずれ) の票, 歩)。cset が None なら候補を絞らない。"""
        seen, layer, votes, ends, depth = {(a, b)}, {(a, b): ()}, Counter(), Counter(), 0
        while layer and not votes and not ends:
            depth += 1
            nxt = {}
            for st, path in layer.items():
                for si in (0, 1):
                    for m in ("前", "次"):
                        w = self._move(st[si], m)
                        if w is None:
                            continue
                        s2 = (w, st[1]) if si == 0 else (st[0], w)
                        if s2 in seen or s2 in nxt:
                            continue
                        nxt[s2] = path + ((si, BACK[m]),)  # s2 から見て、st の語は BACK[m] だけずれている
            for st, path in nxt.items():
                seen.add(st)
                for f in self._facts(0, st[0]):
                    if f[1] != st[1]:
                        continue
                    c, wgt, net = f[2], self.rel._count(f), 0
                    for si, d in path:                    # 運ぶ歩は順に寄らないので、次と前を差し引いてまとめて運ぶ（途中で並びの端を越えないように）
                        rule = self.rule.get((si, d))
                        dc = max(rule, key=rule.get) if rule else None
                        if dc not in ("次", "前", "同じ"):
                            net = None
                            break
                        wgt *= rule[dc] / sum(rule.values())
                        net += 1 if dc == "次" else -1 if dc == "前" else 0
                    if net is None:
                        continue
                    dc = "次" if net > 0 else "前"
                    for k in range(abs(net)):
                        c2 = self._move(c, dc)
                        if c2 is None:
                            if k == abs(net) - 1:
                                ends[(c, dc)] += wgt      # 最後の 1 歩だけ運べなかった
                            c = None
                            break
                        c = c2
                    if c is not None and (cset is None or c in cset):
                        votes[c] += wgt
            layer = nxt if far else {}
        return votes, ends, depth

    def ask(self, a, b, cands, far=True):
        """far=False なら 1 歩だけ運ぶ。True なら何歩でも（数え足し）。"""
        o = self.rel.ask(2, (a, b, None), cands)
        if o["段"] == "2語":
            self.rec.append((a, b, "2語", o["答え"], 0))
            return o
        votes, _, depth = self._carry(a, b, set(cands), far)
        if not votes:
            self.rec.append((a, b, o["段"], o["答え"], 0))
            return o
        cl = list(cands)
        best = max(votes, key=lambda c: (votes[c], -cl.index(c)))
        out = dict(段="ずれを運ぶ", 点=dict(votes), 答え=best, 歩=depth)
        self.rec.append((a, b, out["段"], best, depth))
        return out
