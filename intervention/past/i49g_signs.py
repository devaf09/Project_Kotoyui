"""I49g: 印（＋、− など）ごとに事実を別に持ち、当たり外れだけで答えを探す時の「言う順」を 1 か所にまとめた部品（未確定）。

印ごとに事実を別に持つ:
  3＋7＝10 と 10−7＝3 は、別々の関係の表に入れる（印 → その印のずれを運ぶ部品 Shifter → 関係の表）。
  同じ表に入れると、「5 と 3 の答え」が 8（＋）と 2（−）の 2 つになり、どちらの印の事実か分からなくなるため。数え歌（並び）は、どの印の表にも同じ物を入れる。
  印どうしのつながりは、席の入れ替え（Flip）で数える: ある印の事実を探り当てるたびに、ほかの印の表に、席を入れ替えた物が有るかを数える。
答えを言う順（先生が 合ってる／違う だけ返す時）:
  1. 席の入れ替えの票の多い順（ほかの印の事実から）
  2. その印でずれを運んだ票の多い順
  3. ほかの印として答えた物（その印の事実がまだ 1 つも無い時だけ。知らない印なので、知っている印として言ってみる）
  4. その印の関係の表の点の多い順（下がって答える段）
  5. 最初に言った答えから、数え歌で近い順。最後に、残りを候補の並び順
手で置く物: この順そのもの。票の中身は数えた物。
"""
from __future__ import annotations
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする

from i49e_relation import Relation
from i49g_flip import Flip
from i49g_shift import Shifter


class Signs:
    def __init__(self, song):
        self.song = list(song)
        self.sh, self.fl = {}, {}                         # 印 → Shifter ／ (印, ほかの印) → Flip
        self.rec = []

    def _of(self, sign):
        if sign not in self.sh:
            s = Shifter(Relation(lambda w: None))
            s.hear_song(self.song)
            for other in self.sh:
                self.fl[(sign, other)] = Flip(self.sh[other].rel)
                self.fl[(other, sign)] = Flip(s.rel)
            self.sh[sign] = s
        return self.sh[sign]

    def n_facts(self, sign):
        return len(self._of(sign).rel.tri)

    def order(self, sign, a, b, cands):
        cands = list(cands)
        sh, out, why = self._of(sign), [], []

        def add(cs, name):
            for c in cs:
                if c not in out:
                    out.append(c)
                    why.append(name)

        votes = {}
        for (s, other), fl in self.fl.items():
            if s == sign:
                for c, v in fl.ask(a, b, cands)["点"].items():
                    votes[c] = votes.get(c, 0.0) + v
        add(sorted(votes, key=lambda c: (-votes[c], cands.index(c))), "席の入れ替え")
        o = sh.ask(a, b, cands)
        ranked = [c for c in sorted(cands, key=lambda c: (-o["点"].get(c, 0.0), cands.index(c))) if o["点"].get(c, 0.0) > 0]
        if o["段"] == "ずれを運ぶ":
            add(ranked, "ずれを運ぶ")
        if self.n_facts(sign) == 0:
            for other, so in self.sh.items():
                if other != sign and self.n_facts(other):
                    ans = so.ask(a, b, cands)["答え"]
                    if ans is not None:
                        add([ans], f"{other} として")
        if o["段"] != "ずれを運ぶ":
            add(ranked, "表の下がり方")
        if out:
            lo = hi = out[0]
            near = []
            while lo is not None or hi is not None:
                hi = sh._move(hi, "次") if hi is not None else None
                lo = sh._move(lo, "前") if lo is not None else None
                near += [w for w in (hi, lo) if w is not None and w in cands]
            add(near, "近い順")
        add(cands, "候補の並び順")
        self.rec.append((sign, a, b, out, why))
        return out, why

    def learn(self, sign, tri):
        """合ってると言われた（または聞いた）事実を、その印の表に覚える。ほかの印との席の入れ替えも数える。"""
        tri = tuple(tri)
        self._of(sign).learn(tri)
        for (s, other), fl in self.fl.items():
            if s == sign:
                fl.learn(tri)
