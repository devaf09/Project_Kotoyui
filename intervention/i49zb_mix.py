"""I49zb: いいとこ取りの形（持ち主の言葉「一度いいとこどりをした仕組みを一度作ってみてもいいかも」から。未確定の試し）。

今までの数えで、それぞれが強かった所だけを使う。
  1. 関係の表（I49e）が上の段（聞いた組そのもの・片方だけ借りる・述語側）で答えられる問いは、表の答え（I49s: 表が強い所）
  2. それ以外（もう片方・借りる・答えられない）は、波で解く形（I49k・I49m）の答え（I49s: 解く形が勝つ所）。解く形が答えを出せない時は、表の答えに戻る
  3. 解く形の答えは、選び直し（I49u・I49z の FlowPick）を通せる

  Mix(table, solve, upper=("2語", "片方だけ借りる", "述語側"))
    table: Relation。solve: Unbind か、それを包んだ FlowPick
    ask / learn / tick は Relation と同じ形。used は、どちらの答えを使ったかの数
手で置く物: 表に任せる段の組 upper（I49s の数えで、表が勝った段をそのまま置いた）。
"""
from __future__ import annotations


class Mix:
    def __init__(self, table, solve, upper=("2語", "片方だけ借りる", "述語側")):
        self.table, self.solve, self.upper = table, solve, tuple(upper)
        self.used = dict(表=0, 解く=0)

    def ask(self, si, tri, cands):
        cands = list(cands)
        t = self.table.ask(si, tri, cands)
        s = self.solve.ask(si, tri, cands)                 # 選び直しの板が育つように、毎回どちらにも問う
        if t["段"] in self.upper or s["答え"] is None:
            self.used["表"] += 1
            return t
        self.used["解く"] += 1
        return dict(s, 段="解く")

    def learn(self, tri):
        self.table.learn(tri)
        self.solve.learn(tri)

    def tick(self):
        self.table.tick()
        self.solve.tick()
