"""I49zb: いいとこ取りの形（持ち主の言葉「一度いいとこどりをした仕組みを一度作ってみてもいいかも」から。未確定の試し）。

今までの数えで、それぞれが強かった所だけを使う。
  1. 関係の表（I49e）が上の段（聞いた組そのもの・片方だけ借りる・述語側）で答えられる問いは、表の答え（I49s: 表が強い所）
  2. それ以外（もう片方・借りる・答えられない）は、波で解く形（I49k・I49m）の答え（I49s: 解く形が勝つ所）。解く形が答えを出せない時は、表の答えに戻る
  3. 解く形の答えは、選び直し（I49u・I49z の FlowPick）を通せる

  Mix(table, solve, upper=("2語", "片方だけ借りる", "述語側"))
    table: Relation。solve: Unbind か、それを包んだ FlowPick
    ask / learn / tick は Relation と同じ形。used は、どちらの答えを使ったかの数
  I49zc: Mix(table, solve, upper=None) は、表に任せる段を手で置かない。段ごとに、表と解く形のどちらが当たってきたかを
    答え合わせ（learn）のたびに数え、その段でそれまでに多く当たった方の答えを使う（同じなら表）。数えは問う前の分だけ。
    tally は 段 → [表の当たり, 解く形の当たり, 問いの数]。drop() は、答え合わせが来なかった問いの分を捨てる。
手で置く物: 表に任せる段の組 upper（I49s の数えで、表が勝った段をそのまま置いた）。upper=None の時は無し（同じなら表、だけ）。
"""
from __future__ import annotations


class Mix:
    def __init__(self, table, solve, upper=("2語", "片方だけ借りる", "述語側")):
        self.table, self.solve = table, solve
        self.upper = None if upper is None else tuple(upper)
        self.used = dict(表=0, 解く=0)
        self.tally, self._wait = {}, []

    def ask(self, si, tri, cands):
        cands = list(cands)
        t = self.table.ask(si, tri, cands)
        s = self.solve.ask(si, tri, cands)                 # 選び直しの板が育つように、毎回どちらにも問う
        self._wait.append((si, t["段"], t["答え"], s["答え"]))
        if self.upper is None:
            h = self.tally.get(t["段"], (0, 0, 0))
            use_table = h[0] >= h[1]
        else:
            use_table = t["段"] in self.upper
        if use_table and t["答え"] is None and s["答え"] is not None:
            use_table = False
        if use_table or s["答え"] is None:
            self.used["表"] += 1
            return t
        self.used["解く"] += 1
        return dict(s, 段="解く")

    def learn(self, tri):
        for si, lv, ta, sa in self._wait:
            h = self.tally.setdefault(lv, [0, 0, 0])
            h[0] += ta == tri[si]
            h[1] += sa == tri[si]
            h[2] += 1
        self._wait = []
        self.table.learn(tri)
        self.solve.learn(tri)

    def drop(self):
        """答え合わせが来なかった問いの分を捨てる（I49zzd: 教えられた 3 つ組だけを聞く使い方で、数えが別の 3 つ組とずれないように）。"""
        self._wait = []

    def take(self):
        """答え合わせを待っている問いの分を取り出して空にする（I49zzd の反映: 後でその 3 つ組を聞かせる時に give で戻す）。"""
        w, self._wait = self._wait, []
        return w

    def give(self, waits):
        self._wait = list(waits)

    def tick(self):
        self.table.tick()
        self.solve.tick()
