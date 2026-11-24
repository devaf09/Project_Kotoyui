"""I49zzd: 介入で教わった事を、本体と同じ形の外付けに持つ（未確定の試し）。

I49zzb・I49zzc で、介入の板（鍵 = 2 語の組）は、本体が聞いていない事を教えても効かなかった。この材料では「2 語の組 → 語」の覚えの上限が低く、
本体が聞いて上がる分は、語ごとの覚えから来ていた。ここでは、介入が運ぶ物を、本体と同じ粒で持つ。
I48 の決まりは引き継ぐ: 介入は本体に混ぜず、外付けに持つ。外せば元に戻る。

  TaughtBody(body, ext)
    body: 本体。ext: 本体と同じ形の、空の外付け（教えられた 3 つ組だけを聞く）。
    ask: 本体の答えと、外付けの答えを出す。どちらを使うかは重さを置かず、（本体の答えた段, 外付けの答えた段）ごとに、
         それまでにどちらが多く当たったかで決める（同じなら本体）。数えは問う前の分だけ。
    teach(隠した席, 3 つ組, 出した答え): その 3 つ組を「教えられた」と印を付ける。外付けが聞くのは、その 3 つ組の問いが全部終わった後
         （settle か learn の時）。先に聞かせると、同じ 3 つ組のほかの席の問いで答えを知っている事になるため。
    settle(3 つ組): 本体には聞かせず、印が有れば外付けに聞かせる。どちらが当たったかの数えは、教えが届いた席の問いの分だけ進める
         （教えが届かなかった問いの正解は、知らないはずなので使わない）。
    learn(3 つ組): 本体にも聞かせる。3 つ組の全部を聞くので、数えは全部の席の分を進める。
    tick(): 本体と外付けの両方。rest(): 外付けだけ（本体が聞かない場面の、話の終わり用）。
    last = 直前の問いの (外付けが答えを持ったか, その 2 語の組を教えた事が有るか, 本体の答え, 外付けの答え)。
    tally = (本体の段, 外付けの段) → [本体の当たり, 外付けの当たり, 問い]。used = どちらの答えを使ったかの数。
手で置く物: 同じなら本体。乱数は置かない。
"""
from __future__ import annotations


class TaughtBody:
    def __init__(self, body, ext):
        self.body, self.ext = body, ext
        self.tally, self._wait, self.last = {}, [], None
        self._mark, self._seats, self.keys, self.n = False, set(), set(), 0
        self.used = dict(本体=0, 外付け=0)

    def ask(self, si, tri, cands):
        cands = list(cands)
        o = self.body.ask(si, tri, cands)
        e = self.ext.ask(si, tri, cands) if self.n else dict(段="答えられない", 点={}, 答え=None)
        key = (si,) + tuple(w for j, w in enumerate(tri) if j != si)
        self.last = (e["答え"] is not None, key in self.keys, o["答え"], e["答え"])
        if e["答え"] is None:
            self.used["本体"] += 1
            return o
        cell = (o["段"], e["段"])
        self._wait.append((si, cell, o["答え"], e["答え"]))
        h = self.tally.get(cell, (0, 0, 0))
        if h[1] > h[0]:
            self.used["外付け"] += 1
            return dict(e, 段="外付け")
        self.used["本体"] += 1
        return o

    def teach(self, si, tri, answered):
        self._mark = True
        self._seats.add(si)
        self.keys.add((si,) + tuple(w for j, w in enumerate(tri) if j != si))
        return True

    def settle(self, tri, every=False):
        for si, cell, a, b in self._wait:
            if not every and si not in self._seats:
                continue
            h = self.tally.setdefault(cell, [0, 0, 0])
            h[0] += a == tri[si]
            h[1] += b == tri[si]
            h[2] += 1
        self._wait = []
        if self._mark:
            self.ext.learn(tri)
            self.n += 1
        elif hasattr(self.ext, "drop"):
            self.ext.drop()
        self._mark, self._seats = False, set()

    def learn(self, tri):
        self.settle(tri, every=True)
        self.body.learn(tri)

    def tick(self):
        self.body.tick()
        self.ext.tick()

    def rest(self):
        self.ext.tick()
