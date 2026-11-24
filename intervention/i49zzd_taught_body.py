"""I49zzd: 介入で教わった事を、本体と同じ形の外付けに持つ（未確定の試し）。

I49zzb・I49zzc で、介入の板（鍵 = 2 語の組）は、本体が聞いていない事を教えても効かなかった。この材料では「2 語の組 → 語」の覚えの上限が低く、
本体が聞いて上がる分は、語ごとの覚えから来ていた。ここでは、介入が運ぶ物を、本体と同じ粒で持つ。
I48 の決まりは引き継ぐ: 介入は本体に混ぜず、外付けに持つ。外せば元に戻る。

  TaughtBody(body, ext, fresh=None, reflect=False)
    body: 本体。ext: 本体と同じ形の、空の外付け（教えられた 3 つ組だけを聞く）。
    反映とセーブ（I49zze。持ち主の言葉「外付けを短期記憶とかその場の考え・一時的なスキルと考えれば不自然ではないと思うけど、
      しいて言えば本体への反映と本体からのセーブがあればなおいいかも？」から。未確定の試し）:
      reflect=True なら、話の終わり（rest か tick）に、外付けが聞いた 3 つ組を本体に聞かせ、外付けを空に戻す（fresh() で作り直す）。
      save() は本体の写しを返す。restore(写し) で、本体をその時の姿に戻し、外付けも空にする（「外せば元に戻る」の代わりに「セーブした所まで戻れる」）。
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
    log = 問いごとの (段の組か None, 外付けの答えを使ったか)。数える道具が、害の出た場面を見るために使う。
  Direct(body)   比べ役: 教わった 3 つ組を、本体に直に聞かせる（混ぜる。外せない）。teach / settle / learn / tick は TaughtBody と同じ呼び方。
手で置く物: 同じなら本体。乱数は置かない。
"""
from __future__ import annotations


class TaughtBody:
    def __init__(self, body, ext, fresh=None, reflect=False):
        assert not reflect or fresh is not None
        self.body, self.ext, self.fresh, self.do_reflect = body, ext, fresh, bool(reflect)
        self._held, self.reflected = [], 0
        self.tally, self._wait, self.last = {}, [], None
        self._mark, self._seats, self.keys, self.n = False, set(), set(), 0
        self.used = dict(本体=0, 外付け=0)
        self.log = []

    def ask(self, si, tri, cands):
        cands = list(cands)
        o = self.body.ask(si, tri, cands)
        e = self.ext.ask(si, tri, cands) if self.n else dict(段="答えられない", 点={}, 答え=None)
        key = (si,) + tuple(w for j, w in enumerate(tri) if j != si)
        self.last = (e["答え"] is not None, key in self.keys, o["答え"], e["答え"])
        if e["答え"] is None:
            self.used["本体"] += 1
            self.log.append((None, False))
            return o
        cell = (o["段"], e["段"])
        self._wait.append((si, cell, o["答え"], e["答え"]))
        h = self.tally.get(cell, (0, 0, 0))
        self.log.append((cell, h[1] > h[0]))
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
            self._held.append(tri)
            self.n += 1
        elif hasattr(self.ext, "drop"):
            self.ext.drop()
        self._mark, self._seats = False, set()

    def learn(self, tri):
        self.settle(tri, every=True)
        self.body.learn(tri)

    def reflect(self):
        """外付けが聞いた 3 つ組を本体に聞かせ、外付けを空に戻す。反映した 3 つ組の数を返す。"""
        if hasattr(self.body, "drop"):
            self.body.drop()
        for tri in self._held:
            self.body.learn(tri)
        k = len(self._held)
        if k:
            self.body.tick()
        self.reflected += k
        self.ext, self._held, self.n = self.fresh(), [], 0
        return k

    def save(self):
        import copy
        return copy.deepcopy(self.body)

    def restore(self, snap):
        import copy
        self.body = copy.deepcopy(snap)
        self._held, self.n, self._wait = [], 0, []
        if self.fresh is not None:
            self.ext = self.fresh()

    def tick(self):
        self.body.tick()
        if self.do_reflect:
            self.reflect()
        else:
            self.ext.tick()

    def rest(self):
        if self.do_reflect:
            self.reflect()
        else:
            self.ext.tick()


class Direct:
    def __init__(self, body):
        self.body, self._mark, self.n = body, False, 0

    def ask(self, si, tri, cands):
        return self.body.ask(si, tri, cands)

    def teach(self, si, tri, answered):
        self._mark = True
        return True

    def settle(self, tri, every=False):
        if self._mark:
            self.body.learn(tri)
            self.n += 1
        elif hasattr(self.body, "drop"):
            self.body.drop()
        self._mark = False

    def learn(self, tri):
        self._mark = False
        self.body.learn(tri)

    def tick(self):
        self.body.tick()

    def rest(self):
        self.body.tick()
