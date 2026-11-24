"""I49zx: 介入の板（設計案の 1 つ目の段階。持ち主の言葉「この辺から介入の仕組みをこれに合わせて設計してほしい」から。未確定の試し）。

介入 = 答えを出した後に、外から届く「合っている・違う（違うなら正しい語）」。届かなければ、本体と全く同じ動き。
I48 の決まりを引き継ぐ: 介入は本体に混ぜず、別の板に持つ。外せば元に戻る。だめな答えは、暗い側へ弱く入れる。薄れるだけで、消さない。

  TeachBoard(attr_of, dim=8192, dark=1/1000, half=1024)
    鍵 = 見せた 2 語の組の印（I49zw の PairSolve、広げてから掛けて大きさをそろえた物）。値 = その席の語。
    teach(隠した席, 3 つ組, 出した答え): 正しい語を明るい側に足す。出した答えが違っていたら、その語を暗い側に dark の重さで足す。
    points(隠した席, 問いの 3 つ組, 候補) → 候補 → 点（明るい側との近さ − 暗い側との近さ）。板に何も無ければ空。
    教えるたびに、板の全部が half 回で半分になる速さで薄れる。
    taught(隠した席, 問い): この組そのものを教えた事が有るか（数える用。答え方には使わない）。
  Taught(body, board)   本体に介入の板を外付けする。
    ask: 本体の答えと、板の答え（点が正の候補が有る時だけ）を出す。どちらを使うかは重さを置かず、板が答えを持った問いで
         それまでにどちらが多く当たったかで決める（同じなら本体）。数えは問う前の分だけ。
    learn / tick: 本体と同じ。teach(隠した席, 3 つ組, 出した答え): 板に教える。
    last = 直前の問いの (板が答えを持ったか, 教えた組そのものか, 本体の答え, 板の答え)。tally = [本体の当たり, 板の当たり, 問い]。
手で置く物: dark（持ち主の 1/1,000〜1/10,000 の上の端）、half、dim、同じなら本体。乱数は置かない。
"""
from __future__ import annotations

import numpy as np

from i49zw_dense_pair import PairSolve


class TeachBoard:
    def __init__(self, attr_of, dim=8192, dark=1.0 / 1000, half=1024.0):
        self.pair = PairSolve(attr_of, dim=dim, amp="位相だけ")
        self.dark_w = float(dark)
        self.keep = np.float32(0.5 ** (1.0 / half))
        self.bright = [{}, {}, {}]
        self.dark = [{}, {}, {}]
        self.keys = set()
        self.n = 0

    def _add(self, store, w, p):
        store[w] = p.copy() if w not in store else store[w] + p

    def teach(self, si, tri, answered):
        p = self.pair.sign(si, tri)
        if p is None:
            return False
        for side in (self.bright, self.dark):
            for w in side[si]:
                side[si][w] *= self.keep
        self._add(self.bright[si], tri[si], p)
        if answered is not None and answered != tri[si]:
            self._add(self.dark[si], answered, (p * np.float32(self.dark_w)).astype(np.complex64))
        self.keys.add((si,) + tuple(w for j, w in enumerate(tri) if j != si))
        self.n += 1
        return True

    def taught(self, si, tri):
        return (si,) + tuple(w for j, w in enumerate(tri) if j != si) in self.keys

    def points(self, si, tri, cands):
        if not self.n:
            return {}
        p = self.pair.sign(si, tri)
        if p is None:
            return {}
        s = {}
        for c in cands:
            b, d = self.bright[si].get(c), self.dark[si].get(c)
            if b is None and d is None:
                continue
            s[c] = (0.0 if b is None else float(np.real(np.vdot(p, b)))) - (0.0 if d is None else float(np.real(np.vdot(p, d))))
        return s


class Taught:
    def __init__(self, body, board):
        self.body, self.board = body, board
        self.tally, self._wait, self.last = [0, 0, 0], [], None

    def ask(self, si, tri, cands):
        cands = list(cands)
        o = self.body.ask(si, tri, cands)
        s = self.board.points(si, tri, cands)
        b = max(s, key=lambda c: (s[c], -cands.index(c))) if s and max(s.values()) > 0 else None
        self.last = (b is not None, self.board.taught(si, tri), o["答え"], b)
        if b is None:
            return o
        self._wait.append((si, o["答え"], b))
        if self.tally[1] > self.tally[0]:
            return dict(o, 段="介入の板", 答え=b)
        return o

    def learn(self, tri):
        for si, a, b in self._wait:
            self.tally[0] += a == tri[si]
            self.tally[1] += b == tri[si]
            self.tally[2] += 1
        self._wait = []
        self.body.learn(tri)

    def tick(self):
        self.body.tick()

    def teach(self, si, tri, answered):
        return self.board.teach(si, tri, answered)
