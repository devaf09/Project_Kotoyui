"""I49zx: 介入の板（設計案の 1 つ目の段階。持ち主の言葉「この辺から介入の仕組みをこれに合わせて設計してほしい」から。未確定の試し）。

介入 = 答えを出した後に、外から届く「合っている・違う（違うなら正しい語）」。届かなければ、本体と全く同じ動き。
I48 の決まりを引き継ぐ: 介入は本体に混ぜず、別の板に持つ。外せば元に戻る。だめな答えは、暗い側へ弱く入れる。薄れるだけで、消さない。

  TeachBoard(attr_of, dim=8192, dark=1/1000, half=1024, amp="位相だけ", usual=None, rare=None)
    amp・usual・rare: 鍵の作り方（I49zw の PairSolve にそのまま渡す）。I49zy の直しは amp="そのまま" と、usual（いつもの属性を引く）か rare（めずらしさを掛ける）。
    鍵 = 見せた 2 語の組の印（I49zw の PairSolve、広げてから掛けて大きさをそろえた物）。値 = その席の語。
    teach(隠した席, 3 つ組, 出した答え): 正しい語を明るい側に足す。出した答えが違っていたら、その語を暗い側に dark の重さで足す。
    points(隠した席, 問いの 3 つ組, 候補) → 候補 → 点（明るい側との近さ − 暗い側との近さ）。板に何も無ければ空。
    教えるたびに、板の全部が half 回で半分になる速さで薄れる。
    deny(隠した席, 問いの 3 つ組, 語): 板が広げて出した答えへの「違う」（I49zz、未確定の試し）。その組の印を、その語の「止める側」に stop の重さで足す。
      stop="出した分" にすると、重さを置かず、板がその語に今出している点と同じ量だけ足す（教えた問いの点がちょうど 0 になる。点が 0 以下なら何も足さない）。
      points は 明るい側 − 暗い側 − 止める側。暗い側（本体の間違いを薄く覚える）とは別に持つ。本体には触らない。薄れ方は同じ。
    far(隠した席, 問いの 3 つ組): 「この問いは遠い（板の出番ではない）」の教え（I49zza、未確定の試し）。板がその問いで正の点を出している語の全部を、
      それぞれ出した分だけ止める側に足す（stop の値に関わらず、出した分）。教えた問いでは、板は答えなくなる。止めた語の数を返す。
    taught(隠した席, 問い): この組そのものを教えた事が有るか（数える用。答え方には使わない）。
  Taught(body, board)   本体に介入の板を外付けする。
    ask: 本体の答えと、板の答え（点が正の候補が有る時だけ）を出す。どちらを使うかは重さを置かず、板が答えを持った問いで
         それまでにどちらが多く当たったかで決める（同じなら本体）。数えは問う前の分だけ。
    learn / tick: 本体と同じ。settle(3 つ組): 本体には聞かせず、どちらが当たったかの数えだけ進める（本体が聞かない場面の数え用。I49zzb）。
         数えるのは、教えが届いた席の問いの分だけ（届かなかった問いの正解は、知らないはずなので使わない）。teach(隠した席, 3 つ組, 出した答え): 板に教える。
    last = 直前の問いの (板が答えを持ったか, 教えた組そのものか, 本体の答え, 板の答え)。tally = [本体の当たり, 板の当たり, 問い]。
手で置く物: stop（止める側の重さ 1。明るい側と同じ強さ。"出した分" なら置かない）、dark（持ち主の 1/1,000〜1/10,000 の上の端）、half、dim、同じなら本体。乱数は置かない。
"""
from __future__ import annotations

import numpy as np

from i49zw_dense_pair import PairSolve


class TeachBoard:
    def __init__(self, attr_of, dim=8192, dark=1.0 / 1000, half=1024.0, amp="位相だけ", usual=None, rare=None, stop=1.0):
        self.pair = PairSolve(attr_of, dim=dim, amp=amp, usual=usual, rare=rare)
        self.dark_w = float(dark)
        self.keep = np.float32(0.5 ** (1.0 / half))
        self.bright = [{}, {}, {}]
        self.dark = [{}, {}, {}]
        self.stop_w = stop if stop == "出した分" else float(stop)
        self.stop = [{}, {}, {}]
        self.keys = set()
        self.n = 0

    def _add(self, store, w, p):
        store[w] = p.copy() if w not in store else store[w] + p

    def teach(self, si, tri, answered):
        p = self.pair.sign(si, tri)
        if p is None:
            return False
        for side in (self.bright, self.dark, self.stop):
            for w in side[si]:
                side[si][w] *= self.keep
        self._add(self.bright[si], tri[si], p)
        if answered is not None and answered != tri[si]:
            self._add(self.dark[si], answered, (p * np.float32(self.dark_w)).astype(np.complex64))
        self.keys.add((si,) + tuple(w for j, w in enumerate(tri) if j != si))
        self.n += 1
        return True

    def deny(self, si, tri, word):
        p = self.pair.sign(si, tri)
        if p is None:
            return False
        amount = self.points(si, tri, [word]).get(word, 0.0) if self.stop_w == "出した分" else self.stop_w
        if amount <= 0:
            return False
        for side in (self.bright, self.dark, self.stop):
            for w in side[si]:
                side[si][w] *= self.keep
        self._add(self.stop[si], word, (p * np.float32(amount)).astype(np.complex64))
        self.n += 1
        return True

    def far(self, si, tri):
        p = self.pair.sign(si, tri)
        if p is None:
            return 0
        words = sorted(set(self.bright[si]) | set(self.dark[si]) | set(self.stop[si]), key=str)
        out = {w: v for w, v in self.points(si, tri, words).items() if v > 0}
        if not out:
            return 0
        for side in (self.bright, self.dark, self.stop):
            for w in side[si]:
                side[si][w] *= self.keep
        for w, v in out.items():
            self._add(self.stop[si], w, (p * np.float32(v)).astype(np.complex64))
        self.n += 1
        return len(out)

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
            vs = [side[si].get(c) for side in (self.bright, self.dark, self.stop)]
            if all(v is None for v in vs):
                continue
            x = [0.0 if v is None else float(np.real(np.vdot(p, v))) for v in vs]
            s[c] = x[0] - x[1] - x[2]
        return s


class Taught:
    def __init__(self, body, board):
        self.body, self.board = body, board
        self.tally, self._wait, self.last = [0, 0, 0], [], None
        self._seats = set()

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

    def settle(self, tri, every=False):
        for si, a, b in self._wait:
            if not every and si not in self._seats:
                continue
            self.tally[0] += a == tri[si]
            self.tally[1] += b == tri[si]
            self.tally[2] += 1
        self._wait, self._seats = [], set()

    def learn(self, tri):
        self.settle(tri, every=True)
        self.body.learn(tri)

    def tick(self):
        self.body.tick()

    def teach(self, si, tri, answered):
        self._seats.add(si)
        return self.board.teach(si, tri, answered)
