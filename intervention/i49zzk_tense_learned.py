"""I49zzk: 時制の波を、手で置かずに学習から作る（持ち主の案、未確定の試し）。

持ち主の案: 時制の波形を手で置かず、学習データから関連付ける。時制が有る語ごとに時制の波を持たせ、文の語板の全体に掛ける。
  時制に関係の無い語は影響を受けず、関係の有る語は受ける。動詞の後につなぎの語が有れば、そのたびに時制の波を足す。

作り方（数えて足すだけ。並び順も山の位置も置かない）:
  時の語は、初めて出た順に軸の席を 1 つもらう（意味の無い席）。
  文の中の語（だれが・動詞・つなぎの語）は、一緒に出た時の語の席に 1 を足す。
  語の時制の波 = その語の席ごとの割合 − 全部の語を合わせた席ごとの割合（平均を引く。center=False なら引かない）。
  形の波 = 動詞の波 + つなぎの語の波の足し算。状況波との近さは内積で測る（つなぎの語が無い形は波が 0 に近い）。
比べ役 Whole: 動詞と形をひとまとめ（「時制波ー走る」の形）で数える。動詞の間で分け合わない。

手で置いた物: 平均を引く事、内積で測る事。
乱数は使わない。float32。
"""
from __future__ import annotations

import numpy as np


class Learned:
    """語ごとに、一緒に出た時の語を数えて、時制の波を作る。"""

    def __init__(self, center=True):
        self.center = center
        self.seat = {}       # 時の語 → 軸の席
        self.count = {}      # 語 → {席: 回数}
        self.total = {}      # 席 → 回数（全部の語）

    def units(self, verb, aux):
        """数える単位。ここでは動詞と、つなぎの語の 1 つずつ。"""
        return [("動詞", verb)] + [("つなぎ", a) for a in aux]

    def learn(self, time_word, who, verb, aux):
        s = self.seat.setdefault(time_word, len(self.seat))
        for u in [("だれが", who)] + self.units(verb, aux):
            c = self.count.setdefault(u, {})
            c[s] = c.get(s, 0) + 1
            self.total[s] = self.total.get(s, 0) + 1

    def wave(self, unit):
        """語の時制の波。聞いた事の無い語は 0。"""
        v = np.zeros(len(self.seat), dtype=np.float32)
        c = self.count.get(unit)
        if not c:
            return v
        for s, n in c.items():
            v[s] = n
        v /= v.sum()
        if self.center:
            m = np.array([self.total.get(s, 0) for s in range(len(self.seat))], dtype=np.float32)
            v = v - m / m.sum()
        return v

    def strength(self, unit):
        """時制の影響を受ける強さ（波の長さ）。"""
        return float(np.linalg.norm(self.wave(unit)))

    def form_wave(self, verb, aux):
        v = np.zeros(len(self.seat), dtype=np.float32)
        for u in self.units(verb, aux):
            v = v + self.wave(u)
        return v

    def state_of(self, time_word):
        """時の語が立てる状況波（その語の席に 1）。知らない語は None。"""
        if time_word not in self.seat:
            return None
        v = np.zeros(len(self.seat), dtype=np.float32)
        v[self.seat[time_word]] = 1.0
        return v

    def pick(self, verb, forms, state):
        """forms: {時制: つなぎの語の並び}。状況波に一番近い形の時制。同じ近さなら先に書いた方。"""
        return max(forms, key=lambda t: float(state[:len(self.seat)] @ self.form_wave(verb, forms[t])))


class Whole(Learned):
    """比べ役: 動詞と形をひとまとめで数える（動詞の間で分け合わない）。"""

    def units(self, verb, aux):
        return [("形", verb, tuple(aux))]
