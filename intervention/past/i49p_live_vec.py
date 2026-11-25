"""I49p: 語の波そのものを、聞くたびに動かす形（持ち主の案「周波数を追加し続ければ」「属性数の数だけあるといい」から、未確定）。

今までの形（I49k〜I49n）は、語の「元の向き」は動かさず、別に持った波に相手の元の向きを足していた。
ここでは、語の波は 1 本だけで、初めは元の向き（属性の割合など）。同じ行に出た語の「今の波」を、長さ 1 にそろえて足す。
聞くたびに、一緒に出る語どうしの波が近づく。近づいた波がまた次の語に足されるので、2 つ先の語にも広がる。
  LiveVec(base_of, far=False, bind="なし")
    hear_line(語の並び)   行の中のそれぞれの語に、同じ行のほかの語の今の波（長さ 1）を足す。席は見ない
    hear(3 つ組)          bind="位相" の時は席の組ごとの回し方で回して足す。bind="なし" の時は hear_line と同じ
    of(語) → 今の波 か None
    far=True   ほかの語を遠ざける: 足した数と同じだけ、今までに出た全部の語の波（長さ 1）の平均を引く。
               近づけるだけだと全部の語が同じ向きに寄るので、行に居なかった語からは離す。足し算と引き算と数えだけ。
  LiveAsk(lv, online=True, skip_shown=False)   3 つ組の 1 語を隠した問いに答える。点 = 候補の今の波と、見せた語の今の波の近さ（cos の実部）の合計。
               skip_shown=True なら、見せた語そのものは答えにしない。今の波は自分の元の向きを含むので、席を回さない時は、
               見せた語そのものがいちばん近くなってしまう（初めの数えで見落としていた作りの穴。18 節）。
手で置く物: 足す前に長さ 1 にそろえる事、遠ざける強さ（足した数と同じ）、元の向きを持たない語は 0 から始める事、回し方は I49k と同じ。乱数は置かない。
小さい自作の世界での総当たり（docs の 16 節）では、語 1 つの位相が 4 本以上あれば潰れず、元の向きを足す形より良かった。
"""
from __future__ import annotations
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする

import numpy as np

from i49k_seat_vec import SeatVec


def _unit(x):
    n = float(np.linalg.norm(x))
    return None if n == 0 else x / n


class LiveVec:
    def __init__(self, base_of, far=False, bind="なし"):
        assert bind in ("なし", "位相")
        self.base_of, self.far, self.bind = base_of, far, bind
        self._r = SeatVec(lambda w: None, weight="数", bind=bind)
        self.w = {}                                       # 語 → 今の波
        self.n_add = {}                                   # 語 → 足した語の数
        self.sum_u = None                                 # 今までに出た全部の語の、長さ 1 の波の合計
        self.n_u = 0

    def _get(self, w):
        if w not in self.w:
            b = self.base_of(w)
            self.w[w] = None if b is None else np.asarray(b, dtype=np.complex64).copy()
            u = None if self.w[w] is None else _unit(self.w[w])
            if u is not None:
                self._track(None, u)
        return self.w[w]

    def _track(self, old, new):
        if not self.far:
            return
        if old is None:
            self.n_u += 1
        d = new if old is None else new - old
        self.sum_u = d.astype(np.complex128) if self.sum_u is None else self.sum_u + d

    def _hear(self, words, seats):
        us = []
        for w in words:
            v = self._get(w)
            us.append(None if v is None else _unit(v))
        if sum(u is not None for u in us) == 0:
            return
        mean = None if not self.far or self.n_u == 0 else (self.sum_u / self.n_u).astype(np.complex64)
        new = {}
        for i, w in enumerate(words):
            x, k = None, 0
            for j, u in enumerate(us):
                if j == i or u is None:
                    continue
                y = u if seats is None else self._r.rot(u, (seats[i], seats[j]))
                x = y.copy() if x is None else x + y
                k += 1
            if x is None:
                continue
            if mean is not None:
                x = x - k * mean
            new[w] = (x, k)
        for i, w in enumerate(words):
            if w not in new:
                continue
            x, k = new[w]
            old = us[i]
            self.w[w] = x.astype(np.complex64) if self.w[w] is None else self.w[w] + x
            self.n_add[w] = self.n_add.get(w, 0) + k
            u = _unit(self.w[w])
            if u is not None:
                self._track(old, u)

    def hear_line(self, words):
        words = list(dict.fromkeys(words))
        if len(words) >= 2:
            self._hear(words, None)

    def hear(self, tri):
        t = list(tri)
        if self.bind == "なし" or len(set(t)) < 3:
            self.hear_line(t)
        else:
            self._hear(t, (0, 1, 2))

    def of(self, w):
        return self.w.get(w)


class LiveAsk:
    def __init__(self, lv, online=True, skip_shown=False):
        self.lv, self.online, self.skip_shown = lv, online, skip_shown
        self.rec = []

    def tick(self, n=1):
        pass

    def learn(self, tri):
        if self.online:
            self.lv.hear(tri)

    def ask(self, si, tri, cands):
        cands = list(cands)
        shown = []
        for sj in range(3):
            if sj == si:
                continue
            v = self.lv._get(tri[sj])
            u = None if v is None else _unit(v)
            if u is not None:
                shown.append(u if self.lv.bind == "なし" else self.lv._r.rot(u, (si, sj)))
        out = dict(段="答えられない", 点={}, 答え=None)
        if shown:
            aim = np.sum(shown, axis=0)
            s = {}
            hide = {tri[sj] for sj in range(3) if sj != si} if self.skip_shown else ()
            for c in cands:
                v = None if c in hide else self.lv._get(c)
                u = None if v is None else _unit(v)
                if u is not None:
                    s[c] = float(np.real(np.vdot(aim, u)))
            if s and max(s.values()) > 0:
                out = dict(段="解く", 点=s, 答え=max(s, key=lambda c: (s[c], -cands.index(c))))
        self.rec.append((0, si, tuple(tri[j] for j in range(3) if j != si), out["段"], out["答え"]))
        return out
