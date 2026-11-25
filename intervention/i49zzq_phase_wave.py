"""I49zzq: 状況波を位相の形に置き換える（持ち主の案、未確定の試し）。

持ち主の案:「昨日作った状況波を位相で置き換えて、自然発生的に方向をつくれるかな？複数の波を作って、位相の次元は数千は必要だと思ってるんだけど。」
作り方（数えて足すだけ。向きは置かない）:
  席（前の文の終わりの切り出し）ごとに、DIMS 本の周波数の上の位相を 1 つずつ持たせる。位相そのものに意味は無い（席を見分ける札）。
    位相は黄金比から作る（乱数は使わない）: 席 s、周波数 d の位相 = 2π × ((s+1) × ((d+1) × 黄金比 の小数) の小数)。
  今の文の終わりの切り出しの波 = Σ（その切り出しと席の結び付き × 席の位相の波）。結び付きは i49_situation.Assoc の数えた値（平均を引いた割合）。
  状況波 = 前の文の終わりの切り出しの、席の位相の波の足し算（何文か重ねる時は、古い物ほど fade を掛けて薄くする）。1 本の複素の波。
  合い方 = 今の文の終わりの波と、状況波の、位相をそろえた掛け算の実部の足し算 ÷ DIMS。
  DIMS が大きいほど、席どうしの波が混ざらず、元の数えた値での合い方に近づく。
元の形との違い: 元は「切り出し × 席の表」を持ち、状況は終わりの並びで渡す。ここでは、状況も終わりも、同じ長さ（DIMS）の 1 本の波になる。
手で置いた物: DIMS、位相の作り方、fade。
乱数は使わない。complex64。
"""
from __future__ import annotations

import numpy as np

PHI = 0.6180339887498949


def seat_phases(n_seats, dims):
    """席ごとの位相の波（席 × dims、大きさ 1 の複素数）。"""
    alpha = ((np.arange(1, dims + 1, dtype=np.float64) * PHI) % 1.0)
    th = (np.arange(1, n_seats + 1, dtype=np.float64)[:, None] * alpha[None, :]) % 1.0
    return np.exp(2j * np.pi * th).astype(np.complex64)


class PhaseAssoc:
    """i49_situation.Assoc の数えた値を、位相の波に写す。"""

    def __init__(self, assoc, dims=1024):
        self.a = assoc
        self.dims = dims
        self.e = seat_phases(len(assoc.seats), dims)                 # 席の位相の波
        self.z = (assoc.w.astype(np.complex64) @ self.e) if assoc.w.size else np.zeros((len(assoc.units), dims), dtype=np.complex64)

    def situation(self, prev_tails, fade=1.0):
        """状況波: 前の文の終わり（新しい物が後ろ）の、席の位相の波の足し算。古い物ほど fade を掛ける。"""
        s = np.zeros(self.dims, dtype=np.complex64)
        k = len(prev_tails)
        for i, p in enumerate(prev_tails):
            ps = self.a.ids(p, self.a.six)
            if len(ps):
                s += np.complex64(fade ** (k - 1 - i)) * self.e[ps].sum(axis=0)
        return s

    def wave(self, cur):
        """今の文の終わりの波。"""
        us = self.a.ids(cur, self.a.uix)
        return self.z[us].sum(axis=0) if len(us) else np.zeros(self.dims, dtype=np.complex64)

    def fit(self, sit, cur):
        return float(np.real(np.vdot(sit, self.wave(cur)))) / self.dims

    def score(self, prev, cur):
        return self.fit(self.situation([prev]), cur)
