"""I49l: 語の向きを、行のうねりに合わせて足し算で作る骨組み（持ち主の案・未確定）。

持ち主の案: 「たくさんの波をちょっとづつそろえるよりも、１行ごとにうねりで行全体の語板に合わせて、似ている語が多い場合なるべく少ない周波数の調整で」。
1 行（ひとまとまりの語）を聞くたびに、行の語の元の向きを全部足した 1 本の「うねり」を作り、行の中のそれぞれの語に、うねり（自分の分を除く）を足す。
席は見ない。同じ行によく一緒に出る語どうしが近くなる。

  LineVec(base_of, where="全部")
    where="全部"    うねりの全部の周波数を足す
    where="強い所"  その行のうねりが、いつもより強い周波数だけを足す。「いつも」は、それまでに聞いた行のうねりの強さの、周波数ごとの平均
                    （器の「いつもより外れた分だけ」と同じ形。何本にするかの数は置かない）。初めの行は全部
    hear_line(語の並び)    足すだけ。元の向きの無い語は、うねりの材料にならないが、うねりは受け取る
    of(語) → 作った向き か None
    used                   足した周波数の割合の平均（where="強い所" で、どれだけ少ない周波数で済んだか）
手で置く物: うねりから自分の分を除く事、元の向きを長さ 1 にそろえる事、「いつも」をそれまでの全部の行の平均にした事。数は薄れさせていない。乱数は置かない。
"""
from __future__ import annotations
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする

import numpy as np


class LineVec:
    def __init__(self, base_of, where="全部"):
        assert where in ("全部", "強い所")
        self.base_of, self.where = base_of, where
        self.acc = {}
        self._b = {}
        self.usual = None                                 # 周波数ごとの、うねりの強さの足し合わせ
        self.lines = 0
        self._used = 0.0

    def _base(self, w):
        if w not in self._b:
            v = self.base_of(w)
            if v is not None:
                v = np.asarray(v, dtype=np.complex64)
                n = float(np.linalg.norm(v))
                v = v / n if n > 0 else None
            self._b[w] = v
        return self._b[w]

    def hear_line(self, words):
        words = list(dict.fromkeys(words))
        bs = {w: self._base(w) for w in words}
        have = [b for b in bs.values() if b is not None]
        if len(have) == 0:
            return
        swell = np.sum(have, axis=0)
        amp = np.abs(swell)
        if self.where == "強い所" and self.lines > 0:
            mask = amp > self.usual / self.lines
        else:
            mask = np.ones(len(swell), dtype=bool)
        self.usual = amp.astype(np.float64) if self.usual is None else self.usual + amp
        self.lines += 1
        self._used += float(mask.mean())
        for w in words:
            x = swell if bs[w] is None else swell - bs[w]
            x = np.where(mask, x, 0).astype(np.complex64)
            if not np.any(x):
                continue
            self.acc[w] = x if w not in self.acc else self.acc[w] + x

    @property
    def used(self):
        return self._used / self.lines if self.lines else None

    def of(self, w):
        return self.acc.get(w)
