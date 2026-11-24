"""I49zm: 1 本の波に、属性の波と、語ごとに固有の位相を並べて持つ（持ち主の案、未確定の試し）。

持ち主の言葉:「一つの波の中に属性の波だけではなく、ごそのものを判断区別できる直交の位相、固有の位相を混ぜるまたは２チャンネルで持たせる
仕組みはどうかな？、語板と属性の波を別々にくらべず、属性で大枠、固有いそうで既知との一致、うーん。」
語の元の波 = [属性の波（長さ 1）, mix × 固有の波（長さ 1）] を 1 本に並べた物。SeatVec の base_of にそのまま渡せる。
  属性の側: 「こういう属性の相手と座った」が覚えた波に残る（大枠。聞いた事の無い相手にも広がる）
  固有の側: 「まさにこの語と座った」が覚えた波に残る（既知との一致。ほかの語とはほぼ直交）
固有の波は、語ごとに決まった位相の並び（本数 n_id）。語の番号は、初めて見た順に振る（乱数もハッシュも使わない）。
  位相 = 2π × ((番号 × 黄金比の小数部 × k² + 番号² × (√2 − 1) × k) の小数部)   k = 1..n_id   （I49q の小さい世界と同じ決め方）

  TwoChannel(attr_of, n_id=1024, mix=1.0, wave_of=None)
    of(語) → 並べた波。属性を持たない語は、属性の側が 0（固有の側だけで比べられる）
    wave_of を渡すと、固有の波の代わりにその波（語の波。似た語どうしが近い）を並べる（比べ役）。波の無い語は、その側が 0
    mix=0 なら属性だけと同じ答えになる。
    n_attr・n_wave: 属性の側・語の波の側の長さ。渡しておくと、片方しか持たない語を初めから作れる（渡さなければ、分かるまでその語は None）。
手で置く物: 固有の位相の本数 n_id、混ぜ具合 mix、位相の決め方。
"""
from __future__ import annotations

import numpy as np

G, R2 = 0.6180339887498949, 0.4142135623730951


class TwoChannel:
    def __init__(self, attr_of, n_id=1024, mix=1.0, wave_of=None, n_attr=None, n_wave=None):
        self.attr_of, self.n_id, self.mix, self.wave_of = attr_of, int(n_id), float(mix), wave_of
        self.num = {}                                     # 語 → 初めて見た順の番号（1 から）
        self._v = {}
        self._na = n_attr                                 # 属性の側の長さ（初めて属性を持つ語を見た時に決まる）
        self._nd = self.n_id if wave_of is None else n_wave # 固有の側の長さ（語の波を並べる時は、初めて波を持つ語を見た時に決まる）

    def ident(self, word):
        if self.wave_of is not None:
            v = self.wave_of(word)
            if v is None:
                return None
            v = np.asarray(v, dtype=np.complex64)
            n = float(np.linalg.norm(v))
            return v / n if n > 0 else None
        i = float(self.num.setdefault(word, len(self.num) + 1))
        k = np.arange(1, self.n_id + 1, dtype=np.float64)
        ph = (i * G * k * k + i * i * R2 * k) % 1.0
        return (np.exp(2j * np.pi * ph) / np.sqrt(self.n_id)).astype(np.complex64)

    def of(self, word):
        if word in self._v:
            return self._v[word]
        a = self.attr_of(word)
        if a is not None:
            a = np.asarray(a, dtype=np.complex64)
            n = float(np.linalg.norm(a))
            a = a / n if n > 0 else None
        if a is not None and self._na is None:
            self._na = len(a)
        d = self.ident(word) if self.mix > 0 else None
        if d is not None and self._nd is None:
            self._nd = len(d)
        if a is None and d is None:
            self._v[word] = None
            return None
        if self._na is None or (self.mix > 0 and self._nd is None):   # 長さがまだ分からない側が有る: 分かるまで作らない
            return None
        za = a if a is not None else np.zeros(self._na, dtype=np.complex64)
        if self.mix > 0:                                  # 片方しか持たない語は、無い側を 0 で埋める（長さをそろえる）
            zd = np.float32(self.mix) * d if d is not None else np.zeros(self._nd, dtype=np.complex64)
            v = np.concatenate([za, zd])
        else:
            v = za
        self._v[word] = v.astype(np.complex64)
        return self._v[word]
