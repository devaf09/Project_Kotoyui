"""I49m: 語の「元の向き」を、語の波ではなく属性の割合で作る（持ち主の案「単語ではなく属性でそろえてほしい」、未確定）。

持ち主の案: 単語 1 つに 1 つの属性をピックアップせずに、属性の周波数全体か、属性内容ごとに位相差を持つ形。
ここでは、属性 1 つに周波数 1 本を割り当て、語が持つ属性の割合をそのままその周波数の強さにする（1 つを選ばず、持っている属性を全部使う）。
「どの席で一緒だったか」は、I49k の SeatVec(bind="位相") が位相の回し方で持つ。つまり 強さ = 属性の割合、位相 = 席。

  AttrBase(内容語の板, bands=None)
    内容語の板: [{name, pos, attrs: {帯: {属性: 割合}}}, ...]（語板の属性の帯の表）
    bands: 使う帯の名前の並び。None なら全部
    of(語) → 長さ（属性の数）の向き か None。語は (見出し語, 品詞) の組か、見出し語の文字列
    n_attr                  属性（周波数）の数
  同じ見出し語が板に何度も出る時は、品詞が合う物を先に、無ければ初めに出た物を使う。
手で置く物: 品詞の対応（NOUN→名詞 など）、同じ名前の語の選び方、割合をそのまま強さにする事（帯ごとの重さは付けない）。
"""
from __future__ import annotations

import numpy as np

POS = {"NOUN": "名詞", "PROPN": "固有名詞", "VERB": "動詞", "ADJ": "形容詞", "ADV": "副詞"}


class AttrBase:
    def __init__(self, board, bands=None):
        self.bands = None if bands is None else tuple(bands)
        self.ix = {}                                      # (帯, 属性) → 周波数の番号
        self.by = {}                                      # 見出し語 → [(品詞, {番号: 割合})]
        for w in board:
            row = {}
            for band, d in (w.get("attrs") or {}).items():
                if self.bands is not None and band not in self.bands:
                    continue
                for k, r in d.items():
                    if r:
                        row[self.ix.setdefault((band, k), len(self.ix))] = float(r)
            if row:
                self.by.setdefault(w["name"], []).append((w.get("pos"), row))
        self.n_attr = len(self.ix)
        self._v = {}

    def of(self, word):
        if word in self._v:
            return self._v[word]
        name, pos = (word if isinstance(word, tuple) else (word, None))
        rows = self.by.get(name)
        v = None
        if rows:
            want = POS.get(pos, pos)
            row = next((r for p, r in rows if p == want), rows[0][1])
            v = np.zeros(self.n_attr, dtype=np.complex64)
            for i, r in row.items():
                v[i] = r
        self._v[word] = v
        return v
