"""I48ab: 長いグレイン（文 → 要点 → 文）。語の名前ではなく、語板の属性を周波数に置いた波で数える（足すだけ。勾配の学びはしない）。

1. 入口の波（順方向）
   文の語ごとに、語板の属性の割合を周波数の帯に置く（内容語は語板 v2 の帯、機能語は機能語の板。属性の名前 → 周波数は
   f = (属性の番号 × STEP) mod D。属性の数が D を越えると重なる）。語の位置 k は周波数のシフト（f + k × SHIFT mod D）で重ね、
   1 本の長い波（complex64、長さ D）にする。語の名前そのものは入れない。
   要点の読み出し: 位置 k のシフトを戻した波と、候補の語の属性の波との近さ（cos）でいちばん近い語を選ぶ。読めた語の波をほかの位置から
   引いて読み直す（浮かび上がらせ）も試す。
   要点 = 内容語の原形の並び ＋ 意味の印（過去・否定・丁寧・進行・問い）。
2. 逆回し（要点 → 文）
   隣り合う内容語 i・i+1 の間の詰め物（助詞・語尾など、内容語でない語の表記の並び）を、2 語の属性の波の積
   （周波数の和 (f_a + f_b·MIX) mod D の帯。I48w の和の周波数と同じ考え）と、それぞれの語の波（別のずれ）の重ねから、
   多項の数え上げ（足すだけ、加算の平滑 α）で選ぶ。文末（最後の内容語の後ろ）は、最後の語の波と、意味の印の波（印ごとの周波数）で選ぶ。
   内容語の表記（活用）は、本の数え（原形・次の詰め物の頭 → 表記）から。無い原形は、原形の最後の字と次の詰め物の頭からの書き換えの数え。
3. 基準: 語そのものを鍵にした表（(原形 i, 原形 i+1) → 詰め物。無ければ (原形 i) → 、無ければいちばん多い詰め物）。

手で置く物: CONTENT（内容語の品詞）、FUNC_DEP（内容語の品詞でも詰め物にする係り）、MARKS（機能語の板の機能 → 意味の印）、
帯ごとに上位 TOPK の属性を使うこと、STEP・SHIFT・MIX（周波数の置き方の歩幅。D と互いに素の奇数）、α。
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict

import numpy as np

from i48core.space import TOPK, AttrSpace  # noqa: E402,F401  （実装は i48core へ移した。古い名前で読めるように見せる）
from i48core.text import CONTENT, FUNC_DEP, MARKS, Func, Surface, is_content, norm, read_conllu, split  # noqa: E402,F401

ALPHA = 0.1
OTHER = False                                           # 既定の「その他」の扱い。True なら候補の組の外も答えとして選べる（I48at の「無いが有る」）。
                                                        # 本体（AK.Model.other）や板（.other）に値があればそちらが勝つ。


def other_ok(obj):
    """obj に other（True / False）があればそれ、無ければ（None なら）OTHER。"""
    v = getattr(obj, "other", None)
    return OTHER if v is None else bool(v)


class PairGrain:
    """詰め物の選び方の数え上げ（多項、足すだけ）。クラスは学びの上位 n_cls の詰め物と「その他」。"""

    def __init__(self, D, n_cls=300):
        self.D, self.n_cls = D, n_cls
        self.cls, self.W, self.prior = None, None, None

    def fit(self, rows, counts):
        """rows: (f, v, 詰め物) を順に出す物（一度だけ読む）。counts: 詰め物の数（クラスを決める）。"""
        self.cls = [y for y, _ in counts.most_common(self.n_cls)]
        ci = {y: i for i, y in enumerate(self.cls)}
        other = len(self.cls)
        self.cls.append(None)
        self.W = np.zeros((len(self.cls), self.D), dtype=np.float64)
        pr = np.zeros(len(self.cls))
        for f, v, y in rows:
            i = ci.get(y, other)
            np.add.at(self.W[i], f, v)
            pr[i] += 1
        self.prior = np.log((pr + 1) / (pr.sum() + len(pr)))
        tot = self.W.sum(1, keepdims=True)
        self.logp = np.log((self.W + ALPHA) / (tot + ALPHA * self.D))
        if not other_ok(self):
            self.logp[other] = -1e9                         # 「その他」は言わない
        return self

    def predict(self, f, v, k=1, prior_w=1.0):
        s = prior_w * self.prior + self.logp[:, f] @ v
        o = np.argsort(-s)[:k]
        return [self.cls[i] for i in o]


class WordTable:
    def __init__(self):
        self.t2, self.t1, self.t0 = defaultdict(Counter), defaultdict(Counter), Counter()

    def fit(self, rows):
        """rows: [(原形 i, 原形 i+1 または印の組, 詰め物)]"""
        for a, b, y in rows:
            self.t2[(a, b)][y] += 1
            self.t1[a][y] += 1
            self.t0[y] += 1
        return self

    def predict(self, a, b):
        for c in (self.t2.get((a, b)), self.t1.get(a)):
            if c:
                return c.most_common(1)[0][0]
        return self.t0.most_common(1)[0][0]

