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


def __getattr__(name):
    """PairGrain・WordTable は i48core.boards へ移した（読み込みの輪を避けて、使う時に読む）。"""
    if name in ("PairGrain", "WordTable"):
        import i48core.boards as B
        return getattr(B, name)
    raise AttributeError(name)
