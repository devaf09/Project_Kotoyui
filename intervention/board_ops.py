"""板の操作のモジュール（I27s〜）: 文章の板を 4 つの操作だけで扱う

持ち主（2026-11-08）: 「デジタル判断の部分を減らして、板の操作をモジュール化、ある算術みたいな仕組みで、ある程度自立した判断ができるといいね。
それときずきや違和感、発見板で共通する部分は融合でまとめてみるのはどうかな？同じような反応をする部分は共通した判断になると思うんだけど。」

4 つの操作（算術の演算の帯・照射・差分の相関器・勝者総取りに当たる）:
  束ねる bundle(rows, weight)   行ごとの板（鍵 → 明るさ）を、重み（距離の坂など）で 1 枚に重ねる。
  照らす light(board, key, sat)  板の鍵の明るさを読む。sat を渡すと 1 - exp(-明るさ / sat) で 0〜1 に飽和させる（光の器の吸収の形）。
  差分 react(expected, observed) 期待の明るさと実際の明るさの差。期待より暗い = 違和感（-）、期待より明るい = 発見・気づき（+）。
                                 気づき・違和感・発見は、この 1 つの操作の符号の違いとして持つ（融合した反応の板）。
  勝者 winner(scored)            いちばん明るいものを 1 つ選ぶ（デジタルに残す唯一の判断）。
"""
from __future__ import annotations

import math
from collections import Counter


class Board(Counter):
    """鍵 → 明るさ。鍵は (名詞, 属性) や 属性 など、何でもよい。"""


def bundle(rows, weight=None):
    """rows: [(行の番号, {鍵: 明るさ})]。weight: 行の番号 → 重み（None なら 1）。"""
    b = Board()
    for i, row in rows:
        wt = 1.0 if weight is None else weight(i)
        if wt <= 0:
            continue
        for k, v in row.items():
            b[k] += wt * v
    return b


def light(board, key, sat=None, minus=None):
    """鍵の明るさ。minus（同じ形の板）を引いてから読む（自分の行・自分の作品を除くとき）。"""
    v = board.get(key, 0.0) - (minus.get(key, 0.0) if minus is not None else 0.0)
    v = max(0.0, v)
    if sat is None:
        return v
    return 1.0 - math.exp(-v / sat)


def react(expected, observed):
    """融合した反応: 期待との差。返り値 (大きさ, 種類)。種類は 違和感（期待より暗い）/ 発見（期待より明るい）/ なし。"""
    d = observed - expected
    if abs(d) < 1e-9:
        return 0.0, "なし"
    return abs(d), ("発見" if d > 0 else "違和感")


def winner(scored):
    """[(明るさ, 何か)] から、いちばん明るいもの。同じなら後ろの要素の大きい方（決まった順）。"""
    return max(scored, key=lambda x: (x[0], repr(x[1])))[1] if scored else None
