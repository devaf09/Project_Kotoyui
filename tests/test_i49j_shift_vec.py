"""I49j の関門: 表を持たず、向きのずれだけで残りの 1 つを答える。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
from i49j_shift_vec import ShiftVec  # noqa: E402


def e(i, dim=16):
    x = np.zeros(dim, dtype=np.complex64)
    x[i] = 1.0
    return x


# 「飼う」: 人 → その人の動物。動物の向き = 人の向き + 同じずれ（e(9)）
V = {"太郎": e(0), "花子": e(1), "次郎": e(2), "太郎の犬": e(0) + e(9), "花子の犬": e(1) + e(9), "次郎の犬": e(2) + e(9), "飼う": e(5), "名無し": None}
V = {k: v for k, v in V.items()}
CANDS = ["太郎の犬", "花子の犬", "次郎の犬"]


def filled(mode):
    r = ShiftVec(V.get, mode=mode)
    r.learn(("太郎", "太郎の犬", "飼う"))
    r.learn(("花子", "花子の犬", "飼う"))
    return r


def test_shift_carries_to_an_unseen_word():
    q = ("次郎", None, "飼う")                              # 次郎 は聞いていない
    assert filled("ずれ").ask(1, q, CANDS)["答え"] == "次郎の犬"
    assert filled("真ん中").ask(1, q, CANDS)["答え"] != "次郎の犬"   # 真ん中は、聞いた 2 つの間を指すだけ


def test_blank_when_nothing_heard_or_no_vector():
    r = ShiftVec(V.get)
    assert r.ask(1, ("次郎", None, "飼う"), CANDS)["答え"] is None
    r = filled("ずれ")
    assert r.ask(1, ("名無し", None, "飼う"), CANDS)["答え"] is None
    r.learn(("名無し", "太郎の犬", "飼う"))                  # 向きの無い語は足さない
    assert r.sum[(1, "飼う")][1] == 2


def test_near_mode_learns_nothing():
    r = filled("近さ")
    assert r.sum == {}
    assert r.ask(1, ("次郎", None, "飼う"), CANDS)["答え"] == "次郎の犬"
