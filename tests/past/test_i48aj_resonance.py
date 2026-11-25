"""I48aj（三つの共振・R1 の型）の確かめ（小さな手作りの語板。GiNZA を使わない）。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "intervention"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import i48ab_long_grain as G  # noqa: E402
import i48aj_resonance as AJ  # noqa: E402
from test_i48ab_long_grain import CB, FB  # noqa: E402


def test_ratio_shrinks_rare_board():
    """数え 1 回の板は、β を大きくすると比 1（対数 0）へ寄る。数の多い板は残る。"""
    R = AJ.Ratio(3, 8)
    for _ in range(50):
        R.add(0, np.array([1]), np.array([1.0]))
        R.add(1, np.array([2]), np.array([1.0]))
    R.add(1, np.array([1]), np.array([1.0]))
    R.build(1.0)
    small = float(R.score(np.array([1]), np.array([1.0]))[1])
    R.build(1e5)
    assert abs(float(R.score(np.array([1]), np.array([1.0]))[1])) < abs(small)
    R.build(1.0)
    assert R.score(np.array([1]), np.array([1.0]))[0] > 0 > R.score(np.array([2]), np.array([1.0]))[0]


def test_gap_base_equals_shifted_sentence():
    """基準の型の、間 i から見た波 = 文の波を i × SHIFT 戻した物。段 1 は ±1 の 2 語だけ。"""
    S = G.AttrSpace(CB, G.Func(FB), 256, bands=("B", "C", "D"))
    keys = [("犬", "NOUN"), ("友達", "NOUN"), ("散歩", "NOUN"), ("行く", "VERB")]
    w = AJ.Whole(S, 3)
    assert np.allclose(w.gap(keys, 1), w.at(w.sent(keys), 1), atol=1e-5)
    w1 = AJ.Whole(S, 3, "段1")
    y = w1.word(*keys[1]) + np.roll(w1.word(*keys[2]), S.shift)
    assert np.allclose(w1.gap(keys, 1), y, atol=1e-5)


def test_attend_picks_far_word_sharing_attrs():
    """注視: 焦点の左の語（友達）といちばん属性が重なる、距離 2 以上の語（先生、同じ クラス:人）を選ぶ。"""
    S = G.AttrSpace(CB, G.Func(FB), 256, bands=("B", "C", "D"))
    keys = [("先生", "NOUN"), ("犬", "NOUN"), ("友達", "NOUN"), ("散歩", "NOUN"), ("猫", "NOUN")]
    w = AJ.Whole(S, 3, "注視・強め")
    assert w.attend(keys, 2) == 0
