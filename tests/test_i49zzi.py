"""I49zzi: 器に届かない語の数え（小さな世界）。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("intervention", "tools"):
    sys.path.insert(0, os.path.join(HERE, "..", d))

import numpy as np

import i49zzi_vessel_reach as R


def test_reach_counts_words_without_attributes():
    attrs = {"人": np.array([1, 0, 1], dtype=np.complex64), "食べる": np.array([0, 1, 0], dtype=np.complex64)}
    talks = [([("人", "りんご", "食べる"), ("人", "車", "乗る")], None), ([("犬", "骨", "かじる")], None)]
    d = R.reach(attrs.get, talks)
    assert d["だれが"]["語の種類"] == 2 and d["だれが"]["属性を持つ種類"] == 1
    assert d["だれが"]["回数の割合"] == round(2 / 3, 4) and d["だれが"]["属性の数の平均"] == 2.0
    assert d["何を"]["属性を持つ種類"] == 0 and d["何を"]["属性の数の平均"] is None
    assert d["どうした"]["属性を持つ語の回数"] == 1
    assert d["3つ組ごとの届く席の数"] == {"0": 1, "1": 1, "2": 1, "3": 0}
    assert d["全く届かない3つ組の割合"] == round(1 / 3, 4)
    assert d["話"] == 2 and d["全く届かない話"] == 1
