"""I49k の関門: 座り方から足し算で作った向き。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
from i49k_seat_vec import SeatVec  # noqa: E402


def e(i, dim=28):
    x = np.zeros(dim, dtype=np.complex64)
    x[i] = 1.0
    return x


V = {"魚": e(0), "肉": e(1), "道": e(2), "食べる": e(3), "走る": e(4), "捕る": e(5)}     # ミケ・タマ・ポチ は元の向きが無い
FACTS = [("ミケ", "魚", "食べる"), ("タマ", "魚", "食べる"), ("ミケ", "魚", "捕る"), ("ポチ", "道", "走る"), ("ポチ", "肉", "食べる")]


def cos(a, b):
    return float(np.abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b)))


def made(weight):
    s = SeatVec(V.get, weight=weight)
    for t in FACTS:
        s.hear(t)
    return s


def test_words_without_base_get_a_direction_and_seat_mates_are_near():
    for w in ("数", "偏り"):
        s = made(w)
        assert s.of("ミケ") is not None and s.of("知らない") is None
        assert cos(s.of("ミケ"), s.of("タマ")) > cos(s.of("ポチ"), s.of("タマ"))


def test_seats_are_kept_apart():
    s = SeatVec(V.get, weight="数")
    s.hear(("魚", "肉", "食べる"))
    s.hear(("肉", "魚", "食べる"))
    assert cos(s.of("魚"), s.of("肉")) > 0.3                       # どちらも 食べる と座った
    s2 = SeatVec(V.get, weight="数")
    s2.hear(("魚", "道", "食べる"))                                 # 魚 = だれが
    s2.hear(("道", "肉", "食べる"))                                 # 肉 = 何を。席が違えば、同じ 食べる でも別の向き
    assert cos(s2.of("魚"), s2.of("肉")) < 1e-6


def test_updates_after_hearing_more():
    s = made("数")
    v0 = s.of("タマ").copy()
    s.hear(("タマ", "道", "走る"))
    assert cos(v0, s.of("タマ")) < 0.999
