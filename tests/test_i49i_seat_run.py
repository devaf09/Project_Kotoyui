"""I49i の比べの道具（tools/i49i_seat_run.py）の関門。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
import i49i_seat_run as SR  # noqa: E402


def e(i, dim=16):
    x = np.zeros(dim, dtype=np.complex64)
    x[i] = 1.0
    return x


V = {"魚": e(2), "食べる": e(3), "捕る": e(4), "道": e(7)}
W = {"ミケ", "タマ", "魚", "食べる", "捕る", "道"}
TALKS = [([("ミケ", "魚", "食べる"), ("タマ", "魚", "食べる"), ("ミケ", "魚", "捕る")], W), ([("タマ", "魚", "捕る")], W)]


def test_run_counts_and_seat_helps_the_unseen_pair():
    res = SR.run(V.get, TALKS, [])
    n = res["Relation"]["全体"]["問い"]
    assert n == res["無い語"]["全体"]["問い"] == res["全部"]["全体"]["問い"] == 12
    assert res["向きの無い語を含む問い"]["数"] == 8                        # だれが を隠す問い以外は、ミケ か タマ を見せる
    assert res["無い語"]["座り方の近さを使った回数"] > 0
    c = res["無い語"]["Relation との違い"]
    assert c["当たりが変わった"] == c["良くなった"] + c["悪くなった"]
    assert res["無い語"]["全体"]["一位"] >= res["Relation"]["全体"]["一位"]
