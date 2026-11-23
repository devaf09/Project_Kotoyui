"""I49j の比べの道具の関門。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
import i49j_shift_vec_run as JR  # noqa: E402


def e(i, dim=16):
    x = np.zeros(dim, dtype=np.complex64)
    x[i] = 1.0
    return x


V = {"太郎": e(0), "花子": e(1), "次郎": e(2), "太郎の犬": e(0) + e(9), "花子の犬": e(1) + e(9), "次郎の犬": e(2) + e(9), "飼う": e(5)}
W = set(V)
TALKS = [([("太郎", "太郎の犬", "飼う")], W), ([("花子", "花子の犬", "飼う")], W), ([("次郎", "次郎の犬", "飼う")], W)]


def test_run_has_all_makers_and_the_split_counts_add_up():
    res = JR.run(V.get, TALKS, [])
    assert set(res) == {"Relation", "ずれ", "真ん中", "近さ"}
    for r in res.values():
        n = r["全体"]["問い"]
        assert n == 9
        assert r["述語側の語は聞いたが、2 語の組は初めて"]["問い"] + r["2 語の組を前に聞いた"]["問い"] + r["述語側の語も初めて"]["問い"] == n
    assert res["ずれ"]["全体"]["一位"] >= res["真ん中"]["全体"]["一位"]
    assert "Relation との違い" in res["ずれ"]
