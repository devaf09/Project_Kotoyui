"""I49k の比べの道具の関門。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
import i49k_seat_vec_run as KR  # noqa: E402


def e(i, dim=28):
    x = np.zeros(dim, dtype=np.complex64)
    x[i] = 1.0
    return x


V = {"魚": e(0), "肉": e(1), "道": e(2), "食べる": e(3), "走る": e(4), "捕る": e(5)}
AOZ = [("ミケ", "魚", "食べる"), ("タマ", "魚", "食べる"), ("ミケ", "魚", "捕る"), ("ポチ", "道", "走る"), ("ポチ", "肉", "食べる")]
W = {"ミケ", "タマ", "ポチ", "魚", "肉", "道", "食べる", "走る", "捕る"}
TALKS = [([("タマ", "魚", "捕る")], W), ([("ポチ", "肉", "食べる")], W)]


def test_run_lists_all_makers_and_coverage():
    for w in ("数", "偏り"):
        res = KR.run(V.get, TALKS, AOZ, w)
        assert {"関係の表（語板の向き）", "関係の表（作った向き）", "ずれ", "真ん中", "近さ", "向き"} <= set(res)
        assert res["関係の表（語板の向き）"]["全体"]["問い"] == res["ずれ"]["全体"]["問い"] == 6
        assert res["向き"]["語板には無いが作った向きは有る"] == 3
        assert "Relation との違い" in res["関係の表（作った向き）"]


def test_amount_run_rows():
    import i49k_amount_run as AR
    res = AR.run(V.get, TALKS, AOZ, steps=(2, 1))
    assert set(res) == {"1/2", "1/1"} and res["1/2"]["組"] == 3 and res["1/1"]["組"] == 5
    assert res["1/1"]["作った向きを持つ内容語の割合"] >= res["1/2"]["作った向きを持つ内容語の割合"]
    assert set(res["1/1"]["関係の表"]) == {"一位", "白紙"}
