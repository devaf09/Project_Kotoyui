"""I49h の比べの道具（tools/i49h_tally_run.py）の関門。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
import i49h_tally_run as HR  # noqa: E402


def e(i, dim=16):
    x = np.zeros(dim, dtype=np.complex64)
    x[i] = 1.0
    return x


V = {"猫": e(0), "子猫": e(0) + 0.1 * e(9), "魚": e(2), "食べる": e(3), "捕る": e(4)}


def test_three_makers_see_the_same_questions_and_tally_fills():
    talks = [([("猫", "魚", "食べる")], {"猫", "魚", "食べる", "捕る"}), ([("子猫", "魚", "食べる")], {"子猫", "魚", "食べる", "捕る"})] * 3
    res = HR.run_all(V.get, talks, [])
    n = res["Relation"]["全体"]["問い"]
    assert n == res["選ぶ"]["全体"]["問い"] == res["合わせる"]["全体"]["問い"] > 0
    assert res["Relation"]["流れ"][0]["白紙"] > 0                       # 初めは白紙（聞く前に問う）
    t = res["選ぶ"]["終わりの段ごとの当たり外れ"]["どうした"]
    assert sum(x["当たり"] + x["外れ"] for x in t.values()) > 0
    c = res["合わせる"]["Relation との違い"]
    assert c["当たりが変わった"] == c["良くなった"] + c["悪くなった"]


def test_compare_counts():
    b = [(0, 0, True, "2語"), (0, 1, False, "2語"), (0, 2, False, "答えられない")]
    o = [(0, 0, False, "2語"), (0, 1, True, "借りる"), (0, 2, True, "借りる")]
    c = HR.compare(b, o)
    assert (c["良くなった"], c["悪くなった"], c["段か当たりが変わった"]) == (2, 1, 3)
