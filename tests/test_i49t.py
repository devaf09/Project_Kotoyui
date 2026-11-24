"""I49t の関門: 揺らぎの余地を数える道具。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from test_i49n import NAMES, V  # noqa: E402


def test_room_counts_ranks_gaps_and_repeats():
    import i49t_room_run as T
    W = set(NAMES)
    talks = [([("友達", "魚", "釣る"), ("友達", "魚", "釣る"), ("友達", "魚", "泳ぐ")], W), ([("先生", "ゲーム", "勉強")], W)]
    res = T.run(V.get, V.get, talks, [("友達", "海", "泳ぐ")])
    r = res["白紙から"]["解く・属性（席あり）"]
    assert r["問い"] == 12 and r["同じ問いがまた出た数"] == 4 and r["正解が前と違った割合"] == 0.25
    p = r["正解が入る割合"]
    assert p["1 位まで"] <= p["2 位まで"] <= p["3 位まで"] <= p["5 位まで"] <= 1
    assert abs(sum(r["一位と二位の近さの分かれ方"]) - 1) < 0.01
    assert res["青空の 3 つ組を先に"]["関係の表"]["席ごと"]["だれが"]["三位まで"] >= 0
