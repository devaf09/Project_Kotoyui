"""I49s の関門: 表と解く形の差を、段ごとに分ける道具。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from test_i49n import NAMES, V  # noqa: E402


def test_split_counts_by_stage_and_seat():
    import i49s_where_gap as S
    base = [(0, 0, True, "2語"), (0, 1, False, "述語側"), (0, 2, True, "2語")]
    other = [(0, 0, False, "解く"), (0, 1, True, "解く"), (0, 2, True, "解く")]
    d = S.split(base, other)
    assert d["2語"]["問い"] == 2 and d["2語"]["表だけ当たり"] == 1 and d["2語"]["解くの当たり"] == 0.5
    assert d["述語側・何を"]["解くだけ当たり"] == 1 and d["全部"]["問い"] == 3


def test_run_tool():
    import i49s_where_gap as S
    W = set(NAMES)
    talks = [([("友達", "魚", "釣る"), ("友達", "魚", "釣る")], W), ([("先生", "ゲーム", "勉強")], W)]
    res = S.run(V.get, V.get, talks, [("友達", "海", "泳ぐ")])
    r = res["白紙から"]["解く・属性・席の印"]
    assert r["全部"]["問い"] == 9 and "2語" in r
