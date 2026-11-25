"""I49zd の関門: 語の波を持つ仮の板。"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from test_i49n import NAMES, V  # noqa: E402


def test_run_shapes():
    import i49zd_word_board_run as Z
    W = set(NAMES)
    talks = [([("友達", "魚", "釣る"), ("友達", "魚", "釣る"), ("友達", "海", "泳ぐ")], W), ([("先生", "ゲーム", "勉強")], W)]
    res = Z.run(V.get, V.get, talks, [("友達", "海", "泳ぐ")])
    r = res["白紙から"]
    assert len(r) == 18 and r["板無し"]["答えが変わった数"] == 0
    assert "当たりに変わった数" in r["語の波の板｜九割以内｜席ごと｜半分まで 1 組"]
    for d in r.values():
        assert 0 <= d["正しい物"] <= d["合わせて"] <= 1
