"""I49y の関門: 正解が上位に入っているのに 1 位になれない問いで、何が 1 位を取っているか。"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from test_i49n import NAMES, V  # noqa: E402


def test_who_wins_shapes_and_sums():
    import i49y_who_wins_run as Y
    W = set(NAMES)
    talks = [([("友達", "魚", "釣る"), ("友達", "魚", "泳ぐ"), ("友達", "海", "泳ぐ"), ("先生", "魚", "釣る"), ("友達", "魚", "釣る")], W),
             ([("先生", "ゲーム", "勉強"), ("友達", "ゲーム", "勉強"), ("友達", "魚", "泳ぐ")], W)]
    res = Y.run(V.get, talks, [("友達", "海", "泳ぐ"), ("先生", "海", "釣る")])
    for start in ("白紙から", "青空の 3 つ組を先に"):
        r = res[start]
        assert r["全部の問い"] == 24
        all_ = r["正解が二〜三位だった問い"]["全体"]
        assert all_["問い"] == sum(r["正解が二〜三位だった問い"][s]["問い"] for s in Y.SEATS)
        if all_["問い"]:
            assert abs(all_["一位の方が多く聞いた"] + all_["同じだけ聞いた"] + all_["一位の方が少なく聞いた"] - 1) < 0.01
            assert 0 <= all_["一位が正しいと言える物"] <= 1
        w = r["正解が一位だった問い"]
        assert w["問い"] + all_["問い"] <= 24
