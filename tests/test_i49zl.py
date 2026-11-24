"""I49zl の関門: 誤差で直した後の形での、だれが 1 位を取っているかの数え。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from test_i49n import NAMES, V  # noqa: E402


def test_run_shapes():
    import i49zl_who_wins_after_run as Z
    W = set(NAMES)
    talks = [([("友達", "魚", "釣る"), ("先生", "魚", "釣る"), ("友達", "海", "泳ぐ"), ("友達", "魚", "釣る")], W),
             ([("先生", "ゲーム", "勉強"), ("友達", "魚", "釣る")], W)]
    res = Z.run(V.get, talks, [("友達", "海", "泳ぐ")])
    for start in ("白紙から", "青空の 3 つ組を先に"):
        assert list(res[start]) == ["直さない", "振幅だけ直す", "位相＋振幅を直す"]
        for d in res[start].values():
            assert 0 <= d["正解が二〜三位だった割合"] <= 1 and 0 <= d["正解が一位だった割合"] <= 1
            assert set(d["正解が二〜三位だった問い"]) == {"だれが", "何を", "どうした", "全体"}
