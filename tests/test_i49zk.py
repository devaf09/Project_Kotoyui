"""I49zk の関門: 位相と振幅を直す解く形を、いいとこ取りの形と出た語に重ねる数え。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from test_i49n import NAMES, V  # noqa: E402


def test_duel():
    import i49zk_mix_phase_run as Z
    a = [(0, True, True), (0, False, False), (1, False, False), (2, True, True)]
    b = [(0, True, True), (0, True, True), (1, True, True), (2, False, False)]
    assert Z.duel(a, b) == dict(良くなった=2, 悪くなった=1, z=round(1 / 3 ** 0.5, 2))
    assert Z.duel(a, a)["z"] == 0


def test_run_shapes():
    import i49zk_mix_phase_run as Z
    W = set(NAMES)
    talks = [([("友達", "魚", "釣る"), ("友達", "魚", "釣る"), ("友達", "海", "泳ぐ")], W), ([("先生", "ゲーム", "勉強"), ("友達", "魚", "釣る")], W)]
    saved = []
    res = Z.run(V.get, V.get, talks, [("友達", "海", "泳ぐ")], saved.append)
    for start in ("白紙から", "青空の 3 つ組を先に"):
        r = res[start]
        assert len(r) == 9 and len(r["直の比べ"]) == 4 and saved
        assert "動き" in r["数えて選ぶ｜位相＋振幅＋出た語"] and "段ごとの当たり" in r["数えて選ぶ｜直さない"]
        for k, d in r.items():
            if k != "直の比べ":
                assert 0 <= d["正しい物"] <= d["合わせて"] <= 1
