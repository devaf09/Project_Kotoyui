"""I49zh の関門: いいとこ取りの形の解く側に、振幅を直す形を入れた数え。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from test_i49n import NAMES, V  # noqa: E402


def test_run_shapes():
    import i49zh_mix_amp_run as Z
    W = set(NAMES)
    talks = [([("友達", "魚", "釣る"), ("友達", "魚", "釣る"), ("友達", "海", "泳ぐ")], W), ([("先生", "ゲーム", "勉強"), ("友達", "魚", "釣る")], W)]
    saved = []
    res = Z.run(V.get, V.get, talks, [("友達", "海", "泳ぐ")], None, saved.append)
    for start in ("白紙から", "青空の 3 つ組を先に"):
        r = res[start]
        assert len(r) == 7 and saved
        assert "使った答え" not in r[Z.BASE] and "当たりに変わった数" not in r[Z.BASE]
        d = r["数えて選ぶ・振幅を直す（毎回・席ごと・.01）＋出た語"]
        assert "振幅" in d and "段ごとの当たり" in d and sum(d["使った答え"].values()) > 0
        for d in r.values():
            assert 0 <= d["正しい物"] <= d["合わせて"] <= 1
    again = Z.run(V.get, V.get, talks, [("友達", "海", "泳ぐ")], res, None)       # 出ている版は飛ばす
    assert again["白紙から"]["数えて選ぶ"] is res["白紙から"]["数えて選ぶ"]
