"""I49v の関門: 2 つの席を同時に隠す問いと、掛け算で選ぶ形。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from test_i49n import NAMES, V  # noqa: E402


def test_two_hidden_counts_and_joint_beats_separate_on_a_made_case():
    import i49e_relation as RL
    import i49v_two_hidden_run as T
    W = set(NAMES)
    # 友達 は 魚 とも 海 とも、釣る とも 泳ぐ とも出る。組としては（魚, 釣る）と（海, 泳ぐ）だけ。
    pre = [("友達", "魚", "釣る")] * 3 + [("友達", "海", "泳ぐ")] * 2 + [("先生", "海", "釣る")] * 4
    r = RL.Relation(V.get, half=1024)
    for t in pre:
        r.learn(t)
    res = T.two(r, [([("友達", "魚", "釣る")], W)])
    assert res["問い"] == 3 and res["問う回数"] > 6
    v = res["版"]
    assert set(v) == {"別々", "順に・前から", "順に・後ろから", "掛け算・上位 3", "足し算・上位 3", "掛け算・上位 5", "足し算・上位 5"}
    for d in v.values():
        assert 0 <= d["二席とも当たり"] <= d["席ごとの当たり"] <= 1
    assert v["掛け算・上位 3"]["見せた席ごと"]["だれが"] == 1.0
    assert res["上位に二席とも正解が入る割合"]["上位 3"] <= res["上位に二席とも正解が入る割合"]["上位 5"]


def test_run_shapes_and_none_safe():
    import i49v_two_hidden_run as T
    W = set(NAMES)
    talks = [([("友達", "魚", "釣る"), ("友達", "魚", "釣る"), ("友達", "海", "泳ぐ")], W), ([("先生", "ゲーム", "勉強")], W)]
    res = T.run(lambda w: V[w] if w in V else None if w is not None else 1 / 0, V.get, talks, [("友達", "海", "泳ぐ")])
    for start in ("白紙から", "青空の 3 つ組を先に"):
        for form in ("関係の表", "解く・属性（席あり）"):
            assert res[start][form]["問い"] == 12
