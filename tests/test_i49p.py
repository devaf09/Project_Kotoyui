"""I49p の関門: 語の波そのものを、聞くたびに動かす形。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from i49p_live_vec import LiveAsk, LiveVec  # noqa: E402
from test_i49n import LINES, NAMES, V  # noqa: E402


def cos(a, b):
    return float(np.real(np.vdot(a, b))) / float(np.linalg.norm(a) * np.linalg.norm(b))


def made(**kw):
    lv = LiveVec(V.get, **kw)
    for _ in range(3):
        for ws in LINES:
            lv.hear_line(ws)
    return lv


def test_line_mates_get_nearer_and_base_is_not_touched():
    before = cos(V["学校"], V["勉強"])
    lv = made()
    assert cos(lv.of("学校"), lv.of("勉強")) > before + 0.3
    assert np.allclose(V["学校"], LiveVec(V.get)._get("学校"))
    assert lv.n_add["釣る"] == 6 and lv.of("知らない") is None


def test_word_without_base_gets_a_wave_from_mates():
    lv = LiveVec(lambda w: None if w == "新語" else V.get(w))
    lv.hear_line(["新語", "海", "魚"])
    assert cos(lv.of("新語"), V["海"] + V["魚"]) > 0.99


def test_far_keeps_other_lines_apart():
    near, far = made(), made(far=True)
    assert cos(far.of("釣る"), far.of("先生")) < cos(near.of("釣る"), near.of("先生"))
    assert cos(far.of("釣る"), far.of("魚")) > cos(far.of("釣る"), far.of("先生"))
    assert abs(far.n_u - len(NAMES)) == 0


def test_answer_and_online_learning():
    a = LiveAsk(made(far=True))
    assert a.ask(2, ("海", "魚", None), ["勉強", "ゲーム", "釣る"])["答え"] == "釣る"
    b = LiveAsk(LiveVec(V.get, bind="位相"))
    for _ in range(3):
        b.learn(("先生", "学校", "勉強")), b.learn(("友達", "海", "泳ぐ"))
    assert b.ask(2, ("先生", "学校", None), ["泳ぐ", "勉強"])["答え"] == "勉強"
    assert b.ask(0, (None, "海", "泳ぐ"), ["先生", "友達"])["答え"] == "友達"


def test_run_tool():
    import i49p_live_run as PR
    W = set(NAMES)
    talks = [([("友達", "魚", "釣る")], W), ([("先生", "ゲーム", "勉強")], W)]
    res = PR.run(V.get, V.get, talks, [("友達", "海", "泳ぐ")], LINES, step=2)
    assert set(res) == {"白紙から", "青空の 3 つ組を先に", "本当の行 2 行に 1 行"}
    r = res["本当の行 2 行に 1 行"]
    assert "波に乗っている数" in r["今の波・属性・遠ざける"] and "Relation との違い" in r["今の波・属性・席"]
