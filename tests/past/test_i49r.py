"""I49r の関門: 席の印。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from i49k_seat_vec import SeatVec, Unbind  # noqa: E402
from test_i49n import NAMES, V  # noqa: E402

TRIS = [("先生", "学校", "勉強"), ("友達", "海", "泳ぐ"), ("先生", "魚", "釣る"), ("友達", "学校", "ゲーム")]


def made(mark):
    s = SeatVec(V.get, weight="数", bind="位相", mark=mark)
    for t in TRIS:
        s.hear(t)
    return s


def test_mark_is_fixed_per_seat_and_added_when_heard():
    s, p = made(True), made(False)
    m0 = s.seat_mark(0, "先生")
    assert abs(np.linalg.norm(m0) - 1) < 1e-5 and not np.allclose(m0, s.seat_mark(1, "先生"))
    assert np.allclose(s.of("先生") - p.of("先生"), 2 * m0, atol=1e-5)
    assert s.n_add["先生"] == p.n_add["先生"] + 2


def test_mark_prefers_a_word_that_sat_in_the_hidden_seat():
    # 「海」は 何を の席にしか座っていない。見せた 2 語と一緒に聞いた事の無い問いで、席の印が有ると だれが の常連が上がる
    cands = ["海", "先生"]
    plain = Unbind(made(False)).ask(0, (None, "魚", "泳ぐ"), cands)["点"]
    marked = Unbind(made(True)).ask(0, (None, "魚", "泳ぐ"), cands)["点"]
    gap = lambda d: d.get("先生", 0.0) - d.get("海", 0.0)
    assert gap(marked) > gap(plain)


def test_run_tool():
    import i49r_seat_mark_run as RR
    W = set(NAMES)
    talks = [([("友達", "魚", "釣る")], W), ([("先生", "ゲーム", "勉強")], W)]
    res = RR.run(V.get, V.get, talks, TRIS)
    assert set(res) == {"白紙から", "青空の 3 つ組を先に"}
    assert "Relation との違い" in res["白紙から"]["解く・属性・席の印"]
