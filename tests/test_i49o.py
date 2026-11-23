"""I49o の関門: 行の語の波で、たまたまの分を引く形と、よく出る語を軽くする形。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from i49k_seat_vec import SeatVec  # noqa: E402
from test_i49n import LINES, NAMES, V  # noqa: E402


def made(**kw):
    s = SeatVec(V.get, weight="数", bind="なし", **kw)
    for ws in LINES:
        s.hear_line(ws)
    return s


def test_plain_is_unchanged_and_mass_equals_count():
    s = made()
    assert s.n_add["学校"] == 6 and abs(s.m_add["学校"] - 6) < 1e-9
    assert np.allclose(s.of("釣る"), s._base("海") + s._base("魚"), atol=1e-5)


def test_light_gives_common_words_less_weight():
    s = made(light=True)
    v = s.of("勉強")                                       # 学校（3 行に出る）と、友達・先生を足した物
    c = lambda w: float(np.real(np.vdot(s._base(w), v)))
    assert c("先生") > c("学校") / 2                       # 2 回足した学校が、1 回足した先生の 2 倍より軽い
    assert s.n_add["勉強"] == 4 and 0 < s.m_add["勉強"] != 4


def test_center_removes_the_shared_part():
    s, p = made(center=True), made()
    v = s.of("釣る")
    assert np.allclose(v, p.of("釣る") - p.m_add["釣る"] * (p.tot / p.n_tot), atol=1e-5)
    assert abs(sum(s.m_add.values()) - s.n_tot) < 1e-6


def test_run_tool():
    import i49o_line_load_run as OR
    W = set(NAMES)
    talks = [([("友達", "魚", "釣る")], W), ([("先生", "ゲーム", "勉強")], W)]
    res = OR.run(V.get, V.get, talks, [("友達", "海", "泳ぐ")], LINES, steps=(2, 1))
    assert set(res) == {"白紙から", "青空の 3 つ組を先に", "本当の行 2 行に 1 行", "本当の行 全部"}
    r = res["本当の行 全部"]
    assert r["聞かせた行の数"] == 5 and len([k for k in r if k.startswith("行の語")]) == 8
    assert "Relation との違い" in r["行の語・語の波・引く"]
