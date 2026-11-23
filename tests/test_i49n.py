"""I49n の関門: 行の語だけで語の波を作る。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from i49k_seat_vec import SeatVec, Unbind  # noqa: E402


def wave(i, dim=64):
    k = np.arange(dim)
    return np.exp(2j * np.pi * (((i + 1) * 0.41421356 * k * k) % 1.0)).astype(np.complex64)


NAMES = ["学校", "友達", "勉強", "ゲーム", "海", "魚", "釣る", "泳ぐ", "先生"]
V = {n: wave(i) for i, n in enumerate(NAMES)}
LINES = [["学校", "友達", "勉強"], ["学校", "先生", "勉強"], ["海", "魚", "釣る"], ["海", "友達", "泳ぐ"], ["学校", "友達", "ゲーム"]]


def made():
    s = SeatVec(V.get, weight="数", bind="なし")
    for ws in LINES:
        s.hear_line(ws)
    return s


def test_line_mates_are_added_without_seats():
    s = made()
    assert s.n_add["学校"] == 6 and s.n_add["釣る"] == 2
    assert np.allclose(s.of("釣る"), s._base("海") + s._base("魚"), atol=1e-5)
    assert s.of("知らない") is None


def test_answer_from_line_mates():
    u = Unbind(made())
    assert u.ask(2, ("学校", "先生", None), ["釣る", "泳ぐ", "勉強"])["答え"] == "勉強"
    assert u.ask(2, ("海", "魚", None), ["勉強", "ゲーム", "釣る"])["答え"] == "釣る"


def test_run_tool_three_starts_and_load_stats():
    import i49n_line_words_run as NR
    W = set(NAMES)
    talks = [([("友達", "魚", "釣る")], W), ([("先生", "ゲーム", "勉強")], W)]
    res = NR.run(V.get, V.get, talks, [("友達", "海", "泳ぐ")], LINES)
    assert set(res) == {"白紙から", "青空の 3 つ組を先に", "青空の本当の行を先に"}
    r = res["青空の本当の行を先に"]
    assert set(r) >= {"関係の表", "解く・属性（席あり）", "行の語・属性", "行の語・語の波"}
    st = r["行の語・語の波"]["波に乗っている数"]
    assert st["最大"] >= 6 and 0 < st["波を持つ語の割合"] <= 1
    assert res["白紙から"]["行の語・語の波"]["全体"]["問い"] == 6
