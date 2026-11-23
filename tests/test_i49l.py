"""I49l の関門: 位相で席を回す形、表なしで解く形、行のうねりで作る向き。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from i49k_seat_vec import SeatVec, Unbind  # noqa: E402
from i49l_line_vec import LineVec  # noqa: E402


def wave(i, dim=64):
    """周波数ごとに強さ 1、位相が語ごとに違う向き（乱数は使わない）。"""
    k = np.arange(dim)
    return np.exp(2j * np.pi * (((i + 1) * 0.41421356 * k * k) % 1.0)).astype(np.complex64)


NAMES = ["魚", "肉", "道", "食べる", "走る", "捕る", "ミケ", "タマ", "ポチ"]
V = {n: wave(i) for i, n in enumerate(NAMES)}
FACTS = [("ミケ", "魚", "食べる"), ("タマ", "魚", "食べる"), ("ミケ", "魚", "捕る"), ("ポチ", "道", "走る"), ("ポチ", "肉", "食べる")]


def cos(a, b):
    return float(np.abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b)))


def made(bind, weight="数"):
    s = SeatVec(V.get, weight=weight, bind=bind)
    for t in FACTS:
        s.hear(t)
    return s


def test_phase_binding_keeps_seat_mates_near_and_seats_apart():
    for bind in ("ずらす", "位相"):
        s = made(bind)
        assert cos(s.of("ミケ"), s.of("タマ")) > cos(s.of("ポチ"), s.of("タマ"))
        b = V["魚"] / np.linalg.norm(V["魚"])
        assert cos(s.rot(b, (0, 1)), s.rot(b, (1, 0))) < 0.5        # 席の組が違えば、同じ語でも別の向き


def test_incremental_count_equals_recount():
    s = made("位相")
    again = None
    for k, c in s.ctx["ミケ"].items():
        x = c * s.rot(s._base(k[2]), (k[0], k[1]))
        again = x if again is None else again + x
    assert np.allclose(s.of("ミケ"), again, atol=1e-4)


def test_unbind_answers_without_a_table():
    for bind in ("ずらす", "位相"):
        u = Unbind(made(bind))
        assert u.ask(1, ("タマ", None, "食べる"), ["道", "肉", "魚"])["答え"] == "魚"      # タマ と 食べる の両方と座ったのは 魚
        assert u.ask(2, ("ポチ", "道", None), ["食べる", "捕る", "走る"])["答え"] == "走る"
        assert u.ask(1, ("知らない", None, "見知らぬ"), ["魚"])["答え"] is None


def test_unbind_online_learns_from_blank():
    u = Unbind(SeatVec(V.get, weight="数", bind="位相"), online=True)
    assert u.ask(1, ("ミケ", None, "食べる"), ["魚", "道"])["答え"] is None
    u.learn(("ミケ", "魚", "食べる"))
    assert u.ask(1, ("ミケ", None, "食べる"), ["魚", "道"])["答え"] == "魚"


def test_line_swell_pulls_line_mates_together():
    for where in ("全部", "強い所"):
        lv = LineVec(V.get, where=where)
        for _ in range(3):
            lv.hear_line(["ミケ", "魚", "食べる"])
            lv.hear_line(["タマ", "魚", "食べる"])
            lv.hear_line(["ポチ", "道", "走る"])
        if where == "全部":                                          # 強い所 は、行ごとに足す周波数が違うので、小さな例では近さが逆になる事がある
            assert cos(lv.of("ミケ"), lv.of("タマ")) > cos(lv.of("ミケ"), lv.of("ポチ"))
        assert lv.of("ミケ") is not None and lv.of("知らない") is None
    assert lv.used < 1.0                                             # 強い所 は、全部の周波数は使わない


def test_run_tool_lists_everything():
    import i49l_phase_line_run as LR
    W = set(NAMES)
    talks = [([("タマ", "魚", "捕る")], W), ([("ポチ", "肉", "食べる")], W)]
    st = LR.static(V.get, talks, FACTS)
    assert set(st) == {"座り方・ずらす", "座り方・位相", "行のうねり・全部", "行のうねり・強い所"}
    assert "解く" in st["座り方・位相"] and "解く" not in st["行のうねり・全部"]
    assert 0 < st["行のうねり・強い所"]["足した周波数の割合の平均"] <= 1
    on = LR.online(V.get, talks, FACTS)
    assert on["白紙から"]["解く・位相"]["全体"]["問い"] == on["白紙から"]["関係の表（語板の向き）"]["全体"]["問い"] == 6
    assert on["青空を先に"]["解く・位相"]["全体"]["一位"] >= on["白紙から"]["解く・位相"]["全体"]["一位"]
