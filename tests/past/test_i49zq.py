"""I49zq の関門: 席の数を決めない枠（表と、波で解く形）。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
from test_i49n import NAMES, V  # noqa: E402


def F(**kw):
    import i49zq_frame as Q
    return tuple(kw.get(s) for s in Q.SEATS)


def test_table_levels():
    import i49zq_frame as Q
    t = Q.FrameTable(V.get)
    t.learn(F(が="友達", を="魚", どうした="釣る"))
    t.learn(F(が="先生", を="魚", に="海", どうした="釣る"))
    si = Q.SEATS.index("が")
    o = t.ask(si, F(を="魚", どうした="釣る"), ["友達", "先生", "海"])
    assert o["段"] == "2/2" and o["点"]["友達"] == 1 and o["点"]["先生"] == 1 and o["答え"] == "友達"
    o = t.ask(si, F(を="魚", に="海", どうした="釣る"), ["友達", "先生"])
    assert o["段"] == "3/3" and o["答え"] == "先生"                 # 「に」を見せると、選び分けられる
    o = t.ask(si, F(を="ゲーム", どうした="釣る"), ["友達", "先生"])
    assert o["段"] in ("1/2・借りる", "1/2")
    assert t.ask(si, F(どうした="勉強"), ["友達"])["答え"] in (None, "友達")
    assert Q.FrameTable(V.get).ask(si, F(を="魚"), ["友達"])["段"] == "答えられない"


def test_table_fades_not_deletes():
    import i49zq_frame as Q
    t = Q.FrameTable(V.get, half=2)
    f = F(が="友達", を="魚", どうした="釣る")
    t.learn(f)
    t.tick(2)
    assert abs(t._count(f) - .5) < 1e-6 and f in t.fr


def test_vec_matches_seat_vec_when_three_seats():
    """席が 3 つなら、今までの SeatVec(weight="数", bind="位相") と同じ波、Unbind と同じ答え。"""
    import i49k_seat_vec as KV
    import i49zq_frame as Q
    tris = [("友達", "魚", "釣る"), ("先生", "魚", "釣る"), ("友達", "海", "泳ぐ"), ("先生", "ゲーム", "勉強")]
    sv = KV.SeatVec(V.get, weight="数", bind="位相")
    fv = Q.FrameVec(V.get, n=3)
    for t in tris:
        sv.hear(t)
        fv.hear(t)
    for w in ("友達", "魚", "釣る"):
        assert np.allclose(sv.of(w), fv.of(w), atol=1e-5)
    a = KV.Unbind(sv).ask(0, (None, "魚", "釣る"), ["友達", "先生", "海"])
    b = Q.FrameSolve(fv, online=False).ask(0, (None, "魚", "釣る"), ["友達", "先生", "海"])
    assert a["答え"] == b["答え"] and all(abs(a["点"][c] - b["点"][c]) < 1e-5 for c in a["点"])


def test_solve_uses_extra_seat():
    import i49zq_frame as Q
    fv = Q.FrameVec(V.get)
    s = Q.FrameSolve(fv)
    s.learn(F(が="友達", を="魚", に="海", どうした="釣る"))
    s.learn(F(が="先生", を="魚", に="ゲーム", どうした="釣る"))
    si = Q.SEATS.index("が")
    a = s.ask(si, F(を="魚", に="海", どうした="釣る"), ["友達", "先生"])
    b = s.ask(si, F(を="魚", に="ゲーム", どうした="釣る"), ["友達", "先生"])
    assert a["答え"] == "友達" and b["答え"] == "先生"
    assert s.ask(si, F(), ["友達"])["段"] == "答えられない"


def test_mix_takes_frames():
    import i49zq_frame as Q
    from i49zb_mix import Mix
    m = Mix(Q.FrameTable(V.get), Q.FrameSolve(Q.FrameVec(V.get)), upper=None)
    f = F(が="友達", を="魚", どうした="釣る")
    for _ in range(2):
        for si, w in enumerate(f):
            if w is not None:
                m.ask(si, f, sorted(NAMES))
        m.learn(f)
    assert m.ask(0, f, sorted(NAMES))["答え"] == "友達" and sum(h[2] for h in m.tally.values()) == 6


def test_seat_table_learns_which_seat_to_trust():
    """I49zs: 同じ数の段どうしでは、それまでに多く当たってきた席の段が先に使われる。"""
    import i49zq_frame as Q
    t = Q.FrameSeatTable(V.get)
    si = Q.SEATS.index("が")
    # 「どうした」が同じなら「が」も同じ、「を」が同じでも「が」は決まらない、という材料
    data = [("友達", "魚", "釣る"), ("先生", "魚", "泳ぐ"), ("友達", "海", "釣る"), ("先生", "海", "泳ぐ"),
            ("友達", "ゲーム", "釣る"), ("先生", "ゲーム", "泳ぐ"), ("友達", "学校", "釣る"), ("先生", "学校", "泳ぐ")]
    for s, o, v in data:
        f = F(が=s, を=o, どうした=v)
        t.ask(si, F(を=o, どうした=v), ["友達", "先生"])
        t.learn(f)
    vi, oi = Q.SEATS.index("どうした"), Q.SEATS.index("を")
    assert t._rate(si, ((vi,), False)) > t._rate(si, ((oi,), False))
    o = t.ask(si, F(を="勉強", どうした="泳ぐ"), ["友達", "先生"])
    assert o["答え"] == "先生" and "どうした" in o["段"]
    o = t.ask(si, F(を="魚", どうした="釣る"), ["友達", "先生"])
    assert o["答え"] == "友達" and o["段"] == "が｜を＋どうした"            # 全部同じ枠が有れば、それが先
    assert Q.FrameSeatTable(V.get).ask(si, F(を="魚"), ["友達"])["段"] == "答えられない"


def test_seat_table_fades_and_works_in_mix():
    import i49zq_frame as Q
    from i49zb_mix import Mix
    m = Mix(Q.FrameSeatTable(V.get), Q.FrameSolve(Q.FrameVec(V.get)), upper=None)
    si = Q.SEATS.index("を")
    for _ in range(3):
        m.ask(si, F(が="友達", どうした="釣る"), ["魚", "海"])
        m.learn(F(が="友達", を="魚", どうした="釣る"))
        m.tick()
    assert m.ask(si, F(が="友達", どうした="釣る"), ["魚", "海"])["答え"] == "魚"
    assert m.table.tally and all(h[0] <= h[1] for h in m.table.tally.values())
