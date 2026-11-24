"""I49zj の関門: 語の位相を誤差で動かす形（本番の解く形に入れた物）。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from test_i49n import NAMES, V  # noqa: E402
from test_i49zg import TRIS, _qs  # noqa: E402


def _mk(**kw):
    import i49k_seat_vec as KV
    from i49zj_phase_read import PhaseRead
    return PhaseRead(KV.SeatVec(V.get, weight="数", bind="位相"), **kw)


def test_lr0_equals_unbind():
    import i49k_seat_vec as KV
    u = KV.Unbind(KV.SeatVec(V.get, weight="数", bind="位相"), online=True)
    a = _mk(phase_lr=0.0)
    c = sorted(NAMES)
    for tri in TRIS * 2:
        for si, q in _qs(tri):
            assert u.ask(si, q, c)["答え"] == a.ask(si, q, c)["答え"]
        u.learn(tri)
        a.learn(tri)
    assert a.n_turn == 0 and not a.theta


def test_turn_moves_phase_only_and_helps():
    a = _mk(phase_lr=0.2)
    for tri in TRIS:
        a.learn(tri)
    c = sorted(NAMES)
    tri = ("友達", "魚", "釣る")
    q = (None, "魚", "釣る")
    o = a.ask(0, q, c)
    si, names, g, s, aim, Vm, _ = a._asked.pop()
    other = next(x for x in names if x != "友達")
    s2 = np.where(np.array(names) == other, 9.0, s)             # other が選ばれて外れた事にする
    acc = {w: v.copy() for w, v in a.sv.acc.items()}
    b0 = {w: a.sv._base(w).copy() for w in ("魚", "釣る")}
    before = o["点"]["友達"] - o["点"][other]
    a._turn(si, names, s2, aim, Vm, tri, "友達")
    assert a.n_turn == 1
    for w in acc:                                               # 覚えた波は作り直さない
        assert np.array_equal(acc[w], a.sv.acc[w])
    for w in b0:                                                # 動くのは位相だけ（強さはそのまま）
        assert np.allclose(np.abs(b0[w]), np.abs(a.sv._base(w)), atol=1e-6) and not np.allclose(b0[w], a.sv._base(w))
    o2 = a.ask(0, q, c)
    a._asked.pop()
    assert o2["点"]["友達"] - o2["点"][other] > before


def test_teach_seen_skips_also_right():
    a = _mk(phase_lr=0.2, teach="聞いた組")
    for tri in TRIS:
        a.learn(tri)
    c = sorted(NAMES)
    a.ask(0, (None, "魚", "釣る"), c)
    si, names, g, s, aim, Vm, _ = a._asked.pop()
    assert "先生" in a.seen[(0, ("魚", "釣る"))]                 # 先生・魚・釣る は聞いた事が有る
    s2 = np.where(np.array(names) == "先生", 9.0, s)
    a._turn(si, names, s2, aim, Vm, ("友達", "魚", "釣る"), "友達")
    assert a.n_turn == 0                                        # 聞いた事の有る組なら、外れにしない
    b = _mk(phase_lr=0.2, teach="隠した語")
    for tri in TRIS:
        b.learn(tri)
    b.ask(0, (None, "魚", "釣る"), c)
    si, names, g, s, aim, Vm, _ = b._asked.pop()
    b._turn(si, names, np.where(np.array(names) == "先生", 9.0, s), aim, Vm, ("友達", "魚", "釣る"), "友達")
    assert b.n_turn == 1


def test_run_shapes():
    import i49zj_phase_run as Z
    W = set(NAMES)
    talks = [([("友達", "魚", "釣る"), ("友達", "魚", "釣る"), ("友達", "海", "泳ぐ")], W), ([("先生", "ゲーム", "勉強"), ("先生", "魚", "釣る")], W)]
    saved = []
    res = Z.run(V.get, talks, [("友達", "海", "泳ぐ")], None, saved.append)
    r = res["白紙から"]
    assert len(r) == 10 and saved
    assert "位相を直した回数" in r["位相｜合図 聞いた組｜速さ .1"]["動き"] and "大" in r["位相＋振幅｜合図 隠した語｜速さ .1"]["動き"]
    for d in r.values():
        assert 0 <= d["正しい物"] <= d["合わせて"] <= 1
