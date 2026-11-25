"""I49zg の関門: 読む時の振幅だけを誤差で直す形。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from test_i49n import NAMES, V  # noqa: E402

TRIS = [("友達", "魚", "釣る"), ("先生", "ゲーム", "勉強"), ("友達", "海", "泳ぐ"), ("先生", "魚", "釣る"), ("友達", "ゲーム", "勉強"),
        ("友達", "魚", "釣る"), ("先生", "海", "泳ぐ")]


def _pair(**kw):
    import i49k_seat_vec as KV
    from i49zg_amp_read import AmpRead
    u = KV.Unbind(KV.SeatVec(V.get, weight="数", bind="位相"), online=True)
    a = AmpRead(KV.SeatVec(V.get, weight="数", bind="位相"), **kw)
    return u, a


def _qs(tri):
    for si in range(3):
        yield si, tuple(None if j == si else tri[j] for j in range(3))


def test_lr0_equals_unbind():
    u, a = _pair(lr=0.0)
    c = sorted(NAMES)
    for tri in TRIS * 2:
        for si, q in _qs(tri):
            x, y = u.ask(si, q, c), a.ask(si, q, c)
            assert x["答え"] == y["答え"]
            for k in x["点"]:
                assert abs(x["点"][k] - y["点"][k]) < 1e-4
        u.learn(tri)
        a.learn(tri)
    assert np.all(a.a == 1.0) and a.n_fix == 0


def test_waves_untouched_and_amp_moves():
    for rule in ("外れた時", "毎回"):
        u, a = _pair(rule=rule, lr=0.05, by_seat=True)
        c = sorted(NAMES)
        for tri in TRIS * 3:
            for si, q in _qs(tri):
                u.ask(si, q, c)
                a.ask(si, q, c)
            u.learn(tri)
            a.learn(tri)
        for w in u.sv.acc:                                   # 覚えた波は、振幅を直しても変わらない
            assert np.allclose(u.sv.acc[w], a.sv.acc[w])
        assert a.a.shape[0] == 3 and a.a.dtype == np.float32 and float(a.a.min()) >= 0.0
        sp = a.spread()
        assert a.n_fix > 0 and sp["大"] > 1.0 and 0.0 <= sp["零の割合"] <= 1.0


def test_fix_raises_truth_over_winner():
    from i49zg_amp_read import AmpRead
    import i49k_seat_vec as KV
    a = AmpRead(KV.SeatVec(V.get, weight="数", bind="位相"), lr=0.05)
    g = np.array([[.5, .1, .0, .0], [.1, .5, .0, .0]], dtype=np.float32)
    a.a = np.ones((1, 4), dtype=np.float32)
    a._fix(0, ["甲", "乙"], g, np.array([.6, .6001]), "甲")       # 乙 が選ばれて外れた
    assert a.a[0, 0] > 1.0 and a.a[0, 1] < 1.0 and a.a[0, 2] == 1.0
    before = a.a.copy()
    a._fix(0, ["甲", "乙"], g, np.array([.7, .6]), "甲")          # 当たっていれば直さない
    assert np.array_equal(before, a.a)


def test_run_shapes():
    import i49zg_amp_run as Z
    W = set(NAMES)
    talks = [([("友達", "魚", "釣る"), ("友達", "魚", "釣る"), ("友達", "海", "泳ぐ")], W), ([("先生", "ゲーム", "勉強")], W)]
    saved = []
    res = Z.run(V.get, talks, [("友達", "海", "泳ぐ")], None, saved.append)
    r = res["白紙から"]
    assert len(r) == 13 and saved and r[Z.BASE]["答えが変わった数"] == 0
    assert "振幅" in r["外れた時｜席ごと｜速さ .01"] and "当たりに変わった数" in r["毎回｜振幅 1 組｜速さ .03"]
    for d in r.values():
        assert 0 <= d["正しい物"] <= d["合わせて"] <= 1
