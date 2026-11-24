"""I49zi の関門: 小さい世界で、語の位相を誤差で動かす形。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))


def test_fixed_matches_rebuild():
    """聞くたびに足した覚えた波は、数の表から作り直した物と同じ。"""
    import i49zi_small_grad as Z
    m = Z.Small(6, 8, "固定", 0.0)
    for ids in ([0, 2, 4], [1, 2, 5], [0, 3, 4], [0, 2, 4]):
        m.hear(ids)
    v = m.V.copy()
    m.rebuild()
    assert np.allclose(v, m.V)


def test_fix_moves_only_its_own_part():
    import i49zi_small_grad as Z
    for form in Z.FORMS:
        m = Z.Small(6, 8, form, 0.1)
        for ids in ([0, 2, 4], [1, 3, 5], [1, 2, 5]):
            m.hear(ids)
        phi, amp, V = m.phi.copy(), m.amp.copy(), m.V.copy()
        ids, cands = [0, 2, 4], [0, 1]
        s, aim = m.scores(0, ids, cands)
        s = np.array([0.1, 0.9])                                 # 1 が選ばれて外れた事にする
        t_before, w_before = m.scores(0, ids, cands)[0]
        m.fix(0, ids, cands, s, aim)
        if form == "固定":
            assert m.n_fix == 0 and np.array_equal(phi, m.phi) and np.array_equal(amp, m.amp)
            continue
        assert m.n_fix == 1
        assert np.array_equal(phi, m.phi) == (form == "振幅だけ")
        assert np.array_equal(amp, m.amp) == (form != "振幅だけ")
        assert np.allclose(V, m.V) == (form != "位相・覚えた波を作り直す")
        t_after, w_after = m.scores(0, ids, cands)[0]
        assert (t_after - w_after) > (t_before - w_before)       # 正解と選んだ語の点の差が、正解の側へ動く


def test_run_one_shapes():
    import i49zi_small_grad as Z
    a = Z.run_one(4, 8, "固定", 0.0, 1)
    b = Z.run_one(4, 8, "位相・覚えた波を作り直す", 0.03, 1, "話に合う語")
    for r in (a, b):
        for k in ("流れの当たり", "覚え当て", "広がり当て"):
            assert 0.0 <= r[k] <= 1.0
    assert a["直した回数"] == 0 and a["位相の動き"] == 0.0 and b["位相の動き"] > 0.0
