"""alt2（外れそうだと先に分かるか）の関門。numpy だけで動く。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import alt2_foresee as F  # noqa: E402
import i49a_inner_vessel as V  # noqa: E402


def test_usual_is_rebuilt_exactly_and_counts_are_sane():
    rng = np.random.default_rng(0)
    v = V.Vessel(32, bands=6, longest=64)
    for k in range(400):
        if k % 7 == 0:
            v.tick()
        else:
            v.hear((rng.normal(size=32) + 1j * rng.normal(size=32)).astype(np.complex64))
    z = v.arrays()
    res = F.measure(z, skip=20)
    assert res["聞いた回数"] == int((z["src"] != 0).sum()) - 20
    # 記録から作り直したいつもの外れが、器の中の値と同じ所に着く
    a = 1.0 - np.power(0.5, 1.0 / z["half"].astype(np.float64))
    u = np.ones(6)
    for t in range(len(z["src"])):
        if z["src"][t] != 0:
            u = u + a * (z["sur"][t] - u)
    assert np.allclose(u, v.usual, atol=1e-3)
    assert all(len(s) == 5 for s in res["段"]) and len(res["r"]["いつもの外れ"]) == 6
