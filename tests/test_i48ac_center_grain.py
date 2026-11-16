"""I48ac（中心に置くグレイン）の確かめ（小さな手作りの語板。GiNZA を使わない）。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "intervention"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i48ab_long_grain as G  # noqa: E402
import i48ac_center_grain as CG  # noqa: E402
from test_i48ab_long_grain import CB, FB  # noqa: E402


def space():
    return G.AttrSpace(CB, G.Func(FB), 256, bands=("D",))


def test_center_grain_by_class():
    """左の語のクラス（動物 → の、人 → と）で詰め物を選ぶ。学びに無い「先生」も、同じクラスの「友達」の経験から と。"""
    S = space()
    cg = CG.CenterGrain(S, 1, ["の", "と"])
    for n, y in (("犬", "の"), ("猫", "の"), ("友達", "と")) * 5:
        cg.add(y, [(-1, S.attrs(n, "NOUN")), (1, S.attrs("散歩", "NOUN"))])
    cg.build()
    assert cg.predict([(-1, S.attrs("先生", "NOUN")), (1, S.attrs("散歩", "NOUN"))]) == "と"
    assert cg.predict([(-1, S.attrs("犬", "NOUN")), (1, S.attrs("散歩", "NOUN"))]) == "の"


def test_window_and_relative_shift():
    """窓の外の語は入らない。同じ語でも左（−1）と右（+1）では違う波になる。"""
    S = space()
    cg = CG.CenterGrain(S, 1, ["の"])
    a = S.attrs("犬", "NOUN")
    assert np.allclose(cg.ctx_wave([(2, a)]), 0)
    assert not np.allclose(cg.ctx_wave([(-1, a)]), cg.ctx_wave([(1, a)]))


def test_remember_dark_is_light():
    S = space()
    cg = CG.CenterGrain(S, 1, ["の", "と"])
    r = CG.Remember(cg, CG.DARK_W["千分の 1"])
    ctx = [(-1, S.attrs("犬", "NOUN"))]
    r.bad("と", ctx)
    assert abs(r.nd[cg.ci["と"]] - 1e-3) < 1e-12 and np.isclose(r.dark.sum(), 1e-3 * S.attrs("犬", "NOUN")[1].sum())
