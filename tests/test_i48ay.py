"""I48ay（役の位相鍵）の確かめ（小さな手作りの語板）。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "intervention"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i48ab_long_grain as G  # noqa: E402
import i48ad_lookahead as AD  # noqa: E402
import i48ae_marks as AE  # noqa: E402
import i48ak_teach as AK  # noqa: E402
import i48ay_role_keys as AY  # noqa: E402
from test_i48ab_long_grain import CB, FB, sent  # noqa: E402


def small():
    S = G.AttrSpace(CB, G.Func(FB), 256, bands=("B", "C", "D"))
    fill = AD.Fill(S)
    tr = [sent(n) for n in ("犬", "猫", "友達", "先生") * 3]
    for s in tr:
        fill.see(s)
    return S, fill, tr


def test_keys_distinct_and_wave_unit():
    S, fill, tr = small()
    m = AK.Model(S, fill, "基準").fit(tr, tr[:4])
    rk = AY.RoleKey(S, m.nc, dk=256)
    assert len(set(rk.freq.values())) == len(rk.freq)
    x = AE.items(S, tr[2], "節")[0]
    Z = rk.gap_wave(AK.keys_of(tr[2]), x)
    assert np.isclose(np.linalg.norm(Z), 1.0, atol=1e-5)


def test_plugin_scores_only_above_floor_and_body_untouched():
    S, fill, tr = small()
    m = AK.Model(S, fill, "基準").fit(tr, tr[:4])
    s = tr[2]
    keys = AK.keys_of(s)
    xs = AE.items(S, s, "節")
    g0 = m.gap_scores(keys, xs[0], m.who.fiction(keys))["今"].copy()
    rk = AY.RoleKey(S, m.nc, dk=256).fit(m, tr)
    sc = rk.score(m, keys, xs[0], None, None)
    assert (sc >= 0).all() and sc[-1] == 0
    mm = m.clone()
    mm.extras.append(rk)
    mm.refresh()
    g1 = mm.gap_scores(keys, xs[0], mm.who.fiction(keys))["今"]
    assert np.allclose(g1 - g0, sc, atol=1e-5)
    assert np.allclose(m.gap_scores(keys, xs[0], m.who.fiction(keys))["今"], g0)   # 本体（写しの元）は変わらない
