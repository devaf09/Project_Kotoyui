"""I48av（驚きを場所ごとに効かせる）の確かめ（小さな手作りの語板）。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "intervention"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import i48ab_long_grain as G  # noqa: E402
import i48ad_lookahead as AD  # noqa: E402
import i48ae_marks as AE  # noqa: E402
import i48ak_teach as AK  # noqa: E402
import i48at_surprise as AW  # noqa: E402
import i48av_local_surprise as AV  # noqa: E402
import i48core.learn as LN  # noqa: E402
from test_i48ab_long_grain import CB, FB, sent  # noqa: E402


def test_band_share_goes_to_pushing_bands():
    logp = np.log(np.array([[.5, .1, .4], [.1, .6, .3], [1 / 3, 1 / 3, 1 / 3]]))
    f, v = np.array([0, 1, 2]), np.array([1.0, 1.0, 1.0])
    sc = logp @ v
    share, moved = AV.band_share(f, v, logp, sc, 0)
    assert moved and np.isclose(share.sum(), 1.0) and share[1] > 0 and share[0] == 0


def test_norm_keeps_class_sums():
    m = AV.VModel.__new__(AV.VModel)
    items = [[None, None, 0, 3.0, None, None, None], [None, None, 0, 1.0, None, None, None], [None, None, 1, 5.0, None, None, None]]
    m.adjust_block(items, "ならし")
    assert np.isclose(items[0][3] + items[1][3], 2.0) and np.isclose(items[2][3], 1.0) and np.isclose(items[0][3] / items[1][3], 3.0)


def test_all_hows_fit_small(monkeypatch):
    monkeypatch.setattr(LN, "BLOCK", 4)
    S = G.AttrSpace(CB, G.Func(FB), 256, bands=("B", "C", "D"))
    fill = AD.Fill(S)
    tr = [sent(n) for n in ("犬", "猫", "友達", "先生") * 3]
    for s in tr:
        fill.see(s)
    for how in AV.HOWS:
        m = AV.VModel(S, fill, "基準")
        info = m.fit_weighted(tr, tr[:4], how)
        assert np.isfinite(m.pg.W).all() and info["平均"] > 0
    # 1 つの間を足す時、帯に寄せても、今の点の板に足す量の和は (0) と同じ
    x = AE.items(S, tr[2], "節")[0]
    keys = AK.keys_of(tr[2])
    f, v = m.feat(x)
    c = m.cid(x["y"])
    sc = m.pg.prior + m.pg.logp[:, f] @ v
    d = (f, v, m.pg.logp, sc - sc.max())
    W0 = m.pg.W.sum()
    m.add_gap(keys, x, c, 2.0, m.who.fiction(keys), [], d, "帯")
    assert np.isclose(m.pg.W.sum() - W0, 2.0 * v.sum())


def small_model():
    S = G.AttrSpace(CB, G.Func(FB), 256, bands=("B", "C", "D"))
    fill = AD.Fill(S)
    tr = [sent(n) for n in ("犬", "猫", "友達", "先生") * 3]
    for s in tr:
        fill.see(s)
    return S, fill, tr


def test_add_more_keeps_mass_and_uniform_part(monkeypatch):
    """(4) 足し増し: 足す量の和は (0) と同じ。w ≤ 1 なら (0) と同じ形。"""
    monkeypatch.setattr(LN, "BLOCK", 4)
    S, fill, tr = small_model()
    m = AV.VModel(S, fill, "基準")
    m.fit_weighted(tr, tr[:4], "足し増し")
    x = AE.items(S, tr[2], "節")[0]
    keys = AK.keys_of(tr[2])
    f, v = m.feat(x)
    c = m.cid(x["y"])
    sc = m.pg.prior + m.pg.logp[:, f] @ v
    d = (f, v, m.pg.logp, sc - sc.max())
    for w in (0.5, 3.0):
        W0 = m.pg.W.copy()
        m.add_gap(keys, x, c, w, m.who.fiction(keys), [], d, "足し増し")
        assert np.isclose(m.pg.W.sum() - W0.sum(), w * v.sum())
        if w <= 1:
            dW = np.zeros(S.D)
            np.add.at(dW, f, w * v)
            assert np.allclose(m.pg.W[c] - W0[c], dW)


def test_whole_norm_matches_counts(monkeypatch):
    """(5) 全体ならし: 詰め物ごとの prior の数えが、学び全体のその詰め物の数になる。"""
    monkeypatch.setattr(LN, "BLOCK", 4)
    S, fill, tr = small_model()
    m = AV.VModel(S, fill, "基準")
    info = m.fit_weighted(tr, tr[:4], "全体ならし")
    n = np.zeros(m.nc)
    for s in tr:
        for x in AE.items(S, s, "節"):
            n[m.cid(x["y"])] += 1
    assert np.allclose(m.pr, n) and info["ならしの倍率の最大"] is not None
    m2 = AV.VModel(S, fill, "基準")
    m2.fit_weighted(tr, tr[:4], "同じ")
    assert np.allclose(m2.pr, n)
