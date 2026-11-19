"""I48at（違和感の順・「その他」を選べる形）の確かめ（小さな手作りの語板）。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "intervention"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i48ab_long_grain as G  # noqa: E402
import i48ad_lookahead as AD  # noqa: E402
import i48ae_marks as AE  # noqa: E402
import i48ak_teach as AK  # noqa: E402
import i48ap_units as AP  # noqa: E402
import i48at_recheck as RC  # noqa: E402
from test_i48ab_long_grain import CB, FB, sent  # noqa: E402


def test_oddness_puts_unsure_first():
    rows = [dict(差=5.0, 確から=.9, 食い違い=0, 筋無し=0, 自然さ=1.0),
            dict(差=.1, 確から=.1, 食い違い=9, 筋無し=1, 自然さ=-1.0),
            dict(差=2.0, 確から=.5, 食い違い=3, 筋無し=0, 自然さ=0.0)]
    assert list(RC.oddness(rows)) == [1, 2, 0]


def test_other_switch_unmasks_and_restores():
    S = G.AttrSpace(CB, G.Func(FB), 256, bands=("B", "C", "D"))
    fill = AD.Fill(S)
    tr = [sent(n) for n in ("犬", "猫", "友達", "先生") * 3]
    for s in tr:
        fill.see(s)
    try:
        G.OTHER = False
        m = AP.UModel(S, fill, "基準").fit(tr, tr[:4]).init_units("C")
        m.refresh()
        assert m.pg.logp[-1].max() <= -1e8 and m.rR.L[-1].max() <= -1e8
        s = tr[0]
        rs, _ = m.walk(AK.keys_of(s), AE.items(S, s, "節"))
        assert all(m.nc - 1 not in r["候補"] for r in rs)
        G.OTHER = True
        m.refresh()
        assert np.isfinite(m.pg.logp[-1]).all() and m.pg.logp[-1].max() > -1e8
        rs, _ = m.walk(AK.keys_of(s), AE.items(S, s, "節"))
        assert all(np.isfinite(r["今×三つ"][-1]) and r["今×三つ"][-1] > -1e8 for r in rs)
    finally:
        G.OTHER = False


def test_other_as_model_setting():
    """「その他」の扱いは本体の持ち物（m.other）でも決められ、G.OTHER より勝つ。"""
    S = G.AttrSpace(CB, G.Func(FB), 256, bands=("B", "C", "D"))
    fill = AD.Fill(S)
    tr = [sent(n) for n in ("犬", "猫", "友達", "先生") * 3]
    for s in tr:
        fill.see(s)
    assert G.OTHER is False
    m = AP.UModel(S, fill, "基準").fit(tr, tr[:4]).init_units("C")
    m.other = True
    m.refresh()
    assert m.pg.logp[-1].max() > -1e8 and m.rR.L[-1].max() > -1e8 and m.rR.other is True
    s = tr[0]
    rs, _ = m.walk(AK.keys_of(s), AE.items(S, s, "節"))
    assert all(r["今×三つ"][-1] > -1e8 for r in rs)
    m2 = m.clone()
    m2.other = False
    m2.refresh()
    assert m2.pg.logp[-1].max() <= -1e8 and m.pg.logp[-1].max() > -1e8
