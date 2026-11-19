"""I48bc（新しい語板 1 版目: 使われ方の波）の確かめ（小さな手作りの語板）。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "intervention"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i48ab_long_grain as G  # noqa: E402
import i48ad_lookahead as AD  # noqa: E402
import i48ae_marks as AE  # noqa: E402
import i48ak_teach as AK  # noqa: E402
import i48bc_usage_board as BC  # noqa: E402
from test_i48ab_long_grain import CB, FB, sent  # noqa: E402


def test_usage_wave_reads_counts_and_plugin_is_removable():
    S = G.AttrSpace(CB, G.Func(FB), 256, bands=("B", "C", "D"))
    fill = AD.Fill(S)
    tr = [sent(n) for n in ("犬", "猫", "友達", "先生") * 8]
    for s in tr:
        fill.see(s)
    m = AK.Model(S, fill, "基準").fit(tr, tr[:4])
    ub = BC.UsageBoard(m, dk=256, bands=16).fit(m, tr)
    key = AK.keys_of(sent("友達"))[0]                          # 友達 の後ろは いつも「と」
    nhat, fm, floor = ub.read("後ろ", key)
    assert int(np.argmax(nhat)) == m.cid("と") and fm[m.cid("と")] > floor
    s = sent("友達")
    keys, xs = AK.keys_of(s), AE.items(S, s, "節")
    g0 = m.gap_scores(keys, xs[0], m.who.fiction(keys))["今"].copy()
    mm = m.clone()
    mm.extras.append(ub)
    mm.refresh()
    g1 = mm.gap_scores(keys, xs[0], mm.who.fiction(keys))["今"]
    assert g1[m.cid("と")] - g0[m.cid("と")] > 0                 # 使われ方の波が「と」を押し上げる
    assert np.allclose(m.gap_scores(keys, xs[0], m.who.fiction(keys))["今"], g0)   # 本体は変わらない
