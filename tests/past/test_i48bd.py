"""I48bd（数えの表と波の比べ）の確かめ（小さな手作りの語板）。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "intervention"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import i48ab_long_grain as G  # noqa: E402
import i48ad_lookahead as AD  # noqa: E402
import i48ak_teach as AK  # noqa: E402
import i48bc_usage_board as BC  # noqa: E402
import i48bd_table_vs_wave as BD  # noqa: E402
from test_i48ab_long_grain import CB, FB, sent  # noqa: E402


def test_count_table_matches_counts_and_wave_estimates_them():
    S = G.AttrSpace(CB, G.Func(FB), 256, bands=("B", "C", "D"))
    fill = AD.Fill(S)
    tr = [sent(n) for n in ("犬", "猫", "友達", "先生") * 8]
    for s in tr:
        fill.see(s)
    m = AK.Model(S, fill, "基準").fit(tr, tr[:4])
    ub = BC.UsageBoard(m, dk=256, bands=16).fit(m, tr)
    tab = BD.CountTable(m).fit(m, tr, ub)
    key = AK.keys_of(sent("友達"))[0]
    v, fm, floor = tab.read("後ろ", key)
    assert v[m.cid("と")] == ub.N["後ろ"][key] and v.sum() == ub.N["後ろ"][key]   # 友達 の後ろは いつも「と」
    nhat, _, _ = ub.read("後ろ", key)
    assert abs(nhat[m.cid("と")] - v[m.cid("と")]) < 0.5 * v[m.cid("と")]       # 波の見積もりは数えに近い
    assert BD.band_of(0) == "床未満" and BD.band_of(50) == "〜50" and BD.band_of(501) == "500超"
