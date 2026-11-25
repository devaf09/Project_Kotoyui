"""I48az（間どうしを結ぶ）の確かめ（小さな手作りの語板）。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "intervention"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import i48ab_long_grain as G  # noqa: E402
import i48ad_lookahead as AD  # noqa: E402
import i48ae_marks as AE  # noqa: E402
import i48ak_teach as AK  # noqa: E402
import i48az_link_gaps as AZ  # noqa: E402
from test_i48ab_long_grain import CB, FB, sent  # noqa: E402


def test_chain_is_best_sequence():
    S = G.AttrSpace(CB, G.Func(FB), 256, bands=("B", "C", "D"))
    fill = AD.Fill(S)
    tr = [sent(n) for n in ("犬", "猫", "友達", "先生") * 3]
    for s in tr:
        fill.see(s)
    m = AK.Model(S, fill, "基準").fit(tr, tr[:4])
    tri = AZ.Triple(m, dk=256).fit(tr, tr[:4])
    s = tr[2]
    keys, xs = AK.keys_of(s), AE.items(S, s, "節")
    us, _ = AZ.unary(m, keys, xs)
    seq, und, C = AZ.chain(m, tri, keys, xs, us)
    assert len(seq) == len(xs) and all(c in Ck for c, Ck in zip(seq, C))
    def total(q):
        t = sum(us[k][q[k]] for k in range(len(q))) + tri.phi([tri.HEAD], [q[0]], keys, 0)[0, 0]
        return t + sum(tri.phi([q[k - 1]], [q[k]], keys, k)[0, 0] for k in range(1, len(q)))
    import itertools
    best = max(itertools.product(*C), key=total)
    assert np.isclose(total(seq), total(best), atol=1e-4)


def test_fm_mode_is_bounded():
    S = G.AttrSpace(CB, G.Func(FB), 256, bands=("B", "C", "D"))
    fill = AD.Fill(S)
    tr = [sent(n) for n in ("犬", "猫", "友達", "先生") * 3]
    for s in tr:
        fill.see(s)
    m = AK.Model(S, fill, "基準").fit(tr, tr[:4])
    tri = AZ.Triple(m, dk=256).fit(tr, tr[:4])
    tri.mode = "FM"
    keys = AK.keys_of(tr[2])
    P = tri.phi(list(range(m.nc)), list(range(m.nc)), keys, 1)
    assert (P >= 0).all() and P.max() <= np.log(np.sqrt(tri.n.max())) + 1e-5   # cos ≤ 1 なので log(1 / 床) を越えない
