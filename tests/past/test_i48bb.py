"""I48bb（箱の数・床で起こす）の確かめ（小さな手作りの語板）。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "intervention"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import i48ab_long_grain as G  # noqa: E402
import i48ad_lookahead as AD  # noqa: E402
import i48ae_marks as AE  # noqa: E402
import i48ak_teach as AK  # noqa: E402
import i48bb_boxes as BB  # noqa: E402
from test_i48ab_long_grain import CB, FB, sent  # noqa: E402


def test_n_cls_and_dormant_mask():
    S = G.AttrSpace(CB, G.Func(FB), 256, bands=("B", "C", "D"))
    fill = AD.Fill(S)
    tr = [sent(n) for n in ("犬", "猫", "友達", "先生") * 3]
    for s in tr:
        fill.see(s)
    m = AK.Model(S, fill, "基準")
    m.n_cls = 2
    m.fit(tr, tr[:4])
    assert m.nc == 3                                            # 上位 2 ＋ その他
    mask = np.zeros(m.nc, dtype=bool)
    mask[0] = True
    mm = m.clone()
    mm.extras.append(BB.Dormant(mask))
    mm.refresh()
    s = tr[0]
    rs, _ = mm.walk(AK.keys_of(s), AE.items(S, s, "節"))
    assert all(r["選び"] != 0 for r in rs)                       # 眠っている箱は選ばれない
    assert m.extras == []                                       # 本体は変わらない
