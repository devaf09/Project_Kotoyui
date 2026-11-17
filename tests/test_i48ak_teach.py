"""I48ak・I48al（介入で覚える、別の板）の確かめ。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "intervention"))
import i48ak_teach as AK  # noqa: E402
import i48al_teach_one as AL  # noqa: E402


def test_sep_ratio_raises_taught_bins_only():
    """教えた詰め物の、教えた帯だけが上がる。ほかの帯・教えていない詰め物・「その他」は 0。"""
    logp = np.log(np.full((3, 4), 0.25, dtype=np.float32))
    Wi = np.zeros((3, 4), dtype=np.float32)
    ni = np.zeros(3)
    Wi[0, 1] = 1.0
    ni[0] = 1
    L = AK.sep_ratio(Wi, ni, logp)
    assert L[0, 1] > 0
    assert np.all(L[0, [0, 2, 3]] == 0) and np.all(L[1] == 0) and np.all(L[2] == 0)
    Wi[0, 1] = 3.0
    ni[0] = 3
    L3 = AK.sep_ratio(Wi, ni, logp)[0, 1]
    assert L[0, 1] < L3 < np.log(1 / 0.25)                       # 同じ例を重ねると上がるが、「この帯だけで鳴る」比（1 / 本体の p）を越えない


def test_made_sentences_are_well_formed():
    """自作の教える文: 要点と詰め物の数が合い、どの文にも「と」の間がある。"""
    for axis, text, keys, gaps in AL.TEACH_SET:
        assert len(keys) == len(gaps) and "と" in gaps, text
    keys, gaps = AK.made_sentence("友達", "散歩")
    assert [k for k, _ in keys] == ["友達", "散歩", "行く"] and gaps == ["と", "に", "た。"]


def test_separate_board_keeps_main_and_undo_restores():
    """別の板で教えても本体の板は変わらない。記録の番号で取り消すと、介入の板は空に戻る。"""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import i48ab_long_grain as G
    import i48ad_lookahead as AD
    from test_i48ab_long_grain import CB, FB, sent
    S = G.AttrSpace(CB, G.Func(FB), 256, bands=("B", "C", "D"))
    fill = AD.Fill(S)
    tr = [sent(n) for n in ("犬", "猫", "友達", "先生") * 3]
    for s in tr:
        fill.see(s)
    m = AK.Model(S, fill, "基準").fit(tr, tr[:4])
    W0 = m.pg.W.copy()
    m.sep = True
    keys, gaps = AK.made_sentence("友達", "散歩")
    m.teach(keys, AK.made_items(S, keys, gaps), gaps, True, 1.0, text="友達と散歩に行った")
    m.refresh()
    assert np.array_equal(m.pg.W, W0) and m.ni.sum() > 0 and m.Li is not None
    assert all(e["文"] == "友達と散歩に行った" for e in m.log)
    m.undo([e["番号"] for e in m.log])
    assert m.ni.sum() == 0 and np.allclose(m.Wi, 0, atol=1e-6) and m.Li is None
