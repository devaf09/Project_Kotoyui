"""I48n（受けの句の板）の小さな確かめ（numpy だけで動く）。"""
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "intervention"))
import i48n_echo_board as N  # noqa: E402
from i48_wave_board import Waves  # noqa: E402


def tok(i, w, pos, lemma=None):
    return dict(i=i, text=w, lemma=lemma or w, pos=pos, dep="dep", head=0)


def line(*tw):
    return dict(text="「x」", toks=[tok(i, *x) for i, x in enumerate(tw)])


def test_phrase_shape_and_plain():
    prev = line(("海", "NOUN"), ("に", "ADP"), ("行く", "VERB"), ("。", "PUNCT"))["toks"]
    rep = line(("海", "NOUN"), ("です", "AUX"), ("か", "PART"), ("！", "PUNCT"), ("いい", "ADJ"))["toks"]
    assert N.phrase(prev, rep) == ("X+です+か", "X+か", "海")
    assert N.phrase(prev, line(("海", "NOUN"), ("が", "ADP"), ("好き", "ADJ"), ("。", "PUNCT"))["toks"]) is None   # 中身の語が 2 つ
    assert N.phrase(prev, line(("山", "NOUN"), ("です", "AUX"), ("。", "PUNCT"))["toks"]) is None                 # 相手の行に無い語


def toy(n=80, seed=0):
    """相手の行の最初の名詞を「X です か」で受ける行と、受けない行が半々の資料。"""
    rng = random.Random(seed)
    works = []
    for k in range(n):
        L = []
        for j in range(10):
            a, b = rng.sample(("海", "山", "駅", "店"), 2)
            L.append(line((a, "NOUN"), ("と", "ADP"), (b, "NOUN"), ("に", "ADP"), ("行く", "VERB"), ("！", "PUNCT")))
            if j % 2 == 0:
                L.append(line((a, "NOUN"), ("です", "AUX"), ("か", "PART"), ("、", "PUNCT"), ("いい", "ADJ"), ("。", "PUNCT")))
            else:
                L.append(line(("本", "NOUN"), ("を", "ADP"), ("読む", "VERB"), ("。", "PUNCT")))
        works.append(dict(name=f"w{k}", lines=L))
    return works


def test_pick_first_noun():
    S = [s for w in toy() for s in N.samples(w)]
    P = N.pick_samples(S)
    tr, te = P[: len(P) * 3 // 4], P[len(P) * 3 // 4:]
    ct = N.PickTable(["受ける", "受けない"]).fit(tr)
    r = N.pick_measure(te, [ct.scores(s)[0] / ct.scores(s).sum() for s in te])
    assert r["当たり"] > 0.95                                         # 位置（最初の語）で当てられる
    last = N.pick_measure(te, [float(s["last"]) for s in te])
    assert last["当たり"] < 0.6
    wv = Waves(sorted({s["w"] for s in P}), 128, 0)
    wv.W = wv.W.astype(np.complex64)
    pk = N.PickRelation(wv, ["受ける", "受けない"], 128, ("語",)).fit(tr)
    SC = pk.scoop_scores(te)
    assert N.pick_measure(te, list(SC[:, 0] / SC.sum(1)))["当たり"] > 0.9
