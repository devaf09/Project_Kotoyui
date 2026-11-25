"""I48k（主語の は・が・も を流れから当てる）の小さな確かめ（numpy だけで動く）。"""
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "intervention"))
import i48k_subject_flow as K  # noqa: E402
from i48_wave_board import Waves  # noqa: E402


def tok(i, w, pos, head, dep="dep"):
    return dict(i=i, text=w, lemma=w, pos=pos, dep=dep, head=head)


def subj_line(n, p, v):
    return dict(text="「x」", toks=[tok(0, n, "NOUN", 2), tok(1, p, "ADP", 0), tok(2, v, "VERB", 2, "ROOT")])


def test_samples_read_the_flow():
    w = dict(name="w", lines=[subj_line("猫", "が", "いる"), subj_line("犬", "は", "寝る"), subj_line("猫", "は", "歩く"),
                              subj_line("猫", "も", "寝る")])
    S = K.samples(w)
    assert [(s["w"], s["post"]) for s in S] == [("猫", "が"), ("犬", "は"), ("猫", "は"), ("猫", "も")]
    assert S[0]["flow"]["どこ"] == "初めて" and S[0]["flow"]["前の主語"] == "無し"
    assert S[1]["flow"]["どこ"] == "初めて" and S[1]["flow"]["前の主語"] == "違う" and S[1]["flow"]["相手の主語の助詞"] == "が"
    assert S[2]["flow"]["どこ"] == "自分の前の行" and S[2]["flow"]["2 行前の主語"] == "同じ"
    assert S[3]["flow"]["どこ"] == "相手の前の行" and S[3]["flow"]["前の主語"] == "同じ:は"
    assert S[0]["head"] == "いる"


def toy(n=80, seed=0):
    """新しい名詞は が、すぐ前に出た名詞は は で受ける資料（語と述語だけでは当たらない）。"""
    rng = random.Random(seed)
    works = []
    for k in range(n):
        L = []
        for j in range(12):
            nn = rng.choice(("猫", "犬", "鳥", "魚"))
            v = rng.choice(("いる", "寝る"))
            L += [subj_line(nn, "が", v), subj_line(nn, "は", rng.choice(("いる", "寝る")))]
            if j % 4 == 0:                                       # たまに、別の名詞の後に「も」（数える表の 0 割りを避ける）
                L.append(subj_line(rng.choice(("木", "花")), "も", v))
        works.append(dict(name=f"w{k}", lines=L))
    return works


def test_flow_key_beats_no_flow():
    S = [s for w in toy() for s in K.samples(w)]
    tr, te = S[:1500], S[1500:1800]
    vocab = sorted({s["w"] for s in S} | {s["head"] for s in S})
    wv = Waves(vocab, 256, 0)
    wv.W = wv.W.astype(np.complex64)
    cands = list(K.SUBJ)
    acc = {}
    for keys in (("語", "係"), ("語", "係", "流")):
        rel = K.FlowRelation(wv, cands, 256, keys).fit(tr)
        acc[keys] = K.measure(rel.scoop_scores(te), te, cands)["top1"]
    assert acc[("語", "係", "流")] > 0.95 > acc[("語", "係")] + 0.2
    ct = K.FlowTable(cands, ("係", "流")).fit(tr)
    assert K.measure(np.stack([ct.scores(s) for s in te]), te, cands)["top1"] > 0.95
