"""I48e（関係を名詞と述語に分ける）の小さな確かめ（numpy だけで動く）。"""
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "intervention"))
import i48e_relation_split as E  # noqa: E402
from i48_wave_board import Waves  # noqa: E402


def tok(i, w, pos, head, dep="dep"):
    return dict(i=i, text=w, lemma=w, pos=pos, dep=dep, head=head)


def line(text, toks):
    return dict(text=text, toks=toks)


def test_samples_split_host_post_and_flags():
    w = dict(name="w", lines=[
        line("「猫が歩いている？」", [tok(0, "「", "PUNCT", 1), tok(1, "猫", "NOUN", 3, "nsubj"), tok(2, "が", "ADP", 1),
                                tok(3, "歩く", "VERB", 3, "ROOT"), tok(4, "て", "SCONJ", 3), tok(5, "いる", "AUX", 3),
                                tok(6, "？", "PUNCT", 3), tok(7, "」", "PUNCT", 3)]),
        line("「猫は歩かない」", [tok(0, "猫", "NOUN", 2, "nsubj"), tok(1, "は", "ADP", 0), tok(2, "歩く", "VERB", 2, "ROOT"),
                              tok(3, "ない", "AUX", 2)]),
    ])
    S = E.samples(w)
    assert [(s["w"], s["host"], s["post"]) for s in S] == [("猫", "名", "が"), ("歩く", "述", "て+いる"), ("猫", "名", "は"), ("歩く", "述", "ない")]
    assert S[0]["head"] == "歩く" and S[0]["npost"] == "て+いる"
    assert S[0]["flags"]["問い"] and not S[0]["flags"]["既出"]
    assert S[2]["flags"]["前の行の問い"] and S[2]["flags"]["既出"] and S[2]["flags"]["否定"]   # 自分の外に「ない」
    assert not S[3]["flags"]["否定"]                                                          # 自分の後ろの「ない」は数えない


def toy(n=60, seed=0):
    """名詞の助詞が係る述語で決まる資料（食べる ← を、行く ← に）。"""
    rng = random.Random(seed)
    works = []
    for k in range(n):
        L = []
        for j in range(20):
            v, p = rng.choice((("食べる", "を"), ("行く", "に")))
            nn = rng.choice(("パン", "店", "駅", "米"))
            L.append(line("「x」", [tok(0, nn, "NOUN", 2), tok(1, p, "ADP", 0), tok(2, v, "VERB", 2, "ROOT"), tok(3, "た", "AUX", 2)]))
        works.append(dict(name=f"w{k}", lines=L))
    return works


def test_head_key_finds_particle():
    works = toy()
    S = [s for w in works for s in E.samples(w) if s["host"] == "名"]
    tr, te = S[:800], S[800:1000]
    vocab = sorted({s["w"] for s in S} | {s["head"] for s in S})
    wv = Waves(vocab, 256, 0)
    wv.W = wv.W.astype(np.complex64)
    cands = ["を", "に"]
    acc = {}
    for keys in (("語",), ("語", "係")):
        rel = E.Relation(wv, cands, 256, keys).fit(tr)
        SC = rel.scoop_scores(te)
        acc[keys] = E.measure(SC, te, cands, "名")["top1"]
    assert acc[("語", "係")] > 0.95 > acc[("語",)] + 0.2
    ct = E.CountTable(cands, ("係",)).fit(tr)
    assert E.measure(np.stack([ct.scores(s) for s in te]), te, cands, "名")["top1"] > 0.95


def test_pick_cues_prefers_informative_word():
    S = []
    for i in range(400):
        p = "た" if i % 2 else "る"
        S.append(dict(post=p, cues=(["昨日"] if p == "た" else ["明日"]) + ["する"]))
    assert set(E.pick_cues(S, "post", 2, min_n=10)) == {"昨日", "明日"}
