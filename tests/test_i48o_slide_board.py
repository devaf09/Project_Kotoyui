"""I48o（状況のスライドの板）の小さな確かめ（numpy だけで動く）。"""
import os
import random
import sys
import types

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "intervention"))
import i48o_slide_board as O  # noqa: E402


def tok(i, w, pos, lemma=None, dep="dep", head=0):
    return dict(i=i, text=w, lemma=lemma or w, pos=pos, dep=dep, head=head)


def line(*tw):
    return dict(text="「x」", toks=[tok(i, *x) for i, x in enumerate(tw)])


def toy(n=60, seed=0):
    """話し手 A はいつも丁寧（行く+ます）、話し手 B はいつも丁寧でない（行く）。過去は前の相手の行に合わせる。"""
    rng = random.Random(seed)
    works = []
    for k in range(n):
        L = []
        past = rng.random() < 0.5
        for j in range(12):
            polite = j % 2 == 0
            v = rng.choice(("行く", "見る", "食べる"))
            fs = ([("ます", "AUX")] if polite else []) + ([("た", "AUX")] if past else [])
            L.append(line(("駅", "NOUN"), ("に", "ADP"), (v, "VERB"), *fs, ("ね", "PART"), ("。", "PUNCT")))
        works.append(dict(name=f"w{k}", lines=L))
    return works


def test_reader_keeps_own_politeness_and_follows_past():
    W = toy()
    sb = O.SlideBoard.build(W[:50], D=256, min_count=2)
    rows = [r for w in W[50:] for r in O.reader_rows(w)]
    read = [sb.reader.read(r["partner"], r["own"]) for r in rows]
    pol = [x[2] == r["sit"][2] for x, r in zip(read, rows) if x[2] is not None]
    past = [x[0] == r["sit"][0] for x, r in zip(read, rows) if x[0] is not None]
    assert len(pol) > 0.8 * len(rows) and np.mean(pol) == 1.0       # 丁寧は自分の前の行から
    assert np.mean(past) == 1.0                                      # 過去は相手の今の行（と自分の前の行）から
    assert sb.reader.read(None, None) == (None, None, None)


def test_partial_read_and_save_load(tmp_path):
    W = toy()
    sb = O.SlideBoard.build(W, D=256, min_count=2)
    r, src = sb.scores("行く", "VERB", (1, 0, 1))
    c = sb.cands[int(r.argmax())]
    assert src == "語" and c == "ます+た+ね"
    r2, _ = sb.scores("行く", "VERB", (None, 0, 1))                  # 過去を読めない: 過去の 0 と 1 の両方が出る
    top2 = {sb.cands[int(j)] for j in np.argsort(-r2)[:2]}
    assert top2 == {"ます+た+ね", "ます+ね"}
    p = str(tmp_path / "sb")
    sb.save(p)
    sb2 = O.SlideBoard.load(p)
    r3, _ = sb2.scores("行く", "VERB", (1, 0, 1))
    assert np.allclose(r, r3, atol=1e-5) and sb2.reader.read(*[(0, 0, 1)] * 2) == sb.reader.read(*[(0, 0, 1)] * 2)


def test_batch_matches_single():
    W = toy()
    sb = O.SlideBoard.build(W, D=256, min_count=2)
    ws, ps = ["行く", "見る", "無い語", "食べる"], ["VERB", "VERB", "VERB", "NOUN"]
    sits = [(1, 0, 1), (None, 0, 1), (0, 0, 0), (None, None, None)]
    R, src = sb.batch_scores(ws, ps, sits, B=3)
    for i in range(len(ws)):
        r, s = sb.scores(ws[i], ps[i], sits[i])
        assert s == src[i] and np.allclose(r, R[i], atol=1e-5)


def test_bot_pick_keeps_tail_and_falls_back():
    """chat_bot3 の _slide_pick を、本の表記の数えだけの小さな相手で確かめる（GiNZA を使わない）。"""
    import chat_bot3 as C3
    W = toy()
    sb = O.SlideBoard.build(W, D=256, min_count=2)
    inf = {"行きますね": ("ます+ね", "ね"), "行きました": ("ます+た", ""), "行きましたね": ("ます+た+ね", "ね"), "行くね": ("ね", "ね")}
    me = types.SimpleNamespace(slide=sb, _end_info=inf, lpos={"行く": "VERB"}, slide_log=__import__("collections").Counter(),
                               _slide_note=[], _slide_sit=(1, 0, 1))
    out = [(-1.0, "行くね"), (-2.0, "行きました"), (-3.0, "行きましたね"), (-4.0, "行きますね")]
    K = C3.Bot3._slide_pick(me, "行く", out)
    assert K[0][1] == "行きましたね"                                  # 過去・丁寧へ。語尾の ね は今の 1 番目（行くね）に合わせる
    me._slide_sit = (0, 0, 1)
    assert C3.Bot3._slide_pick(me, "行く", out)[0][1] == "行きますね"
    K = C3.Bot3._slide_pick(me, "知らない語", [(-1.0, "知らない語ね")])
    assert K == [(-1.0, "知らない語ね")] and me.slide_log["出ない（今の語尾）"] == 1   # 表記の数えに無い: 今の候補のまま
