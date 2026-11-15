"""I48l（受けの一言を相手の前の行から当てる）の小さな確かめ（numpy だけで動く）。"""
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "intervention"))
import i48l_uke_flow as U  # noqa: E402
from i48_wave_board import Waves  # noqa: E402


def tok(i, w, pos, head=0, dep="dep"):
    return dict(i=i, text=w, lemma=w, pos=pos, dep=dep, head=head)


def line(*tw):
    return dict(text="「" + "".join(w for w, _ in tw) + "」", toks=[tok(i, w, p) for i, (w, p) in enumerate(tw)])


def test_uke_and_tail():
    a = line(("なるほど", "ADV"), ("、", "PUNCT"), ("楽しい", "ADJ"), ("です", "AUX"), ("ね", "PART"), ("。", "PUNCT"))
    b = line(("猫", "NOUN"), ("、", "PUNCT"), ("かわいい", "ADJ"), ("。", "PUNCT"))         # 名詞を含むかたまりは受けではない
    c = line(("はい", "INTJ"), ("。", "PUNCT"))                                               # 区切りの後に何も無い行は受けではない
    d = line(("映画", "NOUN"), ("を", "ADP"), ("見", "VERB"), ("た", "AUX"), ("？", "PUNCT"))
    assert U.uke(a["toks"]) == "なるほど" and U.uke(b["toks"]) is None and U.uke(c["toks"]) is None
    assert U.tail(a["toks"]) == "です+ね" and U.tail(d["toks"]) == "た"
    S = U.samples(dict(name="w", lines=[d, a, b]))
    assert [s["post"] for s in S] == ["なるほど", U.NO_UKE]
    assert S[0]["tail"] == "た" and S[0]["marks"]["問い"] and not S[0]["marks"]["相手の受け"]
    assert S[1]["marks"]["相手の受け"] and S[1]["marks"]["形容詞"]


def toy(n=60, seed=0):
    """問いには「はい」、うれしい話には「いいですね」、それ以外は受け無しで返す資料。"""
    rng = random.Random(seed)
    works = []
    for k in range(n):
        L = []
        for j in range(16):
            kind = rng.choice(("問い", "うれしい", "ふつう"))
            if kind == "問い":
                L.append(line(("映画", "NOUN"), ("を", "ADP"), ("見る", "VERB"), ("？", "PUNCT")))
            elif kind == "うれしい":
                L.append(line(("旅行", "NOUN"), ("が", "ADP"), ("楽しい", "ADJ"), ("！", "PUNCT")))
            else:
                L.append(line(("駅", "NOUN"), ("に", "ADP"), ("行く", "VERB"), ("。", "PUNCT")))
            u = {"問い": "はい", "うれしい": "いいですね"}.get(kind)
            body = [("本", "NOUN"), ("を", "ADP"), ("読む", "VERB"), ("。", "PUNCT")]
            L.append(line(*([(u, "INTJ"), ("、", "PUNCT")] if u else []), *body))
        works.append(dict(name=f"w{k}", lines=L))
    return works


def test_wave_finds_uke_from_partner_line():
    S = [s for w in toy() for s in U.samples(w) if s["li"] % 2 == 1]        # 相手の行の後の行だけ
    cands = ["はい", "いいですね", U.OTHER, U.NO_UKE]
    tr, te = S[:700], S[700:900]
    vocab = sorted({c for s in S for c in s["cues"]})
    wv = Waves(vocab, 256, 0)
    wv.W = wv.W.astype(np.complex64)
    rel = U.UkeRelation(wv, cands, 256, ("尾", "中", "印")).fit(tr)
    r = U.measure(rel.scoop_scores(te), te, cands)
    assert r["top1"] > 0.95 and r["有無 AUROC"] > 0.95 and r["どの受けか 1 番目"] > 0.95
    ct = U.UkeTable(cands, ("尾", "印")).fit(tr)
    assert U.measure(np.stack([ct.scores(s) for s in te]), te, cands)["top1"] > 0.95


def test_board_save_load_open_and_choose(tmp_path):
    works = toy(80)
    bd = U.UkeBoard.build(works, D=256, seed=0, min_count=5)
    p = str(tmp_path / "uke")
    bd.save(p)
    lb = U.UkeBoard.load(p)
    q_ask = U.query(line(("映画", "NOUN"), ("を", "ADP"), ("見る", "VERB"), ("？", "PUNCT"))["toks"])
    q_flat = U.query(line(("本", "NOUN"), ("を", "ADP"), ("読む", "VERB"), ("。", "PUNCT"))["toks"])
    assert abs(lb.open_prob(q_ask) - bd.open_prob(q_ask)) < 1e-6
    assert lb.open_prob(q_ask) > 0.9 > 0.1 > lb.open_prob(q_flat)      # 問いの後は受けを付け、受けの後の行には付けない（資料のとおり）
    b = lb.brightness(q_ask)
    assert lb.cands[int(b.argmax())] == "はい"
    rng = random.Random(0)
    picks = [lb.choose(q_ask, {}, rng)[0] for _ in range(50)]
    assert picks.count("はい") > 40                                    # 明るさの比で選ぶ（ここでは はい がほぼ全部）
    # 使いすぎた受けは薄れる: はい を見込み（≒1）より多く使った形にはならないので、見込みと同じなら下げない
    u, q = lb.choose(q_ask, {"はい": 3}, random.Random(1))
    assert u == "はい"
    # 見込みより多く使った受けだけが薄れる（明るさを決めた形で確かめる）: 見込み はい .5・いいですね .5 で、はい を全部使った
    q = np.zeros(len(lb.cands))
    q[lb.cix["はい"]] = q[lb.cix["いいですね"]] = 0.5
    lb.brightness = lambda s: q.copy()
    rng3 = random.Random(3)
    share = sum(lb.choose(q_ask, {"はい": 4}, rng3)[0] == "はい" for _ in range(400)) / 400
    assert 0.25 < share < 0.42                                          # .5 × (.5 / 1.0) = .25 を .25 + .5 で割って 1/3。消えはしない
    share0 = sum(lb.choose(q_ask, {"はい": 2, "いいですね": 2}, rng3)[0] == "はい" for _ in range(400)) / 400
    assert 0.42 < share0 < 0.58                                         # 見込みどおりに使っていれば下げない
