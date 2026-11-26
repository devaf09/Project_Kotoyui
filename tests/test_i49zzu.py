"""I49zzu: 相手の行の語と返事の行の語のつながりの数え（小さな作り物の会話）。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("intervention", "tools"):
    sys.path.insert(0, os.path.join(HERE, "..", d))

import i49zzu_reply_words_run as U


def _line(speaker, *words):
    return dict(speaker=speaker, text="", toks=[dict(lemma=w, pos="NOUN") for w in words] + [dict(lemma="は", pos="ADP")])


def _works(n=40):
    """「ご飯」の後は「おいしい」、「仕事」の後は「お疲れ様」が返る会話。"""
    out = []
    for i in range(n):
        ls = []
        for j in range(6):
            a, b = (("ご飯", "おいしい") if (i + j) % 2 else ("仕事", "お疲れ様"))
            ls += [_line("A", a, "今日"), _line("B", b)]
        out.append(dict(name=str(i), lines=ls))
    return out


def test_words_and_pairs():
    w = _works(1)[0]
    assert U.words_of(w["lines"][0]) == ["今日", "仕事"]                 # 中身の語だけ（助詞は入らない）
    assert len(U.pairs_of(w)) == 11 and len(U.pairs_of(w, same=True)) == 0


def test_link_is_found_and_vanishes_when_swapped():
    works = _works()
    train = [x for w in works[4:] for x in U.pairs_of(w)]
    test = [x for w in works[:4] for x in U.pairs_of(w)]
    a = U.WordAssoc(train, min_count=5)
    assert a.score(["ご飯", "今日"], "おいしい") > a.score(["ご飯", "今日"], "お疲れ様")
    assert a.top("仕事", 1) == ["お疲れ様"]
    real = U.two_choice(a, test)["全部"]
    assert real["問い"] > 0 and real["割合"] > 0.9
    assert U.two_choice(a, test, swap=True)["全部"]["割合"] < real["割合"]


def test_run_keeps_counts_consistent():
    r = U.run(_works())
    d = r["相手 → 返事"]["二つから選ぶ"]
    assert d["全部"]["問い"] == d["相手の行にも有った語"]["問い"] + d["相手の行に無かった語"]["問い"]
