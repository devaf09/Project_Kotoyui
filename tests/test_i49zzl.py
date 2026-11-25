"""I49zzl: 本物の文で時制の波を数える道具（小さな作り物の文）。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("intervention", "tools"):
    sys.path.insert(0, os.path.join(HERE, "..", d))

import i49zzl_tense_text_run as T


def _lines():
    out = []
    for who in ("犬が", "猫が", "子供が", "母が", "人が", "彼が"):
        for v in ("歩", "走", "泣"):
            out += ["「昨日%s%sいた。」" % (who, v), "今%s%sいている！" % (who, v), "明日%s%sくだろう？" % (who, v)]
    return out


def test_sentences_and_seats():
    s = list(T.sentences(["「今まで歩いていた。今歩く！」"]))
    assert s == ["今まで歩いていた", "今歩く"]
    assert T.seats_in("今まで今", ["今", "今まで"]) == ["今まで", "今"]


def test_run_seats_finds_endings_and_guesses():
    keep = T.MIN
    T.MIN = 2
    try:
        r, _ = T.run_seats(list(T.sentences(_lines())), ["今", "昨日", "明日"])
    finally:
        T.MIN = keep
    assert r["時の語の有る文"] == 54
    top = {k: [x[0] for x in v[:3]] for k, v in r["席ごとに強い終わりの切り出し"].items()}
    assert any("だろ" in x for x in top["明日"]) and any("てい" in x or "る" == x for x in top["今"])
    g = r["時の語を当てる"]
    assert g["当たり"] > g["一番多い時の語を言い続ける"]


def test_run_auto_and_agreement():
    keep = T.MIN
    T.MIN = 2
    try:
        sents = list(T.sentences(_lines()))
        r1, w1 = T.run_seats(sents, ["今", "昨日", "明日"])
        r2, w2 = T.run_auto(sents, 50)
    finally:
        T.MIN = keep
    assert r2["席"] > 3 and r2["独立に動く向き"]
    ag = T.agreement(w1, w2)
    assert ag["共通の切り出し"] > 3 and ag["相関"] > .5       # 全部の語を席にしても、時の語で作った近さが浮き出る
