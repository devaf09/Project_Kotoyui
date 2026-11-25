"""I49zzk: 時制の波を学習から作る部品（小さな世界）。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("intervention", "tools"):
    sys.path.insert(0, os.path.join(HERE, "..", d))

import numpy as np

import i49zzk_tense_learned as TL
import i49zzk_tense_run as R

FORMS = dict(過去=("た",), 未来=("だろう",))


def _learn(m):
    for who in ("犬", "猫"):
        for verb in ("歩く", "走る"):
            m.learn("さっき", who, verb, ("た",))
            m.learn("これから", who, verb, ("だろう",))
    return m


def test_connecting_words_carry_the_tense_and_others_do_not():
    m = _learn(TL.Learned())
    assert m.strength(("つなぎ", "た")) > 5 * m.strength(("だれが", "犬"))      # 平均を引くと、時制に関係の無い語は弱い
    assert m.strength(("動詞", "歩く")) < 1e-6
    raw = _learn(TL.Learned(center=False))
    assert raw.strength(("だれが", "犬")) > .5                                   # 引かないと、どの語にも波が乗る
    assert m.wave(("つなぎ", "た")).dtype == np.float32


def test_unseen_verb_gets_the_form_from_connecting_words():
    m = _learn(TL.Learned())
    assert m.pick("笑う", FORMS, m.state_of("さっき")) == "過去"
    assert m.pick("笑う", FORMS, m.state_of("これから")) == "未来"
    assert m.state_of("知らない語") is None
    w = _learn(TL.Whole())
    assert w.pick("歩く", FORMS, w.state_of("これから")) == "未来"                # ひとまとめでも、聞いた動詞なら選べる
    assert float(np.abs(w.form_wave("笑う", ("た",))).sum()) == 0.0              # 聞いていない動詞には波が無い


def test_run_small_world():
    a, b = R.run(), R.run()
    assert a == b
    c = a["形を選ぶ"]
    assert c["つなぎの語で持つ"][-1]["出ていない動詞"] == 1.0
    assert c["動詞ごとにひとまとめ"][-1]["出ていない動詞"] < a["当てずっぽう"]
    t = a["形の波どうしの近さ"]
    i, j, k = (t["時制"].index(x) for x in ("過去進行", "過去", "未来"))
    assert t["表"][i][j] > 0 > t["表"][i][k]                                     # 手で並べていないのに、過去進行は過去に近い


def test_spans_learn_every_cut_in_one_hearing():
    assert len(TL.spans("飛んでいる")) == 15 and "でいる" in TL.spans("飛んでいる")
    m = TL.Spans()
    for who in ("犬", "猫"):
        m.learn("さっき", who, None, "歩いた")
        m.learn("今", who, None, "歩いている")
    assert m.strength(("切り出し", "た")) > m.strength(("切り出し", "歩"))       # 時制を運ぶ切り出しが強く、動詞の字は弱い
    forms = dict(過去="飛んだ", 現在進行="飛んでいる")
    assert m.pick(None, forms, m.state_of("今")) == "現在進行"                    # 初めての動詞でも、分け方を決めずに選べる


def test_run_spans():
    r = R.run_spans()
    assert r["形を選ぶ"]["切り出しを全部持つ"][-1]["決まりの内"] == 1.0
    weak = [x for x in r["切り出しごとの強さ"] if x["切り出し"] in ("歩", "食")]
    assert all(x["強さ"] < r["切り出しごとの強さ"][0]["強さ"] / 3 for x in weak)
