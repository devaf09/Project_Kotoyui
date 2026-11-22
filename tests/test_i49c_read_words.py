"""I49c の読み手（tools/i49c_read_words.py）の関門。numpy だけで動く（GiNZA・波の板は使わない）。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
import i49c_read_words as R  # noqa: E402

SENTS = [([("猫", "NOUN"), ("は", "ADP"), ("走る", "VERB")], dict(見出し=True, 空き行=False)),
         ([("ね", "PART")], dict(空き行=True)),
         ([("猫", "NOUN"), ("犬", "NOUN")], {})]


def vecs():
    W = np.array([[3, 4], [0, 2]], dtype=np.complex64)
    return R.WordVecs(W, ["猫", "走る"])


def drain(rd):
    out = []
    while rd.has_next():
        out.append(rd.next())
    return out


def test_vectors_are_unit_rows_and_only_for_content_words_on_the_board():
    wv = vecs()
    assert np.allclose(np.abs(wv.of("猫", "NOUN")), [0.6, 0.8])
    assert np.allclose(np.abs(wv.of("走る", "VERB")), [0.0, 1.0])
    assert wv.of("は", "ADP") is None            # 内容語でない
    assert wv.of("犬", "NOUN") is None           # 板に無い


def test_skip_drops_words_without_vectors_and_carries_marks():
    out = drain(R.WordReader.from_tokens(SENTS, vecs().of, "skip"))
    assert [w for w, _, _ in out] == ["猫", "走る", "猫"]
    assert out[0][2].get("見出し") and out[0][2].get("文頭")
    assert out[1][2].get("文の終わり")
    assert out[2][2].get("空き行")              # 丸ごと飛んだ文の空き行は、次の文の初めの語へ持ち越す
    assert all(v is not None for _, v, _ in out)


def test_none_passes_every_word_with_marks():
    out = drain(R.WordReader.from_tokens(SENTS, vecs().of, "none"))
    assert [w for w, _, _ in out] == ["猫", "は", "走る", "ね", "猫", "犬"]
    assert [v is None for _, v, _ in out] == [False, True, False, True, False, True]
    assert out[3][2].get("空き行") and out[3][2].get("文頭") and out[3][2].get("文の終わり")


def test_reader_moves_only_when_called_and_stops():
    rd = R.WordReader.from_tokens(SENTS, vecs().of, "skip")
    assert rd.i == 0 and rd.has_next()
    drain(rd)
    assert not rd.has_next()
    try:
        rd.next()
        assert False
    except StopIteration:
        pass


def test_counts():
    c = R.counts(SENTS, vecs().of)
    assert c["文"] == 3 and c["語"] == 6 and c["ベクトルを持つ語"] == 3 and c["ベクトルを持つ語が0の文"] == 1


def test_works_with_the_thinking_loop():
    import i49c_think_loop as TL
    from i49a_inner_vessel import Vessel
    W = np.eye(4, dtype=np.complex64)
    wv = R.WordVecs(W, ["a", "b", "c", "d"])
    sents = [([(w, "NOUN") for w in "abcd"], {})] * 3
    rd = R.WordReader.from_tokens(sents, wv.of, "none")
    loop = TL.Loop(Vessel(4, bands=4), rd)
    loop.run(200)
    assert not rd.has_next() and loop.counts()["次へ"] == 12
