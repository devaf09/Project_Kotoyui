"""I49d の下ごしらえ（tools/i49d_assoc.py の連想の口と印、tools/i49c_read_words.py の印の移し方）の関門。GiNZA・波の板は使わない。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
import i49c_read_words as R  # noqa: E402
import i49d_assoc as A  # noqa: E402


def test_near_returns_k_nearest_in_order_including_self():
    M = np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    a = A.Assoc(["a", "b", "c", "d"], M)
    r = a.near(np.array([1, 0, 0], dtype=np.float32), 3)
    assert [x[0] for x in r] == ["a", "b", "c"]
    assert r[0][2] >= r[1][2] >= r[2][2]
    assert np.isclose(np.linalg.norm(r[1][1]), 1.0)
    assert a.near(np.zeros(3), 2) == []


def test_near_works_with_complex_rows_and_ignores_phase():
    M = np.array([[1, 0], [0, 1]], dtype=np.complex64)
    a = A.Assoc(["x", "y"], M)
    r = a.near(np.array([1j, 0], dtype=np.complex64), 1)
    assert r[0][0] == "x" and np.isclose(r[0][2], 1.0)


def test_from_wordvecs_keeps_names_and_rows():
    wv = R.WordVecs(np.array([[3, 4], [0, 2]], dtype=np.complex64), ["猫", "走る"])
    a = A.Assoc.from_wordvecs(wv)
    assert a.names == ["猫", "走る"] and np.allclose(np.abs(a.vec("猫")), [0.6, 0.8])


class Tok:
    def __init__(self, lemma, pos, inf=""):
        self.lemma_, self.pos_ = lemma, pos
        self.morph = {"Inflection": [inf]} if inf else {}


def test_token_marks():
    toks = [Tok("走る", "VERB"), Tok("て", "SCONJ"), Tok("いる", "VERB"), Tok("ます", "AUX", "助動詞-マス;連用形-一般"),
            Tok("ぬ", "AUX", "助動詞-ヌ;終止形"), Tok("た", "AUX", "助動詞-タ;終止形-一般")]
    mk = A.token_marks(toks)
    assert mk[2] == {"進行"} and mk[3] == {"丁寧"} and mk[4] == {"否定"} and mk[5] == {"過去"} and mk[0] == set()


def test_reader_moves_marks_to_previous_word_or_sentence():
    wv = R.WordVecs(np.eye(2, dtype=np.complex64), ["猫", "走る"])
    sents = [([("猫", "NOUN", set()), ("走る", "VERB", set()), ("た", "AUX", {"過去"})], {}),
             ([("た", "AUX", {"過去"}), ("猫", "NOUN", set())], {})]
    prev = list(R.WordReader.from_tokens(sents, wv.of, "skip", "prev").items)
    assert prev[1][0] == "走る" and prev[1][2].get("過去")
    assert not prev[2][2].get("過去")                    # 前にベクトルを持つ語が無いので落とす
    sent = list(R.WordReader.from_tokens(sents, wv.of, "skip", "sent").items)
    assert sent[0][2].get("過去") and sent[2][2].get("過去") and not sent[1][2].get("過去")
