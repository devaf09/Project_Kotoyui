"""I49i の関門: 向きの無い語を、座り方の近さで借りられるようにする。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
from i49e_relation import Relation  # noqa: E402
from i49i_seat import SeatRelation  # noqa: E402


def e(i, dim=16):
    x = np.zeros(dim, dtype=np.complex64)
    x[i] = 1.0
    return x


V = {"魚": e(2), "食べる": e(3), "捕る": e(4), "肉": e(5), "走る": e(6), "道": e(7)}   # ミケ・タマ・ポチ は向きが無い
FACTS = [("ミケ", "魚", "食べる"), ("タマ", "魚", "食べる"), ("ミケ", "魚", "捕る"), ("ポチ", "道", "走る"), ("ポチ", "肉", "食べる")]


def filled(cls, **kw):
    r = cls(V.get, **kw)
    for t in FACTS:
        r.learn(t)
    return r


def test_seat_near_counts_shared_seats():
    r = filled(SeatRelation)
    assert r.seat_near("ミケ", "タマ") > r.seat_near("ポチ", "タマ") > 0     # ミケとタマは 魚・食べる を共にする。ポチとタマは 食べる だけ
    assert r.seat_near("ミケ", "魚") == 0.0                                  # 席が違えば 0
    assert r.seat_near("ミケ", "知らない") == 0.0
    assert abs(r.seat_near("ミケ", "タマ") - r.seat_near("タマ", "ミケ")) < 1e-9


def test_no_vector_word_can_borrow():
    q = ("タマ", None, "捕る")                                               # タマ が 捕る のは聞いていない
    cands = ["道", "肉", "魚"]
    a = filled(Relation).ask(1, q, cands)
    b = filled(SeatRelation).ask(1, q, cands)
    assert a["段"] == "述語側" and b["段"] == "片方だけ借りる"                # 元の表は借りられず、述語側まで下がる
    assert b["答え"] == "魚"


def test_words_with_vectors_are_unchanged_unless_all():
    r = filled(SeatRelation)
    assert r.near("魚", "肉") == 0.0                                         # 向きが有る語どうしは |cos| のまま
    r2 = filled(SeatRelation, where="全部")
    assert r2.near("魚", "肉") > 0.0                                         # 全部 なら、座り方（食べる の 何を）で近くなる


def test_cache_follows_new_facts():
    r = filled(SeatRelation)
    s0 = r.seat_near("ポチ", "タマ")
    r.learn(("ポチ", "魚", "食べる"))
    assert r.seat_near("ポチ", "タマ") > s0
