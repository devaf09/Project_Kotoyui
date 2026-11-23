"""I49e の関係の表（intervention/i49e_relation.py）の関門。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "intervention"))
from i49e_relation import Relation  # noqa: E402


def e(i, dim=16):
    x = np.zeros(dim, dtype=np.complex64)
    x[i] = 1.0
    return x


V = {"猫": e(0), "子猫": e(0) + 0.1 * e(9), "鳥": e(1), "魚": e(2), "小魚": e(2) + 0.1 * e(10), "食べる": e(3), "捕る": e(4), "空": e(5)}


def make(half=1024.0):
    return Relation(V.get, half=half)


def test_exact_pair_answers_first():
    r = make()
    r.learn(("猫", "魚", "食べる"))
    o = r.ask(2, ("猫", "魚", None), ["食べる", "捕る"])
    assert o["段"] == "2語" and o["答え"] == "食べる"
    assert r.ask(0, (None, "魚", "食べる"), ["鳥", "猫"])["答え"] == "猫"


def test_borrows_one_side_before_falling_to_one_word():
    r = make()
    r.learn(("猫", "魚", "食べる"))
    for _ in range(2):
        r.learn(("鳥", "魚", "捕る"))
    # 1 語の表（魚 →）なら「捕る」が多い。子猫は猫に近いので、片方だけ借りる段が先に「食べる」と答える
    o = r.ask(2, ("子猫", "魚", None), ["食べる", "捕る"])
    assert o["段"] == "片方だけ借りる" and o["答え"] == "食べる"
    assert o["点"]["食べる"] > 0.9 and o["点"]["捕る"] == 0.0


def test_falls_down_the_levels_and_stays_blank_when_nothing():
    r = make()
    r.learn(("猫", "魚", "食べる"))
    assert r.ask(2, ("空", "魚", None), ["食べる", "捕る"])["段"] == "述語側"      # 空 は猫に近くない → 魚 の 1 語の表
    assert r.ask(2, ("猫", "空", None), ["食べる", "捕る"])["段"] == "もう片方"    # 空 は見た事が無い → 猫 の 1 語の表
    assert r.ask(2, ("子猫", "小魚", None), ["食べる", "捕る"])["段"] == "借りる"  # どちらも見た事が無い → 2 語とも借りる
    o = r.ask(2, ("空", "鳥", None), ["食べる", "捕る"])
    assert o["段"] == "答えられない" and o["答え"] is None                         # 白紙。無理には答えない
    assert r.tally()["答えられない"] == 1 and len(r.rec) == 4


def test_words_without_vectors_are_used_only_as_themselves():
    r = make()
    r.learn(("ぽち", "魚", "食べる"))
    assert r.ask(2, ("ぽち", "魚", None), ["食べる", "捕る"])["段"] == "2語"
    assert r.ask(2, ("たま", "小魚", None), ["食べる", "捕る"])["段"] == "答えられない"


def test_memory_fades_but_is_not_deleted():
    r = make(half=4.0)
    r.learn(("猫", "魚", "食べる"))
    r.tick(8)                                              # 半減期 2 回ぶん
    r.learn(("猫", "魚", "捕る"))
    o = r.ask(2, ("猫", "魚", None), ["食べる", "捕る"])
    assert o["答え"] == "捕る" and abs(o["点"]["食べる"] - 0.25) < 1e-6 and o["点"]["捕る"] == 1.0
    r.learn(("猫", "魚", "食べる"))                        # 薄れた数に足す
    assert abs(r.ask(2, ("猫", "魚", None), ["食べる", "捕る"])["点"]["食べる"] - 1.25) < 1e-6
