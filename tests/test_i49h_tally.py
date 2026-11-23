"""I49h 段の当たり外れで下がる順を決める部品（intervention/i49h_tally.py）の関門。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "intervention"))
from i49e_relation import Relation  # noqa: E402
from i49h_tally import TallyRelation  # noqa: E402


def e(i, dim=16):
    x = np.zeros(dim, dtype=np.complex64)
    x[i] = 1.0
    return x


V = {"猫": e(0), "子猫": e(0) + 0.1 * e(9), "犬": e(1), "魚": e(2), "骨": e(3), "食べる": e(4), "かじる": e(5), "見る": e(6)}
C = ["食べる", "かじる", "見る"]


def test_before_any_check_it_answers_like_the_plain_table():
    a, b = Relation(V.get), TallyRelation(V.get)
    for r in (a, b):
        r.learn(("猫", "魚", "食べる"))
    assert a.ask(2, ("子猫", "魚", None), C)["答え"] == b.ask(2, ("子猫", "魚", None), C)["答え"]
    assert a.ask(2, ("子猫", "魚", None), C)["段"] == b.ask(2, ("子猫", "魚", None), C)["段"]


def test_learn_checks_the_waiting_answers_per_level():
    r = TallyRelation(V.get)
    r.learn(("猫", "魚", "食べる"))
    r.ask(2, ("子猫", "魚", None), C)
    r.learn(("子猫", "魚", "食べる"))
    t = r.tally_levels()["どうした"]
    assert t["片方だけ借りる"] == dict(当たり=1, 外れ=0) and t["2語"] == dict(当たり=0, 外れ=0)


def test_a_level_that_keeps_missing_is_passed_over():
    r = TallyRelation(V.get, mode="選ぶ")
    r.learn(("猫", "魚", "食べる"))
    r.learn(("犬", "魚", "見る"))
    r.learn(("犬", "魚", "見る"))                          # 述語側（魚）だけなら「見る」が多い
    r.miss[2]["片方だけ借りる"], r.hit[2]["述語側"] = 3, 3   # 片方だけ借りる は外れ続け、述語側 は当たり続けた、という数えを置く
    o = r.ask(2, ("子猫", "魚", None), C)
    assert o["段"] == "述語側" and o["答え"] == "見る"
    assert Relation.ask(r, 2, ("子猫", "魚", None), C)["段"] == "片方だけ借りる"


def test_mixing_gives_an_answer_too():
    r = TallyRelation(V.get, mode="合わせる")
    r.learn(("猫", "魚", "食べる"))
    r.ask(2, ("子猫", "魚", None), C)
    r.learn(("子猫", "魚", "食べる"))
    o = r.ask(2, ("子猫", "骨", None), C)
    assert o["答え"] == "食べる"
