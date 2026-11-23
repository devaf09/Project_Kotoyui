"""I49g 席の入れ替えの部品（intervention/i49g_flip.py）の関門。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "intervention"))
from i49e_relation import Relation  # noqa: E402
from i49g_flip import Flip  # noqa: E402

CANDS = [str(i) for i in range(19)]


def make():
    add = Relation(lambda w: None)
    for a in range(10):
        for b in range(10):
            add.learn((str(a), str(b), str(a + b)))
    return Flip(add)


def test_before_any_found_fact_it_has_no_vote():
    assert make().ask("10", "7", CANDS)["答え"] is None


def test_one_found_fact_is_enough_to_answer_the_next():
    fl = make()
    fl.learn(("10", "7", "3"))
    assert fl.ask("11", "4", CANDS)["答え"] == "7" and fl.ask("2", "1", CANDS)["答え"] == "1"


def test_which_turn_is_right_is_counted():
    fl = make()
    for c, b, a in (("10", "7", "3"), ("11", "4", "7"), ("5", "2", "3")):
        fl.learn((c, b, a))
    assert fl.hit[(2, 1, 0)] == 3 and fl.miss[(2, 1, 0)] == 0 and fl.miss[(0, 1, 2)] == 3
