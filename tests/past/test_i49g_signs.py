"""I49g 印ごとに事実を持つ部品（intervention/past/i49g_signs.py）の関門。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "intervention"))
from i49g_signs import Signs  # noqa: E402

SONG = [str(i) for i in range(19)]


def plus(sg):
    for a in range(10):
        for b in range(10):
            sg.learn("＋", (str(a), str(b), str(a + b)))


def test_same_pair_keeps_two_answers_apart():
    sg = Signs(SONG)
    sg.learn("＋", ("5", "3", "8"))
    sg.learn("−", ("5", "3", "2"))
    assert sg.order("＋", "5", "3", SONG)[0][0] == "8" and sg.order("−", "5", "3", SONG)[0][0] == "2"


def test_unknown_sign_is_first_tried_as_the_known_one():
    sg = Signs(SONG)
    plus(sg)
    order, why = sg.order("−", "10", "7", SONG)
    assert order[0] == "17" and why[0] == "＋ として" and sorted(order) == sorted(SONG)


def test_after_one_found_fact_the_turn_answers_first():
    sg = Signs(SONG)
    plus(sg)
    sg.learn("−", ("10", "7", "3"))
    order, why = sg.order("−", "11", "4", SONG)
    assert order[0] == "7" and why[0] == "席の入れ替え"
