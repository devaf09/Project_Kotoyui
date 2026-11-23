"""I49g ずれを運ぶ部品（intervention/i49g_shift.py）の関門。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "intervention"))
from i49e_relation import Relation  # noqa: E402
from i49g_shift import Shifter  # noqa: E402

SONG = [str(i) for i in range(19)]
CANDS = SONG


def make(facts, song=True):
    sh = Shifter(Relation(lambda w: None))
    if song:
        sh.hear_song(SONG)
    for a, b in facts:
        sh.learn((str(a), str(b), str(a + b)))
    return sh


def test_song_gives_next_and_before():
    sh = make([])
    assert sh.between("3", "4") == "次" and sh.between("4", "3") == "前" and sh.between("3", "3") == "同じ" and sh.between("3", "5") is None


def test_heard_fact_is_answered_as_is():
    sh = make([(3, 3), (4, 3)])
    assert sh.ask("4", "3", CANDS)["段"] == "2語"


def test_carries_the_shift_to_an_unheard_sum():
    sh = make([(1, 1), (2, 1), (1, 2), (3, 3)])           # 「左が次なら答えも次」「右が次なら答えも次」を 1 回ずつ数えた
    o = sh.ask("4", "3", CANDS)
    assert o["段"] == "ずれを運ぶ" and o["答え"] == "7"
    assert sh.ask("3", "2", CANDS)["答え"] == "5"          # 前へも運べる（3＋3 の右が前）


def test_without_song_or_rule_it_falls_back():
    assert make([(1, 1), (2, 1), (3, 3)], song=False).ask("4", "3", CANDS)["段"] != "ずれを運ぶ"
    assert make([(3, 3)]).ask("4", "3", CANDS)["段"] != "ずれを運ぶ"      # 並びは知っているが、決まりをまだ数えていない


def test_counts_on_over_many_steps():
    sh = make([(a, b) for a in range(4) for b in range(4)])
    o = sh.ask("9", "6", CANDS)
    assert o["答え"] == "15" and o["歩"] == 9
    assert sh.ask("9", "6", CANDS, far=False)["段"] != "ずれを運ぶ"


def test_seats_a_new_number_heard_once_as_an_answer():
    def world(fill):
        sh = Shifter(Relation(lambda w: None), fill=fill)
        sh.hear_song([str(i) for i in range(9)])
        for a in range(9):
            for b in range(9 - a):
                sh.learn((str(a), str(b), str(a + b)))
        sh.learn(("4", "5", "9"))
        return sh
    cands = [str(i) for i in range(10)]
    assert world(False).ask("2", "7", cands)["答え"] != "9"
    sh = world(True)
    assert sh.filled == [("8", "9")] and sh.ask("2", "7", cands)["答え"] == "9" and sh.ask("9", "0", cands)["答え"] == "9"
    assert all(set(c) <= {"次", "前"} for c in sh.rule.values())      # 埋めた並びで決まりを汚していない
