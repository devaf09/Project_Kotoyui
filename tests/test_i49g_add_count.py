"""I49g 足し算の数え（tools/i49g_add_count.py）の関門。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
import i49g_add_count as AC  # noqa: E402
from i49e_relation import Relation  # noqa: E402


def test_facts_are_the_hundred_sums_once_each():
    f = AC.facts()
    assert len(set(f)) == 100 and all(int(a) + int(b) == int(c) for a, b, c in f)
    assert sum(AC.held(t) for t in f) == 20


def test_second_round_is_all_right_and_first_is_not():
    r = AC.stream(AC.VECS["なし"], rounds=2)["周ごと"]
    assert r[0]["当たり"] == 0.0 and r[1]["当たり"] == 1.0 and r[1]["段"] == {"2語": 1.0}


def test_song_vectors_make_near_misses():
    plain, song = AC.holdout(AC.VECS["なし"]), AC.holdout(AC.VECS["数え歌"])
    assert plain["当たり"] == 0.0 and song["当たり"] == 0.0
    assert song["外れの大きさ"] < plain["外れの大きさ"]


def test_digits_need_every_fact_and_whole_numbers_are_blank():
    rel = Relation(AC.VECS["なし"])
    for t in AC.facts():
        rel.learn(t)
    assert AC.add_by_digits(rel, "58", "67") == "125" and AC.add_by_digits(rel, "99", "99") == "198"
    assert AC.ask_c(rel, "58", "67", [str(i) for i in range(199)])["答え"] is None
    assert AC.add_by_digits(Relation(AC.VECS["なし"]), "12", "34") is None


def test_waves_are_kept_at_start_and_end():
    out, arr = AC.waves(AC.VECS["なし"], rounds=2)
    assert out["周ごと"][0]["違った"] > out["周ごと"][1]["違った"] == 0.0
    assert out["周ごと"][1]["正解を聞いた時の外れ"]["速い波"] < out["周ごと"][0]["正解を聞いた時の外れ"]["速い波"]
    assert {"最初の問い・大きさ", "最後の問い・感度", "最後の問い・違和感"} <= set(arr)


def test_doubt_stands_only_for_unseen_sums():
    d = AC.doubt_count(AC.VECS["数え歌"])
    assert d["見せた"]["借りた"] == 0.0 and d["見せた"]["迷い_最大"] == 0.0
    assert d["見せていない"]["借りた"] == 1.0 and d["見せていない"]["迷い_最小"] > 0.0


def test_shift_part_solves_unseen_sums_only_with_the_song():
    assert AC.shift_count(True)["見せていない問い"]["当たり"] == 1.0
    assert AC.shift_count(False)["見せていない問い"]["当たり"] == 0.0


def test_shift_rule_is_counted_not_placed():
    o = AC.other_ops()
    assert o["引き算"]["当たり"] == 1.0 and o["引き算"]["決まり"]["右が次"] == {"前": 27}
    assert o["掛け算"]["当たり"] < 0.5


def test_beyond_the_learned_range_needs_the_names_of_numbers():
    o = AC.beyond(limit=12, top=24, every=3)
    assert o["数え歌 0〜24"]["当たり"] == 1.0 and o["数え歌 0〜12"]["当たり"] == 0.0
    assert o["聞きながら・答えの小さい順"]["当たり"] > o["聞きながら・混ぜた順"]["当たり"] > 0.0


def test_char_song_needs_thirty_to_settle_the_nines():
    o = AC.char_song(tops=(10, 20, 30))
    assert o["10"]["間違えた"] == 8 and o["20"]["作れなかった"] == 7 and o["30"]["正しく作れた"] == o["30"]["問い"]


def test_right_or_wrong_only_still_grows_with_the_song():
    y, n = AC.yes_no(True), AC.yes_no(False)
    assert y["二十五問ごと"][0]["言った答えの数の平均"] > y["二十五問ごと"][-1]["言った答えの数の平均"] == 1.0
    assert n["全体"]["一回目で当たり"] < 0.1 < y["全体"]["一回目で当たり"]
