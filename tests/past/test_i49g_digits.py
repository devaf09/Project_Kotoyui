"""I49g 数え歌を文字で見る部品（intervention/past/i49g_digits.py）の関門。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "intervention"))
from i49g_digits import CharSong  # noqa: E402


def song(top):
    cs = CharSong()
    cs.hear([str(i) for i in range(top + 1)])
    return cs


def test_makes_names_it_never_heard():
    cs = song(50)
    assert cs.next("57") == "58" and cs.next("59") == "60" and cs.next("88") == "89"
    assert cs.next("23") == "24" and cs.next("9") == "10"


def test_cannot_go_past_what_the_characters_allow():
    assert song(50).next("99") is None                    # 9 の次の字を知らない


def test_too_short_a_song_is_not_enough():
    assert song(8).next("57") is None
