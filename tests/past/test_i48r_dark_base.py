"""I48r（--dark_base: 行のまるごとの暗さを場のふだんの指摘の割合と比べる）の小さな確かめ（GiNZA を使わない）。"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "intervention"))
import chat_bot3 as C3  # noqa: E402


def bot(dark_base):
    b = C3.Bot3.__new__(C3.Bot3)                    # 本を読まずに、行の経験の所だけを使う
    b.uke, b._last_nov, b.dark_base = None, None, dark_base
    return b


def say(b, line, pointed=None):
    b.__dict__.setdefault("line_said", Counter())[b._exp_key(line)] += 1
    if pointed:
        b.mark_line(line, src=pointed)


def test_default_is_unchanged():
    b = bot(False)
    say(b, "海に行く。", pointed="先生")
    assert b.dark_line("海に行く。")                       # この会話で先生に指摘された: 暗い（今まで通り）
    b.conv_dark = set()                                    # 次の会話
    assert b.dark_line("海に行く。") and b.dark_cut() == 0.5   # 1 回言って 1 回指摘: (1.1)/2 > 0.5 で暗いまま


def test_dark_base_uses_field_rate_and_partner_only():
    b = bot(True)
    for i in range(8):                                     # 先生がほとんどの行に何か言う場（8 行のうち 7 行）
        say(b, f"行{i}。", pointed="先生" if i < 7 else None)
    assert abs(b.dark_cut() - 7 / 8) < 1e-9
    assert not b.dark_line("行0。")                        # 先生の指摘だけでは、この会話でも暗くしない。割合 .55 < .875
    say(b, "山に行く。", pointed="相手")
    assert b.dark_line("山に行く。")                       # 相手に通じなかった行は、この会話では暗い
    b.conv_dark = set()
    assert not b.dark_line("山に行く。")                   # 次の会話: (1.1)/2 = .55 < ふだんの割合 .89
    for _ in range(3):
        say(b, "変な行。", pointed="相手")
    b.conv_dark = set()
    assert not b.dark_line("変な行。")                     # 指摘の多い場（ふだん .92）では、毎回指摘でも 3 回では (3.1)/4 = .78 に届かない


def test_dark_base_low_field_rate():
    b = bot(True)
    for i in range(9):                                     # 指摘の少ない場（10 行のうち 1 行）
        say(b, f"行{i}。")
    say(b, "変な行。", pointed="相手")
    b.conv_dark = set()
    assert abs(b.dark_cut() - 0.1) < 1e-9 and b.dark_line("変な行。") and not b.dark_line("行0。")   # .55 > .1: 暗い
