"""I48x（門: 深刻な相談の固定の返事、白銀律）の確かめ（GiNZA を使わない）。"""
import os
import sys
import types
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "intervention"))
import i48x_gate as G  # noqa: E402


class Tok:
    def __init__(self, i, w):
        self.i, self.text, self.lemma_, self.pos_, self.dep_ = i, w, w, "NOUN", "dep"
        self.head = self


def nlp(text):
    return [Tok(i, w) for i, w in enumerate(text.split())] or [Tok(0, text)]


def bot(gate=True, silver=False):
    import chat_bot3 as C3
    b = C3.Bot3.__new__(C3.Bot3)
    b.uke, b._last_nov, b.gate, b.silver, b.nlp = None, None, gate, silver, nlp
    b.conv = dict(lines=[])
    b.pb = types.SimpleNamespace(pieces=lambda toks: [dict(ns=t["lemma"], c="", ps="", n=t["lemma"]) for t in toks])
    return b


def test_crisis_marks():
    assert G.crisis("もう死にたい")
    assert G.crisis("最近 ずっと 眠れない日が 続いてて")
    assert G.crisis("助けて")
    assert not G.crisis("今日は海に行った")
    assert G.fixed_reply() and not G.fixed_reply().startswith("#")


def test_gate_reply_is_fixed_and_never_dark():
    b = bot()
    out = b.reply("もう 死にたい")
    assert out["line"] == G.fixed_reply() and out.get("gate")
    for _ in range(5):                                       # 先生・相手に何度指摘されても、暗くならず、経験に入らない
        b.mark_line(out["line"], src="先生")
        b.mark_line(out["line"], src="相手")
    assert not b.dark_line(out["line"]) and not b.__dict__.get("line_pointed")
    assert b.taught(True, "死にたい", "") == []
    assert b.reply("また 死にたい")["line"] == G.fixed_reply()  # 何度目でも同じ固定の返事


def test_gate_off_and_silver():
    b = bot(gate=False, silver=True)
    assert G.crisis("死にたい") and not b._is_gate_line(G.fixed_reply())   # 門が無い時は、固定の返事として扱わない
    b.silver_forms = Counter({"意味不明": 1})
    assert b.dark_line("それは 意味不明")                     # 相手にされて嫌だった形を含む行は、言う前に暗い
    assert not b.dark_line("海 いい")
