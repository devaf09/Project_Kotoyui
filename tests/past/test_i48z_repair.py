"""I48z（A-1: 聞き返しへの言い直しを後の段で替えない、台本の行を経験に入れない）の確かめ（GiNZA を使わない）。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "intervention"))


class Tok:
    def __init__(self, i, w, pos="NOUN"):
        self.i, self.text, self.lemma_, self.pos_, self.dep_ = i, w, w, pos, "dep"
        self.head = self


def nlp(text):
    return [Tok(i, w) for i, w in enumerate(text.split())] or [Tok(0, text)]


def toks(text):
    return [dict(i=i, text=w, lemma=w, pos="NOUN", dep="dep", head=i) for i, w in enumerate(text.split())]


def bot(dark=(), head=""):
    import chat_bot3 as C3
    b = C3.Bot3.__new__(C3.Bot3)
    b.nlp = nlp
    b.idf = {"海": 2.0, "空": 1.0}
    b.dark_line = lambda line: line in dark
    b.restate_head = lambda: head
    b.value = lambda t, line: -len(line)                      # 短い方が点が大きい（確かめ用）
    b._chunk = "光"
    b.conv = dict(lines=[dict(who="人", text="空 と 海", toks=toks("空 と 海")), dict(who="ボット", text="パン は 光", toks=toks("パン は 光")),
                         dict(who="人", text="光って？", toks=toks("光って？")), dict(who="ボット", text="X", toks=toks("X"))])
    return b


def test_repair_alt_picks_undark_form():
    b = bot(dark={"ええ、パンは光ってこと。"}, head="ええ")
    out = b._repair_alt(dict(line="ええ、パンは光ってこと。", why="相手の聞き返し → …", repair=True))
    assert out["repair"] and out["line"] == "光のこと。"      # 候補: 頭を外した形・「海の、光のこと。」・「光のこと。」のうち点の大きい方
    assert b.conv["lines"][-1]["text"] == "光のこと。"


def test_repair_alt_all_dark_keeps_or_short():
    b = bot(dark={"パンは光ってこと。", "海の、光のこと。", "光のこと。"})
    out = b._repair_alt(dict(line="パンは光ってこと。", why="w", repair=True))
    assert out["line"] in ("パンは光ってこと。", "光のこと。")


def test_skip_line():
    import chat_bot3 as C3
    b = C3.Bot3.__new__(C3.Bot3)
    assert not b._is_skip_line("うちの猫の名前、覚えてる？")     # 既定は空
    b.skip_lines = frozenset({"うちの猫の名前、覚えてる？"})
    assert b._is_skip_line("「うちの猫の名前、覚えてる？」")
