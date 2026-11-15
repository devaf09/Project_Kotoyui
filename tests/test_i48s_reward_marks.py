"""I48s（会話の報酬の印）の小さな確かめ（GiNZA を使わない）。"""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "intervention"))
import i48s_reward_marks as M  # noqa: E402


def tok(i, w, pos, lemma=None, dep="dep", head=0):
    return dict(i=i, text=w, lemma=lemma or w, pos=pos, dep=dep, head=head)


def T(*tw):
    return [tok(i, *x) for i, x in enumerate(tw)]


def test_thanks_and_new_words():
    assert M.thanks(T(("ありがとう", "INTJ"), ("！", "PUNCT")))
    assert M.thanks(T(("助かっ", "VERB", "助かる"), ("た", "AUX")))
    assert not M.thanks(T(("海", "NOUN"), ("に", "ADP"), ("行く", "VERB")))
    assert M.new_words(T(("海", "NOUN"), ("山", "NOUN")), {"海"}) == 1


def test_resonance_and_aha():
    near = types.SimpleNamespace(near=lambda a, b: {a, b} == {"海", "浜"})
    p = T(("海", "NOUN"), ("に", "ADP"), ("行く", "VERB"))
    assert M.resonance(T(("浜", "NOUN"), ("いい", "ADJ")), p, near) == (False, True)       # 同じ語なし、近い語あり: 共鳴だけ
    assert M.resonance(T(("海", "NOUN"), ("いい", "ADJ")), p, near) == (True, False)       # そのまま返し
    H = [T(("猫", "NOUN"), ("が", "ADP"), ("好き", "ADJ"))]
    assert M.aha(T(("猫", "NOUN"), ("と", "ADP"), ("海", "NOUN")), p, H)                    # 前の行の語と、今の行の語を結ぶ
    assert not M.aha(T(("海", "NOUN"), ("いい", "ADJ")), p, H)


def test_odd_uses_book_counts():
    book = [dict(lines=[dict(toks=T(("海", "NOUN", None, "obl", 2), ("に", "ADP", None, "case", 0), ("行く", "VERB", None, "ROOT", 2)))] * 20)]
    od = M.Odd(book)
    d, where, _ = od.worst(T(("海", "NOUN", None, "obl", 2), ("を", "ADP", None, "case", 0), ("行く", "VERB", None, "ROOT", 2)))
    assert d > 0.5 and where[0] == "述語に合わない助詞"                                    # 本では「海に行く」だけ: 「海を行く」は不自然
    d2, _, _ = od.worst(T(("海", "NOUN", None, "obl", 2), ("に", "ADP", None, "case", 0), ("行く", "VERB", None, "ROOT", 2)))
    assert d2 < 0.5


def test_bot_bind_value():
    """chat_bot3 の --bind: 結んだ行を言い、相手が受けて続けた経験が、その行の点を上げる。"""
    import chat_bot3 as C3
    b = C3.Bot3.__new__(C3.Bot3)
    b.uke, b.bind = None, True
    b.pb = types.SimpleNamespace(pieces=lambda toks: [])           # 所の無い行: 行のまるごとで数える
    b.conv = dict(lines=[dict(who="人", text="猫が好き", toks=T(("猫", "NOUN"), ("が", "ADP"), ("好き", "ADJ"))),
                         dict(who="ボット", text="いいね", toks=T(("いい", "ADJ"),)),
                         dict(who="人", text="海に行く", toks=T(("海", "NOUN"), ("に", "ADP"), ("行く", "VERB"))),
                         dict(who="ボット", text="猫と海", toks=T(("猫", "NOUN"), ("と", "ADP"), ("海", "NOUN")))])
    v0 = b._bind_value(b.conv["lines"][-1]["toks"], "猫と海")
    b._bind_said()
    assert b._last_bind is not None and b.bind_log["結び"] == 1      # 今の行（海）と前の行（猫）の語を結んだ
    b._dl().resolve("結び", True)                                   # 相手が受けて続けた（_reply の中でする足し方と同じ）
    assert b._bind_value(b.conv["lines"][-1]["toks"], "猫と海") > v0
    b.conv["lines"].append(dict(who="人", text="うん", toks=T(("うん", "INTJ"),)))
    b.conv["lines"].append(dict(who="ボット", text="海", toks=T(("海", "NOUN"),)))
    b._bind_said()
    assert b._last_bind is None                                     # 前の行の語だけ: 結びではない
    k = b._bind_keys(b.conv["lines"][-1]["toks"], "海")
    v = b._bind_value(b.conv["lines"][-1]["toks"], "海")
    b._dl().resolve("結び", True)                                   # 結びでない返事が受けられても、結びの良は付かない
    assert b._bind_value(b.conv["lines"][-1]["toks"], "海") == v and b.delayed.said["結び"][k[0]] == 1
