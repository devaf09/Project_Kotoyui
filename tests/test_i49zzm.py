"""I49zzm: 前の文の終わりを席にする道具（小さな作り物の文）。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("intervention", "tools"):
    sys.path.insert(0, os.path.join(HERE, "..", d))

import i49zzm_prev_tail_run as M


def _lines():
    # 丁寧な行は丁寧なまま、くだけた行はくだけたまま続く世界。語の中身は毎回違う。
    who = ("犬", "猫", "鳥", "馬", "牛", "魚", "虫", "人")
    out = []
    for i, a in enumerate(who):
        for j, b in enumerate(who):
            if i != j:
                out.append("%sが来ました。%sも来ます。" % (a, b))
                out.append("%sが来たぞ。%sも来るぞ。" % (a, b))
    return out


def test_marks_and_pairs():
    assert M.marks("来ましたよ") == dict(丁寧=True, 過去=True, 問い=False)
    assert M.marks("行くのか")["問い"] and not M.marks("行くぞ")["過去"]
    inside, across = M.pairs_of(["あれは犬だ。そうですか。", "", "猫です。", "鳥だよ。"])
    assert inside == [("あれは犬だ", "そうですか")] and across == [("猫です", "鳥だよ")]


def test_learned_wave_follows_the_carried_style():
    keep = (M.MIN, M.SEATS)
    M.MIN, M.SEATS = 2, 200
    try:
        inside, _ = M.pairs_of(_lines())
        r = M.run_one(inside)
    finally:
        M.MIN, M.SEATS = keep
    assert r["二つから選ぶ"]["学んだ波で選ぶ"] > .6             # 丁寧さが続く世界なら選べる（別の組も同じ丁寧さの時は五分なので、1 にはならない）
    assert r["続き方"]["丁寧"]["倍"] > 1.5
    a = M.Assoc(inside, seats=200, min_count=2)
    assert a.score("犬が来ました", "馬も来ます") > a.score("犬が来ました", "馬も来るぞ")
