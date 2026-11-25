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


def test_lags_show_style_that_returns_every_other_utterance():
    # 2 人の会話。A はいつも丁寧、B はいつもくだけている世界。丁寧さは 1 個前では続かず、2 個前から続く。
    who = ("犬", "猫", "鳥", "馬", "牛", "魚", "虫", "人")
    lines = []
    for i, a in enumerate(who):
        for j, b in enumerate(who):
            if i != j:
                lines += ["%sが来ました。" % a, "%sも来るぞ。" % b, "%sは帰ります。" % b, "%sも帰るぞ。" % a, "%sを見ました。" % a, "%sも見たぞ。" % b, ""]
    keep = (M.MIN, M.SEATS)
    M.MIN, M.SEATS = 2, 200
    try:
        runs = M.runs_of(lines)
        r = M.run_lags(runs, max_lag=2)
    finally:
        M.MIN, M.SEATS = keep
    assert len(runs) == 56 and all(len(x) == 6 for x in runs)
    assert r["続き方"]["1"]["丁寧"]["倍"] < .1 and r["続き方"]["2"]["丁寧"]["倍"] > 1.5
    assert r["二つから選ぶ"]["問い"] > 0 and set(r["二つから選ぶ"]["重ねる"]) == {"1 個前まで", "2 個前まで"}


def test_breaks_are_seen_when_the_scene_style_changes():
    # 続きごとに場の口調が決まっている世界（丁寧な場と、くだけた場が交互）。切れ目の直後は、前の発言と合わない。
    who = ("犬", "猫", "鳥", "馬", "牛", "魚", "虫", "人")
    lines = []
    k = 0
    for i, a in enumerate(who):
        for j, b in enumerate(who):
            if i == j:
                continue
            end = ("ました。", "ます。") if k % 2 == 0 else ("たぞ。", "るぞ。")
            lines += ["%sが来%s" % (a, end[0]), "%sも来%s" % (b, end[1]), "%sを見%s" % (a, end[0]), "%sも見%s" % (b, end[1]), ""]
            k += 1
    keep = (M.MIN, M.SEATS)
    M.MIN, M.SEATS = 2, 200
    try:
        r = M.run_breaks(M.runs_of(lines))
    finally:
        M.MIN, M.SEATS = keep
    assert r["切れ目の直後の発言"] == 27 and r["続きの中の発言"] == 56
    assert r["続きの中の方が合う割合"]["学んだ波"] > .9
