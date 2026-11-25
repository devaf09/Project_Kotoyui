"""I49zzo: 状況波を入れ替えて、出る終わりが変わるかを見る道具（小さな作り物の文）。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("intervention", "tools"):
    sys.path.insert(0, os.path.join(HERE, "..", d))

import i49zzm_prev_tail_run as ZM
import i49zzo_steer_run as S


def _lines():
    who = ("犬", "猫", "鳥", "馬", "牛", "魚", "虫", "人")
    out = []
    for i, a in enumerate(who):
        for j, b in enumerate(who):
            if i != j:
                out += ["%sが来ました。%sも行きますよ。" % (a, b), "%sが来たんだ。%sも行くんだ。" % (a, b)]
    return out


def test_marks_have_old_style():
    assert S.marks("ござります")["古風"] and S.marks("ござります")["丁寧"] and not S.marks("行くんだ")["古風"]


def test_changing_the_situation_changes_the_ending():
    keep = (ZM.MIN, ZM.SEATS, S.COUNT, S.TOP)
    ZM.MIN, ZM.SEATS, S.COUNT, S.TOP = 2, 200, 2, 1
    try:
        inside, _ = ZM.pairs_of(_lines())
        r = S.steer(ZM.Assoc(inside), S.candidates([c for _, c in inside]))
    finally:
        ZM.MIN, ZM.SEATS, S.COUNT, S.TOP = keep
    sh = r["状況ごとの上位の割合"]
    assert sh["間柄: 丁寧"]["丁寧"] == 1.0 and sh["間柄: くだけ"]["丁寧"] == 0.0     # 状況を入れ替えると、出る終わりが替わる
    ex = r["例"]
    assert "ます" in ex["間柄: 丁寧"]["字ごと"]["行"][0] and "んだ" in ex["間柄: くだけ"]["字ごと"]["行"][0]


def test_across_pairs_do_not_cross_breaks():
    runs = ZM.runs_of(["犬だ。", "猫です。", "", "鳥だよ。", "馬か。"])
    assert set(S.across_pairs(runs)) == {("犬だ", "猫です"), ("鳥だよ", "馬か")}


def test_rewrite_changes_only_the_ending():
    import i49zzl_tense_text_run as ZL
    keep = (ZM.MIN, ZM.SEATS, S.SEEN)
    ZM.MIN, ZM.SEATS, S.SEEN = 2, 200, 2
    try:
        lines = _lines()
        inside, _ = ZM.pairs_of(lines)
        table = S.joins(ZL.distinct(list(ZL.sentences(lines))))
        a = ZM.Assoc(inside)
        polite = S.rewrite("牛も行くんだ", ("ました", "ますよ"), a, table)[0][0]
        casual = S.rewrite("牛も行きますよ", ("たんだ", "くんだ"), a, table)[0][0]
    finally:
        ZM.MIN, ZM.SEATS, S.SEEN = keep
    assert polite == "牛も行きますよ" and casual == "牛も行くんだ"      # 頭はそのままで、終わりだけが状況に合わせて替わる
    assert S.rewrite("あ", ("ました",), a, table) == []
