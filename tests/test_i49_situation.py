"""I49 状況波の部品（i49_situation）: まとめの口 SituationWave を、小さな作り物の文で確かめる。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("intervention", "tools"):
    sys.path.insert(0, os.path.join(HERE, "..", d))

import i49_situation as SIT


def _lines():
    # 丁寧な行は丁寧なまま、くだけた行はくだけたまま続く世界。場の切れ目（空の行）ごとに、丁寧な場とくだけた場が入れ替わる。
    who = ("犬", "猫", "鳥", "馬", "牛", "魚", "虫", "人")
    out = []
    for i, a in enumerate(who):
        for j, b in enumerate(who):
            if i != j:
                out += ["%sが来ました。%sも行きますよ。" % (a, b), "%sは来ますか。%sも来ました。" % (b, a), "",
                        "%sが来たんだ。%sも行くんだ。" % (a, b), "%sは来るのか。%sも来たんだ。" % (b, a), ""]
    return out


def _wave():
    keep = (SIT.MIN, SIT.SEATS, SIT.SEEN)
    SIT.MIN, SIT.SEATS, SIT.SEEN = 2, 200, 2
    try:
        return SIT.SituationWave(_lines())
    finally:
        SIT.MIN, SIT.SEATS, SIT.SEEN = keep


def test_cutting_parts():
    assert list(SIT.sentences(["「犬が来た。猫も来ます！」"])) == ["犬が来た", "猫も来ます"]
    assert SIT.tail_of("今まで犬が歩いていた") == "が歩いていた"
    assert SIT.tail_of("さっき、犬が来た", seats=("さっき",)) == "犬が来た"
    assert SIT.spans("来た") == ["来", "来た", "た"]
    assert SIT.distinct(["a", "b", "a"]) == ["a", "b"]
    assert sorted(SIT.golden_order(5)) == [0, 1, 2, 3, 4]
    assert SIT.marks("来ましたよ") == dict(丁寧=True, 過去=True, 問い=False)
    assert SIT.marks("ござります", old=True)["古風"]


def test_runs_do_not_cross_breaks():
    runs = SIT.runs_of(["犬だ。", "猫です。", "", "鳥だよ。", "馬か。"])
    assert set(SIT.across_pairs(runs)) == {("犬だ", "猫です"), ("鳥だよ", "馬か")}


def test_wave_fits_the_same_tone():
    w = _wave()
    polite, casual = ("ました", "ますよ"), ("たんだ", "くんだ")
    for mode in SIT.MODES:
        assert set(w.assoc) == set(SIT.MODES)
        assert w.fit(polite, "牛も行きますよ", mode) > w.fit(polite, "牛も行くんだ", mode)
        assert w.fit(casual, "牛も行くんだ", mode) > w.fit(casual, "牛も行きますよ", mode)
    assert w.rank(polite, ["行くんだ", "行きますよ"])[0] == "行きますよ"
    assert w.rank(casual, ["行きますよ", "行くんだ"])[0] == "行くんだ"


def test_wave_rewrites_only_the_ending():
    w = _wave()
    keep = SIT.SEEN
    SIT.SEEN = 2
    try:
        assert w.rewrite("牛も行くんだ", ("ました", "ますよ"))[0][0] == "牛も行きますよ"
        assert w.rewrite("牛も行きますよ", ("たんだ", "くんだ"))[0][0] == "牛も行くんだ"
        assert w.rewrite("あ", ("ました",)) == []
    finally:
        SIT.SEEN = keep


def test_null_world_has_no_preference():
    # 帰無の世界: 丁寧とくだけが、前の文と関係なく同じ割合で続く。どちらの状況でも、合い方に差が出ない。
    who = ("犬", "猫", "鳥", "馬", "牛", "魚", "虫", "人")
    lines = []
    for i, a in enumerate(who):
        for j, b in enumerate(who):
            if i != j:
                for x in ("%sが来ました", "%sが来たんだ"):
                    for y in ("%sも行きますよ", "%sも行くんだ"):
                        lines.append(x % a + "。" + y % b + "。")
    keep = (SIT.MIN, SIT.SEATS)
    SIT.MIN, SIT.SEATS = 2, 200
    try:
        w = SIT.SituationWave(lines)
    finally:
        SIT.MIN, SIT.SEATS = keep
    d = w.fit(("ました",), "牛も行きますよ") - w.fit(("ました",), "牛も行くんだ")
    assert abs(d) < 1e-4
