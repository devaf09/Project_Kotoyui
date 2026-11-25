"""I49zzq: 状況波を位相の形に置き換える部品（小さな作り物の文）。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("intervention", "tools"):
    sys.path.insert(0, os.path.join(HERE, "..", d))

import i49_situation as SIT
import i49zzq_phase_run as Q
from i49zzq_phase_wave import PhaseAssoc, seat_phases


def _lines(null=False):
    who = ("犬", "猫", "鳥", "馬", "牛", "魚", "虫", "人")
    out = []
    for i, a in enumerate(who):
        for j, b in enumerate(who):
            if i != j:
                if null:      # 何も続かない世界: 丁寧とくだけが、前の文と関係なく同じ割合で続く
                    out += [x % a + "。" + y % b + "。" for x in ("%sが来ました", "%sが来たんだ") for y in ("%sも行きますよ", "%sも行くんだ")]
                else:
                    out += ["%sが来ました。%sも行きますよ。" % (a, b), "%sが来たんだ。%sも行くんだ。" % (a, b)]
    return out


def _assoc(lines):
    keep = (SIT.MIN, SIT.SEATS)
    SIT.MIN, SIT.SEATS = 2, 200
    try:
        return SIT.Assoc(SIT.pairs_of(lines)[0])
    finally:
        SIT.MIN, SIT.SEATS = keep


def test_seat_phases_are_unit_waves_without_random():
    e = seat_phases(50, 256)
    assert e.dtype == np.complex64 and e.shape == (50, 256)
    assert np.allclose(np.abs(e), 1.0, atol=1e-5)
    assert np.array_equal(e, seat_phases(50, 256))
    assert Q.mix(e) < .05                                   # 別の席の波とは、ほとんど混ざらない


def test_phase_form_agrees_with_the_counted_form():
    a = _assoc(_lines())
    p = PhaseAssoc(a, 2048)
    for prev, cur in (("ました", "牛も行きますよ"), ("ました", "牛も行くんだ"), ("たんだ", "牛も行くんだ"), ("たんだ", "牛も行きますよ")):
        assert abs(p.score(prev, SIT.tail_of(cur)) - a.score(prev, SIT.tail_of(cur))) < .15 * max(abs(a.score(prev, SIT.tail_of(cur))), 1.0)
    assert p.score("ました", "行きますよ") > p.score("ました", "行くんだ")
    assert p.score("たんだ", "行くんだ") > p.score("たんだ", "行きますよ")


def test_situation_is_one_wave_and_fades():
    p = PhaseAssoc(_assoc(_lines()), 1024)
    polite, casual = p.situation(["ました", "ますよ"]), p.situation(["たんだ", "くんだ"])
    assert polite.shape == (1024,) and polite.dtype == np.complex64
    assert p.fit(polite, "行きますよ") > p.fit(polite, "行くんだ") and p.fit(casual, "行くんだ") > p.fit(casual, "行きますよ")
    mixed = p.situation(["ました", "たんだ"], fade=.2)        # 古い丁寧は薄れ、新しいくだけが勝つ
    assert p.fit(mixed, "行くんだ") > p.fit(mixed, "行きますよ")
    assert not np.any(p.situation(["zzz"]))                 # 席に無い終わりは、波を作らない


def test_null_world_has_no_preference():
    p = PhaseAssoc(_assoc(_lines(null=True)), 2048)
    s = p.situation(["ました"])
    assert abs(p.fit(s, "行きますよ") - p.fit(s, "行くんだ")) < .02


def test_tool_runs_on_small_world():
    keep = (SIT.MIN, SIT.SEATS, Q.COUNT)
    SIT.MIN, SIT.SEATS, Q.COUNT = 2, 200, 2
    try:
        r = Q.run(SIT.pairs_of(_lines() * 2 + ["%sが来ました。%sも行きますよ。" % (a, b) for a in "犬猫" for b in "鳥馬"])[0], dims=(32, 512))
    finally:
        SIT.MIN, SIT.SEATS, Q.COUNT = keep
    assert set(r["位相の形"]) == {"32", "512"}
    assert r["位相の形"]["512"]["元の形と同じ方を選んだ割合"] >= r["位相の形"]["32"]["元の形と同じ方を選んだ割合"] - .05
