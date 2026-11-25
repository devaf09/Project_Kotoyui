"""I49zzr: 状況波をずらして書き替え、違和感チェックに通す部品（小さな作り物の文）。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("intervention", "tools"):
    sys.path.insert(0, os.path.join(HERE, "..", d))

import i49_situation as SIT
import i49zzr_air as AIR
import i49zzr_air_run as R
from i49zzq_phase_wave import PhaseAssoc


def _lines():
    who = ("犬", "猫", "鳥", "馬", "牛", "魚", "虫", "人")
    out = []
    for i, a in enumerate(who):
        for j, b in enumerate(who):
            if i != j:
                out += ["%sが来ました。" % a, "%sも行きますよ。" % b, "", "%sが来たんだ。" % a, "%sも行くんだ。" % b, ""]
    return out


def _small(fn):
    keep = (SIT.MIN, SIT.SEATS, SIT.SEEN, R.COUNT)
    SIT.MIN, SIT.SEATS, SIT.SEEN, R.COUNT = 2, 200, 2, 2
    try:
        return fn()
    finally:
        SIT.MIN, SIT.SEATS, SIT.SEEN, R.COUNT = keep


def test_odd_check_sees_only_the_seam():
    chk = AIR.OddCheck(["犬が来ました", "猫も行きますよ"], n=3)
    assert chk.passes("犬が来ました", 3)                       # 材料に有る並び
    assert not chk.passes("犬が来ますよ", 3)                   # 「来ます」は材料に無い
    assert chk.unseen("犬が来ますよ", 3) == (1, 2)             # 「が来ま」は有り、「来ます」が無い
    assert not chk.passes("犬が来ました", 3, least=2)          # 自分の 1 回を除くと、見た事が無い
    assert AIR.OddCheck(["ab"], n=3).grams("ab", 1) == [AIR.HEAD + "ab", "ab" + AIR.END]


def test_shift_moves_the_situation():
    def go():
        lines = _lines()
        a = SIT.Assoc(list(SIT.across_pairs(SIT.runs_of(lines))))
        p = PhaseAssoc(a, 512)
        dirs, share = AIR.air_directions(p, 2)
        assert dirs.shape == (2, 512) and dirs.dtype == np.complex64 and share[0] >= share[1]
        zero = np.zeros(512, dtype=np.complex64)
        up, down = AIR.shifted(zero, dirs[0], 1, 1.0), AIR.shifted(zero, dirs[0], -1, 1.0)
        # 一番強い向きは、丁寧とくだけを分ける。正と負で、合う終わりが入れ替わる
        d_up = p.fit(up, "行きますよ") - p.fit(up, "行くんだ")
        d_down = p.fit(down, "行きますよ") - p.fit(down, "行くんだ")
        assert d_up * d_down < 0 and abs(d_up) > 1e-4
        table = SIT.joins(SIT.distinct(list(SIT.sentences(lines))))
        polite = up if d_up > 0 else down
        casual = down if d_up > 0 else up
        assert AIR.rewrites("牛も行くんだ", polite, p, table)[0][0] == "牛も行きますよ"
        assert AIR.rewrites("牛も行きますよ", casual, p, table)[0][0] == "牛も行くんだ"
        assert AIR.rewrites("あ", polite, p, table) == [("あ", 0.0, 0)]
    _small(go)


def test_tool_runs_and_counts():
    r = _small(lambda: R.run(_lines(), talk=("牛も行くんだ", "馬が来ました"), dims=256, pick=10))
    assert len(r["会話"]) == 2 * 2 * len(r["向きの取り分"])
    assert {"チェック無し", "チェック有り", "チェック無しは通るか", "打ち消しが変わった"} <= set(r["会話"][0])
    t = r["材料の文での数え"]
    assert t["書き替え"] == t["抜いた文"] * 2 * len(r["向きの取り分"]) and 0 <= t["変わった行のうち通る割合"] <= 1
    assert [c["見る並びの長さ"] for c in r["違和感チェックの確かめ"]] == [3, 4, 5]
    assert R.has_neg("今日は行きません") and not R.has_neg("今日は行きます")
