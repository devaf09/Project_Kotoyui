"""I49zzn: 口調から話し手と関係を読む数え（小さな作り物の会話）。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("intervention", "tools"):
    sys.path.insert(0, os.path.join(HERE, "..", d))

import i49zzn_tone_speaker_run as N


def _talks(n=40):
    # 家族 2 人（父・娘）と初対面 1 人（客）。家族どうしはくだけて、客には丁寧に話す。客はいつも丁寧。
    things = ("犬", "猫", "鳥", "馬", "牛", "魚", "虫", "花")
    talks = []
    for k in range(n):
        u = []
        for i in range(12):
            t = things[(k + i) % len(things)]
            who, to = (("父", "娘"), ("娘", "客"), ("客", "父"), ("娘", "父"), ("父", "客"), ("客", "娘"))[i % 6]
            polite = who == "客" or to == "客"
            text = "%sを見ました。" % t if polite else "%sを見たぞ。" % t
            u.append(dict(who=who, tails=N.tails(text), to=[to], kind="Family", fam=["父", "娘"]))
        talks.append(u)
    return talks


def test_tone_changes_with_the_addressee():
    r = N.by_addressee(_talks())
    assert r["家族宛て"]["丁寧の割合"] == 0.0 and r["初対面の人宛て"]["丁寧の割合"] == 1.0
    assert r["両方に 10 回以上話した人"] == 2 and r["そのうち家族宛ての方が丁寧でない人"] == 2


def test_relation_is_read_from_tone():
    keep = N.MIN
    N.MIN = 2
    try:
        r = N.guess_relation(_talks())
    finally:
        N.MIN = keep
    assert r["当たり"] == 1.0 and r["多い方を言い続ける"] == 0.5


def test_speaker_guess_and_carry_run():
    talks = _talks()
    g = N.guess_speaker(talks)["Family"]
    assert g["問い"] == 40 * 6 and 0.0 <= g["当たり"] <= 1.0
    c = N.carry_by_speaker(talks, max_lag=3)
    assert c["1"]["同じ話し手"]["組"] == 0 and c["1"]["違う話し手"]["組"] == 40 * 11
    assert c["3"]["同じ話し手"]["組"] + c["3"]["違う話し手"]["組"] == 40 * 9
