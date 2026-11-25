"""I49zzj: 状況波（時制）の部品（小さな世界）。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("intervention", "tools"):
    sys.path.insert(0, os.path.join(HERE, "..", d))

import numpy as np

import i49zzj_tense_run as R
import i49zzj_tense_wave as TW


def _board():
    return TW.TenseBoard(R.FORMS)


def test_waves_form_a_gradient():
    b = _board()
    c = [TW.cos(b.wave("未来"), b.wave(t)) for t in b.order]
    assert abs(c[0] - 1.0) < 1e-5 and all(x > y for x, y in zip(c, c[1:]))   # 並びで遠いほど、波も遠い
    assert b.wave("過去").dtype == np.float32


def test_time_word_learns_the_tense_it_came_with():
    b = _board()
    tw = TW.TimeWords(b)
    for form in ("歩いた", "走った", "食べていた"):
        tw.learn("さっき", form)
    assert b.nearest(tw.wave("さっき"))[0] == "過去"
    assert tw.wave("知らない語") is None
    s = TW.Situation()
    s.hear(tw.wave("さっき"))
    assert b.pick_form("笑う", s.state) == "笑った"                           # 学習に出ていない動詞でも、近い形を選ぶ


def test_situation_fades_and_follows_the_newer_wave():
    b = _board()
    s = TW.Situation(half=0.5)
    s.hear(b.wave("過去"))
    s.hear(b.wave("未来"))
    assert b.nearest(s.state)[0] == "未来"
    assert TW.cos(s.state, b.wave("過去")) > TW.cos(b.wave("未来"), b.wave("過去"))   # 前の分は消えずに残る


def test_two_tenses_on_both_sides_merge_when_the_bumps_are_wide():
    # 現在進行と過去進行の両方と出る語は、山が広いと間の「過去」に寄る。山が狭ければ寄らない。周波数 1 つで持つと必ず寄る。
    keep = TW.WIDTH
    try:
        for width, want in ((8.0, True), (4.0, False)):
            TW.WIDTH = width
            b = _board()
            tw = TW.TimeWords(b)
            for form in ("歩いている", "歩いていた"):
                tw.learn("ずっと", form)
            assert (b.nearest(tw.wave("ずっと"))[0] == "過去") is want
            assert b.nearest(tw.wave("ずっと", single=True))[0] == "過去"
    finally:
        TW.WIDTH = keep


def test_run_is_the_same_every_time():
    a, b = R.run(), R.run()
    assert a == b and a["学習の文"] == 768
    assert a["形を選ぶ"]["山の形のまま"]["割合"] > a["形を選ぶ"]["当てずっぽう"]
