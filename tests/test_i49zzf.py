"""I49zzf: 属性の帯ごとの、近さへの効き方の数え（小さな世界）。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("intervention", "tools"):
    sys.path.insert(0, os.path.join(HERE, "..", d))

import numpy as np

import i49m_attr_base as AB
import i49zzf_attr_bands as F


def _world():
    # 帯「種類」は使われ方と合う（食べ物 / 乗り物）。帯「色」は使われ方と合わない。帯「物」はどの語も持つ。
    kinds = dict(りんご="食べ物", みかん="食べ物", パン="食べ物", 車="乗り物", 船="乗り物", 電車="乗り物")
    colors = dict(りんご="赤", みかん="黄", パン="白", 車="赤", 船="白", 電車="黄")
    board = [dict(name=w, pos="名詞", attrs={"種類": {kinds[w]: 1.0}, "色": {colors[w]: 1.0}, "物": {"物": 1.0}}) for w in kinds]
    verb = dict(食べ物="食べる", 乗り物="乗る")
    talks = [([("人", w, verb[k])], None) for w, k in kinds.items()]
    return AB.AttrBase(board), talks


def test_rank_share():
    assert F.rank_share(np.array([.9, .8, .1, .2]), np.array([True, True, False, False])) == 1.0
    assert F.rank_share(np.array([.5, .5, .5, .5]), np.array([True, False, True, False])) == .5
    assert F.rank_share(np.array([.1, .2]), np.array([True, True])) is None


def test_band_report_finds_the_band_that_matches_usage():
    base, talks = _world()
    d = F.band_report(base, talks)["何を"]
    assert d["語"] == 6 and d["組"] == 15 and d["同じ使われ方の組"] == 6
    b = d["帯"]
    assert b["種類"]["その帯だけの見分け"] == 1.0                      # 使われ方と合う帯は、それだけで見分ける
    assert b["色"]["その帯だけの見分け"] < .5                          # 合わない帯は見分けない
    assert b["物"]["その帯だけの見分け"] == .5                         # どの語も持つ帯は、どの組も同じ近さ
    assert b["色"]["その帯を抜いた見分け"] >= d["全部の帯の見分け"]    # 合わない帯を抜くと、見分けは下がらない
    assert abs(sum(v["近さの取り分"] for v in b.values()) - d["cosの平均"]) < 1e-3
    assert abs(sum(v["力の取り分"] for v in b.values()) - 1.0) < 1e-3
