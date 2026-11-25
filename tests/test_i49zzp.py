"""I49zzp: 算数の文字の波（小さな世界で、部品の動きを確かめる）。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("intervention", "tools"):
    sys.path.insert(0, os.path.join(HERE, "..", d))

import i49zzp_arith_run as R
from i49zzp_arith_wave import CharWave


def _learn(lines, offsets=True):
    w = CharWave(offsets=offsets)
    for x in lines:
        w.learn(x)
    return w


def test_worlds():
    add, sub = R.world()
    assert len(add) == 55 and len(sub) == 55 and all(len(x) == 5 for x in add + sub)
    add, sub = R.world(full=True)
    assert len(add) == 100 and "9+9=18" in add and "3-5=-2" in sub
    add, sub = R.ring_world()
    assert "7+5=2" in add and "3-5=8" in sub
    lines = add + sub
    null = R.null_world(lines)
    assert sorted(R.ans(x) for x in null) == sorted(R.ans(x) for x in lines)     # 答えの出る回数は同じ
    assert [R.head(x) for x in null] == [R.head(x) for x in lines] and null != lines


def test_seats_and_waves_are_centered():
    w = _learn(["1+2=3", "2+1=3"])
    assert (2, "2") in w.seat and (-4, "1") in w.seat
    units, m = w.waves()
    assert m.dtype.name == "float32" and units == sorted(units)
    assert abs(float(m.sum(axis=1).max())) < 1e-5          # 割合から平均の割合を引くので、行の合計は 0
    assert _learn(["1+2=3"], offsets=False).seat.keys() == {(0, c) for c in "1+2=3"}


def test_small_world_orders_the_digits():
    add, sub = R.world()
    r = R.run(add + sub)
    d = r["数の差ごとの近さ"]
    assert d["1"] > d["5"] > d["9"]                         # 差が小さい数ほど波が近い
    assert r["位相"]["一番強い向きと数の順の合い方"] == 1.0
    assert r["位相"]["位相を読める"]


def test_ring_world_pairs_carry_nothing():
    # 1 の位だけの世界: どの 2 文字の組も同じ回数だけ出るので、数字の波は全部同じになる（2 文字ずつの数えには、式の関係が入らない）。
    add, sub = R.ring_world()
    for off in (True, False):
        r = R.run(add + sub, off)
        assert r["位相"]["数字の波の散らばり"] < 1e-3 and not r["位相"]["位相を読める"]
        assert min(r["数の差ごとの近さ"].values()) > .999


def test_answer_handles_strings():
    w = _learn(["1+2=3", "9+9=18", "3-5=-2"])
    got = w.answer("9+9=", ["3", "18", "-2"])
    assert got[0][0] == "18" and len(got) == 3
    assert w.answer("1+2=", "3")[0][0] == "3"


def test_no_random_same_result_twice():
    add, sub = R.world(full=True)
    assert R.run(add + sub) == R.run(add + sub)


def test_hex_world():
    assert R.fmt(30, R.HEX) == "1E" and R.fmt(-2, R.HEX) == "-2" and R.num("1E", R.HEX) == 30 and R.num("-A", R.HEX) == -10
    add, sub = R.world(True, R.HEX)
    assert len(add) == 256 and "F+F=1E" in add and "3-5=-2" in sub
    add, sub = R.world(False, R.HEX)
    r = R.run(add + sub, True, R.HEX)
    assert len(r["数の差ごとの近さ"]) == 15 and r["位相"]["一番強い向きと数の順の合い方"] == 1.0
    add, sub = R.ring_world(R.HEX)
    assert "F+3=2" in add
    assert R.run(add + sub, True, R.HEX)["位相"]["数字の波の散らばり"] < 1e-3     # 16 進でも、回数の偏りが無ければ波は全部同じ
