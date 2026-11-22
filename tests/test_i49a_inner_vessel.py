"""I49a の内面の骨組みの関門（物差しの 1: 壊れていないか）。numpy だけで動く。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "intervention"))
import i49a_inner_vessel as V  # noqa: E402


def unit(dim, i):
    x = np.zeros(dim, dtype=np.complex64)
    x[i] = 1.0
    return x


def test_bounded_even_with_huge_inputs():
    v = V.Vessel(16, bands=8)
    rng = np.random.default_rng(0)
    for _ in range(500):
        v.hear((rng.normal(size=16) * 1e6).astype(np.complex64))
    a = v.arrays()
    assert np.isfinite(a["act"]).all() and np.isfinite(a["gain"]).all()
    assert a["act"].max() <= float(v.cap) * (1 + 1e-4)
    assert a["gain"].max() <= float(v.gmax) * (1 + 1e-4) and a["gain"].min() >= 1 / float(v.gmax) * (1 - 1e-4)
    assert a["張り付き"].sum() > 0                      # 止め役が働いた事は見張りに残る


def test_quiets_without_input_fast_bands_first():
    v = V.Vessel(8, bands=6, longest=64)
    for _ in range(200):
        v.hear(unit(8, 0))
    before = v.act().copy()
    for _ in range(20):
        v.tick()
    after = v.act()
    assert (after < before).all()
    ratio = after / before
    assert ratio[0] < ratio[-1]                          # 速い波ほど先に薄れ、遅い波は残る


def test_think_stops_at_budget_and_when_empty():
    v = V.Vessel(8, bands=4)
    assert v.think(50) == 0                              # 何も無ければ回さない
    v.hear(unit(8, 0))
    assert v.think(50) == 50
    assert v.t == 51


def test_log_has_every_tick_and_is_reproducible():
    def run():
        v = V.Vessel(8, bands=5)
        for k in range(30):
            v.hear(unit(8, k % 3)) if k % 4 else v.tick()
        v.settle(unit(8, 5))
        return v
    a, b = run().arrays(), run().arrays()
    assert len(a["src"]) == 31 and a["act"].shape == (31, 5)
    assert a["src"][-1] == 2 and set(a["src"].tolist()) == {0, 1, 2}
    for k in ("src", "act", "gain", "sur"):
        assert np.array_equal(a[k], b[k])


def test_surprise_falls_on_repeat_and_jumps_on_new():
    v = V.Vessel(8, bands=6, longest=32)
    first = v.hear(unit(8, 0)).mean()
    for _ in range(40):
        last = v.hear(unit(8, 0)).mean()
    new = v.hear(unit(8, 1)).mean()
    assert first == 1.0 and last < 0.01 and new > 0.9


def test_same_input_different_state_gives_different_reaction():
    a, b = V.Vessel(8, bands=4), V.Vessel(8, bands=4)
    for _ in range(20):
        a.hear(unit(8, 0))
        b.hear(unit(8, 1))
    assert a.hear(unit(8, 0)).mean() < 0.01 < b.hear(unit(8, 0)).mean()


def test_watchdog_counts_dead_ticks():
    v = V.Vessel(8, bands=4)
    for _ in range(10):
        v.tick()
    assert v.watch["死んだ"] == 10 and v.watch["刻み"] == 10


def test_gain_rises_when_quiet():
    v = V.Vessel(8, bands=4, longest=16)
    for _ in range(100):
        v.hear(unit(8, 0))
    g0 = v.gain.copy()
    for _ in range(50):
        v.tick()
    assert (v.gain >= g0).all() and (v.gain > g0).any()
