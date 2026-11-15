"""I48x（感情・型・報酬を足す土台）の確かめ。帯を足すたびに、重ならない置き方がこのテストで確かめられる。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "intervention"))
import i48x_mesh as X  # noqa: E402

EMOTIONS = ("喜び", "悲しみ", "信頼", "嫌悪", "恐れ", "怒り", "驚き", "期待")
TYPES = ("感嘆・反復", "Yes, And", "横へ広げる問い", "時間軸の問い", "言い換えて褒める", "受けだけ")


def table():
    t = X.BandTable(2048)
    for who in ("自分", "相手"):
        for e in EMOTIONS:
            t.add(f"{who}:{e}", "感情")
    for k in TYPES:
        t.add("型:" + k, "型")
    return t


def test_band_table_no_overlap():
    t = table()
    assert t.check() and len(t.rows) == 22 and max(t.freqs()) < 2048
    t.add("型:新しい型", "型")                                   # 帯を足す時は 1 行足すだけ。置き方は自動で、重ならない
    assert t.check()


def test_stimulus_direction_fixed_strength_learned():
    s = X.Stimulus()
    s.link("感謝された", "自分:喜び", +1, w0=0.1)
    s.link("拒まれた", "自分:喜び", -1, w0=0.1)
    st = s.apply({"感謝された": 1.0}, {})
    assert abs(st["自分:喜び"] - 0.1) < 1e-9
    for _ in range(50):
        s.learn({"感謝された": 1.0}, {}, {"自分:喜び": 0.6})
    assert abs(s.apply({"感謝された": 1.0}, {})["自分:喜び"] - 0.6) < 0.02     # 強さは答えに近づく
    assert s.apply({"拒まれた": 1.0}, {"自分:喜び": 0.5})["自分:喜び"] < 0.5    # 向きは置いたまま


def test_delayed_next_and_end():
    d = X.Delayed()
    d.register("結び", d.NEXT)
    d.register("落ち着き", d.END)
    v0 = d.value("結び", ["a"])
    d.said_keys("結び", ["a"], eligible=True)
    d.resolve("結び", True)
    assert d.value("結び", ["a"]) > v0
    d.said_keys("結び", ["b"], eligible=False)
    d.resolve("結び", True)                                       # 対象でない返事には付かない
    assert d.good["結び"]["b"] == 0
    for k in ("x", "y"):
        d.said_keys("落ち着き", [k])
    d.resolve("落ち着き", True, weight=1.0)                       # 会話の終わりに、返事全体へ薄く配る
    assert d.good["落ち着き"]["x"] == 0.5 and d.good["落ち着き"]["y"] == 0.5


def test_gate_chain_order():
    ch = X.GateChain()
    ch.add("深刻な相談", lambda c: (X.FIXED, "固定の返事") if "死にたい" in c else (X.PASS, None))
    ch.add("白銀律", lambda c: (X.NARROW, "暗くする") if "嫌" in c else (X.PASS, None))
    assert ch.run("死にたいし嫌") == (X.FIXED, "固定の返事", "深刻な相談")    # 固定が出たら、そこで止める
    assert ch.run("嫌なこと")[0] == X.NARROW and ch.run("海")[0] == X.PASS
