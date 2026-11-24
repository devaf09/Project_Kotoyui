"""I49q の関門: 小さい自作の世界の総当たりの道具。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))


def test_world_is_fixed_and_classes_do_not_overlap():
    import i49q_small_world as Q
    heard, plaus, odd, half, ok, seats = Q.world(10)
    assert Q.world(10)[0] == heard and len(heard) == 219
    assert not set(heard) & set(plaus) and not (set(odd) | set(half)) & ok
    assert "私" in seats[0] and "する" in seats[2]


def test_more_phases_help_in_the_small_world():
    import i49q_small_world as Q
    rows, info = Q.run(10, (1, 16), (1,), ["今の波に足す", "元の波を足す"])
    get = lambda f, k: next(r["当て"] for r in rows if r["形"] == f and r["本数"] == k and not r["席"])
    assert info["語"] == 52 and get("今の波に足す", 16) > get("今の波に足す", 1) and get("元の波を足す", 16) > 0.9


def test_mix_tool_runs():
    import i49q_small_world_mix as M
    rows = M.run(10, (16,), (0, 5))
    get = lambda f, m: next(r["当て"] for r in rows if r["形"] == f and r["混ぜた語"] == m)
    assert get("元の波を足す", 0) > get("元の波を足す", 5)
    assert len(rows) == 2 * len(M.FORMS)


def test_attr_and_gate_tools_run_on_a_tiny_world():
    import i49q_small_world_attr as A
    import i49q_small_world_gate as Gt
    rows = A.run(T=16, Ms=(0,), passes=1)
    get = lambda name: next(r["当て"] for r in rows if r["形"].startswith(name))
    assert get("属性 19 本") >= get("属性 4 本（話 16 個で 1 つ）・聞かない")
    assert any(r["足した本数"] for r in rows if "外れた時に足す" in r["形"])
    assert len(Gt.run(T=16, Ms=(0,), passes=1)) == 20


def test_stream_world_orders_forms_like_the_real_count():
    import i49q_small_world_stream as St
    talks = St.make(talks=150)
    words = sorted({w for _, ws in talks for w in ws})
    idx = {w: i for i, w in enumerate(words)}
    B = St.base_dense(words, 64)
    r = {name: St.stream(mk(len(words), B), talks, idx)[0] for name, mk in St.FORMS}
    assert St.make(talks=150)[0] == talks[0]
    assert r["関係の表"] > r["元の向きを足す・席なし"] > r["今の波に足す・席なし"]
    assert r["元の向きを足す・席あり"] > r["元の向きを足す・席なし"]


def test_grow_adds_a_frequency_on_a_miss():
    import i49q_small_world_grow as Gr
    import i49q_small_world_stream as St
    talks = St.make(talks=40)
    words = sorted({w for _, ws in talks for w in ws})
    idx = {w: i for i, w in enumerate(words)}
    g = Gr.Grow(len(words), St.base_dense(words, 32), 64, "外れ")
    rate = St.stream(g, talks, idx)[0]
    assert 0 < g.added <= 40 * 14 and 0 < rate < 1
