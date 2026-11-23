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
