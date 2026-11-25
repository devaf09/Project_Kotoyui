"""I49e の聞くだけの数え（tools/past/i49e_listen_count.py）の関門: 番の並べ方、作品の境、番の種類ごとの数え。"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import i49e_listen_count as LC  # noqa: E402


def _book(n):
    t = []
    for _ in range(n):
        t += [("本", "本", [("猫", "NOUN"), ("山", "NOUN")]), ("A1", "A", [("猫", "NOUN")]),
              ("B1", "B", [("猫", "NOUN"), ("犬", "NOUN")]), ("A2", "A", [("犬", "NOUN")])]
    return t


def test_listen_kinds_and_book_boundary():
    r = LC.listen([_book(3), _book(3)], None, True)
    k = r["番の種類ごと"]
    assert set(k) == {"本→A1", "A1→B1", "B1→A2", "A2→本"}
    assert k["A1→B1"]["拾われた割合"] == 1.0 and k["A2→本"]["拾われた割合"] == 0.0
    assert k["本→A1"]["語の数"] == 2 * 3 * 2                # 作品の境では直前の番を切るので、作品 2 の初めの本は数えない
    assert k["A2→本"]["語の数"] == 2 * 2


def test_teachers_only_skips_pages():
    r = LC.listen([_book(2)], None, False)
    assert set(r["番の種類ごと"]) == {"A1→B1", "B1→A2"}       # A2 → 次の A1 は同じ話し手なので数えない
