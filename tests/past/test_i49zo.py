"""I49zo の関門: 全部の問いと答えをテキストに出す事、違和感の点のまとめ。"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from test_i49n import NAMES, V  # noqa: E402


def _rows(tmp_path):
    import i49zo_dump_answers as Z
    W = set(NAMES)
    talks = [([("友達", "魚", "釣る"), ("先生", "魚", "釣る"), ("友達", "海", "泳ぐ"), ("友達", "魚", "釣る")], W),
             ([("先生", "ゲーム", "勉強"), ("友達", "魚", "釣る")], W)]
    rows = Z.dump(V.get, V.get, talks, [("友達", "海", "泳ぐ")])
    p = str(tmp_path / "a.tsv")
    Z.write_rows(p, rows)
    return Z, rows, Z.read_rows(p)


def test_dump_and_read(tmp_path):
    Z, rows, back = _rows(tmp_path)
    assert len(rows) == 2 * 18 and len(back) == len(rows)
    for r in back:
        assert [r[s] for s in Z.SEATS].count("？") == 1 and r["席"] in Z.SEATS
        assert Z.filled(r, r["正解"]).count("？") == 0
    assert {r["始め方"] for r in back} == {"白紙から", "青空を先に"}


def test_look_counts(tmp_path):
    Z, _, back = _rows(tmp_path)
    for i, r in enumerate(back):                                   # 答えの違う問いを作る
        r["属性だけ"], r["属性＋固有の位相"] = ("魚", r["正解"]) if i % 2 else ("魚", "魚")
    calls = []

    def score(tri):
        calls.append(tri)
        return 2 if "魚" in tri and tri.count("魚") > 1 else 4

    st, done = Z.look(back, score, n=5)
    assert st["取った問い"] == min(5, st["答えが違った問い"]) == st["見終わった問い"]
    assert len(calls) == len(set(calls))                           # 同じ組は 1 回だけ
    assert st["新しい方が当たり"]["問い"] == st["見終わった問い"]
    p = str(tmp_path / "t.txt")
    Z.write_text(p, done, "属性だけ", "属性＋固有の位相")
    assert len(open(p, encoding="utf-8").read().splitlines()) == len(done) + 1


def test_time_cap(tmp_path):
    Z, _, back = _rows(tmp_path)
    for r in back:
        r["属性だけ"], r["属性＋固有の位相"] = "魚", "海"
    st, done = Z.look(back, lambda tri: 3, minutes=0)
    assert st["見終わった問い"] == 0 and done == []


def test_tuple_words(tmp_path):
    """語が組（見出し語と品詞）でも、テキストに出せる。"""
    import i49zo_dump_answers as Z
    T = lambda w: (w, "名詞")
    get = lambda w: V.get(w[0])
    talks = [([(T("友達"), T("魚"), T("釣る")), (T("先生"), T("魚"), T("釣る"))], {T(n) for n in NAMES})]
    rows = Z.dump(get, get, talks, [])
    p = str(tmp_path / "a.tsv")
    Z.write_rows(p, rows)
    back = Z.read_rows(p)
    assert back[0]["正解"].endswith("／名詞") and Z.plain(back[0]["正解"]) in NAMES
