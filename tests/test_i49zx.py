"""I49zx の関門: 介入の板（別に持つ・外せば元に戻る・暗い側・薄れるだけ）。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from test_i49zw import attr  # noqa: E402


class Fixed:
    """いつも同じ語を答える本体（板との違いを見るための物）。"""

    def __init__(self, word):
        self.word, self.heard = word, 0

    def ask(self, si, tri, cands):
        return dict(段="本体", 点={self.word: 1.0}, 答え=self.word)

    def learn(self, tri):
        self.heard += 1

    def tick(self):
        pass


def test_no_teaching_means_same_as_body():
    import i49zx_teach_board as X
    t = X.Taught(Fixed("振る"), X.TeachBoard(attr, dim=1024))
    o = t.ask(2, ("太郎", "りんご", None), ["食べる", "振る"])
    assert o == dict(段="本体", 点={"振る": 1.0}, 答え="振る") and t.last[0] is False
    t.learn(("太郎", "りんご", "食べる"))
    assert t.tally == [0, 0, 0] and t.body.heard == 1


def test_board_fixes_taught_pair_and_spreads_to_similar_pair():
    import i49zx_teach_board as X
    b = X.TeachBoard(attr, dim=1024)
    assert b.teach(2, ("太郎", "りんご", "食べる"), "振る")
    assert b.taught(2, ("太郎", "りんご", None)) and not b.taught(2, ("花子", "みかん", None))
    s = b.points(2, ("太郎", "りんご", None), ["食べる", "振る"])
    assert s["食べる"] > .9 and s["振る"] < 0                        # 明るい側に正しい語、暗い側に違った答え
    s = b.points(2, ("花子", "みかん", None), ["食べる", "振る"])
    assert s["食べる"] > .5                                          # 似た組にも効く
    s = b.points(2, ("太郎", "金づち", None), ["食べる", "振る"])
    assert abs(s["食べる"]) < .2                                     # 片方だけ同じ組には、ほとんど効かない
    assert abs(s["振る"]) <= b.dark_w + 1e-6                         # 暗い側は弱い


def test_board_is_used_only_after_it_has_won_more():
    import i49zx_teach_board as X
    t = X.Taught(Fixed("振る"), X.TeachBoard(attr, dim=1024))
    t.teach(2, ("太郎", "りんご", "食べる"), "振る")
    q = ("花子", "みかん", None)
    o = t.ask(2, q, ["食べる", "振る"])
    assert o["答え"] == "振る" and t.last == (True, False, "振る", "食べる")   # まだ数えが無いので、本体のまま
    t.learn(("花子", "みかん", "食べる"))
    assert t.tally == [0, 1, 1]
    assert t.ask(2, q, ["食べる", "振る"])["答え"] == "食べる"                # 板の方が当たってきたので、板を使う


def test_board_fades_but_never_deletes():
    import i49zx_teach_board as X
    b = X.TeachBoard(attr, dim=1024, half=1)
    b.teach(2, ("太郎", "りんご", "食べる"), "食べる")
    first = float(np.linalg.norm(b.bright[2]["食べる"]))
    b.teach(2, ("太郎", "金づち", "振る"), "振る")
    assert abs(float(np.linalg.norm(b.bright[2]["食べる"])) - first / 2) < 1e-4 and "食べる" in b.bright[2]
    assert not b.dark[2]                                              # 合っていた答えは、暗い側に入れない


def test_run_shapes_and_body_unchanged():
    import i49zp_text_talks_run as ZP
    import i49zx_teach_board_run as R
    lines = ["太郎がりんごを食べる", "花子がみかんをかじる", "太郎が金づちを振る", "花子がりんごを食べる", "太郎がみかんを食べる", "",
             "花子が金づちを振る", "太郎がりんごをかじる", "花子がみかんを食べる", "太郎が金づちを振る"]
    talks, _, _ = ZP.talks_of(lines)
    get = lambda w: attr(w[0])
    res = R.run(get, get, talks, rates=(0.5, 1.0), ks=(0,), dim=1024)
    d = res["元の順"]
    assert d["介入 100%"]["教えた数"] >= d["介入 50%"]["教えた数"] > 0
    for name in ("介入 50%", "介入 100%"):
        s = d[name]
        assert s["本体"] == d["板なし"]["本体"] and s["本体の答えが板なしと違った数"] == 0     # 板を付けても本体は変わらない（外せば元に戻る）
        assert 0 <= s["板を付けた形"] <= 1
        assert s["教えた組そのもの"]["問い"] + s["似た組"]["問い"] <= s["問い"]


def test_teach_try_before_and_after(tmp_path):
    import i49zx_teach_board as X
    import i49zx_teach_try as T
    ask = tmp_path / "ask.tsv"
    ask.write_text("教える\t太郎\tりんご\t？\n似た問い\t花子\tみかん\t？\n関係ない\t太郎\t金づち\t？\n読めない行\n", encoding="utf-8")
    teach = tmp_path / "teach.tsv"
    teach.write_text("1\t食べる\n", encoding="utf-8")
    qs, tc = T.read_ask(str(ask)), T.read_teach(str(teach))
    assert len(qs) == 3 and qs[0][1] == 2 and tc == {0: ["食べる"]}
    get = lambda w: attr(w[0])
    body = Fixed(("振る", "VERB"))
    rows = T.try_run(body, X.TeachBoard(get, dim=1024), qs, {})
    assert [r["一回目"] for r in rows] == ["振る"] * 3 and "板の答え" not in rows[0]
    rows = T.try_run(body, X.TeachBoard(get, dim=1024), qs, tc)
    assert rows[0]["教え"].startswith("違う") and rows[0]["板の答え"] == "食べる" and rows[0]["教えた組そのもの"]
    assert rows[1]["板の答え"] == "食べる" and not rows[1]["教えた組そのもの"]      # 似た問いへ広がる
    assert all(r["外すと一回目と同じ"] for r in rows)                                # 板を外せば元の答え
    assert "教える前の答え" in T.to_md(rows, True) and "上位 3" in T.to_md(rows, False)
    teach.write_text("3\t振る\n3\tかじる\n", encoding="utf-8")          # 出した答えは合っていて、別の語も合っている
    b2 = X.TeachBoard(get, dim=1024)
    rows = T.try_run(body, b2, qs, T.read_teach(str(teach)))
    assert rows[2]["教え"] == "合っている。かじる も合っている" and not b2.dark[2]      # 合っていた答えは暗い側に入れない
    assert set(b2.bright[2]) == {("振る", "VERB"), ("かじる", "VERB")}
    ask.write_text("候補\t花子\tみかん、金づち\t食べる、かじる\n教える\t太郎\tりんご\t？\n", encoding="utf-8")
    ex = T.read_extra(str(ask))
    assert ("かじる", "VERB") in ex[2] and ("金づち", "NOUN") in ex[1] and len(T.read_ask(str(ask))) == 1
    assert ("かじる", "VERB") in T.cands_of(T.read_ask(str(ask)), {}, ex)[2]
