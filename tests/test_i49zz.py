"""I49zz の関門: 板が広げて出した答えへの「違う」を、止める側に持つ。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import test_i49zy as ZY  # noqa: E402


def attr(w):
    """I49zy の小さな世界に、読み物と歌だけが持つ属性（「著作」）を足す。実物で 本 と 歌 が近かった事の写し。"""
    v = ZY.attr(w)
    if v is not None and (w in ZY.KINDS["読み物"] or w in ZY.KINDS["歌"]):
        v = v.copy()
        v[170:180] = 4
    return v


def _board():
    import i49zw_dense_pair as D
    import i49zx_teach_board as X
    b = X.TeachBoard(attr, amp="そのまま", rare=D.rare_of(attr, ZY.SEATS))
    b.teach(2, ("学生", "本", "読む"), "読む")
    return b


def test_stop_side_stops_far_question_and_keeps_taught_pair():
    b, cs = _board(), ["読む", "歌う"]
    far, same, near = ("学生", "歌", None), ("学生", "本", None), ("父", "本", None)
    before = {q: b.points(2, q, cs)["読む"] for q in (far, same, near)}
    assert before[far] > .2                                           # 直す前: 属性が近いので、遠い問いにも 読む を出す
    assert b.deny(2, far, "読む") and not b.dark[2] and "読む" in b.stop[2]
    after = {q: b.points(2, q, cs)["読む"] for q in (far, same, near)}
    assert after[far] < 0                                             # 止まる
    assert after[same] > .3 and after[near] > 0                       # 教えた組と、その似た組は残る（弱くはなる）
    assert after[same] < before[same]


def test_stop_side_also_reaches_similar_words():
    b, cs = _board(), ["読む", "歌う"]
    q = ("学生", "手紙", None)
    before = b.points(2, q, cs)["読む"]
    b.deny(2, ("学生", "歌", None), "読む")
    assert before > .2 and b.points(2, q, cs)["読む"] < before        # 止める印も近さで広がる（巻き込みが有る事を、そのまま書く）
    b.teach(2, ("学生", "手紙", "読む"), "読む")
    assert b.points(2, q, cs)["読む"] > .5                             # その組そのものを教えれば、戻る


def test_stop_fades_like_the_rest():
    import i49zx_teach_board as X
    b = X.TeachBoard(attr, half=1)
    b.deny(2, ("学生", "歌", None), "読む")
    first = float(np.linalg.norm(b.stop[2]["読む"]))
    b.teach(2, ("学生", "本", "読む"), "読む")
    assert abs(float(np.linalg.norm(b.stop[2]["読む"])) - first / 2) < 1e-4


def test_teach_try_deny_line(tmp_path):
    import i49zx_teach_try as T
    get = lambda w: attr(w[0])
    seats = [[(w, "NOUN") for w in ZY.SEATS[0]], [(w, "NOUN") for w in ZY.SEATS[1]], []]
    qs = [("教える", 2, (("学生", "NOUN"), ("本", "NOUN"), None)), ("遠い問い", 2, (("学生", "NOUN"), ("歌", "NOUN"), None))]
    teach = tmp_path / "t.tsv"
    teach.write_text("1\t読む\n2\t×読む\n", encoding="utf-8")
    tc = T.read_teach(str(teach))
    assert tc == {0: ["読む"], 1: ["×読む"]} and ("読む", "VERB") in T.cands_of(qs, tc)[2] and ("×読む", "VERB") not in T.cands_of(qs, tc)[2]

    class Body:
        def ask(self, si, tri, cands):
            return dict(段="本体", 点={("歌う", "VERB"): 1.0}, 答え=("歌う", "VERB"))

    rows = T.try_run(Body(), T.board_of(get, "めずらしさ", seats), qs, tc)
    assert rows[0]["板の答え"] == "読む" and rows[1]["板の答え"] is None and rows[1]["教え"] == "板の 読む は違う"
    rows = T.try_run(Body(), T.board_of(get, "めずらしさ", seats), qs, {0: ["読む"]})
    assert rows[1]["板の答え"] == "読む"                                # 「違う」を教えなければ、遠い問いに 読む のまま
