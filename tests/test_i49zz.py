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


def test_stop_by_own_output_brings_denied_point_to_zero():
    import i49zw_dense_pair as D
    import i49zx_teach_board as X
    cs = ["読む", "歌う"]
    far, same, tegami = ("学生", "歌", None), ("学生", "本", None), ("学生", "手紙", None)
    out = {}
    for stop in (1.0, "出した分"):
        b = X.TeachBoard(attr, amp="そのまま", rare=D.rare_of(attr, ZY.SEATS), stop=stop)
        b.teach(2, ("学生", "本", "読む"), "読む")
        assert b.deny(2, far, "読む")
        out[stop] = {q: b.points(2, q, cs)["読む"] for q in (far, same, tegami)}
        if stop == "出した分":
            assert abs(out[stop][far]) < 1e-3                          # 教えた問いの点が、ちょうど 0 になる
            assert not b.deny(2, far, "読む")                          # もう点が出ていないので、重ねて引かない
    assert out["出した分"][same] > out[1.0][same] and out["出した分"][tegami] > out[1.0][tegami]      # 引き過ぎない分、巻き込みが小さい


def test_teach_try_denies_after_teaching_so_own_output_is_seen():
    import i49zx_teach_try as T
    get = lambda w: attr(w[0])
    seats = [[(w, "NOUN") for w in ZY.SEATS[0]], [(w, "NOUN") for w in ZY.SEATS[1]], []]
    qs = [("遠い問い", 2, (("学生", "NOUN"), ("歌", "NOUN"), None)), ("教える", 2, (("学生", "NOUN"), ("本", "NOUN"), None))]

    class Body:
        def ask(self, si, tri, cands):
            return dict(段="本体", 点={("歌う", "VERB"): 1.0}, 答え=("歌う", "VERB"))

    b = T.board_of(get, "めずらしさ", seats, stop="出した分")
    rows = T.try_run(Body(), b, qs, {0: ["×読む"], 1: ["読む"]})        # 「違う」の行が先に書いてあっても、教えの後に入る
    assert rows[0]["教え"] == "板の 読む は違う" and rows[0]["板の答え"] is None and ("読む", "VERB") in b.stop[2]
    assert rows[1]["板の答え"] == "読む"


def test_far_stops_every_word_the_board_gives_there():
    import i49zw_dense_pair as D
    import i49zx_teach_board as X
    cs = ["読む", "書く", "歌う"]
    b = X.TeachBoard(attr, amp="そのまま", rare=D.rare_of(attr, ZY.SEATS))
    b.teach(2, ("学生", "本", "読む"), "読む")
    b.teach(2, ("父", "手紙", "書く"), "書く")
    far, same = ("学生", "歌", None), ("学生", "本", None)
    before = b.points(2, same, cs)["読む"]
    assert b.points(2, far, cs)["読む"] > .2 and b.points(2, far, cs)["書く"] > 0
    assert b.far(2, far) == 2 and not b.dark[2]
    s = b.points(2, far, cs)
    assert abs(s["読む"]) < 1e-3 and abs(s["書く"]) < 1e-3              # 教えた問いでは、どの語の点も 0 になる（次の語が出てこない）
    assert b.far(2, far) == 0                                          # もう何も出していないので、重ねて止めない
    assert .3 < b.points(2, same, cs)["読む"] < before                  # 教えた組は残る（弱くはなる）


def test_teach_try_far_line(tmp_path):
    import i49zx_teach_try as T
    get = lambda w: attr(w[0])
    seats = [[(w, "NOUN") for w in ZY.SEATS[0]], [(w, "NOUN") for w in ZY.SEATS[1]], []]
    qs = [("遠い問い", 2, (("学生", "NOUN"), ("歌", "NOUN"), None)), ("教える", 2, (("学生", "NOUN"), ("本", "NOUN"), None))]

    class Body:
        def ask(self, si, tri, cands):
            return dict(段="本体", 点={("歌う", "VERB"): 1.0}, 答え=("歌う", "VERB"))

    tc = {0: ["遠い"], 1: ["読む", "書く"]}
    assert ("遠い", "VERB") not in T.cands_of(qs, tc)[2]
    rows = T.try_run(Body(), T.board_of(get, "めずらしさ", seats), qs, tc)
    assert rows[0]["教え"] == "遠い（板の出番ではない。止めた語 2）" and rows[0]["板の答え"] is None and rows[1]["板の答え"] in ("読む", "書く")


def test_unheard_run_board_carries_what_body_never_heard():
    import i49zp_text_talks_run as ZP
    import i49zx_teach_board_run as R
    from test_i49zw import attr as small
    lines = ["太郎が金づちを振る", "花子が金づちを振る", "",
             "花子が金づちを振る", "太郎が金づちを振る", "花子が金づちを振る", "",
             "太郎がりんごを食べる", "花子がみかんをかじる", "太郎がみかんを食べる", "",
             "花子がりんごをかじる", "太郎がりんごを食べる", "花子がみかんをかじる", "太郎がみかんを食べる"]
    talks, _, _ = ZP.talks_of(lines)
    assert len(talks) == 4
    get = lambda w: small(w[0])
    res = R.run_unheard(get, get, talks, rates=(1.0,), ks=(0,), dim=1024)
    d = res["元の順"]
    base, s = d["本体だけ（後半を聞かない）"], d["介入 100%"]
    assert s["本体"] == base["本体"] and s["本体の答えが板なしと違った数"] == 0       # 本体は後半を聞かず、板を付けても変わらない
    assert d["本体が後半も聞く・板なし（比べ役）"]["問い"] == base["問い"] == s["問い"]
    assert s["教えた数"] > 0 and s["板が答えを持った問い"] > 0
    assert s["板が答えを持てばいつも板（比べ役）"] > base["本体"]                    # 本体が聞いていない事を、板が運ぶ
    assert s["板を付けた形"] >= base["本体"]


def test_ceiling_counts_pair_memory_limit():
    import i49zp_text_talks_run as ZP
    import i49zx_teach_board_run as R
    from test_i49zw import attr as small
    lines = ["太郎が金づちを振る", "花子が金づちを振る", "",
             "花子が金づちを振る", "太郎が金づちを振る", "花子が金づちを振る", "",
             "太郎がりんごを食べる", "花子がみかんをかじる", "太郎がみかんを食べる", "",
             "花子がりんごをかじる", "太郎がりんごを食べる", "花子がみかんをかじる", "太郎がみかんを食べる"]
    talks, _, _ = ZP.talks_of(lines)
    get = lambda w: small(w[0])
    d = R.run_ceiling(get, get, talks, ks=(0,))["元の順"]
    a, b, c = d["同じ組が後半でもう出た問い"], d["同じ組がまだ出ていない問い"], d["全部"]
    assert a["問い"] + b["問い"] == c["問い"] and a["問い"] > 0
    assert b["正解がその組で前に出た語のどれか"] == 0 and b["一番多く出た語で答えた時の当たり"] == 0
    assert a["一番多く出た語で答えた時の当たり"] <= a["正解がその組で前に出た語のどれか"] <= 1
    assert a["その組で前に出た語の種類の平均"] >= 1
