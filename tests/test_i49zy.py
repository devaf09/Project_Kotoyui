"""I49zy の関門: 組の印を作る前に、席ごとの「いつもの属性」を引く。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))

# どの語も持つ属性（0〜19）が多い小さな世界。種類の属性は 20 から 10 個ずつ、語だけの属性は 100 から 4 個ずつ
KINDS = {"人": ("学生", "父", "母", "先生"), "動物": ("犬", "猫", "鳥", "馬"), "読み物": ("本", "雑誌", "新聞", "手紙"), "歌": ("歌", "曲", "童謡", "校歌"),
         "道具": ("金づち", "のこぎり", "はさみ", "ほうき")}
WORDS = [w for ws in KINDS.values() for w in ws]


def attr(w):
    if w not in WORDS:
        return None
    k = next(i for i, ws in enumerate(KINDS.values()) if w in ws)
    v = np.zeros(180, dtype=np.complex64)
    v[:20] = 1
    v[20 + 10 * k:30 + 10 * k] = 1
    i = WORDS.index(w)
    v[100 + 4 * i:104 + 4 * i] = 1
    return v


SEATS = [KINDS["人"] + KINDS["動物"], KINDS["読み物"] + KINDS["歌"] + KINDS["道具"], []]


def _near(p):
    cos = lambda x, y: float(np.real(np.vdot(x, y)))
    a = p.sign(2, ("学生", "本", None))
    return {k: cos(a, p.sign(2, t + (None,))) for k, t in dict(同じ=("学生", "本"), 片方が似た語=("学生", "雑誌"), 両方が似た語=("父", "雑誌"),
                                                             片方だけ同じ=("学生", "歌"), 両方違う=("父", "曲")).items()}


def test_usual_is_mean_of_unit_waves_per_seat():
    import i49zw_dense_pair as D
    u = D.usual_of(attr, [KINDS["人"], ["本", "本", "歌", "無い語"], []])
    assert u[2] is None and abs(float(np.linalg.norm(u[0])) - 1) < .2
    x, y = attr("本"), attr("歌")
    assert np.allclose(u[1], (x / np.linalg.norm(x) + y / np.linalg.norm(y)) / 2, atol=1e-6)     # 同じ語は 1 回、属性の無い語は数えない


def test_rare_is_one_minus_share_of_words_having_the_attribute():
    import i49zw_dense_pair as D
    r = D.rare_of(attr, SEATS + [["無い語"]])
    assert r[2] is None and r[3] is None
    assert np.allclose(r[0][:20], 0) and np.allclose(r[0][20:40], .5) and abs(float(r[0][100]) - 7 / 8) < 1e-6   # みんなが持つ属性は 0
    assert np.allclose(r[1][20:40], 1)                                                                   # その席のどの語も持たない属性は 1（使われない）


def test_old_sign_does_not_tell_similar_pair_from_one_shared_word():
    import i49zw_dense_pair as D
    n = _near(D.PairSolve(attr, amp="位相だけ"))
    assert n["片方だけ同じ"] > .05 and n["両方が似た語"] - n["片方だけ同じ"] < .1     # 今までの形: 1 語が同じだけの組にも、正の近さが残る
    assert _near(D.PairSolve(attr))["片方だけ同じ"] > .5                              # 大きさをそろえないと、どの組もほとんど同じ印


def test_with_usual_sign_follows_both_words():
    import i49zw_dense_pair as D
    n = _near(D.PairSolve(attr, usual=D.usual_of(attr, SEATS)))
    assert abs(n["同じ"] - 1) < 1e-4
    assert n["片方が似た語"] > .4 and n["両方が似た語"] > .15
    assert n["片方だけ同じ"] < 0 and n["両方違う"] < 0                               # 違う種類の語は、負の近さになる


def test_with_rare_sign_follows_both_words_and_stays_sparse():
    import i49zw_dense_pair as D
    p = D.PairSolve(attr, rare=D.rare_of(attr, SEATS))
    n = _near(p)
    assert abs(n["同じ"] - 1) < 1e-4
    assert n["片方が似た語"] > .4 and n["両方が似た語"] > .15
    assert abs(n["片方だけ同じ"]) < .02 and abs(n["両方違う"]) < .02                 # 1 語が同じだけの組には、引かれない


def test_board_with_fix_does_not_answer_far_question():
    import i49zw_dense_pair as D
    import i49zx_teach_board as X
    boards = dict(今まで=X.TeachBoard(attr), 引く=X.TeachBoard(attr, amp="そのまま", usual=D.usual_of(attr, SEATS)),
                  めずらしさ=X.TeachBoard(attr, amp="そのまま", rare=D.rare_of(attr, SEATS)))
    cs = ["読む", "歌う"]
    for name, b in boards.items():
        b.teach(2, ("学生", "本", "読む"), "読む")
        far = b.points(2, ("学生", "歌", None), cs)["読む"]
        assert far > .05 if name == "今まで" else far < .02, name                   # 今までの形は、遠い問いにも「読む」を出す
        assert b.points(2, ("学生", "本", None), cs)["読む"] > .99, name
        if name != "今まで":
            assert b.points(2, ("学生", "雑誌", None), cs)["読む"] > .4 and b.points(2, ("父", "雑誌", None), cs)["読む"] > .15, name


def test_teach_try_key_forms(tmp_path):
    import i49zx_teach_try as T
    get = lambda w: attr(w[0])
    seats = [[(w, "NOUN") for w in SEATS[0]], [(w, "NOUN") for w in SEATS[1]], [("読む", "VERB")]]
    qs = [("教える", 2, (("学生", "NOUN"), ("本", "NOUN"), None)), ("似た問い", 2, (("父", "NOUN"), ("雑誌", "NOUN"), None)),
          ("遠い問い", 2, (("学生", "NOUN"), ("歌", "NOUN"), None))]

    class Body:
        def ask(self, si, tri, cands):
            return dict(段="本体", 点={("歌う", "VERB"): 1.0}, 答え=("歌う", "VERB"))

    out = {}
    for key in ("今まで", "引く", "めずらしさ"):
        rows = T.try_run(Body(), T.board_of(get, key, seats), qs, {0: ["読む"]})
        out[key] = rows
        assert rows[0]["板の答え"] == "読む" and abs(rows[0]["教えた組との一番の近さ"] - 1) < 1e-3
        assert "教えた組との一番の近さ" in T.to_md(rows, True)
    assert out["今まで"][2]["板の答え"] == "読む"                                     # 今までの形は、遠い問いにも答えてしまう
    for key in ("引く", "めずらしさ"):
        assert out[key][1]["板の答え"] == "読む" and out[key][2]["教えた組との一番の近さ"] < .02      # 似た問いには広がり、遠い問いには近くない
        assert out[key][1]["教えた組との一番の近さ"] > out[key][2]["教えた組との一番の近さ"] + .1
    assert out["引く"][2]["板の答え"] is None                                         # 引く形は、遠い問いが負の近さになり、板は答えない
