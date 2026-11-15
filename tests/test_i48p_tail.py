"""I48p（終助詞の板と --tail、先生の文の分け方）の小さな確かめ（numpy だけで動く）。"""
import os
import sys
import types
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "intervention"))
import i48p_tail_board as P  # noqa: E402
import i48p_teacher_cats as T  # noqa: E402


def test_core_of():
    assert P.core_of("よ", "よ") == "∅" and P.core_of("ます+ね", "ね") == "ます" and P.core_of("∅", "") == "∅" and P.core_of("た", "") == "た"


def tok(i, w, pos, lemma=None):
    return dict(i=i, text=w, lemma=lemma or w, pos=pos, dep="dep", head=0)


def line(text, *tw):
    return dict(text=text, toks=[tok(i, *x) for i, x in enumerate(tw)])


def toy():
    L = []
    for k in range(30):
        L.append(line("「行くよ。」", ("行く", "VERB"), ("よ", "PART"), ("。", "PUNCT")))
        L.append(line("「行く。」", ("行く", "VERB"), ("。", "PUNCT")))
        L.append(line("「いいね。」", ("いい", "ADJ"), ("ね", "PART"), ("。", "PUNCT")))
        L.append(line("「行く？」", ("行く", "VERB"), ("？", "PUNCT")))                              # 問い: 数えない
        L.append(line("「行ったり。」", ("行く", "VERB", "行っ"), ("たり", "PART"), ("。", "PUNCT")))   # たり で終わる: 数えない
        L.append(line("「行ったりする。」", ("行く", "VERB", "行っ"), ("たり", "PART"), ("する", "VERB"), ("。", "PUNCT")))
        L.append(line("「行きます。」", ("行く", "VERB", "行き"), ("ます", "AUX"), ("。", "PUNCT")))   # 丁寧: 数えない
    return [dict(name="w", lines=L)]


def test_board_rate_dist_and_finals(tmp_path):
    W = toy()
    fin = P.final_parts(W)
    assert "よ" in fin and "ね" in fin and "たり" not in fin
    tb = P.TailBoard(min_count=5).fit([r for w in W for r in P.rows(w)], fin)
    r, n = tb.rate("行く", "VERB")
    assert n == 60 and abs(r - 0.5) < 1e-9                        # 行くよ・行く（問い・丁寧・たり で終わる行は数えない。行ったりする は する の行）
    assert tb.dist("行く", "VERB").most_common(1)[0][0] == "よ" and tb.dist("いい", "ADJ").most_common(1)[0][0] == "ね"
    p = str(tmp_path / "tb.json")
    tb.save(p)
    tb2 = P.TailBoard.load(p)
    assert tb2.rate("行く", "VERB") == tb.rate("行く", "VERB") and tb2.cands == tb.cands and tb2.finals == tb.finals


def test_dim_like_uke():
    q = P.dim(Counter(よ=3, ね=1), Counter(よ=4))                 # よ を使い切っている: 見込み .75 ÷ 使った割合 1.0 を掛ける
    assert abs(q["よ"] - 0.75 * 0.75) < 1e-9 and abs(q["ね"] - 0.25) < 1e-9


def test_bot_tail_pick():
    import chat_bot3 as C3
    W = toy()
    tb = P.TailBoard(min_count=5).fit([r for w in W for r in P.rows(w)], P.final_parts(W))
    inf = {"行く": ("∅", ""), "行くよ": ("よ", "よ"), "行くね": ("ね", "ね"), "行くの": ("の", "の"), "行きます": ("ます", "")}
    me = types.SimpleNamespace(tail=tb, _end_info=inf, lpos={"行く": "VERB"}, tail_log=Counter(), _tail_note=[], _tail_used=Counter(),
                               _tail_u=(0.0, 0.0))
    out = [(-1.0, "行く"), (-2.0, "行くよ"), (-3.0, "行くの")]
    assert C3.Bot3._tail_pick(me, "行く", out) == [(-2.0, "行くよ")]   # 付ける数（0.0 < 割合）、よ を選ぶ
    me._tail_u = (0.99, 0.0)
    assert C3.Bot3._tail_pick(me, "行く", out) == out                 # 割合より大きい数: 付けない
    me._tail_u = (0.0, 0.0)
    assert C3.Bot3._tail_pick(me, "行く", [(-1.0, "行くの"), (-2.0, "行く")]) == [(-1.0, "行くの"), (-2.0, "行く")]   # 問いの の は今のまま
    assert C3.Bot3._tail_pick(me, "行く", [(-1.0, "行きます")]) == [(-1.0, "行きます")]                              # 丁寧は触らない
    assert C3.Bot3._tail_pick(me, "行く", [(-1.0, "行く")]) == [(-1.0, "行くよ")] and me.tail_log["付けた（つないだ）"] == 1   # 本に無い: 原形をつなぐ
    assert me.tail_log["付けた（本の表記）"] == 1


def test_teacher_cats_drop_narration():
    t = "相手は海が好きだと言っているから、もう少し気持ちを込めて返そう。"
    assert T.END not in T.cats(t) and "ている" in t                  # 先生の地の文の「ている」は語尾・形にしない
    assert T.END in T.cats("「行く。」だけだと素っ気ないよ。") and T.curt("「行く。」だけだと素っ気ないよ。")
    assert not T.curt("「素っ気ない」って言わずに、優しく返そう。")    # 言い換えの例の中だけ
