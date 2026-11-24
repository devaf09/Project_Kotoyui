"""I49zp の関門: 「A が B を C する」のテキストを、切り出しを通さずに話にする。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from test_i49n import NAMES, V  # noqa: E402


def test_cut():
    import i49zp_text_talks_run as Z
    assert Z.cut("友達が魚を釣る") == (("友達", "魚", "釣る"), None)
    assert Z.cut("友達が海に行く")[1] == "「を」が無い"
    assert Z.cut("友達が箱に魚を入れる")[1] == "ほかの助詞が入る"
    assert Z.cut("友達が魚を箱に入れる")[1] == "ほかの助詞が入る"
    assert Z.cut("友達が犬を撫でる") == (("友達", "犬", "撫でる"), None)
    assert Z.cut("友達が先生の魚を釣る")[1] == "「の」でつないだ語"
    assert Z.cut("友達が勉強をする")[1] == "〜をする"
    assert Z.cut("雨、止む")[1] == "「が」が無い"


def test_scenes_and_talks():
    import i49zp_text_talks_run as Z
    a = ["友達が魚を釣る", "友達が魚を見る", "先生が魚を釣る", "友達が海を見る", "先生が海を見る", "友達が魚を運ぶ"]
    b = ["犬がゲームを見る", "犬がゲームを運ぶ", "犬が魚を見る", "犬が海を見る", "犬がゲームを釣る", "犬が魚を運ぶ"]
    sc = Z.scenes(a + b)
    assert sc == [list(range(6)), list(range(6, 12))]                # 出る人が替わる所で切れる
    assert Z.scenes(a[:3] + [""] + a[3:]) == [[0, 1, 2], [4, 5, 6]]     # 空行は必ず切る
    talks, why, ex = Z.talks_of(a + ["友達が海に行く", "友達が魚を釣る"] + b)
    assert why["「を」が無い"] == 1 and why["同じ話の中の同じ行"] == 1
    assert all(len(t[0]) >= 2 for t in talks) and talks[0][0][0] == (("友達", "NOUN"), ("魚", "NOUN"), ("釣る", "VERB"))


def test_order_has_no_random():
    import i49zp_text_talks_run as Z
    assert Z.order(10, 0) == list(range(10))
    p = Z.order(10, 1)
    assert sorted(p) == list(range(10)) and p != list(range(10)) and p == Z.order(10, 1) and p != Z.order(10, 2)


def test_run_shapes():
    import i49zp_text_talks_run as Z
    lines = ["友達が魚を釣る", "先生が魚を釣る", "友達が海を泳ぐ", "友達が魚を釣る", "先生がゲームを勉強", "友達が魚を見る", "",
             "先生が海を泳ぐ", "友達がゲームを釣る", "先生が魚を釣る"]
    talks, _, _ = Z.talks_of(lines)
    get = lambda w: V.get(w[0])
    res = Z.run(get, get, talks, ks=(0, 1))
    assert list(res) == ["元の順", "並べ替え 1"]
    for d in res.values():
        for name, *_ in Z.VERS:
            assert 0 <= d[name]["正しい物"] <= 1 and 0 <= d[name]["後ろ 2 割"] <= 1
    c = Z.cover(talks, get, get)
    assert c["語の種類"] >= 4 and 0 <= c["属性が無い語を含む行の割合"] <= 1
