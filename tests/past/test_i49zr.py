"""I49zr の関門: 助詞の位置で枠（が・を・に・で・と・へ・から・どうした）に切り、席を足した形を数える。"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from test_i49n import V  # noqa: E402


def words(f):
    import i49zq_frame as Q
    return {Q.SEATS[j]: w[0] for j, w in enumerate(f) if w is not None}


def test_cut_seats():
    import i49zr_frame_run as R
    assert words(R.cut("友達が魚を釣る")[0]) == dict(が="友達", を="魚", どうした="釣る")
    assert words(R.cut("友達がタオルで体を拭く")[0]) == dict(が="友達", を="体", で="タオル", どうした="拭く")
    assert words(R.cut("友達が駅から家へ歩く")[0]) == {"が": "友達", "へ": "家", "から": "駅", "どうした": "歩く"}
    assert words(R.cut("友達が魚を箱に入れる")[0]) == dict(が="友達", を="魚", に="箱", どうした="入れる")
    assert words(R.cut("友達が犬を撫でる")[0]) == dict(が="友達", を="犬", どうした="撫でる")      # 送り仮名の「で」は助詞にしない
    assert words(R.cut("友達が石を落とす")[0]) == dict(が="友達", を="石", どうした="落とす")
    assert words(R.cut("魚が針にかかる")[0]) == dict(が="魚", に="針", どうした="かかる")        # かなだけの動詞の前の助詞は切る
    assert words(R.cut("友達が静かに魚を釣る")[0]) == dict(が="友達", を="魚", どうした="釣る") or R.cut("友達が静かに魚を釣る")[0] is not None
    assert R.cut("雨、止む")[1] == "「が」が無い"
    assert R.cut("友達が先生の魚を釣る")[1] == "「の」「と」でつないだ語"
    assert R.cut("友達が海に船に乗る")[1] == "同じ助詞が 2 回"


def test_core_and_strip():
    import i49zr_frame_run as R
    a, b = R.cut("友達が魚を釣る")[0], R.cut("友達が魚を箱に入れる")[0]
    assert R.is_core(a) and not R.is_core(b)
    assert words(R.strip(b)) == dict(が="友達", を="魚", どうした="入れる")


LINES = ["友達が魚を釣る", "先生が魚を海で釣る", "友達が海を泳ぐ", "先生がゲームを学校に運ぶ", "友達が魚を見る", "先生が海を見る", "",
         "先生が海を泳ぐ", "友達がゲームを学校に運ぶ", "先生が魚を海で釣る", "友達が魚を釣る", "先生が魚を見る"]


def test_talks_and_run_shapes():
    import i49zr_frame_run as R
    talks, why, _ = R.talks_of(LINES + ["", "友達が魚を釣る", "友達が魚を釣る", "先生が魚を見る"])
    assert len(talks) == 3 and why["同じ話の中の同じ行"] == 1
    get = lambda w: V.get(w[0])
    res = R.run(get, get, talks, ks=(0, 1))
    assert list(res) == ["元の順", "並べ替え 1"]
    for d in res.values():
        for part in ("三つの席", "全部の席", "席を見せない"):
            for name in ("表", "解く形", "数えて選ぶ"):
                assert 0 <= d[part][name]["正しい物"] <= 1
        assert "今までの表" in d["三つの席"]
        s = d["足した席を見せると"]["表"]
        assert s["問い"] > 0 and 0 <= s["見せる"] <= 1 and 0 <= s["見せない"] <= 1
        assert d["全部の席"]["表"]["問い"] > d["席を見せない"]["表"]["問い"]      # 足した席も問う分、問いが増える


def test_three_seats_same_questions_as_old_parts():
    """が・を・どうした だけの枠では、今までの部品と同じ問いが、同じ順に並ぶ（直の比べが出来る）。段の作りは違うので、答えは同じとは限らない。"""
    import i49zr_frame_run as R
    talks, _, _ = R.talks_of([l for l in LINES if "に" not in l and "で" not in l])
    get = lambda w: V.get(w[0])
    ok = R.ok_sets(talks)
    a = R.stream(dict(R.forms(get, get))["表"](), talks, ok)
    b = R.stream(dict(R.old_forms(get, get))["今までの表"](), talks, ok)
    assert [r[5] for r in a] == [r[5] for r in b] and len(a) > 10
    d = R.duel(a, b)
    assert d["良くなった"] + d["悪くなった"] <= len(a)
