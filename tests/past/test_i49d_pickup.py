"""I49d の選ぶ口の重さの見本（tools/past/i49d_pickup.py）の関門: 拾われた数え、少ない語の戻し、AUC。"""
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import i49d_pickup as P  # noqa: E402


def _line(*ws):
    return dict(text="", toks=[dict(lemma=l, pos=p, text=l) for l, p in ws])


def test_table_counts_pickup_and_backs_off(tmp_path):
    ask = _line(("猫", "NOUN"), ("見る", "VERB"), ("猫", "NOUN"))      # 同じ発言の同じ語は 1 回
    ans = _line(("猫", "NOUN"), ("可愛い", "ADJ"))
    data = [dict(name=f"c{i}", lines=[ask, ans] * 10) for i in range(20)]
    p = tmp_path / "rpc.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tb = P.PickupTable.build(str(p))
    assert tb.value("猫", "NOUN") > 0.9                   # 返事にいつも出る
    assert tb.value("見る", "VERB") < 0.1                  # 返事に出ない
    assert tb.value("犬", "NOUN") == tb.value("無い語", "NOUN")  # 少ない語は品詞の割合
    assert tb.value("猫", None) is None
    tb.save(str(tmp_path / "t.json"))
    assert P.PickupTable.load(str(tmp_path / "t.json")).value("猫", "NOUN") == tb.value("猫", "NOUN")
    res = P.check(str(p), tb)
    assert res["試しの語の数"] > 0


def test_auc_ranks_and_ties():
    assert P.auc([0.1, 0.9], [False, True]) == 1.0
    assert P.auc([0.5, 0.5], [False, True]) == 0.5
    assert P.auc([1, 2], [True, True]) is None
