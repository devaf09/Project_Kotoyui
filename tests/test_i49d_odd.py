"""I49d の違和感の口（tools/i49d_odd.py）の関門のうち、型を学ばずに確かめられる物（展開②の表、文の切り方、順位の割合）。"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import i49d_odd as O  # noqa: E402


def test_split_sents_drops_punct_only():
    toks = [("雨", "NOUN", "雨"), ("だ", "AUX", "だ"), ("。", "PUNCT", "。"), ("。", "PUNCT", "。"), ("ね", "PART", "ね")]
    S = O.split_sents(toks)
    assert len(S) == 2 and S[0][-1][2] == "。" and S[1][0][0] == "ね"


def test_continue_table_counts_and_backs_off(tmp_path):
    lines = [dict(toks=[dict(lemma="そう", pos="ADV", text="そう"), dict(lemma="だ", pos="AUX", text="だ"), dict(lemma="。", pos="PUNCT", text="。"),
                        dict(lemma="行く", pos="VERB", text="行く"), dict(lemma="？", pos="PUNCT", text="？")])] * 30
    data = [dict(name=f"c{i}", lines=lines) for i in range(20)]
    p = tmp_path / "rpc.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tb = O.ContinueTable.build(str(p))
    keep = tb.value([("だ", "AUX", "だ"), ("。", "PUNCT", "。")])
    stop = tb.value([("行く", "VERB", "行く"), ("？", "PUNCT", "？")])
    assert keep > 0.9 and stop < 0.1                      # 「だ。」の後は続け、「行く？」の後は渡す
    other = tb.value([("食べる", "VERB", "食べる"), ("？", "PUNCT", "？")])
    assert other < 0.1                                    # 知らない語は（品詞, 記号）に戻す
    tb.save(str(tmp_path / "t.json"))
    tb2 = O.ContinueTable.load(str(tmp_path / "t.json"))
    assert tb2.value([("だ", "AUX", "だ"), ("。", "PUNCT", "。")]) == keep


def test_pct_uses_the_reference_order():
    sc = O.OddScorer(None, None, {k: np.arange(10, dtype=float) for k in O.KINDS})
    assert sc.pct(O.KINDS[0], -1) == 0.0 and sc.pct(O.KINDS[0], 4.5) == 0.5 and sc.pct(O.KINDS[0], 99) == 1.0
