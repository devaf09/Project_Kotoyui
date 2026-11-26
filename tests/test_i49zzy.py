"""I49zzy: 語幹の終わりの字で流れを選び、前の字を見て流れの表記を選ぶ（小さな作り物の材料）。"""
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("intervention", "tools", os.path.join("intervention", "past")):
    sys.path.insert(0, os.path.join(HERE, "..", d))

import i49zzy_form as Y
import i49zzy_form_run as R

SY = ("文末", "品詞:VERB", "台詞:はい")
NOUN = ("文末", "品詞:ADJ", "台詞:はい")


def _its():
    """「読む・飲む」は過去を聞いた。「住む」は言い切りだけ。「食べる」は過去と丁寧。断定の「だ」は文の中で「に」になる。"""
    its = []
    for lem, stem in (("読む", "読ん"), ("飲む", "飲ん"), ("噛む", "噛ん")):
        its += [(SY, lem, ("だ",), stem, ("だ",))] * 3 + [(SY, lem, (), lem, ())]
    its += [(SY, "食べる", ("た",), "食べ", ("た",))] * 6 + [(SY, "見る", ("た",), "見", ("た",))] * 6 + [(SY, "食べる", ("ます",), "食べ", ("ます",))] * 3
    its += [(SY, "住む", (), "住む", ())] * 2
    its += [(NOUN, "静か", ("だ",), "静か", ("に",))] * 20            # 断定の「だ」の、文の中の形
    return [its]


def test_past_of_an_unheard_mu_verb_takes_da_and_is_spelled_da():
    f = Y.Former(_its())
    assert f.rev.table("住む", ("過去",)) == ("た",)                  # I42 のまま: 動詞の表で一番多い「た」
    assert f.end.flow(f.rev, "住む", ("過去",)) == ("だ",)            # 直し: 終わりが「む」の語の表
    assert f.build("住む", ("過去",), fix=True) == "住んだ"
    assert f.build("読む", ("過去",), fix=False) == "読んに"          # I42 のまま: 断定の「だ」の形と混ざる
    assert f.build("読む", ("過去",), fix=True) == "読んだ"
    assert f.build("食べる", ("過去",), fix=True) == "食べた" and f.build("食べる", ("丁寧",), fix=True) == "食べます"


def test_reference_forms():
    assert R.reference("読む", "五段-マ行")[0] == {"過去": "読んだ", "丁寧": "読みます", "否定": "読まない"}
    assert R.reference("書く", "五段-カ行")[0]["過去"] == "書いた" and R.reference("行く", "五段-カ行")[0]["過去"] == "行った"
    assert R.reference("食べる", "下一段-バ行")[0]["否定"] == "食べない"
    assert R.reference("話す", "五段-サ行")[0] == {"過去": "話した", "丁寧": "話します", "否定": "話さない"}
    assert R.reference("勉強する", "サ行変格")[0]["丁寧"] == "勉強します"
    assert R.reference("来る", "カ行変格")[0] is None


def test_count_splits_by_kind_and_known():
    f = Y.Former(_its())
    refs = {w: R.reference(w, "五段-マ行") for w in ("読む", "住む")}
    d = R.count(f, Counter({"読む": 2, "住む": 1, "来る": 5}), refs)
    assert d["直した形（細かい表を先）"]["全部"]["過去"] == [3, 3] and d["I42 のまま"]["全部"]["過去"] == [3, 0]
    assert d["直した形（細かい表を先）"]["五段のマ行"]["過去"] == [3, 3] and "材料に有った語" in d["直した形（掛け合わせ）"]
