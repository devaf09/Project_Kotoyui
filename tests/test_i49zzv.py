"""I49zzv: 意味の印で骨組みを決め、丁寧の有り無しだけを替える道具（小さな作り物の会話）。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("intervention", "tools"):
    sys.path.insert(0, os.path.join(HERE, "..", d))

import i49zzv_frame_polite_run as V

# (行の文, [(表記, 見出し, 品詞, 係り)])
FORMS = {"行く": [("行く", "行く", "VERB", "ROOT")],
         "行きます": [("行き", "行く", "VERB", "ROOT"), ("ます", "ます", "AUX", "aux")],
         "行かない": [("行か", "行く", "VERB", "ROOT"), ("ない", "ない", "AUX", "aux")],
         "行きません": [("行き", "行く", "VERB", "ROOT"), ("ませ", "ます", "AUX", "aux"), ("ん", "ぬ", "AUX", "aux")],
         "見る": [("見る", "見る", "VERB", "ROOT")],
         "見ます": [("見", "見る", "VERB", "ROOT"), ("ます", "ます", "AUX", "aux")],
         "見ない": [("見", "見る", "VERB", "ROOT"), ("ない", "ない", "AUX", "aux")],
         "見ません": [("見", "見る", "VERB", "ROOT"), ("ませ", "ます", "AUX", "aux"), ("ん", "ぬ", "AUX", "aux")]}


HEADS = ["明日", "今日", "昨日", "来週", "今夜", "毎朝", "週末", "午後", "正月", "来月", "今年", "去年"]


def _line(speaker, form, head="明日"):
    toks = [dict(i=0, text=head, lemma=head, pos="NOUN", dep="obl", head=1)]
    for t, l, p, d in FORMS[form]:
        toks.append(dict(i=len(toks), text=t, lemma=l, pos=p, dep=d, head=1))
    return dict(speaker=speaker, text=head + form, toks=toks)


def _works(n=400):
    """会話ごとに、丁寧か、そうでないかが決まっている。行の形と頭の語は、会話ごとに違う並びにする（状況波が数えられるだけの種類を作る）。"""
    names = list(FORMS)
    out = []
    for i in range(n):
        pick = [f for f in names if ("ま" in f) == (i % 2 == 0)]
        out.append(dict(name=str(i), lines=[_line("AB"[j % 2], pick[(i * i + 3 * j * j + i * j + i // 4) % len(pick)], HEADS[(i * 5 + j * 7 + i * j) % 12])
                                            for j in range(8)]))
    return out


def test_last_pred_takes_only_lines_ending_with_the_predicate():
    g = V.last_pred(_line("A", "行きません"), None)
    assert g is not None and g[1] == "行く" and g[2] == ("ます", "ぬ") and g[5] == "明日"
    broken = _line("A", "行く")
    broken["text"] += "よね"
    assert V.last_pred(broken, None) is None


def test_walk_tracks_previous_and_own_lines():
    rows = list(V.walk(dict(lines=[_line("A", "行く"), _line("B", "見ます"), _line("A", "見ない")])))
    assert rows[0][2] is None and rows[1][2] == "明日行く" and rows[1][3] is None
    assert rows[2][2] == "明日見ます" and rows[2][3] == "明日行く"


def test_negation_never_flips_by_meaning_and_counts_are_consistent():
    r = V.run(_works(), hold=0.2)
    c = r["数"]
    assert c["行"] > 0 and c["組めた"] <= c["行"] and c["打ち消しの行"] > 0
    assert c["意味の印で打ち消しが入れ替わった"] == 0
    assert c["元と同じ形"] >= 0.8 * c["組めた"]
    for pc in r["丁寧の有り無しを選ぶ"].values():
        assert pc["状況波が当たり"] <= pc["問い"]
    for pc in r["丁寧の有り無しを選ぶ"].values():                      # 作り物では、会話ごとに丁寧が揃うので、状況波で選べる
        assert pc["割合"]["前の行の丁寧の印に合わせる"] > 0.9 and pc["割合"]["状況波"] > 0.9 and pc["割合"]["いつも多い方"] < 0.6
