"""I49zzy: 持ち主の文の動詞を、過去・丁寧・否定の形に組めるか（I42 のまま／直した形。`intervention/i49zzy_form.py`。未確定の試し）。

輪の最後（組んだ 3 つ組を文の形にする所）の前提の確かめ。形を組む部品は、解析済みの会話（雑談など）から作る。
動詞は「A が B を C する」のテキストから取る（tools/i49zp_text_talks_run.py と同じ切り方。語は出さず、数だけ出す）。
物差し: 形態素の辞書（sudachipy）の活用の種類から、決まりで作った形（過去・丁寧・否定）。物差しは数えるためだけで、組む側には使わない。
  物差しを作れない語（辞書で動詞 1 語にならない、来る など）は数えない。
分けて数える: 形を組む部品の材料にその語が有ったか／活用の種類（一段、する、五段の行）。
手で置いた物: 物差しの決まり（音便の表）。乱数は使わない。
  外した行で元と同じ形: 解析済みの会話の 1 割を外し、本物の行の述語を元と同じ意味の組から組み直して、元の形と同じかを数える（直しが本物の行を壊さないかの確かめ）。
引数: --text 材料（カンマ区切り） --cache 解析済み JSON --out 記録.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "intervention"), os.path.join(HERE, "..", "intervention", "past")]

import i42_fused_flow as F  # noqa: E402
import i49zp_text_talks_run as ZP  # noqa: E402
import i49zzy_form as Y  # noqa: E402

ROW = {"カ": "かきいた", "ガ": "がぎいだ", "サ": "さしした", "タ": "たちった", "ナ": "なにんだ", "バ": "ばびんだ", "マ": "まみんだ", "ラ": "らりった", "ワ": "わいった"}
MEANS = ("過去", "丁寧", "否定")
FORMS = (("I42 のまま", False), ("直した形（細かい表を先）", "細かい表を先"), ("直した形（掛け合わせ）", "掛け合わせ"))


def reference(word, kind):
    """活用の種類（辞書の言い方）から、決まりで作った形。→ ({意味: 形}, 種類の名前) か (None, 種類)。word は終止形。"""
    if kind.startswith(("上一段", "下一段")):
        st = word[:-1]
        return {"過去": st + "た", "丁寧": st + "ます", "否定": st + "ない"}, "一段"
    if kind.startswith("サ行変格") and word.endswith("する"):
        st = word[:-2]
        return {"過去": st + "した", "丁寧": st + "します", "否定": st + "しない"}, "する"
    if kind.startswith("五段") and len(kind) > 3 and kind[3] in ROW:
        r, st = ROW[kind[3]], word[:-1]
        past = "った" if word in ("行く", "いく") else r[2:]
        return {"過去": st + past, "丁寧": st + r[1] + "ます", "否定": st + r[0] + "ない"}, "五段の" + kind[3] + "行"
    return None, kind


def kind_of(tk, lem):
    """辞書で引いた (終止形の最後の語の前まで, 最後の語, 活用の種類)。動詞で終わらない時は None。"""
    from sudachipy import tokenizer
    ms = list(tk.tokenize(lem, tokenizer.Tokenizer.SplitMode.C))
    if not ms or ms[-1].part_of_speech()[0] != "動詞":
        return None
    return "".join(x.surface() for x in ms[:-1]), ms[-1].surface(), ms[-1].part_of_speech()[4]


def count(former, verbs, refs):
    """verbs: 語 → 回数。refs: 語 → ({意味: 形}, 種類)。→ 形 → 分け方 → 意味 → (数, 合う)。"""
    out = {n: defaultdict(lambda: {m: [0, 0] for m in MEANS}) for n, _ in FORMS}
    for lem, n in verbs.items():
        ref, kind = refs.get(lem, (None, None))
        if ref is None:
            continue
        known = "材料に有った語" if lem in former.surf.forms else "材料に無かった語"
        for name, fix in FORMS:
            for m in MEANS:
                hit = former.build(lem, (m,), fix=fix) == ref[m]
                for k in ("全部", known, kind):
                    c = out[name][k][m]
                    c[0] += n
                    c[1] += n * hit
    return {n: {k: v for k, v in sorted(d.items())} for n, d in out.items()}


def same_form(works, hold=0.1):
    """作品の 1 割を外し、外した行の述語を、元と同じ意味の組から組み直す。→ 語の類 → 形 → (数, 元と同じ)。本物の行での確かめ。"""
    import i49_situation as SIT
    order = SIT.golden_order(len(works))
    k = max(int(len(works) * hold), 1)
    former = Y.Former(F.items([works[i] for i in order[k:]]))
    out = {pos: {n: [0, 0] for n, _ in FORMS} for pos in ("VERB", "ADJ")}
    for its in F.items([works[i] for i in order[:k]]):
        for sy, lem, fl, stem, fs in its:
            pos = F._pos(sy)
            if pos not in out:
                continue
            m = F.meaning(fl, fs, pos)
            for name, fix in FORMS:
                c = out[pos][name]
                c[0] += 1
                c[1] += former.build(lem, m, pos, "台詞:はい" in sy, fix=fix) == stem + "".join(fs)
    return out


def read(path):
    for enc in ("utf-8", "cp932"):
        try:
            return [l.strip() for l in open(path, encoding=enc).read().splitlines()]
        except UnicodeDecodeError:
            continue
    raise ValueError("読めない文字の並び")


def main():
    from sudachipy import dictionary
    ap = argparse.ArgumentParser()
    for k in ("text", "cache", "out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    verbs = Counter()
    for f in a.text.split(","):
        for t in ZP.talks_of(read(f))[0]:
            verbs.update(tri[2][0] for tri in t[0])
    works = json.load(open(a.cache, encoding="utf-8"))
    works = works["works"] if isinstance(works, dict) and "works" in works else works
    former = Y.Former(F.items(works))
    tk = dictionary.Dictionary().create()
    refs, none = {}, 0
    for lem, n in verbs.items():
        k = kind_of(tk, lem)
        ref, kind = reference(k[0] + k[1], k[2]) if k else (None, None)
        refs[lem] = (ref, kind)
        none += n * (ref is None)
    res = dict(動詞の数=sum(verbs.values()), 違う動詞=len(verbs), 物差しを作れない=none, 外した行で元と同じ形=same_form(works), **count(former, verbs, refs))
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    for name, _ in FORMS:
        print(name, {m: round(c[1] / max(c[0], 1), 3) for m, c in res[name]["全部"].items()})
    print(res["外した行で元と同じ形"])


if __name__ == "__main__":
    main()
