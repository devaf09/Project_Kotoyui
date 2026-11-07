"""介入の研究 実験I27f: 値の帯から関連性で文を作り、もう一度縮めて値の帯が同じに戻るかを見る（縮めて伸ばして同じ内容か）

持ち主（2026-11-07）: 「圧縮展開、文章走査するならバッファ板付でまとめて、文章を新たに生成するなら関連性で作ったほうが良いのかな」。
I27e で、ネストがあれば文は完全に戻った。ここではネスト（並びの順）を捨て、値の帯（誰が誰に、どの係り方で係るか）だけから文を作る。
  - 値の帯: 文の係り受けの木の、各板（語）に係った語とその係り方（GiNZA の dep）。語の表記（活用した形）は板に残す。並びの順は持たない。
  - 作り方（関連性）: 板ごとに、係った語を「前に置く語」と「後ろに置く語」（助詞・助動詞・て など、dep が case / aux / mark / fixed / cop と読点・句点）に分ける。
    並べる順は、ほかの話（1 話抜き）の文から学ぶ。2 通り:
      平均: 同じ係り方・同じ見出しの語が兄弟の中のどこに来たか（0 = 先頭、1 = 末尾）の平均。
      対: 兄弟の 2 語の組ごとに、どちらが前に来たかを数え、「前に来やすさ」の和で並べる。
    どちらも、見出しで見たことが無ければ係り方と品詞、それも無ければ係り方だけで引く。かぎかっこは落とす。
  - 比べ:
      元の順（ネストあり、I27e）・でたらめな順・関連性の順。
      作った文を GiNZA で読み直し（もう一度縮める）、値の帯（語 → 係った語と係り方の組）が元とどれだけ同じかを数える（F1）。
      上限の目安: 元の文をいまの GiNZA で読み直したときの F1（解析の版の違いの分）。
  - 名詞句の中の修飾（I27c の曖昧な所）が、読み直しでも同じ語に係ったかを別に数える。
物差し: 文がそのまま同じになった割合、値の帯の F1（内容の語の組 / 助詞の組）、曖昧な所の係りが保たれた割合。作った文は非公開の置き場（童話の語を含むため）。
結果: ../Project_Kotoyui_private/intervention/results/i27f.jsonl、作った文: <非公開>/data/tales/i27f_generated.md
実行: python intervention/i27f_generate.py --tales <tales_v1_parsed.json> [--gen <md>]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from i27c_compress_units import LEVELS, candidates, modifiers, units  # noqa: E402

AFTER = {"case", "aux", "mark", "fixed", "cop"}
CONTENT = {"NOUN", "PROPN", "PRON", "VERB", "ADJ", "ADV", "NUM"}
QUOTE = {"「", "」", "『", "』", "（", "）"}


def tree(toks):
    kids = defaultdict(list)
    roots = []
    for i, w in enumerate(toks):
        if w["text"] in QUOTE:
            continue
        h = w["head"]
        while h != i and toks[h]["text"] in QUOTE:                   # かぎかっこに係るものは、その先へ
            h = toks[h]["head"] if toks[h]["head"] != h else i
        if h == i:
            roots.append(i)
        else:
            kids[h].append(i)
    return kids, roots


def side(w):
    if w["dep"] in AFTER:
        return "後"
    if w["pos"] == "PUNCT":
        return "後"
    return "前"


def learn_order(tales, skip):
    """（係り方・見出し）/（係り方・品詞）/ 係り方 → 兄弟の中の位置（0〜1）の平均。"""
    acc = [defaultdict(list), defaultdict(list), defaultdict(list)]
    for ti, t in enumerate(tales):
        if ti == skip:
            continue
        for lv in LEVELS:
            for s in t["parsed"].get(lv, []):
                toks = s["toks"]
                kids, _ = tree(toks)
                for h, ks in kids.items():
                    for sd in ("前", "後"):
                        sib = sorted(k for k in ks if side(toks[k]) == sd)
                        for r, k in enumerate(sib):
                            p = r / (len(sib) - 1) if len(sib) > 1 else 0.5
                            w = toks[k]
                            acc[0][(sd, w["dep"], w["lemma"])].append(p)
                            acc[1][(sd, w["dep"], w["pos"])].append(p)
                            acc[2][(sd, w["dep"])].append(p)
    return [{k: sum(v) / len(v) for k, v in a.items()} for a in acc]


def learn_pairs(tales, skip):
    """兄弟の組の前後: 鍵の組 → [前に来た数, 数]。鍵は（係り方・見出し）/（係り方・品詞）/ 係り方 の 3 段。"""
    acc = [defaultdict(lambda: [0, 0]) for _ in range(3)]
    for ti, t in enumerate(tales):
        if ti == skip:
            continue
        for lv in LEVELS:
            for s in t["parsed"].get(lv, []):
                toks = s["toks"]
                kids, _ = tree(toks)
                for h, ks in kids.items():
                    for sd in ("前", "後"):
                        sib = sorted(k for k in ks if side(toks[k]) == sd)
                        for x in range(len(sib)):
                            for y in range(x + 1, len(sib)):
                                wx, wy = toks[sib[x]], toks[sib[y]]
                                for lvl, (kx, ky) in enumerate(zip(keys(sd, wx), keys(sd, wy))):
                                    acc[lvl][(kx, ky)][0] += 1
                                    acc[lvl][(kx, ky)][1] += 1
                                    acc[lvl][(ky, kx)][1] += 1
    return acc


def keys(sd, w):
    return ((sd, w["dep"], w["lemma"]), (sd, w["dep"], w["pos"]), (sd, w["dep"]))


def before(pairs, sd, wx, wy):
    for lvl, (kx, ky) in enumerate(zip(keys(sd, wx), keys(sd, wy))):
        c = pairs[lvl].get((kx, ky))
        if c and c[1] > 0:
            return (c[0] + 0.5) / (c[1] + 1)
    return 0.5


def score(order, sd, w):
    for tab, key in zip(order, ((sd, w["dep"], w["lemma"]), (sd, w["dep"], w["pos"]), (sd, w["dep"]))):
        if key in tab:
            return tab[key]
    return 0.5


def realize(toks, how, order=None, rng=None):
    """値の帯（木の親子と係り方）だけから並びを作る。how: 元 / でたらめ / 関連。返り値: トークン番号の並び。"""
    kids, roots = tree(toks)

    def rank(sd, ks):
        if how == "元":
            return sorted(ks)
        if how == "でたらめ":
            ks = ks[:]
            rng.shuffle(ks)
            return ks
        if how == "関連（対）":                                      # 兄弟の組ごとに「前に来やすさ」を足して並べる
            return sorted(ks, key=lambda k: -sum(before(order, sd, toks[k], toks[j]) for j in ks if j != k))
        return sorted(ks, key=lambda k: score(order, sd, toks[k]))

    def walk(i):
        ks = kids.get(i, [])
        pre = rank("前", [k for k in ks if side(toks[k]) == "前"])
        post = rank("後", [k for k in ks if side(toks[k]) == "後"])
        out = []
        for k in pre:
            out += walk(k)
        out.append(i)
        for k in post:
            out += walk(k)
        return out
    seq = []
    for r in (sorted(roots) if how != "でたらめ" else rng.sample(roots, len(roots))):
        seq += walk(r)
    return seq


def edges(toks):
    """値の帯の組: (係った語の見出し, 係り方, 係り先の見出し)。内容の語の組と、助詞・助動詞の組に分ける。"""
    c, f = Counter(), Counter()
    for i, w in enumerate(toks):
        h = w["head"]
        if h == i or w["pos"] == "PUNCT" or w["text"] in QUOTE:
            continue
        hl = toks[h]["lemma"]
        if w["pos"] in CONTENT:
            c[(w["lemma"], w["dep"], hl)] += 1
        else:
            f[(w["lemma"], w["dep"], hl)] += 1
    return c, f


def f1(a, b):
    m = sum((a & b).values())
    p, r = m / max(1, sum(b.values())), m / max(1, sum(a.values()))
    return 2 * p * r / max(1e-9, p + r)


def to_toks(doc):
    return [dict(i=t.i, text=t.text, lemma=t.lemma_, pos=t.pos_, dep=t.dep_, head=t.head.i) for t in doc]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tales", required=True)
    ap.add_argument("--gen", default=None)
    ap.add_argument("--out", default="../Project_Kotoyui_private/intervention/results/i27f.jsonl")
    a = ap.parse_args()
    import spacy                                                     # CPU で動かす
    nlp = spacy.load("ja_ginza")
    tales = json.load(open(a.tales, encoding="utf-8"))
    rng = random.Random(0)
    hows = ("元", "でたらめ", "関連", "関連（対）")
    tot = {h: dict(same=0, fc=0.0, ff=0.0, amb_kept=0) for h in hows + ("読み直し",)}
    n = n_amb = 0
    lines = ["# I27f 値の帯から作った文（非公開: 童話の語を含む）", ""]
    for ti, t in enumerate(tales):
        order = learn_order(tales, ti)
        pairs = learn_pairs(tales, ti)
        lines += [f"## {t['name']}", ""]
        for si, s in enumerate(t["parsed"]["20"]):
            toks = s["toks"]
            oc, of = edges(toks)
            tok2u, us = units(toks)
            amb = []
            for ui, gu in modifiers(toks, tok2u, us):
                cand, _ = candidates(toks, tok2u, us, ui)
                if len(cand) >= 2 and gu in cand:
                    amb.append((toks[us[ui]["head"]]["lemma"], toks[us[gu]["head"]]["lemma"]))
            n += 1
            n_amb += len(amb)
            ref = "".join(w["text"] for w in toks if w["text"] not in QUOTE)
            texts = {"読み直し": "".join(w["text"] for w in toks)}
            for h in hows:
                seq = realize(toks, h, pairs if h == "関連（対）" else order, rng)
                texts[h] = "".join(toks[k]["text"] for k in seq)
            for h, txt in texts.items():
                nt = to_toks(nlp(txt))
                nc, nf = edges(nt)
                r = tot[h]
                r["same"] += txt.replace("「", "").replace("」", "") == ref if h == "読み直し" else txt == ref
                r["fc"] += f1(oc, nc)
                r["ff"] += f1(of, nf)
                for mlem, hlem in amb:
                    r["amb_kept"] += any(w["lemma"] == mlem and nt[w["head"]]["lemma"] == hlem for w in nt)
            if texts["関連（対）"] != ref:
                lines.append(f"- 元: {s['text']}  \n  関連（平均）: {texts['関連']}  \n  関連（対）: {texts['関連（対）']}")
        lines.append("")
    rec = dict(exp="I27f", sentences=n, ambiguous=n_amb,
               res={h: dict(same=r["same"] / n, f1_content=r["fc"] / n, f1_function=r["ff"] / n,
                            amb_kept=r["amb_kept"] / max(1, n_amb)) for h, r in tot.items()})
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if a.gen:
        os.makedirs(os.path.dirname(os.path.abspath(a.gen)), exist_ok=True)
        open(a.gen, "w", encoding="utf-8").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
