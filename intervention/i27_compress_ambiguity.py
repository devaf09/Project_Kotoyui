"""介入の研究 実験I27: 縮めると曖昧になる所を見つける — 4 段の長さ（20・12・5・1 行）の話で、修飾の係り先の候補がいくつあるかと、長い版の話の板が決めるかを数える
I27a で、短い段の話の板は意図の読みを下げた（縮めたこと自体で曖昧さが生まれる）。持ち主の 20 → 12 → 5 → 1 行は、縮めた結果の例として使える。
  - 修飾する語 M: 名詞に係る形容詞・動詞の連体（GiNZA の acl / amod）と、の で名詞に係る名詞（nmod）。
  - 係り先の候補: M の右にある名詞を順に見て、名詞句が続く間（名詞が の・という で次の名詞に係る間）の名詞の中心をすべて集める。
    途中で別の動詞・形容詞が来たら止める。候補が 2 つ以上なら、その M は「構造の上で曖昧」。
  - 読み 3 つ: 近い方（いちばん近い候補）、GiNZA の係り先、長い版（20 行）の話の板（I27a: 候補の語の板に「修飾:M」があるもの。1 つだけにあれば決まる）。
  - 縮めて曖昧になる所: 短い段で構造の上で曖昧で、話の板が決め、それが近い方と違う所（近い方で読むと、長い版と食い違う）。
物差し: 段ごとの、修飾 1 つあたりの曖昧な割合、話の板が決めた割合、近い方と食い違う数。場所の一覧は非公開の置き場（童話の文を含むため）。
結果: ../Project_Kotoyui_private/intervention/results/i27.jsonl、場所の一覧: <非公開>/data/tales/i27_places.md
実行: python intervention/i27_compress_ambiguity.py --tales <tales_v1_parsed.json> --places <非公開>/data/tales/i27_places.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from i26_ambig_judge import compound  # noqa: E402
from i27a_story_check import story_boards  # noqa: E402

LEVELS = ("20", "12", "5", "1")
NOUN = ("NOUN", "PROPN", "PRON")


def modifiers(toks):
    for i, w in enumerate(toks):
        h = w["head"]
        if h == i or toks[h]["pos"] not in NOUN:
            continue
        if (w["pos"] in ("ADJ", "VERB") and w["dep"] in ("acl", "amod")) or (w["pos"] in NOUN and w["dep"] == "nmod"):
            yield i


def candidates(toks, i):
    """M の右の名詞句の名詞の中心（複合語の前の部分は除く）。"""
    out = []
    for j in range(i + 1, len(toks)):
        w = toks[j]
        if w["pos"] in ("VERB", "ADJ") and w["dep"] not in ("compound", "fixed"):
            break
        if w["pos"] not in NOUN or w["dep"] == "compound":
            continue
        out.append(j)
        if w["dep"] != "nmod":                                        # 名詞句の中心まで来た
            break
    return out


def story_has(sb, toks, j, mlem):
    return max(sb.get(k, {}).get("修飾:" + mlem, 0) for k in {compound(toks, j), toks[j]["lemma"]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tales", required=True)
    ap.add_argument("--places", default=None)
    ap.add_argument("--out", default="../Project_Kotoyui_private/intervention/results/i27.jsonl")
    a = ap.parse_args()
    tales = json.load(open(a.tales, encoding="utf-8"))
    res = {lv: Counter() for lv in LEVELS}
    lines = ["# I27 縮めると曖昧になる所（非公開: 童話の文を含む）", ""]
    for t in tales:
        sb = story_boards(t, ("20",))
        lines += [f"## {t['name']}", ""]
        for lv in LEVELS:
            for s in t["parsed"].get(lv, []):
                toks = s["toks"]
                for i in modifiers(toks):
                    c = res[lv]
                    c["修飾"] += 1
                    cand = candidates(toks, i)
                    if toks[i]["head"] not in cand:
                        c["候補の外（GiNZA の係り先が名詞句の外）"] += 1
                    if len(cand) < 2:
                        continue
                    c["曖昧"] += 1
                    mlem = toks[i]["lemma"]
                    ev = {j: story_has(sb, toks, j, mlem) for j in cand}
                    near, gz = cand[0], toks[i]["head"]
                    pos = [j for j in cand if ev[j] > 0]
                    if len(pos) == 1:
                        st = pos[0]
                        c["話の板が決めた"] += 1
                        if st != near:
                            c["決めた: 近い方と違う"] += 1
                        if st != gz:
                            c["決めた: GiNZA と違う"] += 1
                        if lv != "20" and st != near:
                            lines.append(f"- [{lv} 行] {s['text']}  \n  「{toks[i]['text']}」: 近い方 = {compound(toks, near)}、"
                                         f"GiNZA = {compound(toks, gz)}、長い版の話の板 = {compound(toks, st)}")
                    elif len(pos) > 1:
                        c["話の板: 両方にある"] += 1
                    else:
                        c["話の板: どちらにも無い"] += 1
        lines.append("")
    rec = dict(exp="I27", levels={lv: dict(c) for lv, c in res.items()})
    for lv, c in res.items():
        n = max(1, c["修飾"])
        rec["levels"][lv]["曖昧の割合"] = c["曖昧"] / n
        rec["levels"][lv]["決めた割合（曖昧のうち）"] = c["話の板が決めた"] / max(1, c["曖昧"])
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if a.places:
        os.makedirs(os.path.dirname(a.places), exist_ok=True)
        open(a.places, "w", encoding="utf-8").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
