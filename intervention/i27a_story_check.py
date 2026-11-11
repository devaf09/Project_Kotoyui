"""介入の研究 実験I27a: 圧縮と展開を比べて、曖昧な係り先の A / B を決める（持ち主の 3 案の 2、2026-11-06）

  - 圧縮: 話（4 段の長さすべての文）を、語ごとの「話の中の板」に縮める。名詞 N の板には、話の中で
        N を修飾した語（形容詞・動詞の連体）、N が が / は になった述語、N が を になった述語、N を の で修飾した名詞、を数えて入れる。
        動詞 V の板には、V に連用で係った語（〜ながら・〜て）を入れる。I25 の仮の語板と同じ「話の中の差分の板」（語板のスレッドの案）。
  - 展開と比べ: 曖昧な例文を A（意図）と B（誤読）の両方で読み、それぞれ「修飾する語 M が候補の語に付く」という属性を作る。
        話の板を展開して、その属性がすでにある方を選ぶ（食い違わない方）。どちらにも無ければ、近い方のまま。
  - I26 の「文脈」（係り受けを修飾する語から 2 歩たどる）は、M → N の向きだけを見ていた。ここでは「竹が光る」（N が主語）のように逆の向きも数える。
  - 例文そのものは話の文に入っていない（別に作られた文）。同じ語の組が話の中に出るかで決まる。
物差し: I26 と同じ（75 個で意図を選ぶ割合、直す / 壊す）。
結果: ../Project_Kotoyui_private/intervention/results/i27a.jsonl
実行: python intervention/i27a_story_check.py --tales <tales_v1_parsed.json> --labels <i24_ambig_labels.json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from i26_ambig_judge import compound  # noqa: E402


def story_boards(tale, levels=("20", "12", "5", "1")):
    """話の中の板: 語（複合語の表記と見出し）→ 属性の数え。"""
    sb = defaultdict(Counter)
    for lv in levels:
        for s in tale["parsed"].get(lv, []):
            toks = s["toks"]
            for i, w in enumerate(toks):
                h = w["head"]
                if h == i:
                    continue
                hw = toks[h]
                keys_h = {compound(toks, h), hw["lemma"]}
                keys_w = {compound(toks, i), w["lemma"]}
                if w["pos"] in ("ADJ", "VERB") and w["dep"] in ("acl", "amod") and hw["pos"] in ("NOUN", "PROPN"):
                    for k in keys_h:
                        sb[k]["修飾:" + w["lemma"]] += 1
                if w["pos"] in ("NOUN", "PROPN") and w["dep"] in ("nsubj", "obj", "obl") and hw["pos"] in ("VERB", "ADJ"):
                    role = {"nsubj": "する", "obj": "される", "obl": "で・に"}[w["dep"]]
                    for k in keys_w:
                        sb[k][f"{role}:{hw['lemma']}"] += 1
                        sb[k]["修飾:" + hw["lemma"]] += 1                # 「竹が光る」も 光る が 竹 に付く属性として数える
                if w["pos"] in ("NOUN", "PROPN") and w["dep"] == "nmod" and hw["pos"] in ("NOUN", "PROPN"):
                    for k in keys_h:
                        sb[k]["の:" + w["lemma"]] += 1
                        sb[k]["修飾:" + w["lemma"]] += 1
                if w["dep"] in ("advcl", "advmod") and hw["pos"] == "VERB":
                    for k in keys_h:
                        sb[k]["連用:" + w["lemma"]] += 1
                        sb[k]["修飾:" + w["lemma"]] += 1
    return sb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tales", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--levels", default="20,12,5,1")
    ap.add_argument("--out", default="../Project_Kotoyui_private/intervention/results/i27a.jsonl")
    a = ap.parse_args()
    tales = json.load(open(a.tales, encoding="utf-8"))
    labels = json.load(open(a.labels, encoding="utf-8"))
    levels = tuple(a.levels.split(","))
    boards = [story_boards(t, levels) for t in tales]
    items = []
    for ti, t in enumerate(tales):
        for k, it in enumerate(t["ambig"]):
            lab = labels.get(f"{ti}.{k}")
            if not lab or lab[0] != "attach":
                continue
            toks = it["ginza"]
            tx = [w["text"] for w in toks]
            m = tx.index(lab[1])
            ia = [i for i, w in enumerate(tx) if w == lab[2] and i != m][0]
            ib = [i for i, w in enumerate(tx) if w == lab[3] and i != m][0]
            mlem = toks[m]["lemma"]
            sc = {}
            for name, j in (("意図", ia), ("誤読", ib)):
                keys = {compound(toks, j), toks[j]["lemma"]}
                sc[name] = max(boards[ti][key]["修飾:" + mlem] for key in keys)
            near = "意図" if abs(ia - m) < abs(ib - m) else "誤読"
            items.append(dict(id=f"{ti}.{k}", near=near, story=sc))
    far = {"意図": "誤読", "誤読": "意図"}
    ch = [far[x["near"]] if x["story"][far[x["near"]]] > x["story"][x["near"]] else x["near"] for x in items]
    dec = [x for x in items if x["story"]["意図"] != x["story"]["誤読"]]
    rec = dict(exp="I27a", levels=list(levels), n=len(items),
               near_acc=sum(x["near"] == "意図" for x in items) / len(items),
               acc=sum(c == "意図" for c in ch) / len(items),
               fixed=sum(1 for x, c in zip(items, ch) if x["near"] == "誤読" and c == "意図"),
               broke=sum(1 for x, c in zip(items, ch) if x["near"] == "意図" and c == "誤読"),
               story_decided=len(dec), story_decided_intent=sum(x["story"]["意図"] > x["story"]["誤読"] for x in dec),
               items=items)
    print(json.dumps({k: v for k, v in rec.items() if k != "items"}, ensure_ascii=False))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
