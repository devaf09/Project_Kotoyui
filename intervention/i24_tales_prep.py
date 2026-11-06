"""介入の研究 実験I24: 案 2（文の圧縮と解凍）の下ごしらえ — 童話ファイルを読み、文に切り、GiNZA にかけ、語板の被覆を数える

持ち主（2026-11-06）が用意した童話ファイル（非公開の置き場 data/tales/tales_v1.txt）。12 話 × 4 段の長さ（20・12・5・1 行）+ 曖昧な例文 10 個ずつ。
  1. 読み込み: 書式が 2 通り（浦島太郎・桃太郎は表の形、残りは ■ と【例文】の形）なので、どちらも同じ形にそろえる。
  2. 文に切る（。と 」で終わる所）。GiNZA（語板を作ったときと同じ道具）で語・品詞・係り受け。
  3. 語板 v2 の被覆: 内容語（名詞・動詞・形容詞）のうち語板がある割合。知らない語を話ごとに並べる。
  4. 例文 120 個の GiNZA の読み（係り先）を出す。正しい係り先・誤った係り先は、この出力を見て手で付ける（i24_ambig_labels.json）。
童話の文はコードにも PR にも書かない。出力は非公開の置き場だけ。
実行: python intervention/i24_tales_prep.py --tales <tales_v1.txt> --boards <content_boards.json> --out <非公開の置き場>/data/tales/
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter

LEVELS = ("20", "12", "5", "1")
CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ"}


def split_sents(text):
    text = re.sub(r"^\s*\d+\.\s*", "", text, flags=re.M)              # 「1. 」の番号
    text = text.replace("\n", "")
    out, buf, depth = [], "", 0
    for ch in text:
        buf += ch
        if ch == "「":
            depth += 1
        elif ch == "」":
            depth = max(0, depth - 1)
        if ch == "。" and depth == 0:
            out.append(buf.strip())
            buf = ""
    if buf.strip():
        out.append(buf.strip())
    return [s for s in out if s]


def read_tales(path):
    s = open(path, encoding="utf-8").read()
    # 話の切れ目: 「〇〇の物語（4段階の長さ）」 か 「【n】 〇〇」
    heads = [(m.start(), m.group(1)) for m in re.finditer(r"^(\S+?)の物語（4段階の長さ）", s, flags=re.M)]
    heads += [(m.start(), m.group(1).strip()) for m in re.finditer(r"^【\d+】\s*(.+)$", s, flags=re.M)]
    heads.sort()
    tales = []
    for k, (p, name) in enumerate(heads):
        body = s[p:heads[k + 1][0] if k + 1 < len(heads) else len(s)]
        name = re.sub(r"（.*?）", "", name).strip()
        vers = {}
        for lv in LEVELS:
            m = re.search(lv + r"行バージョン\s*\n?(.*?)(?=(?:■\s*)?(?:\d+行バージョン|文法的|=====)|$)", body, flags=re.S)
            vers[lv] = split_sents(m.group(1)) if m else []
        tales.append(dict(name=name, versions=vers, ambig=read_ambig(body)))
    return tales


def read_ambig(body):
    items = []
    if "【例文】" in body:
        for m in re.finditer(r"【例文】(.+?)\n\s*\[意図\](.+?)\n\s*\[誤読\](.+?)\n\s*\[原因\](.+?)(?:\n|$)", body):
            items.append(dict(sent=m.group(1).strip(), intent=m.group(2).strip(), misread=m.group(3).strip(), cause=m.group(4).strip()))
        return items
    i = body.find("曖昧性の原因")
    if i < 0:
        return items
    rows = [x.strip() for x in body[i + len("曖昧性の原因"):].split("\n") if x.strip()]
    k = 0
    while k + 4 < len(rows):
        if rows[k].isdigit():
            items.append(dict(sent=rows[k + 1], intent=rows[k + 2], misread=rows[k + 3], cause=rows[k + 4]))
            k += 5
        else:
            k += 1
    return items


def score_baselines(tales, labels):
    """係り先の例文（attach）で、GiNZA の読みと「近い方に係る」が意図・誤読のどちらを選ぶか。"""
    res = Counter()
    for ti, t in enumerate(tales):
        for k, it in enumerate(t["ambig"]):
            lab = labels.get(f"{ti}.{k}")
            if not lab or lab[0] != "attach":
                continue
            g = it["ginza"]
            tx = [w["text"] for w in g]
            m = tx.index(lab[1])
            a = [i for i, w in enumerate(tx) if w == lab[2] and i != m][0]
            b = [i for i, w in enumerate(tx) if w == lab[3] and i != m][0]
            h, got = m, "どちらでもない"
            for _ in range(20):                                       # 係り受けを上へたどり、先に着いた方
                h = g[h]["head"]
                if h in (a, b):
                    got = "意図" if h == a else "誤読"
                    break
                if g[h]["head"] == h:
                    break
            res["GiNZA:" + got] += 1
            res["近い方:" + ("意図" if abs(a - m) < abs(b - m) else "誤読")] += 1
            res["n"] += 1
    return dict(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tales", required=True)
    ap.add_argument("--boards", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", default=None, help="i24_ambig_labels.json（手で付けた係り先）")
    a = ap.parse_args()
    import spacy
    nlp = spacy.load("ja_ginza")
    names = set(b["name"] for b in json.load(open(a.boards, encoding="utf-8")))
    tales = read_tales(a.tales)
    summary = dict(tales=[], unknown={})
    allc = Counter()
    for t in tales:
        parsed = {}
        cnt = Counter()
        unk = Counter()
        for lv, ss in t["versions"].items():
            parsed[lv] = []
            for s in ss:
                d = nlp(s)
                toks = [dict(i=w.i, text=w.text, lemma=w.lemma_, pos=w.pos_, dep=w.dep_, head=w.head.i) for w in d]
                parsed[lv].append(dict(text=s, toks=toks))
                for w in d:
                    if w.pos_ in CONTENT_POS:
                        cnt["content"] += 1
                        if w.lemma_ in names:
                            cnt["known"] += 1
                        else:
                            unk[w.lemma_] += 1
        for it in t["ambig"]:
            d = nlp(it["sent"])
            it["ginza"] = [dict(i=w.i, text=w.text, lemma=w.lemma_, pos=w.pos_, dep=w.dep_, head=w.head.i) for w in d]
        t["parsed"] = parsed
        allc.update(cnt)
        summary["tales"].append(dict(name=t["name"], n_sents={lv: len(v) for lv, v in t["versions"].items()}, n_ambig=len(t["ambig"]),
                                     coverage=cnt["known"] / max(1, cnt["content"]), n_unknown_types=len(unk)))
        summary["unknown"][t["name"]] = unk.most_common(30)
    summary["coverage_all"] = allc["known"] / max(1, allc["content"])
    if a.labels:
        labels = json.load(open(a.labels, encoding="utf-8"))
        summary["kinds"] = dict(Counter(v[0] for k, v in labels.items() if not k.startswith("_")))
        summary["baselines"] = score_baselines(tales, labels)
    os.makedirs(a.out, exist_ok=True)
    json.dump(tales, open(os.path.join(a.out, "tales_v1_parsed.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(summary, open(os.path.join(a.out, "i24_summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "unknown"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
