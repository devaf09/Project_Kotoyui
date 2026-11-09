"""介入の研究 実験I34a: 会話の板（問い → 答えの型）と、聞き返しの手がかりを数える

持ち主（2026-11-09）の 1: 短い会話で問いに関係の無い語板を拾う。初めて話す人なら前提知識が無い。
  相手の属性で知らない所があれば聞き返してよい。自分の語板に無い属性なら「無い」と言ってよい。
材料: 隣り合う台詞（「」『』で始まる行が 2 つ続く所）。話し手は分からないが、多くは替わる。
数えるもの（手の一覧は型の名前を付けるためだけ。選ぶのは数えた割合）:
  問いの型   はい/いいえ（？・か で終わり、問いの語が無い）、問いの語（何・どこ・誰・いつ・どう・なぜ）、問いでない。
  答えの型   始めの感動詞（うん・はい・ええ・いや・いいえ など、品詞 INTJ の語そのもの）、
             問いの述語を繰り返すか（同じ原形）、その述語を否定で返すか、問いの名詞を「って」「？」で返すか（聞き返し）、
             答えも問いか（問い返し）。
  P(答えの型 | 問いの型) と、聞き返しが起きる時の問いの名詞の珍しさ（ほかの作品に出る割合）。
Web 小説は _wn。結果は非公開の置き場だけ。本文はコードに書かない。
実行: python intervention/i34_dialogue_board.py --cache <解析済み JSON> --stats <jsonl>
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

ASK = ("何", "どこ", "誰", "いつ", "どう", "なぜ", "どうして", "どれ", "どちら", "どんな", "いくら", "いくつ")
NOUN = ("NOUN", "PROPN")


def isq(text):
    return text.lstrip("　 ").startswith(("「", "『"))


def body(text):
    t = text.strip("　 ")
    return t[1:].rstrip("」』") if t[:1] in "「『" else t


def q_type(toks, text):
    b = body(text).rstrip("…。 ")
    wh = [t["lemma"] for t in toks if t["lemma"] in ASK]
    ends_q = b.endswith(("？", "?", "か", "の", "かな", "かい"))
    if wh and ends_q:
        return "問いの語:" + wh[0]
    if ends_q and b.endswith(("？", "?", "か", "かい")):
        return "はい/いいえ"
    return "問いでない"


def a_shape(qt, atoks, atext, q_preds, q_nouns):
    out = []
    first = next((t for t in atoks if t["pos"] not in ("PUNCT", "SYM")), None)
    if first is not None and first["pos"] == "INTJ":
        out.append("始め:" + first["lemma"])
    a_preds = {t["lemma"] for t in atoks if t["pos"] in ("VERB", "ADJ")}
    neg = any(t["lemma"] in ("ない", "ぬ") and t["pos"] == "AUX" for t in atoks)
    if q_preds & a_preds:
        out.append("述語を繰り返す・否定" if neg else "述語を繰り返す")
    b = body(atext)
    if any(n in b and (n + "って" in b or n + "？" in b or n + "?" in b) for n in q_nouns):
        out.append("聞き返し")
    if b.rstrip("…。 ").endswith(("？", "?")):
        out.append("問い返し")
    if not out:
        out.append("その他")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--stats", required=True)
    a = ap.parse_args()
    works = json.load(open(a.cache, encoding="utf-8"))
    df = Counter()
    for w in works:
        df.update({t["lemma"] for ln in w["lines"] for t in ln["toks"] if t["pos"] in NOUN})
    W = len(works)
    by_q = defaultdict(Counter)
    nq = Counter()
    start = defaultdict(Counter)
    rare = defaultdict(list)
    for w in works:
        L = w["lines"]
        for i in range(len(L) - 1):
            if not (isq(L[i]["text"]) and isq(L[i + 1]["text"])):
                continue
            qt = q_type(L[i]["toks"], L[i]["text"])
            qp = {t["lemma"] for t in L[i]["toks"] if t["pos"] in ("VERB", "ADJ")}
            qn = {t["text"] for t in L[i]["toks"] if t["pos"] in NOUN and len(t["text"]) >= 2}
            sh = a_shape(qt, L[i + 1]["toks"], L[i + 1]["text"], qp, qn)
            nq[qt.split(":")[0]] += 1
            for s in sh:
                by_q[qt.split(":")[0]][s.split(":")[0]] += 1
                if s.startswith("始め:"):
                    start[qt.split(":")[0]][s] += 1
            # 聞き返しが起きた所と起きなかった所の、問いの名詞の「ほかの作品に出る割合」
            for t in L[i]["toks"]:
                if t["pos"] in NOUN and len(t["text"]) >= 2:
                    r = (df[t["lemma"]] - 1) / (W - 1)
                    ab = body(L[i + 1]["text"])
                    echoed = (t["text"] + "って") in ab or (t["text"] + "？") in ab
                    rare["聞き返された名詞" if echoed else "聞き返されなかった名詞"].append(r)
    rec = dict(exp="I34a", 隣の台詞の組=sum(nq.values()), 問いの型の数=dict(nq),
               答えの型の割合={q: {s: round(v / nq[q], 3) for s, v in c.most_common()} for q, c in by_q.items()},
               始めの感動詞={q: {s: round(v / nq[q], 3) for s, v in c.most_common(8)} for q, c in start.items()},
               問いの名詞がほかの作品に出る割合={k: dict(n=len(v), 平均=round(sum(v) / len(v), 3) if v else None,
                                                ほぼ出ない_0_05未満=round(sum(x < 0.05 for x in v) / len(v), 3) if v else None)
                                         for k, v in rare.items()})
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    with open(a.stats, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
