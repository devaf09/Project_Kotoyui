"""I48w: 人の会話の返事の「働き」を数える（持ち主の型の 5 つの答え合わせ。数えるだけ）。

働き（上から順に当てる。1 つの返事に 1 つ）:
  時間軸の問い : 問い（I48e の is_question）で、過去・仮定・先の印がある（行の状況の過去、または TIME の語）。
  同じ話題の問い: 問いで、相手の前の行の中身の語を持つ（深掘り）。
  横へ広げる問い: 問いで、相手の前の行に無い中身の語だけを持つ（または中身の語が無い）。
  言い換えて褒める: 問いでなく、PRAISE の語がある（粗い。語の目印に頼る）。
  Yes, And    : 問いでなく、受けの一言（I48l の uke）で始まり、相手の前の行に無い中身の語を足している。
  感嘆・反復   : 問いでなく、受けの一言で始まるか感動詞を持ち、新しい中身の語を足していない（相手の語を返す・感嘆だけ）。
  その他       : どれでもない。
置いた語（原形）: TIME（仮定・先の印）、PRAISE（褒めの語）。どちらも粗く、見分けの弱い働きがある。
穴（{単語}）: 返事に、相手の前の行の中身の語と同じ原形がある。丁寧: 行の機能語に です・ます・ござる（I48o の line_sit）。

実行: python intervention/i48w_reply_function.py --cache <parsed.json> [--name RPC] --out <json>
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i48e_relation_split as E  # noqa: E402
import i48l_uke_flow as U  # noqa: E402
import i48o_slide_board as O  # noqa: E402

TIME = ("たら", "ば", "なら", "もし", "次", "これから", "今度", "将来", "昔", "前", "最初", "きっかけ", "いつか")
PRAISE = ("すごい", "凄い", "素敵", "すてき", "素晴らしい", "偉い", "さすが", "立派", "えらい")
KINDS = ("時間軸の問い", "同じ話題の問い", "横へ広げる問い", "言い換えて褒める", "Yes, And", "感嘆・反復", "その他")
QS = KINDS[:3]


def plain_len(text):
    return len("".join(ch for ch in text if ch not in "「」『』 　\n"))


def function(ln, prev):
    toks = ln["toks"]
    P = {t["lemma"] for t in prev["toks"] if t["pos"] in E.CONTENT} if prev else set()
    C = [t["lemma"] for t in toks if t["pos"] in E.CONTENT]
    new = [w for w in C if w not in P]
    lem = {t["lemma"] for t in toks} | {t["text"] for t in toks}
    if E.is_question(ln):
        sit = O.line_sit(toks) or (0, 0, 0)
        if sit[0] or any(w in lem for w in TIME):
            return "時間軸の問い"
        return "同じ話題の問い" if any(w in P for w in C) else "横へ広げる問い"
    if any(w in lem for w in PRAISE):
        return "言い換えて褒める"
    opener = U.uke(toks) is not None
    intj = any(t["pos"] == "INTJ" for t in toks)
    if opener and new:
        return "Yes, And"
    if (opener or intj) and not new:
        return "感嘆・反復"
    return "その他"


def rows(work):
    L = work["lines"]
    out = []
    for i, ln in enumerate(L):
        if i == 0 or not ln["toks"]:
            continue
        prev = L[i - 1]
        P = {t["lemma"] for t in prev["toks"] if t["pos"] in E.CONTENT}
        out.append(dict(i=i, f=function(ln, prev), n=plain_len(ln.get("text", "")),
                        slot=any(t["lemma"] in P for t in ln["toks"] if t["pos"] in E.CONTENT),
                        polite=bool((O.line_sit(ln["toks"]) or (0, 0, 0))[2])))
    return out


def measure(works):
    R = {w["name"]: rows(w) for w in works}
    allr = [r for v in R.values() for r in v]
    short = [r for r in allr if 9 <= r["n"] <= 20]
    res = {}
    for name, rs in (("全部", allr), ("9〜20 字", short)):
        c = Counter(r["f"] for r in rs)
        res[name] = dict(行=len(rs), 働き={k: round(c[k] / max(1, len(rs)), 4) for k in KINDS},
                         問い=round(sum(c[k] for k in QS) / max(1, len(rs)), 4),
                         穴あり=round(sum(r["slot"] for r in rs) / max(1, len(rs)), 4), 丁寧=round(sum(r["polite"] for r in rs) / max(1, len(rs)), 4))
    # 同じ話し手の、前の自分の返事（2 行前）の働き → 今の働き
    trans = defaultdict(Counter)
    gaps, between = Counter(), Counter()
    for name, rs in R.items():
        by_i = {r["i"]: r for r in rs}
        for r in rs:
            p = by_i.get(r["i"] - 2)
            if p is not None:
                trans["問い" if p["f"] in QS else p["f"]]["問い" if r["f"] in QS else r["f"]] += 1
        for parity in (0, 1):                                 # 話し手ごとに、問いと問いの間に挟まる自分の返事の数と、その中身
            own = [by_i[i] for i in sorted(by_i) if i % 2 == parity]
            last_q = None
            for k, r in enumerate(own):
                if r["f"] in QS:
                    if last_q is not None:
                        gap = k - last_q - 1
                        gaps[min(gap, 5)] += 1
                        mid = own[last_q + 1:k]
                        between["受けか肯定だけ" if mid and all(m["f"] in ("感嘆・反復", "Yes, And") for m in mid) else
                                ("挟まず" if not mid else "ほかも挟む")] += 1
                    last_q = k
    res["前の自分の返事の働き → 今の働き"] = {a: {b: round(n / sum(c.values()), 3) for b, n in c.most_common()} for a, c in trans.items()}
    g = sum(gaps.values())
    res["問いと問いの間の自分の返事の数"] = {str(k) + ("以上" if k == 5 else ""): round(v / max(1, g), 3) for k, v in sorted(gaps.items())}
    res["問いと問いの間に挟まる物"] = {k: round(v / max(1, g), 3) for k, v in between.items()}
    pq = trans.get("問い", Counter())
    res["問いの後にまた問い"] = round(pq["問い"] / max(1, sum(pq.values())), 4)
    res["問いの前の自分の返事が問いでない割合"] = round(1 - res["問いの後にまた問い"], 4)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    works = json.load(open(a.cache, encoding="utf-8"))
    works = works["works"] if isinstance(works, dict) else works
    res = measure(works)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
