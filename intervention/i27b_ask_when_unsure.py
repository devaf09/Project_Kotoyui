"""介入の研究 実験I27b: 迷ったときだけ聞き返す（持ち主の誤読の判断の 3 案の 3、2026-11-06）

外の AI を呼ぶ代わりに、I24 の係り先の印を「聞けば返ってくる答え」とし、何回聞けばどこまで上がるかを測る。
  - 基本の読み: 近い方に係る + 話の板（20 行版）と食い違わない方（I27a、0.87）。
  - 迷いの強さ: 話の板で決まらなかった例文について、一般常識の語板（I26、v2.3b）が遠い方をどれだけ推すか（遠い方の合い方 − 近い方の合い方）。
    種類が連用（〜ながら・〜て）のものは、近い方がいつも正しかったので（I26）、迷いを低くする。
  - 迷いの強い順に k 個だけ聞き、聞いた例文は答えに従う。
  - 比べ: でたらめな順に聞く（100 回の平均）、誤りだけを知っている理想の順。
入力: I26 の結果（items に compat と kind）、I27a の結果（items に story）。
結果: ../Project_Kotoyui_private/intervention/results/i27b.jsonl
実行: python intervention/i27b_ask_when_unsure.py --i26 <i26.jsonl の 1 行（v2.3b）> --i27a <i27a.jsonl の 1 行（20 行版）>
"""
from __future__ import annotations

import argparse
import json
import os
import random


def load(path, pick):
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if pick(r):
            return r
    raise SystemExit(f"見つからない: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--i26", required=True)
    ap.add_argument("--i26_boards", default="v2.3b")
    ap.add_argument("--i27a", required=True)
    ap.add_argument("--out", default="../Project_Kotoyui_private/intervention/results/i27b.jsonl")
    a = ap.parse_args()
    r26 = load(a.i26, lambda r: r.get("boards") == a.i26_boards)
    r27 = load(a.i27a, lambda r: r.get("levels") == ["20"])
    c26 = {x["id"]: x for x in r26["items"]}
    far = {"意図": "誤読", "誤読": "意図"}
    items = []
    for x in r27["items"]:
        n, f = x["near"], far[x["near"]]
        s = x["story"]
        choice = f if s[f] > s[n] else n
        decided = s["意図"] != s["誤読"]
        y = c26[x["id"]]
        doubt = -1e9 if decided else (y["compat"][f] - y["compat"][n]) - (10.0 if y["kind"].startswith("連用") else 0.0)
        items.append(dict(id=x["id"], choice=choice, doubt=doubt))
    N = len(items)
    base = sum(x["choice"] == "意図" for x in items)
    order = sorted(range(N), key=lambda i: -items[i]["doubt"])
    wrong = [i for i in range(N) if items[i]["choice"] != "意図"]

    def acc_after(asked):
        return (base + sum(1 for i in asked if items[i]["choice"] != "意図")) / N
    ks = list(range(0, 62))
    curve = {k: acc_after(order[:k]) for k in ks}
    rng = random.Random(0)
    undecided = [i for i in range(N) if items[i]["doubt"] > -1e8]
    rand = {k: sum(acc_after(rng.sample(undecided, min(k, len(undecided)))) for _ in range(100)) / 100 for k in ks}
    ideal = {k: acc_after(wrong[:k]) for k in ks}
    rec = dict(exp="I27b", n=N, base=base / N, n_wrong=len(wrong), curve=curve, random=rand, ideal=ideal,
               first_hits=[items[i]["choice"] != "意図" for i in order[:15]])
    for k in (0, 3, 5, 10, 15, 20, 30):
        print(k, round(curve[k], 3), round(rand[k], 3), round(ideal[k], 3))
    print("迷いの強い順の先頭 15 個が誤りだったか:", rec["first_hits"])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
