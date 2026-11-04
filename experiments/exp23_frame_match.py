"""実験23: 返事選びに「文の形」（述語と格）の合い方を足す — 切り貼り感を減らせるか（持ち主の選んだ 2、2026-11-04）

着想: 持ち主「単語を切り貼りして会話している感じ」。いまの返事選びは語の袋（順番も役割も見ない）で動いている（説明のページ 2026-11-04）。
      相手の発言と候補の返事で、文の形が合うかを点に足す。
文の形（形態素解析 fugashi + UniDic から、係り受けの解析は使わない軽い形）:
  格の組   名詞 + 格助詞（が・を・に・で・へ・と・から）の組。例 散歩が・犬を
  述語     動詞・形容詞・形状詞の原形（補助的な動詞は除く）
  問いの型 相手の発言の疑問詞（どこ・いつ・誰・何・どう・なぜ・どれ・いくつ）
足す点:
  F 格の組の重なり: 相手と候補で同じ「名詞 + 格」がある数（語の重なりより厳しい: 同じ語が同じ役で出る）
  P 述語の共鳴: 「前の発言の述語 x → 次の発言の述語 y」の正の PMI（教えの会話で数える。語の共鳴の、述語だけの版）
  Q 問いの答えの型: どこ → 候補に地名・場所の語、いつ → 時の語、誰 → 人の語、何 → 名詞の目的語、いくつ → 数、どう・なぜ → 理由・様子の言い方
物差し: 実験06・07 と同じ 10 択（同じ seed で同じ問、3,000 問、ほかの会話から・同じ会話から）。
        今の芯（重なり A + 共鳴 B = kotoyui 0.1.0 の芯）に F・P・Q を重みを付けて足す（重みは試しの前半で選び、後半で測り、逆も測って平均）。
材料: RealPersonaChat・JMultiWOZ（CC BY-SA 4.0）、会話ごとに教え 9 : 試し 1。
結果: results/exp23.txt
実行: CUDA_VISIBLE_DEVICES= python experiments/exp23_frame_match.py --seed 0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import kotoyui as K  # noqa: E402

CASES = ("が", "を", "に", "で", "へ", "と", "から")
QWORD = {"どこ": "place", "いつ": "time", "誰": "person", "だれ": "person", "何": "thing", "なに": "thing", "なん": "thing",
         "どう": "how", "なぜ": "why", "なんで": "why", "どうして": "why", "どれ": "thing", "いくつ": "num", "何人": "num", "何回": "num", "いくら": "num"}
TIME_W = re.compile(r"(時|日|曜|週|月|年|朝|昼|夜|夕方|最近|昔|今度|来週|先週|毎日|毎週|いつも|頃|ごろ|前|後)")
WHY_W = re.compile(r"(から|ので|ため|せい|おかげ|理由)")
HOW_W = re.compile(r"(感じ|ように|みたい|そう|ます|です)")
_fr = {}


def frame(t):
    """発言 → (格の組の集まり, 述語の集まり, 地名・人名・数の有無)。同じ発言は 1 回だけ解析する。"""
    if t not in _fr:
        ts = K.tokens(t)
        pairs, preds, kinds = set(), set(), set()
        for i, m in enumerate(ts):
            if m[2] == "名詞" and i + 1 < len(ts) and ts[i + 1][0] in CASES and ts[i + 1][3] == "格助詞":
                pairs.add((m[1], ts[i + 1][0]))
            if m[2] in ("動詞", "形容詞", "形状詞") and m[3] != "非自立可能":
                preds.add(m[1])
            if "地名" in m[4] or m[1] in ("店", "駅", "家", "公園", "海", "山", "近く", "場所", "会社", "学校"):
                kinds.add("place")
            if "人名" in m[4] or m[1] in ("友達", "家族", "母", "父", "妻", "夫", "子供", "先輩", "上司", "彼", "彼女"):
                kinds.add("person")
            if m[2] == "名詞" and m[3] == "数詞":
                kinds.add("num")
            if m[2] == "名詞" and i + 1 < len(ts) and ts[i + 1][0] in ("を", "が", "は"):
                kinds.add("thing")
        if TIME_W.search(t):
            kinds.add("time")
        if WHY_W.search(t):
            kinds.add("why")
        if HOW_W.search(t):
            kinds.add("how")
        _fr[t] = (pairs, preds, kinds)
    return _fr[t]


def qtype(t):
    for w, k in QWORD.items():
        if w in t:
            return k
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/exp23.txt")
    a = ap.parse_args()
    t0 = time.perf_counter()
    mat, tg = K.load_tables()
    convs = K.load_conversations()
    # 述語の共鳴: 前の発言の述語 → 次の発言の述語（教えの会話で数える）
    cp, cx, cy, n = Counter(), Counter(), Counter(), 0
    for _, turns in convs["train"]:
        F = [frame(t)[1] for t in turns]
        for i in range(len(F) - 1):
            n += 1
            cx.update(F[i])
            cy.update(F[i + 1])
            for x in F[i]:
                for y in F[i + 1]:
                    cp[(x, y)] += 1
    ppmi = lambda x, y: max(0.0, math.log(cp[(x, y)] * n / (cx[x] * cy[y]))) if cp.get((x, y), 0) >= 3 else 0.0
    print(f"述語の組 {len(cp)}（{time.perf_counter() - t0:.0f} 秒）", flush=True)
    rng = random.Random(a.seed)                                         # 実験06・07 と同じ問の作り方
    pool = [t for _, turns in convs["test"] for t in turns]
    cases = []
    for src, turns in convs["test"]:
        for i in range(3, len(turns), 4):
            negs = []
            while len(negs) < 9:
                t2 = rng.choice(pool)
                if t2 != turns[i] and t2 not in turns:
                    negs.append(t2)
            same = [t for j, t in enumerate(turns) if abs(j - i) > 1 and t != turns[i]]
            cases.append((src, turns[max(0, i - 3):i], turns[i], negs, rng.sample(same, min(9, len(same)))))
    rng.shuffle(cases)
    cases = cases[:a.n]

    def feats(u, c):
        uw, cw_ = K.analyze(u)[0], K.analyze(c)[0]
        fu, fc = frame(u), frame(c)
        P = 0.0
        if fu[1] and fc[1]:
            P = sum(ppmi(x, y) for x in fu[1] for y in fc[1]) / math.sqrt(len(fu[1]) * len(fc[1]))
        q = qtype(u)
        return dict(A=K.overlap(uw, cw_), B=mat.resonance(uw, cw_), F=float(len(fu[0] & fc[0])), P=P, Q=float(q is not None and q in fc[2]))
    rec = dict(exp="23", **vars(a), n_pred_pairs=len(cp), runs=[], version=K.__version__)
    for hard in (False, True):
        qs = [[feats(ctx[-1], c) for c in [gold] + (same if hard else negs)] for _, ctx, gold, negs, same in cases]
        half = len(qs) // 2

        def acc(ws, part):
            ok = 0
            for fs in part:
                s = [sum(w * f[k] for k, w in ws.items()) + 1e-6 * rng.random() for f in fs]
                ok += all(s[0] > v for v in s[1:])
            return ok / max(1, len(part))
        base = dict(A=1.0, B=1.0)
        grid = [0.0, 0.1, 0.3, 1.0]
        configs = [("重なり + 共鳴（0.1.0 の芯）", None), ("+ 格の組の重なり F", ["F"]), ("+ 述語の共鳴 P", ["P"]), ("+ 問いの答えの型 Q", ["Q"]), ("+ F・P・Q", ["F", "P", "Q"])]
        for name, keys in configs:
            if keys is None:
                r = dict(hard=hard, name=name, top1=acc(base, qs), weights=base)
            else:
                res = []
                for fit, test in ((qs[:half], qs[half:]), (qs[half:], qs[:half])):
                    best = max(([dict(base, **dict(zip(keys, ws)))] for ws in __import__("itertools").product(grid, repeat=len(keys))),
                               key=lambda wl: acc(wl[0], fit))[0]
                    res.append((acc(best, test), best))
                r = dict(hard=hard, name=name, top1=sum(x[0] for x in res) / 2, weights=[x[1] for x in res])
            rec["runs"].append(r)
            print("同じ会話から" if hard else "ほかの会話から", name, round(100 * r["top1"], 1), flush=True)
        hasq = [i for i, (_, ctx, *_r) in enumerate(cases) if qtype(ctx[-1])]
        rec[f"n_question_{int(hard)}"] = len(hasq)
    rec["total_s"] = time.perf_counter() - t0
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
