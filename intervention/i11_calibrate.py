"""介入の研究 実験I11: 確率に経験の量を入れて、言い方（きっと・たぶん・わからない）と正解率を合わせる（較正）

I10 では、親（科）の明るさをそのまま確率にしたため、知っている子が 4 つしかないのに言い切りすぎた（「きっと」の正解 0.63）。
持ち主（2026-11-05）「自覚や不安感は確率で持つべき。このままだとなんでも『たぶん』になってしまう」。
方法（I10 と同じ世界・同じ文の作り方）:
  - 子の分布 = 子の光の器の明るさを観点ごとに 1 に正規化したもの（子 1 つ = 1 票）。
  - 親の想定 = (子の分布の和 + α × 祖先の分布) / (子の数 + α)。α = 祖先をどれだけの子の数ぶん信じるか（経験の量に応じた薄め）。
    子が多い親ほど親の分布が強く、子が少ない親ほど祖先に寄る。
  - α の決め方: 固定（0・0.5・1・2・4）と、**自分で決める**（知っている子を 1 つずつ抜いて、親から想定したときに本当の値に付く確率の対数が最も大きくなる α を選ぶ。
    隠した品種は使わない）。
  - 言い方の区切りも自分で決める: 知っている子を抜いた想定で、1 位の確率が p 以上のときの正解率が 0.8 以上になる最小の p を「きっと」、
    0.5 以上になる最小の p を「たぶん」の区切りにする（words_self）。固定の区切り（0.7・0.4）とも比べる。
  - 物差し: 本当の値に付けた確率の対数（平均、大きいほど良い）、較正の誤差（確率の区切りごとに「付けた確率の平均」と「正解率」の差の平均、小さいほど良い）、
    言い方ごとの正解率（きっと ≥ 0.7、たぶん 0.4〜0.7、わからない < 0.4）。
結果: results/i11.jsonl
実行: CUDA_VISIBLE_DEVICES= python intervention/i11_calibrate.py --seed 0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from i10_hierarchy import FAMILIES, VIEWS, SLOTS, SLOT, make_sentences, read, fit_board, word_of  # noqa: E402

ALPHAS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]


def vdist(b, v):
    ids = [SLOT[(v, x)] for x in VIEWS[v]]
    p = np.asarray(b)[ids] + 1e-9
    return p / p.sum()


def predict(kids, root, v, alpha):
    s = sum(vdist(k, v) for k in kids) if kids else np.zeros(len(VIEWS[v]))
    p = (s + alpha * root[v]) / (len(kids) + alpha) if (kids or alpha) else np.ones(len(VIEWS[v])) / len(VIEWS[v])
    return p / p.sum()


def p_true(p, v, truth):
    return float(sum(p[VIEWS[v].index(x)] for x in truth))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--out", default="../Project_Kotoyui_private/intervention/results/i11.jsonl")
    a = ap.parse_args()
    torch.set_num_threads(2)
    rng = random.Random(a.seed)
    t0 = time.perf_counter()
    fams = list(FAMILIES)
    rows = defaultdict(list)                                        # 方法 → [(本当の値の確率, 1 位の確率, 1 位が正解か)]
    chosen, cuts = [], []
    for hold_i in range(5):
        boards, parent_of = {}, {}
        hidden = {f: list(FAMILIES[f])[hold_i] for f in fams}
        for f in fams:
            for c, at in FAMILIES[f].items():
                if c == hidden[f]:
                    parent_of[c] = f
                    continue
                cnt = np.zeros(len(SLOTS))
                for toks in make_sentences(c, f, at, a.n, rng):
                    ch, par, attrs = read(toks, fams)
                    if par:
                        parent_of[ch] = par
                    for s in attrs:
                        cnt[SLOT[s]] += 1
                boards[c] = fit_board(cnt, a.seed * 100 + hold_i, a.steps)
        kids = {f: [c for c in boards if parent_of.get(c) == f] for f in fams}
        allk = [c for f in fams for c in kids[f]]
        root = {v: (sum(vdist(boards[c], v) for c in allk) / max(len(allk), 1)) for v in VIEWS}
        # α を自分で決める: 知っている子を 1 つずつ抜いて、親から想定したときの本当の値（その子の明るさの主な値）の対数確率
        best = None
        for al in ALPHAS:
            ll = []
            for f in fams:
                for c in kids[f]:
                    others = [boards[k] for k in kids[f] if k != c]
                    for v in VIEWS:
                        tgt = vdist(boards[c], v)
                        p = predict(others, root, v, al)
                        ll.append(float((tgt * np.log(p + 1e-6)).sum()))
            m = float(np.mean(ll))
            if best is None or m > best[0]:
                best = (m, al)
        chosen.append(best[1])
        # 言い方の区切りも自分で決める: 知っている子を抜いた想定で、1 位の確率が p 以上のときの正解率が 0.8 以上になる最小の p を「きっと」、0.5 以上を「たぶん」の区切りにする
        loo = []
        for f in fams:
            for c in kids[f]:
                others = [boards[k] for k in kids[f] if k != c]
                for v in VIEWS:
                    pr = predict(others, root, v, best[1])
                    tv = vdist(boards[c], v)
                    top = int(np.argmax(pr))
                    loo.append((float(pr[top]), tv[top] >= 0.3))
        def cut(target):
            for th in np.arange(0.3, 1.0, 0.05):
                sel = [ok for pt, ok in loo if pt >= th]
                if len(sel) >= 5 and np.mean(sel) >= target:
                    return float(th)
            return 1.01
        cuts.append((cut(0.8), cut(0.5)))
        for f in fams:
            c = hidden[f]
            if parent_of.get(c) != f:
                continue
            kb = [boards[k] for k in kids[f]]
            for v in VIEWS:
                truth = FAMILIES[f][c][v]
                for name, al in [(f"α={x}", x) for x in ALPHAS] + [("α を自分で決める", best[1])]:
                    p = predict(kb, root, v, al)
                    top = int(np.argmax(p))
                    rows[name].append((p_true(p, v, truth), float(p[top]), VIEWS[v][top] in truth, cuts[-1]))
    summary = {}
    for name, rs in rows.items():
        cut_words = defaultdict(list)                               # 自分で決めた区切りでの言い方
        for pt, ptop, ok, (c_sure, c_maybe) in rs:
            cut_words["きっと" if ptop >= c_sure else ("たぶん" if ptop >= c_maybe else "わからない")].append(ok)
        rs = [r[:3] for r in rs]
        bins = defaultdict(list)
        for pt, ptop, ok in rs:
            bins[min(int(ptop * 5), 4)].append((ptop, ok))
        ece = sum(len(b) * abs(np.mean([x[0] for x in b]) - np.mean([x[1] for x in b])) for b in bins.values()) / len(rs)
        words = defaultdict(list)
        for pt, ptop, ok in rs:
            words[word_of(ptop)].append(ok)
        summary[name] = dict(logp=float(np.mean([math.log(max(pt, 1e-6)) for pt, _, _ in rs])), p_true=float(np.mean([pt for pt, _, _ in rs])),
                             top1=float(np.mean([ok for _, _, ok in rs])), ece=float(ece),
                             words={k: (float(np.mean(v)), len(v)) for k, v in words.items()},
                             words_self={k: (float(np.mean(v)), len(v)) for k, v in cut_words.items()})
    rec = dict(exp="I11", seed=a.seed, n=a.n, summary=summary, chosen_alpha=chosen, cuts=cuts, total_s=time.perf_counter() - t0)
    print(json.dumps(rec, ensure_ascii=False), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
