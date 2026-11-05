"""介入の研究 実験I13: 融合（劣化あり）— 子の語板を融合して親を作るとき、劣化があっても想定が保てるか

持ち主の方針（2026-11-05）: 光の融合（劣化あり）を優先する特性に入れる。
I10 では、親の語板 = 子の器の明るさの和（劣化なし）だった。ここでは融合に劣化を入れる:
  - 融合のしかた:
      明るさの和（インコヒーレント）: 子の出口の明るさを足す。劣化 = 子ごとの損失（η 倍）と、明るさのむら（各属性に 1 + σ·乱数 を掛ける）。
      場の和（コヒーレント）: 子の出口の光の場（複素数）をそのまま重ねてから明るさにする。子どうしの位相がそろっていないと干渉して崩れる。
  - 深さ: 品種 → 科（1 回の融合）、科 → 果物全体（2 回目の融合。劣化が重なる）。
  - 物差し: I10 と同じく、隠した品種の本当の値に付けた確率（親から想定）と、果物全体から想定した確率。
結果: ../Project_Kotoyui_private/intervention/results/i13.jsonl
実行: CUDA_VISIBLE_DEVICES= python intervention/i13_fusion.py --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from i02_reverse import PortDevice, VAL0  # noqa: E402
from i10_hierarchy import FAMILIES, VIEWS, SLOTS, SLOT, make_sentences, read  # noqa: E402


def fit_board_field(counts, seed, steps):
    torch.manual_seed(seed)
    dev = PortDevice(1, seed=seed)
    q = torch.tensor(counts / counts.sum(), dtype=torch.float32)
    opt = torch.optim.Adam(dev.parameters(), lr=0.02)
    for _ in range(steps):
        W = (dev.forward_field(dev.ports).abs()[:, VAL0:VAL0 + len(SLOTS)] ** 2)[0]
        p = torch.cat([W, (1 - W.sum()).clamp(min=1e-9)[None]])
        loss = -(torch.cat([q, torch.zeros(1)]) * torch.log(p + 1e-9)).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return dev.forward_field(dev.ports)[0, VAL0:VAL0 + len(SLOTS)].numpy()


def vprob(b, v):
    ids = [SLOT[(v, x)] for x in VIEWS[v]]
    p = np.asarray(b)[ids] + 1e-9
    return p / p.sum()


def fuse(items, how, eta, sigma, rng):
    """items: 子の場（複素数）のリスト。戻り値: 融合した親の「場」と「明るさ」。"""
    if how == "場の和":
        f = sum(np.sqrt(eta) * x for x in items)
        b = np.abs(f) ** 2
    else:
        b = sum(eta * np.abs(x) ** 2 for x in items)
        f = np.sqrt(b)                                              # 次の融合へは明るさの平方根を場として渡す（位相は失われる）
    b = b * np.clip(1 + sigma * rng.standard_normal(b.shape), 0, None)   # 明るさのむら
    return f if how == "場の和" else np.sqrt(b), b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--out", default="../Project_Kotoyui_private/intervention/results/i13.jsonl")
    a = ap.parse_args()
    torch.set_num_threads(2)
    rng = random.Random(a.seed)
    nrng = np.random.default_rng(a.seed)
    t0 = time.perf_counter()
    fams = list(FAMILIES)
    conds = [("明るさの和", 1.0, 0.0), ("明るさの和", 0.5, 0.0), ("明るさの和", 1.0, 0.2), ("明るさの和", 1.0, 0.5), ("明るさの和", 0.5, 0.5),
             ("場の和", 1.0, 0.0), ("場の和", 1.0, 0.2)]
    res = defaultdict(lambda: defaultdict(list))
    for hold_i in range(5):
        fieldsd, parent_of = {}, {}
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
                fieldsd[c] = fit_board_field(cnt, a.seed * 100 + hold_i, a.steps)
        for how, eta, sigma in conds:
            key = f"{how} η={eta} σ={sigma}"
            par_f, par_b = {}, {}
            for f in fams:
                kids = [fieldsd[c] for c in fieldsd if parent_of.get(c) == f]
                par_f[f], par_b[f] = fuse(kids, how, eta, sigma, nrng) if kids else (np.zeros(len(SLOTS)), np.zeros(len(SLOTS)))
            _, root_b = fuse([par_f[f] for f in fams], how, eta, sigma, nrng)   # 2 回目の融合（劣化が重なる）
            for f in fams:
                c = hidden[f]
                for v in VIEWS:
                    truth = FAMILIES[f][c][v]
                    for lvl, b in (("親", par_b[f]), ("祖先", root_b)):
                        p = vprob(b, v)
                        res[key][lvl].append(float(sum(p[VIEWS[v].index(x)] for x in truth)))
    summary = {k: {lvl: float(np.mean(v)) for lvl, v in d.items()} for k, d in res.items()}
    rec = dict(exp="I13", seed=a.seed, n=a.n, summary=summary, total_s=time.perf_counter() - t0)
    print(json.dumps(rec, ensure_ascii=False), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
