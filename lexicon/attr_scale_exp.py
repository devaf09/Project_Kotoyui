"""属性を増やしたとき、板の位相を「画素ごと」に持つ器と「周波数成分」で持つ器を比べる（持ち主 2026-11-05）

「属性を増やす = メモリアクセスが増える、これも周波数成分保持で変わるかも、属性数多めの語板で試してほしい」。
語板は合成: 一語あたり約 105 の属性が光る（語板 v1 の試し作りの平均）、属性の人気は Zipf、強さはばらばら。
属性 A を 2,000 / 4,000 / 8,000 と増やし、画素 n は A が収まる 2 のべき（2,048 / 4,096 / 8,192）。
比べる器: 画素ごと（重み = 板 K 枚 × n）と、周波数成分 m 個（重み = K × 2m）。
物差し: 全変動（出口の割合と欲しい割合の差）、光の総量、1 回の学びの秒、器の重み（1 語あたり、バイト）。
実行: python lexicon/attr_scale_exp.py --out <結果.json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import board_light as BL  # noqa: E402
from content_light import FourierDevice  # noqa: E402


def synth(B, A, k, rng):
    pop = 1.0 / np.arange(1, A + 1) ** 0.8
    pop /= pop.sum()
    Q = np.zeros((B, A), dtype=np.float32)
    for i in range(B):
        idx = rng.choice(A, size=k, replace=False, p=pop)
        w = rng.pareto(1.5, size=k) + 0.05
        Q[i, idx] = w / w.sum()
    return Q


def run(Q, n, K, steps, m=None, seed=0, lr=0.03):
    B, A = Q.shape
    dev = FourierDevice(B, n, K, m, seed=seed) if m else BL.BatchDevice(B, n=n, plates=K, seed=seed)
    opt = torch.optim.Adam(dev.parameters(), lr=lr)
    q = torch.tensor(Q)
    qz = torch.cat([q, torch.zeros(B, 1)], 1)
    t0 = time.perf_counter()
    for _ in range(steps):
        W = dev.bright(A)
        p = torch.cat([W, (1 - W.sum(1, keepdim=True)).clamp(min=1e-9)], 1)
        loss = -(qz * torch.log(p + 1e-9)).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    sec = (time.perf_counter() - t0) / steps
    with torch.no_grad():
        W = dev.bright(A).numpy()
    tv = float(np.mean(0.5 * np.abs(W / W.sum(1, keepdims=True) - Q).sum(1)))
    n_w = K * 2 * m if m else K * n
    return dict(器の重み_1語=n_w, 重みのバイト_1語=n_w * 4, 全変動=round(tv, 4),
                光の総量=round(float(W.sum(1).mean()), 3), 学び1回の秒=round(sec, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--attrs", type=int, nargs="*", default=[2000, 4000, 8000])
    ap.add_argument("--k", type=int, default=105)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--fracs", type=float, nargs="*", default=[1 / 32, 1 / 8, 1 / 4, 1 / 2])
    a = ap.parse_args()
    torch.set_num_threads(4)
    rng = np.random.default_rng(0)
    res = dict(設定=vars(a))
    for A in a.attrs:
        n = 1 << int(np.ceil(np.log2(A + BL.VAL0)))
        Q = synth(a.B, A, a.k, rng)
        r = {"画素ごと": run(Q, n, a.K, a.steps)}
        print(A, n, "画素ごと", r["画素ごと"], flush=True)
        for f in a.fracs:
            m = int(n * f)
            r[f"周波数成分{m}"] = run(Q, n, a.K, a.steps, m=m)
            print(A, n, m, r[f"周波数成分{m}"], flush=True)
        res[f"属性{A}_画素{n}"] = r
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
