"""介入の研究 実験I14: 差分検証を光で — 2 枚の語板の差を、出口の明るさの引き算・光の干渉だけで測る

持ち主の方針（2026-11-05）: 処理系として、光の一括処理・融合（劣化あり）・消去・差分検証・逆回しの特性を優先する。後半はデジタルの重みを減らす。
I1〜I3 の原因の比べでは、2 枚の語板の差を「観点ごとに明るさを 1 に正規化してから、値ごとの差の絶対値を足す」（デジタル）で測っていた。
ここでは、I1 の語板（光の器、既定の設定で学んだもの）をそのまま使い、差の測り方だけを変える:
  tv      : これまでと同じ（観点ごとに正規化した明るさの差、デジタル）。
  明るさの引き算: 2 枚に同じ光を通し、出口の明るさを差し引く（2 つの受光器の差）。Δ_v = Σ|b1 − b2| / Σ(b1 + b2)（観点の範囲の和、正規化は全体の明るさだけ）。
  干渉   : 2 枚の出口の光の場（複素数）を逆の位相で重ねる（干渉計）。Δ_v = Σ|f1 − f2|² / Σ(|f1|² + |f2|²)。
          同じ場なら 0、重ならない場なら 1、位相が逆なら 2。明るさが同じでも位相が違えば差が出る。
  干渉 + 位相合わせ: 片方の場に位相を 1 つ掛けて（位相器 1 つ）、差がいちばん小さくなる位相で重ねる。
          Δ_v = (Σ|f1|² + Σ|f2|² − 2|Σ f1* f2|) / Σ(|f1|² + |f2|²)。
  物差し: 原因の比べの組で、反例（観点の差 ≥ 0.5 で味の差 < 0.2）・支持（観点の差 ≥ 0.5 で味の差 ≥ 0.5）の分け方が tv とどれだけ一致するか。介入の問いの正解率（否定、1 語抜き）。
結果: ../Project_Kotoyui_private/intervention/results/i14.jsonl
実行: CUDA_VISIBLE_DEVICES= python intervention/i14_light_diff.py --seed 0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i01_small_world as I1  # noqa: E402


def fields(B):
    """語板ごとの、出口の光の場（複素数、属性の範囲）。式は I1 の Devices.brightness と同じで、絶対値の 2 乗をとる前。"""
    D = B.D
    idx = torch.arange(len(B.counts))
    with torch.no_grad():
        ph = D.phase[idx]
        u = torch.nn.functional.pad(D.inp.expand(len(idx), -1), (0, D.L - D.n))
        for k in range(D.plates):
            u = torch.cat([u[:, :D.n] * torch.exp(1j * ph[:, :, k]), u[:, D.n:]], 1)
            u = torch.fft.ifft(torch.fft.fft(u, dim=-1, norm="ortho") * D.H, dim=-1, norm="ortho")
        f = torch.fft.fft(u[:, :D.n], dim=-1, norm="ortho")
    return f[:, I1.VAL0:I1.VAL0 + len(I1.SLOTS)].numpy()


def make_dist(kind, B, F):
    L = B.light()

    def ids(v):
        return [I1.SLOT[(v, x)] for x in I1.VIEWS[v]]

    def dist(B_, r1, r2, v):
        j = ids(v)
        if kind == "明るさの引き算":
            b1, b2 = L[r1, j], L[r2, j]
            return float(np.abs(b1 - b2).sum() / max((b1 + b2).sum(), 1e-12))
        f1, f2 = F[r1, j], F[r2, j]
        e = (np.abs(f1) ** 2).sum() + (np.abs(f2) ** 2).sum()
        if kind == "干渉":
            return float((np.abs(f1 - f2) ** 2).sum() / max(e, 1e-12))
        return float((e - 2 * abs((np.conj(f1) * f2).sum())) / max(e, 1e-12))   # 干渉 + 位相合わせ
    return dist


def qa(B):
    Q = I1.questions()
    per = defaultdict(lambda: [0, 0])
    effs = {}
    for q in Q:
        if q["word"] not in effs:
            effs[q["word"]] = I1.effects(I1.pairs(B, exclude_word=q["word"]), "deny")
        per[q["kind"]][0] += I1.answer(B, q, effs[q["word"]]) == q["ans"]
        per[q["kind"]][1] += 1
    return {k: v[0] / v[1] for k, v in per.items()}


def classify(P):
    out = []
    for _, d, dt, src in P:
        if src != "語板どうし":
            continue
        for v in I1.CAUSE_VIEWS:
            out.append("反例" if d[v] >= 0.5 and dt < 0.2 else ("支持" if d[v] >= 0.5 and dt >= 0.5 else "なし"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_train", type=int, default=50)
    ap.add_argument("--out", default="../Project_Kotoyui_private/intervention/results/i14.jsonl")
    a = ap.parse_args()
    torch.set_num_threads(2)
    t0 = time.perf_counter()
    args = argparse.Namespace(mode="light", theta=0.5, consolidate=1, reread=1, split=1, n_train=a.n_train, seed=a.seed, baselines=0, lm_steps=0)
    I1.run(args)
    B = I1.LAST_B
    F = fields(B)
    rec = dict(exp="I14", seed=a.seed, n_train=a.n_train, n_boards=len(B.counts))
    I1.DIST = None
    base_cls = classify(I1.pairs(B))
    rec["tv"] = dict(qa=qa(B), agree=1.0, n_pairs=len(base_cls))
    for kind in ("明るさの引き算", "干渉", "干渉 + 位相合わせ"):
        I1.DIST = make_dist(kind, B, F)
        cls = classify(I1.pairs(B))
        rec[kind] = dict(qa=qa(B), agree=float(np.mean([x == y for x, y in zip(cls, base_cls)])) if cls else None,
                         n_ref=cls.count("反例"), n_sup=cls.count("支持"))
        sample = [(w, {k: round(x, 2) for k, x in d.items()}, round(dt, 2)) for w, d, dt, s in I1.pairs(B) if s == "語板どうし"][:3]
        rec[kind]["sample"] = sample
    I1.DIST = None
    rec["tv"]["sample"] = [(w, {k: round(x, 2) for k, x in d.items()}, round(dt, 2)) for w, d, dt, s in I1.pairs(B) if s == "語板どうし"][:3]
    rec["total_s"] = time.perf_counter() - t0
    print(json.dumps(rec, ensure_ascii=False, default=float), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=float) + "\n")


if __name__ == "__main__":
    main()
