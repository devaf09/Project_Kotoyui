"""語板 v1f: v1 と同じ語板・同じ確かめで、器の作り方だけを実験の結果に合わせて変える（2026-11-05）

v1 からの変更（design_exp.py, attr_scale_exp.py, v1f_init_exp.py の結果から）:
  1. 板の位相を周波数成分 m = n/4 = 512 個で持つ（学ぶ重みは画素ごとの半分、4,096 / 語）。n/8 以下は壊れる。
  2. 初めの板 = 設計（周波数で振幅を直す往復 100 回）の位相を、最後の板の低い周波数に写したもの。乱数から始めない。
  3. 学びは 300 回（v1 は 600 回）。
  4. 融合の比べに「親の割合を設計して学ぶ（周波数で融合）」を加える。
確かめは v1 と同じ（content_light.py の関数を使う）。
結果: <dir>/light_v1f_s<seed>.json, light_v1f_bright_s<seed>.npy
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
import board_light as BL  # noqa: E402
from content_light import FourierDevice, odd_sentence_test, PRIV  # noqa: E402
from design_exp import full_target, gs  # noqa: E402


def designed_device(Q, n, K, m, seed):
    """設計の位相を最後の板の低い周波数 m 個に写した器。"""
    psi, _ = gs(full_target(Q, n), 100, np.random.default_rng(seed))
    phi = np.unwrap(psi - 2 * math.pi * np.arange(n) / n, axis=1)
    c = np.fft.rfft(phi, axis=1, norm="ortho")[:, :m] / math.sqrt(n)
    dev = FourierDevice(len(Q), n, K, m, seed=seed)
    with torch.no_grad():
        dev.coef.zero_()
        dev.coef[:, K - 1, :, 0] = torch.tensor(c.real, dtype=torch.float32)
        dev.coef[:, K - 1, :, 1] = torch.tensor(c.imag, dtype=torch.float32)
    return dev


def fit(dev, Q, steps, lr=0.03):
    B, A = Q.shape
    q = torch.cat([torch.tensor(Q), torch.zeros(B, 1)], 1)
    opt = torch.optim.Adam(dev.parameters(), lr=lr)
    for _ in range(steps):
        W = dev.bright(A)
        p = torch.cat([W, (1 - W.sum(1, keepdim=True)).clamp(min=1e-9)], 1)
        loss = -(q * torch.log(p + 1e-9)).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return dev


def train_v1f(Q, n, K, m, steps, seed, batch=500):
    B, A = Q.shape
    out = np.zeros((B, A), dtype=np.float32)
    for s in range(0, B, batch):
        dev = fit(designed_device(Q[s:s + batch], n, K, m, seed + s), Q[s:s + batch], steps)
        with torch.no_grad():
            out[s:s + batch] = dev.bright(A).numpy()
        print("  学んだ", s + len(Q[s:s + batch]), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(PRIV, "data", "boards", "v1"))
    ap.add_argument("--n_boards", type=int, default=5000)
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--plates", type=int, default=4)
    ap.add_argument("--m", type=int, default=512)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(4)
    rng = random.Random(a.seed)
    boards = json.load(open(os.path.join(a.dir, "content_boards.json"), encoding="utf-8"))[:a.n_boards]
    boards = [b for b in boards if b["attrs"]]
    triples = json.load(open(os.path.join(a.dir, "test_triples.json"), encoding="utf-8"))
    for b in boards:
        d = b["attrs"].get("D")
        b["func"] = max(d, key=d.get).split(":", 1)[1] if d else "クラスなし"
    cls_of = {b["name"]: b["func"] for b in boards if b["pos"] == "名詞" and b["func"] != "クラスなし"}
    S, SLOT = BL.slots_of(boards)
    A = len(S)
    Q = np.stack([BL.target(b, SLOT, A) for b in boards]).astype(np.float32)
    print("語板", len(boards), "属性", A, flush=True)
    res = dict(設定=vars(a), 語板=len(boards), 属性=A, 器="周波数成分 m 個 + 設計を写した初めの板")
    t0 = time.perf_counter()
    W = train_v1f(Q, a.n, a.plates, a.m, a.steps, a.seed)
    res["学ぶ秒"] = round(time.perf_counter() - t0, 1)
    res["器の重み_1枚"] = a.plates * 2 * a.m
    res["学びの正確さ_全変動"] = round(float(np.mean([BL.tv(W[i], b, S, SLOT) for i, b in enumerate(boards)])), 4)
    res["光の総量"] = round(float(W.sum(1).mean()), 3)
    print("学んだ", res["学ぶ秒"], res["学びの正確さ_全変動"], flush=True)
    res["逆回し"] = {q: BL.reverse_test(W, boards, S, SLOT, qv, rng) for q, qv in
                    [("C（動的）", ["C"]), ("A+C", ["A", "C"]), ("B+C", ["B", "C"])]}
    keep = defaultdict(list)
    for i, b in enumerate(boards):
        keep[b["func"]].append(i)
    sub = sorted(i for k, v in keep.items() if len(v) >= 5 and k != "クラスなし" for i in v[:40])
    sb = [boards[i] for i in sub]
    dev_sub = fit(designed_device(Q[sub], a.n, a.plates, a.m, a.seed), Q[sub], a.steps)
    res["融合"] = BL.fusion_test(dev_sub, W[sub], sb, S, SLOT, Q[sub], a.n, a.steps, a.seed)
    # 周波数で融合: 親の割合（子の平均）を設計し、同じ器で学ぶ
    groups = defaultdict(list)
    for i, b in enumerate(sb):
        groups[BL.label_of(b)].append(i)
    names = [k for k, v in groups.items() if len(v) >= 2]
    truth = np.stack([Q[sub][groups[k]].mean(0) for k in names]).astype(np.float32)
    for st in (0, 100, a.steps):
        dp = fit(designed_device(truth, a.n, a.plates, a.m, a.seed + 9), truth, st)
        with torch.no_grad():
            Wp = dp.bright(A).numpy()
        res["融合"][f"周波数で融合（設計 + {st} 回）"] = round(float(np.mean(
            [0.5 * np.abs(Wp[j] / Wp[j].sum() - truth[j] / truth[j].sum()).sum() for j in range(len(names))])), 3)
    res["消去"] = BL.erase_test(W[sub], sb, S)
    res["崩した文_光"] = odd_sentence_test(W, boards, S, SLOT, triples, cls_of, random.Random(1))
    res["崩した文_数えた割合"] = odd_sentence_test(Q, boards, S, SLOT, triples, cls_of, random.Random(1))
    json.dump(res, open(os.path.join(a.dir, f"light_v1f_s{a.seed}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    np.save(os.path.join(a.dir, f"light_v1f_bright_s{a.seed}.npy"), W.astype(np.float16))
    print(json.dumps({k: v for k, v in res.items() if k != "設定"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
