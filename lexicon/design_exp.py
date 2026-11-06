"""学ばずに器を作る 2 つの方法の小さな実験（持ち主 2026-11-05）

「学習する前に、周波数をいじって調整する方法と、一旦画素マップに変換する方法の 2 つも小さく実験して検討してほしい」。
器を簡単にした形で考える: 板 1 枚の直後にレンズ（板の位相 φ、口の波 e^{i2πy/n}）→ 出口の場 = FFT(e^{iψ})、ψ = φ + 2πy/n。
  方法 1 周波数で直接組む（周波数をいじって調整）: 出口で欲しい明るさ q から、振幅 √q を周波数に置き、
         「周波数で振幅を √q に直す ↔ 画素で振幅を 1 に直す」を往復して位相を決める（Gerchberg–Saxton）。学び（勾配）は使わない。
  方法 2 画素の地図にしてから位相を取る: 周波数に √q（位相はばらばら）を置いて逆 FFT した画素の地図から、位相だけを取る（往復 1 回）。
  比べ: 勾配で学ぶ（今の方法、600 回）。また、方法 1 で作った位相を学びの初めの板にした場合（何回で追いつくか）。
本物の器（板の後に回折で広がってからレンズ、board_light.BatchDevice）に入れた場合も測る（回折の分だけずれる）。
融合も試す:
  (a) 周波数で融合: 子の q の平均を、方法 1 で組み直す。
  (b) 画素の地図で融合: 子の画素の地図（e^{iψ}）を足し、位相だけを取る。
  (c) 位相の平均（前の試しで壊れた方法）、(d) 光の和（理想）。
物差し: 出口の割合と欲しい割合の差（全変動。観点ごとではなく全体）、光の総量。
実行: python lexicon/design_exp.py --boards <content_boards.json>
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
import board_light as BL  # noqa: E402

VAL0 = BL.VAL0


def full_target(Q, n):
    """出口の周波数 VAL0.. に q を置いた、長さ n の欲しい明るさ（合計 1）。"""
    T = np.zeros((len(Q), n))
    T[:, VAL0:VAL0 + Q.shape[1]] = Q
    return T / T.sum(1, keepdims=True)


def simple_out(psi):
    """簡単な器: 出口の明るさ = |FFT(e^{iψ})|² / n（合計 1）。"""
    F = np.fft.fft(np.exp(1j * psi), axis=-1) / math.sqrt(psi.shape[-1])
    return np.abs(F) ** 2


def gs(T, iters, rng, psi0=None):
    """方法 1: 往復で位相を決める。iters = 1 は方法 2（画素の地図から位相を取るだけ）。"""
    n = T.shape[1]
    amp = np.sqrt(T)
    F = amp * np.exp(1j * rng.uniform(0, 2 * math.pi, T.shape))
    if psi0 is not None:
        F = np.fft.fft(np.exp(1j * psi0), axis=-1) / math.sqrt(n)
        F = amp * np.exp(1j * np.angle(F))
    hist = []
    for it in range(iters):
        u = np.fft.ifft(F, axis=-1) * math.sqrt(n)                   # 画素の地図（振幅がばらばら）
        psi = np.angle(u)                                            # 位相だけ残す（板にできるのは位相だけ）
        G = np.fft.fft(np.exp(1j * psi), axis=-1) / math.sqrt(n)
        F = amp * np.exp(1j * np.angle(G))                           # 周波数で振幅を √q に直す
        if it + 1 in (1, 5, 20, 100, 300):
            hist.append((it + 1, psi.copy()))
    return psi, hist


def err(out, T):
    """出口の割合（欲しい出口の中で正規化）と欲しい割合の差、欲しい出口に届いた光の量。"""
    m = T > 0
    on = (out * m).sum(1)
    a = (out * m) / (on[:, None] + 1e-12)
    return float(np.mean(0.5 * np.abs(a - T).sum(1))), float(on.mean())


def real_device(psi, n, A, plates=1):
    """本物の器（board_light.BatchDevice）の最後の板に設計した位相を置き、ほかの板は 0。"""
    y = np.arange(n)
    phi = psi - 2 * math.pi * y / n
    ph = torch.zeros(len(psi), n, plates)
    ph[:, :, plates - 1] = torch.tensor(phi, dtype=torch.float32)
    dev = BL.BatchDevice(len(psi), n=n, plates=plates, phase=ph)
    with torch.no_grad():
        return dev, dev.bright(A).numpy()


def train_from(Q, n, plates, steps, init_phase=None, seed=0, lr=0.03):
    B, A = Q.shape
    dev = BL.BatchDevice(B, n=n, plates=plates, seed=seed, phase=init_phase)
    q = torch.tensor(Q, dtype=torch.float32)
    opt = torch.optim.Adam(dev.parameters(), lr=lr)
    curve = {}
    for s in range(1, steps + 1):
        W = dev.bright(A)
        p = torch.cat([W, (1 - W.sum(1, keepdim=True)).clamp(min=1e-9)], 1)
        loss = -(torch.cat([q, torch.zeros(B, 1)], 1) * torch.log(p + 1e-9)).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if s in (1, 30, 100, 300, 600):
            with torch.no_grad():
                Wn = dev.bright(A).numpy()
            curve[s] = Wn
    return dev, curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boards", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--n_boards", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(4)
    rng = np.random.default_rng(a.seed)
    boards = [b for b in json.load(open(a.boards, encoding="utf-8")) if b["attrs"]]
    S, SLOT = BL.slots_of(boards)
    A = len(S)
    pick = boards[:a.n_boards]
    Q = np.stack([BL.target(b, SLOT, A) for b in pick])
    T = full_target(Q, a.n)
    res = dict(語板=len(pick), 属性=A, 画素=a.n, 一語の属性の平均=float((Q > 0).sum(1).mean()))

    # 方法 1・2（簡単な器）
    t0 = time.perf_counter()
    psi, hist = gs(T, 300, rng)
    res["方法1_周波数で組む_秒"] = round(time.perf_counter() - t0, 2)
    res["簡単な器"] = {}
    for it, p in hist:
        e, on = err(simple_out(p), T)
        res["簡単な器"][f"往復{it}回" + ("（方法2: 画素の地図から位相）" if it == 1 else "")] = dict(差=round(e, 4), 届いた光=round(on, 3))
    # 本物の器（回折あり）
    res["本物の器_板1枚"] = {}
    for it, p in hist:
        _, W = real_device(p, a.n, A, plates=1)
        e = float(np.mean([0.5 * np.abs(W[i] / W[i].sum() - Q[i]).sum() for i in range(len(Q))]))
        res["本物の器_板1枚"][f"往復{it}回"] = dict(差=round(e, 4), 届いた光=round(float(W.sum(1).mean()), 3))
    # 勾配で学ぶ（比べ）と、方法 1 を初めの板にした場合
    res["勾配で学ぶ_板4枚"] = {}
    t0 = time.perf_counter()
    _, curve = train_from(Q, a.n, 4, 600, seed=a.seed)
    res["勾配で学ぶ_板4枚_秒"] = round(time.perf_counter() - t0, 1)
    for s, W in curve.items():
        res["勾配で学ぶ_板4枚"][f"{s}回"] = round(float(np.mean([0.5 * np.abs(W[i] / W[i].sum() - Q[i]).sum() for i in range(len(Q))])), 4)
    y = np.arange(a.n)
    init = torch.zeros(len(Q), a.n, 4)
    init[:, :, 3] = torch.tensor(hist[-1][1] - 2 * math.pi * y / a.n, dtype=torch.float32)
    _, curve2 = train_from(Q, a.n, 4, 300, init_phase=init, seed=a.seed)
    res["方法1を初めの板にして学ぶ_板4枚"] = {f"{s}回": round(float(np.mean([0.5 * np.abs(W[i] / W[i].sum() - Q[i]).sum() for i in range(len(Q))])), 4)
                                     for s, W in curve2.items()}

    # 融合
    groups = defaultdict(list)
    for i, b in enumerate(pick):
        d = b["attrs"].get("D")
        groups[max(d, key=d.get) if d else "なし"].append(i)
    groups = {k: v for k, v in groups.items() if len(v) >= 2 and k != "なし"}
    names = list(groups)
    Tp = np.stack([T[groups[k]].mean(0) for k in names])
    psi_c = hist[-1][1]
    out_c = simple_out(psi_c)
    fus = {}
    fus["光の和（理想）"] = err(np.stack([out_c[groups[k]].sum(0) for k in names]), Tp)
    fus["周波数で融合（親の q を組み直す、往復 300 回）"] = err(simple_out(gs(Tp, 300, rng)[0]), Tp)
    pix = np.stack([np.exp(1j * psi_c[groups[k]]).sum(0) for k in names])
    fus["画素の地図で融合（地図を足して位相を取る）"] = err(simple_out(np.angle(pix)), Tp)
    fus["画素の地図で融合 + 往復 20 回で直す"] = err(simple_out(gs(Tp, 20, rng, psi0=np.angle(pix))[0]), Tp)
    fus["位相の平均"] = err(simple_out(np.stack([psi_c[groups[k]].mean(0) for k in names])), Tp)
    res["融合_簡単な器"] = {k: dict(差=round(v[0], 4), 届いた光=round(v[1], 3)) for k, v in fus.items()}
    res["融合_親の数"] = len(names)
    out = a.out or os.path.join(os.path.dirname(a.boards), "design_exp.json")
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
