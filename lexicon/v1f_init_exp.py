"""語板 v1f の器の作り方を決める小さな比べ（2026-11-05）

v1 の実験から: (1) 周波数成分 n/4 で重みは半分、(2) 設計（往復）を初めの板にすると学びは 6 分の 1。
両方を合わせる形を比べる。
  画素ごと・乱数から 600 回（v1）
  画素ごと・設計から 100 回
  成分 n/4・乱数から 600 回
  成分 n/4・設計の位相を低い周波数に写して初めにする（写すと崩れるかを見る）
  設計 + 成分の補正: 最後の板 = 設計の位相（q から毎回作り直せるので重みに数えない）+ 成分 m 個の補正（学ぶ）
実行: python lexicon/v1f_init_exp.py --boards <content_boards.json>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import board_light as BL  # noqa: E402
from content_light import FourierDevice  # noqa: E402
from design_exp import full_target, gs  # noqa: E402


class BaseFourierDevice(FourierDevice):
    """板の位相 = 決まった土台（設計）+ 低い周波数成分 m 個の補正。"""

    def __init__(self, B, n, plates, m, base, seed=0, zero=True):
        super().__init__(B, n, plates, m, seed=seed)
        self.register_buffer("base", base)
        if zero:
            with torch.no_grad():
                self.coef.zero_()

    @property
    def phase(self):
        return self.base + FourierDevice.phase.fget(self)


def design_phase(Q, n, plates, iters=100, seed=0):
    T = full_target(Q, n)
    psi, _ = gs(T, iters, np.random.default_rng(seed))
    base = torch.zeros(len(Q), n, plates)
    base[:, :, plates - 1] = torch.tensor(psi - 2 * math.pi * np.arange(n) / n, dtype=torch.float32)
    return base


def fit(dev, Q, steps, lr=0.03):
    B, A = Q.shape
    q = torch.cat([torch.tensor(Q), torch.zeros(B, 1)], 1)
    opt = torch.optim.Adam(dev.parameters(), lr=lr)
    t0 = time.perf_counter()
    for _ in range(steps):
        W = dev.bright(A)
        p = torch.cat([W, (1 - W.sum(1, keepdim=True)).clamp(min=1e-9)], 1)
        loss = -(q * torch.log(p + 1e-9)).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        W = dev.bright(A).numpy()
    tv = float(np.mean(0.5 * np.abs(W / W.sum(1, keepdims=True) - Q).sum(1)))
    return dict(全変動=round(tv, 4), 光の総量=round(float(W.sum(1).mean()), 3), 秒=round(time.perf_counter() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boards", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--K", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(4)
    boards = [b for b in json.load(open(a.boards, encoding="utf-8")) if b["attrs"]]
    S, SLOT = BL.slots_of(boards)
    A = len(S)
    Q = np.stack([BL.target(b, SLOT, A) for b in boards[:a.B]]).astype(np.float32)
    n, K, B = a.n, a.K, len(Q)
    t0 = time.perf_counter()
    base = design_phase(Q, n, K)
    res = dict(語板=B, 属性=A, 設計の秒=round(time.perf_counter() - t0, 1))
    m4, m8 = n // 4, n // 8
    runs = {
        "画素ごと・乱数から600回": (lambda: BL.BatchDevice(B, n=n, plates=K, seed=0), 600, K * n),
        "画素ごと・設計から100回": (lambda: BL.BatchDevice(B, n=n, plates=K, phase=base.clone()), 100, K * n),
        f"成分{m4}・乱数から600回": (lambda: FourierDevice(B, n, K, m4, seed=0), 600, K * 2 * m4),
        f"設計+成分{m4}の補正・100回": (lambda: BaseFourierDevice(B, n, K, m4, base), 100, K * 2 * m4),
        f"設計+成分{m4}の補正・300回": (lambda: BaseFourierDevice(B, n, K, m4, base), 300, K * 2 * m4),
        f"設計+成分{m8}の補正・300回": (lambda: BaseFourierDevice(B, n, K, m8, base), 300, K * 2 * m8),
    }
    # 設計の位相を低い周波数に写して初めにする
    proj = torch.tensor(np.fft.rfft(np.unwrap(base[:, :, K - 1].numpy(), axis=1), axis=1, norm="ortho")[:, :m4] / math.sqrt(n))

    def proj_dev():
        d = FourierDevice(B, n, K, m4, seed=0)
        with torch.no_grad():
            d.coef.zero_()
            d.coef[:, K - 1, :, 0] = proj.real.float()
            d.coef[:, K - 1, :, 1] = proj.imag.float()
        return d
    runs[f"成分{m4}・設計を写して100回"] = (proj_dev, 100, K * 2 * m4)
    for name, (mk, steps, nw) in runs.items():
        r = fit(mk(), Q, steps)
        r["学ぶ重み_1語"] = nw
        res[name] = r
        print(name, r, flush=True)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
