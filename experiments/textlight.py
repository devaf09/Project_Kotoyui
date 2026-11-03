"""光で言葉を扱う部品（実験02 から）: 単位の列を帯に並べて光の場に入れる光路と、分担の 2DTG

出どころ: SN2-Parts-2D の experiments/text/exp40_inputs.py の EncNet（入れ方 "bands"）と exp58_split.py の分担の 2DTG（担当の決め打ち = route、実験70）を、
          同じ著作権者の MIT のコードから、使う形だけに書き直したもの（光の部品 sn2_parts_2d.py に入っていないため。SN2-Parts-2D 側の課題、NOTES.md）。
入れ方: 場（64 × 64）を縦に n 本の帯に分け、位置 m の帯に、その単位の番号で決まる向きの平面波を置く。位置ごとの明るさを学習する（直前ほど明るく、SN2 実験33）。
        向きは番号 × 黄金比の歩幅で 64 × 64 の向きに散らす（SN2 の stride と同じ）。
読み方: レンズの後の窓 K 個の明るさの模様と、答えごとの符号（学習する）の近さ（符号読み、SN2 実験22・42）。
"""
from __future__ import annotations

import importlib.util
import math
import os

import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_spec = importlib.util.spec_from_file_location("sn2_parts_2d_v0_11_0", os.path.join(ROOT, "vendor", "sn2_parts_2d_v0_11_0.py"))
p2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p2)
L = 64


def stride(N):
    """番号を散らす歩幅（N と互いに素な、黄金比に近い奇数）。SN2-Parts-2D の exp31_text2d.stride と同じ。"""
    g = int(N * 0.6180339887) | 1
    while math.gcd(g, N) != 1:
        g += 2
    return g


class BandNet(torch.nn.Module):
    """単位の番号 (B, n) → 光の場（n 本の帯）→ 板 ① × layers → レンズ → 窓 K 個 → 答え C 個の点数（符号読み）。
    学習する数: 光の部分 = 板の A・φ、位置ごとの明るさ n 個。デジタルの部分 = 符号 C × K、温度、窓の背景光（Net2D の bias、使わないが数える）。"""

    def __init__(self, n, V, C, layers=8, patches=16, K=32, seed=0):
        super().__init__()
        self.n = n
        self.net = p2.Net2D(L, K, ["xy"] * layers, patches=patches, seed=seed)
        self.codes = torch.nn.Parameter(torch.randn(C, K, generator=torch.Generator().manual_seed(seed)))
        self.temp = torch.nn.Parameter(torch.tensor(10.0))
        self.w = torch.nn.Parameter(0.3 * (torch.arange(n, dtype=torch.float32) - n + 1))
        pos = (torch.arange(V) * stride(L * L)) % (L * L)
        self.register_buffer("kx", (pos % L).float())
        self.register_buffer("ky", (pos // L).float())
        yy = torch.arange(L, dtype=torch.float32)
        self.register_buffer("yy", yy[:, None].expand(L, L).clone())
        self.register_buffer("xx", yy[None, :].expand(L, L).clone())
        M = (torch.arange(L)[None, :] * n // L == torch.arange(n)[:, None]).float()[:, :, None].expand(n, L, L)
        self.register_buffer("M", M.contiguous())

    def field(self, X):
        ph = 2 * math.pi * (self.kx[X][..., None, None] * self.xx + self.ky[X][..., None, None] * self.yy) / L
        W = torch.polar(torch.ones_like(ph), ph)
        a = torch.exp(self.w)
        a = a / a.pow(2).mean().sqrt()
        u = (W * self.M[None] * a[None, :, None, None]).sum(1)
        return u / (u.abs().pow(2).mean((-2, -1), keepdim=True) + 1e-12).sqrt()

    def forward(self, X):
        I = self.net.screen(self.field(X))
        return p2.code_scores(p2.read_windows(I, self.net.det), self.codes, 1.0) * self.temp

    def params(self):
        """(光の部分, デジタルの部分) の学習する数。"""
        light = self.net.A.numel() + self.net.phi.numel() + self.w.numel()
        digital = self.codes.numel() + self.temp.numel() + self.net.gain.numel() + self.net.bias.numel()
        return light, digital


class Split2DTG(torch.nn.Module):
    """分担の 2DTG: 直前の単位の番号の余り（G 個）で担当を決め打ちし（route、SN2 実験70）、担当の BandNet だけが答える。
    ハードでは「どの光路を照らすか」のスイッチ。各 2DTG は自分の担当の文脈だけで別々に学ぶ。"""

    def __init__(self, n, V, C, G=16, layers=8, patches=16, K=32, seed=0):
        super().__init__()
        self.G = G
        self.grains = torch.nn.ModuleList([BandNet(n, V, C, layers, patches, K, seed * 100 + g) for g in range(G)])

    def group(self, X):
        return X[:, -1] % self.G

    def logits(self, X):
        out = None
        g = self.group(X)
        for gi in range(self.G):
            mk = g == gi
            if mk.any():
                o = self.grains[gi](X[mk])
                if out is None:
                    out = torch.empty(len(X), o.shape[1], device=o.device)
                out[mk] = o
        return out

    def params(self):
        ps = [m.params() for m in self.grains]
        return sum(p[0] for p in ps), sum(p[1] for p in ps)


def train_split(model, X, y, steps=3000, lr=0.01, batch=256, seed=0, dev="cuda", log=None):
    """各 2DTG を、自分の担当の文脈だけで別々に学ぶ（Adam、学習率 lr → 0）。戻り値: 学習の時間（秒）。"""
    import time
    g = model.group(X)
    t = 0.0
    for gi, m in enumerate(model.grains):
        own = (g == gi).nonzero().squeeze(1)
        if len(own) == 0:
            continue
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        gen = torch.Generator().manual_seed(seed * 100 + gi)
        t0 = time.perf_counter()
        for _ in range(steps):
            i = own[torch.randint(len(own), (batch,), generator=gen)]
            loss = torch.nn.functional.cross_entropy(m(X[i].to(dev)), y[i].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
        if str(dev).startswith("cuda"):
            torch.cuda.synchronize()
        t += time.perf_counter() - t0
        if log:
            log(f"2DTG {gi + 1}/{model.G} 学習 {len(own)} 件 {t:.0f} 秒")
    return t


# =============================================================================================
# 4f 系 ＋ 光の非線形（実験03 から）
# =============================================================================================
# 出どころ: SN2-Parts-2D の nl_parts.py（NL の sqrt）、experiments/exp75_4f.py（Multi4f）、experiments/text/exp84_word4f.py（Word4f）、
#           exp40_inputs.py の EncNet の入れ方 "spread"（コミット e52ab26 のころ、同じ著作権者の MIT）を、使う形だけに書き直したもの。
#           4f 系 = レンズ → 周波数の面の板（振幅と位相を学習）→ レンズ。光が通るだけで畳み込みを計算する（CNN の畳み込みの層と同じ）。
#           段の間で明るさを測って光を出し直す（平方根 = 位相を捨てる非線形）。SN2 の単語の課題で、光路 1 本 36.1%（3-gram 36.4%）、
#           足し合わせ 37.4%（GRU 36.9% を超えた、SN2 実験84・85）。


class SpreadField(torch.nn.Module):
    """単位の番号 (B, n) → 光の場。位置ごとに決まったでたらめの位相の模様を掛けた平面波を、面全体に重ねる（拡散、SN2 実験78 で帯より 10 点良い）。"""

    def __init__(self, n, V):
        super().__init__()
        self.n = n
        self.w = torch.nn.Parameter(0.3 * (torch.arange(n, dtype=torch.float32) - n + 1))   # 位置ごとの明るさ（光の部分）
        pos = (torch.arange(V) * stride(L * L)) % (L * L)
        self.register_buffer("kx", (pos % L).float())
        self.register_buffer("ky", (pos // L).float())
        yy = torch.arange(L, dtype=torch.float32)
        self.register_buffer("yy", yy[:, None].expand(L, L).clone())
        self.register_buffer("xx", yy[None, :].expand(L, L).clone())
        gr = torch.Generator().manual_seed(12345)
        self.register_buffer("ref", 2 * math.pi * torch.rand(n, L, L, generator=gr))

    def forward(self, X):
        ph = 2 * math.pi * (self.kx[X][..., None, None] * self.xx + self.ky[X][..., None, None] * self.yy) / L + self.ref[None]
        a = torch.exp(self.w)
        a = a / a.pow(2).mean().sqrt()
        u = (torch.polar(torch.ones_like(ph), ph) * a[None, :, None, None]).sum(1)
        return u / (u.abs().pow(2).mean((-2, -1), keepdim=True) + 1e-12).sqrt()


class Multi4f(torch.nn.Module):
    """4f 系 × 幅 width を stages 段。段 s: 周波数の面に幅の数だけ板を並べ、出口の場を複素数の重みで足す → 平方根の非線形（最後の段の後は除く）
    → レンズ → 明るさ（平均 1）。"""

    def __init__(self, stages=6, width=8, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.stages = stages
        self.phase = torch.nn.Parameter(0.1 * torch.randn(stages, width, L, L, generator=g))
        self.logamp = torch.nn.Parameter(torch.zeros(stages, width, L, L))
        self.mix = torch.nn.Parameter(torch.randn(stages, width, 2, generator=g) / width ** 0.5)

    def screen(self, u):
        for s in range(self.stages):
            U = torch.fft.fft2(u, norm="ortho")[:, None]
            filt = torch.polar(torch.exp(self.logamp[s].clamp(max=3)), self.phase[s])[None]
            v = torch.fft.ifft2(U * filt, norm="ortho")
            w = torch.complex(self.mix[s, :, 0], self.mix[s, :, 1])
            u = p2.normalize((v * w[None, :, None, None]).sum(1))
            if s < self.stages - 1:     # 平方根: 明るさを測って、その平方根の振幅・位相 0 の光を出し直す
                I = p2.intensity(u)
                u = p2.normalize((I / (I.mean((-2, -1), keepdim=True) + 1e-12)).clamp(min=0).sqrt().to(torch.complex64))
        I = p2.intensity(p2.lens(u))
        return I / (I.mean((-2, -1), keepdim=True) + 1e-12)


class Text4f(torch.nn.Module):
    """単位の列 → 拡散 → 4f 系 + 平方根 → 窓 K 個 → 答え C 個の符号読み（SN2 の Word4f と同じ形）。"""

    def __init__(self, n, V, C, K=256, stages=6, width=8, seed=0):
        super().__init__()
        self.field = SpreadField(n, V)
        self.body = Multi4f(stages, width, seed)
        self.register_buffer("det", p2.window_grid(L, L, K, 4))
        self.codes = torch.nn.Parameter(torch.randn(C, K, generator=torch.Generator().manual_seed(seed)))
        self.temp = torch.nn.Parameter(torch.tensor(10.0))

    def forward(self, X):
        w = p2.read_windows(self.body.screen(self.field(X)), self.det)
        return p2.code_scores(w, self.codes, 1.0) * self.temp

    def params(self):
        light = self.field.w.numel() + sum(q.numel() for q in self.body.parameters())
        return light, self.codes.numel() + self.temp.numel()


class Split4f(torch.nn.Module):
    """分担の 2DTG の中身を Text4f にしたもの（直前の単位の番号の余り G 個で担当を決め打ち）。Split2DTG と同じ使い方。"""

    def __init__(self, n, V, C, G=4, K=256, stages=4, width=4, seed=0):
        super().__init__()
        self.G = G
        self.grains = torch.nn.ModuleList([Text4f(n, V, C, K, stages, width, seed * 100 + g) for g in range(G)])

    group = Split2DTG.group
    logits = Split2DTG.logits
    params = Split2DTG.params
