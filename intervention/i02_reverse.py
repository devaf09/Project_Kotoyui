"""介入の研究 実験I2: 境目の分からない 2 つの物を 1 枚の語板に融合し、属性を 1 つずつ逆から入れて、関連と分離を調べる

持ち主の案（2026-11-04）: 「リンゴに複数の要素を持たせて、逆からそれぞれ 1 つずつ属性を入れていくと、出力がどうなるか。
  A と B の境目が分かっていない 2 つの板を融合して、共通する属性・しない属性を逆から入れて、関連を調べる（特に分離）」。
  前の案: 「特定のグレインに逆向きに属性を入れると 2 つの光の経路にまとまる場合、そのグレインは 2 つの関係を持つから 2 つに分ける」。
器: 実験09・I1 の光の器（画素ごとの位相の板 K 枚 + 回折、0 を詰めた 2 倍の場、レンズ）。入口に「口」（周波数 1〜M の波）を M 個持てる。
    順: 口 m の波 → 板 → レンズ → 属性 j の明るさ（周波数 VAL0 + j）。
    逆: 属性 j の点の光をレンズの後ろから入れ、板を逆にたどる（随伴）→ 入口の場 B_j（縦 n 画素）。
        口 m に戻る光の量 |<口 m|B_j>|² は、順の明るさと同じ（器の性質）。
    往復: 逆で入口に戻った光を、(a) 口の成分だけ残して（口に射影して）、(b) そのまま、もう一度順に通す → 属性 j から「関連して出てくる属性」。
融合のしかた（板の学び方）:
    marg: 口 1 つ、2 つの物の文を混ぜた属性の割合を学ぶ（I1 と同じ。境目の情報は入らない）。
    mix:  口 M 個、文ごとに「どの口から来たか」を隠れたまま、文の属性の組の起こりやすさを最大にする（混合のモデル）。
          文 s の起こりやすさ = Σ_m (1/M) Π_{属性 a ∈ s} 明るさ_m(a)。境目は教えない（文がどちらの物かは使わない）。
組（境目の正解は評価にだけ使う）:
    果物と人（リンゴ/果物 + リンゴ/人、共通の属性なし）、熟した・未熟（トマト・イチゴ・バナナ、共通: 種類・形・される）、
    3 つ（リンゴ/果物 + リンゴ/人 + レモン）、比べ: 1 つの物を半分に分けて融合（境目のない融合。リンゴ/果物・トマト/完熟）。
分けるかどうかの判断（口 2 つ）: 口ごとの明るさで、両方の口に光がある観点のうち値の割合が半分以上違うもの（競合する観点）と、
    片方の口にしか光がない観点（片方だけの観点）を数える。競合 2 つ以上、または競合 1 つ + 片方だけ 1 つ以上なら「分ける」。
    （色違いのリンゴ = 競合 1・片方だけ 0 → 分けない。文の型で割れただけ = 競合 0 → 分けない。I1 の見直しと同じ考え）
物差し:
    分離の度合い Δ: 属性ごとの「逆の手がかり」（口の分布、または入口の場）の似ている度合いの表から、2 つ（3 つの組は 3 つ）に分け（スペクトル分割）、
      同じ組の中の似ている度合いの平均 − 違う組の間の平均。境目のある融合で大きく、比べ（半分ずつ）で小さければ、分けるかどうかを自覚できる。
    分け方の正しさ: 片方の物だけの属性を、正しく 2 つに分けられたか（純度）。共通の属性は口の分布のエントロピー（両方に広がるか）。
    往復の関連: 属性 j から往復で出てくる明るさのうち、同じ物の属性に出る割合（j 自身と共通の属性は除く）。
結果: results/i02.jsonl
実行: CUDA_VISIBLE_DEVICES= python intervention/i02_reverse.py --seed 0
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from i01_small_world import SENSES, SLOTS, SLOT, attrs_of, make_sentence  # noqa: E402

VAL0 = 8


class PortDevice(torch.nn.Module):
    """口 M 個の光の器。式は実験09 の Devices と同じ（1 台）。"""

    def __init__(self, M, n=64, plates=8, gap=32.0, seed=0):
        super().__init__()
        self.M, self.n, self.plates, self.L = M, n, plates, 2 * n
        fx = torch.fft.fftfreq(self.L)
        self.register_buffer("H", torch.exp(1j * gap * torch.sqrt((2 * math.pi) ** 2 - (2 * math.pi * fx) ** 2)).to(torch.complex64))
        g = torch.Generator().manual_seed(seed)
        self.phase = torch.nn.Parameter(math.pi * (torch.rand(n, plates, generator=g) < 0.5).float())
        y = torch.arange(n, dtype=torch.float32)
        self.register_buffer("ports", torch.stack([torch.exp(1j * 2 * math.pi * (m + 1) * y / n) / math.sqrt(n) for m in range(M)]).to(torch.complex64))

    def forward_field(self, u0):
        """入口の場 (B, n) → レンズの後の場 (B, n)。"""
        u = torch.nn.functional.pad(u0, (0, self.L - self.n))
        for k in range(self.plates):
            u = torch.cat([u[:, :self.n] * torch.exp(1j * self.phase[:, k]), u[:, self.n:]], 1)
            u = torch.fft.ifft(torch.fft.fft(u, dim=-1, norm="ortho") * self.H, dim=-1, norm="ortho")
        return torch.fft.fft(u[:, :self.n], dim=-1, norm="ortho")

    def backward_field(self, v):
        """レンズの後ろの場 (B, n) を逆にたどる（forward_field の随伴）→ 入口の場 (B, n)。"""
        u = torch.fft.ifft(v, dim=-1, norm="ortho")
        u = torch.nn.functional.pad(u, (0, self.L - self.n))
        for k in reversed(range(self.plates)):
            u = torch.fft.ifft(torch.fft.fft(u, dim=-1, norm="ortho") * self.H.conj(), dim=-1, norm="ortho")
            u = torch.cat([u[:, :self.n] * torch.exp(-1j * self.phase[:, k]), u[:, self.n:]], 1)
        return u[:, :self.n]

    def bright(self):
        """口ごとの属性の明るさ (M, 属性)。"""
        return self.forward_field(self.ports).abs()[:, VAL0:VAL0 + len(SLOTS)] ** 2


def train(dev, sents, how, steps, lr=0.02):
    """sents: 文ごとの属性の番号のリスト。"""
    opt = torch.optim.Adam(dev.parameters(), lr=lr)
    if how == "marg":
        q = torch.zeros(len(SLOTS))
        for s in sents:
            for j in s:
                q[j] += 1
        q = q / q.sum()
    else:
        X = torch.zeros(len(sents), len(SLOTS))
        for i, s in enumerate(sents):
            for j in s:
                X[i, j] += 1
    for _ in range(steps):
        W = dev.bright()                                            # (M, A)
        if how == "marg":
            p = torch.cat([W[0], (1 - W[0].sum()).clamp(min=1e-9)[None]])
            loss = -(torch.cat([q, torch.zeros(1)]) * torch.log(p + 1e-9)).sum()
        else:
            ll = X @ torch.log(W + 1e-9).T                          # (文, M)
            loss = -(torch.logsumexp(ll, 1) - math.log(dev.M)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return float(loss.detach())


def spectral_split(S, k):
    """似ている度合いの表 S から k 組に分ける（正規化ラプラシアンの固有ベクトル + 小さな k 平均）。"""
    S = np.clip(S, 0, None)
    d = S.sum(1) + 1e-9
    Lm = np.eye(len(S)) - S / np.sqrt(d[:, None] * d[None, :])
    w, V = np.linalg.eigh(Lm)
    X = V[:, :k]
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    best = None
    rng = np.random.RandomState(0)
    for _ in range(20):
        C = X[rng.choice(len(X), k, replace=False)]
        for _ in range(30):
            lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), 1)
            C = np.array([X[lab == c].mean(0) if (lab == c).any() else C[c] for c in range(k)])
        sc = ((X - C[lab]) ** 2).sum()
        if best is None or sc < best[0]:
            best = (sc, lab)
    return best[1]


def delta(S, lab):
    n = len(S)
    win, bet = [], []
    for i in range(n):
        for j in range(i + 1, n):
            (win if lab[i] == lab[j] else bet).append(S[i, j])
    if not win or not bet:
        return 0.0
    return float(np.mean(win) - np.mean(bet))


def purity(lab, truth):
    """片方の物だけの属性について、分け方と正解の一致（組の名前の入れ替えは最善を取る）。"""
    idx = [i for i, t in enumerate(truth) if t >= 0]
    if not idx:
        return float("nan")
    ks = sorted(set(truth[i] for i in idx))
    best = 0
    for perm in itertools.permutations(sorted(set(lab)), len(ks)) if len(set(lab)) >= len(ks) else [tuple(sorted(set(lab)))]:
        mp = dict(zip(perm, ks))
        best = max(best, sum(mp.get(lab[i], -9) == truth[i] for i in idx))
    return best / len(idx)


GROUPS = {   # 名前 → 融合する意味のリスト（"半分" は 1 つの意味の文を 2 つに分けたもの）
    "リンゴ: 果物 + 人": ["リンゴ/果物", "リンゴ/人"],
    "トマト: 完熟 + 未熟": ["トマト/完熟", "トマト/未熟"],
    "イチゴ: 完熟 + 未熟": ["イチゴ/完熟", "イチゴ/未熟"],
    "バナナ: 完熟 + 未熟": ["バナナ/完熟", "バナナ/未熟"],
    "3 つ: 果物 + 人 + レモン": ["リンゴ/果物", "リンゴ/人", "レモン"],
    "比べ: リンゴ/果物を半分ずつ（赤と緑がある）": ["リンゴ/果物", "リンゴ/果物"],
    "比べ: トマト/完熟を半分ずつ": ["トマト/完熟", "トマト/完熟"],
}


def run_group(name, senses, a, rng):
    sents, owner = [], []
    for gi, s in enumerate(senses):
        for _ in range(a.n_sent):
            toks = make_sentence(s, rng)
            sents.append(sorted(SLOT[x] for x in attrs_of(toks, SENSES[s][0])))
            owner.append(gi)
    present = sorted({j for s in sents for j in s})
    # 正解: 属性ごとに、どの物だけにあるか（-1 = 共通）
    have = [set() for _ in senses]
    for s, o in zip(sents, owner):
        have[o] |= set(s)
    truth = []
    for j in present:
        own = [gi for gi in range(len(senses)) if j in have[gi]]
        truth.append(own[0] if len(own) == 1 else -1)
    k = len(senses)
    out = []
    for how, M in [("marg", 1)] + [("mix", m) for m in a.ports]:
        torch.manual_seed(a.seed)
        dev = PortDevice(M, n=a.n, plates=a.plates, seed=a.seed)
        loss = train(dev, sents, how, a.steps)
        with torch.no_grad():
            W = dev.bright().numpy()                                 # (M, A)
            e = torch.zeros(len(present), dev.n, dtype=torch.complex64)
            for r, j in enumerate(present):
                e[r, VAL0 + j] = 1
            B = dev.backward_field(e)                                # 逆: 入口の場 (属性, n)
            amp = B @ dev.ports.conj().T                             # 口への戻り (属性, M)
            portp = (amp.abs() ** 2).numpy()                         # = 順の明るさ
            # 往復: (a) 口に射影して順へ、(b) そのまま順へ
            rt_port = dev.forward_field(amp @ dev.ports).abs().numpy()[:, VAL0:VAL0 + len(SLOTS)] ** 2
            rt_free = dev.forward_field(B).abs().numpy()[:, VAL0:VAL0 + len(SLOTS)] ** 2
            Bn = B / (B.abs().pow(2).sum(1, keepdim=True).sqrt() + 1e-9)
            S_field = (Bn @ Bn.conj().T).abs().numpy()               # 入口の場の重なり
        q = portp / (portp.sum(1, keepdims=True) + 1e-12)
        S_port = (q @ q.T) / (np.linalg.norm(q, axis=1)[:, None] * np.linalg.norm(q, axis=1)[None] + 1e-12)
        res = dict(group=name, how=how, M=M, loss=loss, n_attr=len(present))
        for cue, S in (("port", S_port), ("field", S_field)):
            if cue == "port" and M == 1:
                continue
            lab = spectral_split(S, k)
            res[f"delta_{cue}"] = delta(S, lab)
            res[f"purity_{cue}"] = purity(list(lab), truth)
        # 往復の関連: 同じ物の属性に出る割合（j 自身と共通の属性は除く）
        for nm, RT in (("rt_port", rt_port), ("rt_free", rt_free)):
            fr = []
            for r, j in enumerate(present):
                if truth[r] < 0:
                    continue
                same = sum(RT[r, jj] for rr, jj in enumerate(present) if jj != j and truth[rr] == truth[r])
                other = sum(RT[r, jj] for rr, jj in enumerate(present) if jj != j and truth[rr] >= 0 and truth[rr] != truth[r])
                if same + other > 0:
                    fr.append(same / (same + other))
            res[nm] = float(np.mean(fr)) if fr else float("nan")
        # 共通の属性の口の広がり（エントロピー / log M）
        if M > 1:
            ent = [float(-(q[r] * np.log(q[r] + 1e-12)).sum() / math.log(M)) for r in range(len(present))]
            res["entropy_common"] = float(np.mean([e_ for e_, t in zip(ent, truth) if t < 0])) if any(t < 0 for t in truth) else float("nan")
            res["entropy_own"] = float(np.mean([e_ for e_, t in zip(ent, truth) if t >= 0])) if any(t >= 0 for t in truth) else float("nan")
        # 口どうしの「競合する観点」と「片方だけの観点」（口 2 つのとき。分けるかどうかの判断に使う）
        if M == 2:
            from i01_small_world import VIEWS
            comp, excl = [], []
            for v, xs in VIEWS.items():
                ids = [SLOT[(v, x)] for x in xs]
                s0, s1 = W[0, ids].sum(), W[1, ids].sum()
                h0, h1 = s0 >= 0.05 * W[0].sum(), s1 >= 0.05 * W[1].sum()
                if h0 and h1:
                    if 0.5 * np.abs(W[0, ids] / s0 - W[1, ids] / s1).sum() >= 0.5:
                        comp.append(v)
                elif h0 != h1 and min(s0 / W[0].sum(), s1 / W[1].sum()) < 0.01:
                    excl.append(v)
            res["compete"], res["exclusive"] = comp, excl
            res["split"] = len(comp) >= 2 or (len(comp) >= 1 and len(excl) >= 1)
        # 表に残す: 属性ごとの口の分布と、往復で上位の属性
        res["attrs"] = [dict(attr="{}:{}".format(*SLOTS[j]), own=truth[r], port=[round(float(x), 3) for x in q[r]] if M > 1 else None,
                             rt_top=["{}:{}".format(*SLOTS[jj]) for jj in np.argsort(-rt_port[r])[:4]])
                        for r, j in enumerate(present)]
        res["port_bright"] = [[round(float(x), 3) for x in W[m]] for m in range(M)]
        out.append(res)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_sent", type=int, default=50, help="物 1 つあたりの文")
    ap.add_argument("--ports", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--plates", type=int, default=8)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--out", default="../Project_Kotoyui_private/intervention/results/i02.jsonl")
    a = ap.parse_args()
    torch.set_num_threads(4)
    rng = random.Random(a.seed)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    t0 = time.perf_counter()
    for name, senses in GROUPS.items():
        for res in run_group(name, senses, a, rng):
            res.update(seed=a.seed, n_sent=a.n_sent, steps=a.steps, n=a.n, plates=a.plates)
            print(json.dumps({k: v for k, v in res.items() if k not in ("attrs", "port_bright")}, ensure_ascii=False), flush=True)
            with open(a.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
    print(f"{time.perf_counter() - t0:.0f} 秒")


if __name__ == "__main__":
    main()
