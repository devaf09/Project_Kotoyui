"""実験09: 「次の発言の共鳴」を語板（1 語 = 1 台の光の器）で作る — Sin-CNN3 のライブラリ sincnn3 の器の形

着想: 持ち主の案内（2026-11-02「Sin-CNN3 をライブラリ化した、試して使いやすいほうを」）。数え上げの共鳴（PMI）は
      「前の発言の語 x ごとの、次の発言に出やすい語 y の表」を足し合わせたもの。sincnn3 の KnowledgeBase は
      「1 語 = 1 台の器、出口の明るさ = 割合、融合 = 明るさの足し算」なので、形がそのまま合う。
      実験07・08 の「1 台の大きな光路に語の袋を入れる」形とは逆に、語ごとに小さな器を持つ。
方法: 語 x の器 = sincnn3.LightDevice と同じもの（画素ごとの位相の板 16 枚 + 角スペクトル法の回折 + 0 を詰めた 2 倍の場、縦 n = 128）。
      入口に周波数 1 の波を入れ、レンズの後の周波数 3〜n−1 の明るさ = x の「次に出やすい語」の上位 k 語（k = n − 3 = 125）の割合。
      割合の正解 = 正の PMI を上位 k 語の和で割ったもの。語ごとの和 S_x（デジタル 1 個）を掛け戻すと、上位 k 語に切った PMI になる。
      学び方は sincnn3.Grain.fit と同じ（明るさ + 残りの光を確率と見た交差エントロピー、Adam、500 回）。器はたくさんの語をまとめて計算する（中身は同じ式）。
      --mode table: 光を使わず、上位 k 語に切った PMI そのもので 10 択を測る（光の器で届く上の目安。CPU だけ）。
      上位 k 語の選び方: --rank pmi（PMI の大きい順）か weight（回数 × PMI の大きい順 = 合計への効きが大きい順）。
注意: 器の出口 j がどの語かの対応表（語 x ごとに k 語）は持つ（sincnn3 の値の名前と同じ、学習はしない）。
比べる（同じ 10 択、実験06・07 と同じ問の作り方・同じ seed で同じ問）: でたらめ、語の重なり、数え上げの共鳴（PMI、切らない）、重なり + PMI（kotoyui 0.1.0 の芯）。
材料: RealPersonaChat・JMultiWOZ（CC BY-SA 4.0）、会話ごとに教え 9 : 試し 1（kotoyui の表）。
出どころ: 器の式は Sin-CNN3 の sincnn3/device.py・knowledge.py（版 0.1.0、同じ著作権者の MIT）を、たくさんの器をまとめて計算する形に書き直したもの。
結果: results/exp09.txt
実行: python experiments/exp09_word_grains.py --mode light --rank weight --device cuda --out results/exp09.txt
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

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import kotoyui as K  # noqa: E402


class Devices(torch.nn.Module):
    """sincnn3.LightDevice を W 台まとめたもの。板の位相 (W, n, plates) が学習する数。式は LightDevice.forward・brightness と同じ。"""

    def __init__(self, W, n=128, plates=16, gap=64.0, seed=0):
        super().__init__()
        self.n, self.plates, self.L = n, plates, 2 * n                      # 0 を詰めて 2 倍に広げた場（端から回り込まない）
        fx = torch.fft.fftfreq(self.L)
        self.register_buffer("H", torch.exp(1j * gap * torch.sqrt((2 * math.pi) ** 2 - (2 * math.pi * fx) ** 2)).to(torch.complex64))
        g = torch.Generator().manual_seed(seed)
        self.phase = torch.nn.Parameter(math.pi * (torch.rand(W, n, plates, generator=g) < 0.5).float())   # 0 か π で始める（sincnn3 の既定）
        y = torch.arange(n, dtype=torch.float32)
        self.register_buffer("inp", (torch.exp(1j * 2 * math.pi * y / n) / math.sqrt(n)).to(torch.complex64))  # 周波数 1 の波

    def brightness(self, idx):
        """器 idx (B,) に周波数 1 の波を通した、レンズの後の明るさ (B, n)。"""
        ph = self.phase[idx]
        u = torch.nn.functional.pad(self.inp.expand(len(idx), -1), (0, self.L - self.n))
        for k in range(self.plates):
            u = torch.cat([u[:, :self.n] * torch.exp(1j * ph[:, :, k]), u[:, self.n:]], 1)          # 板 k（場の前半だけ）
            u = torch.fft.ifft(torch.fft.fft(u, dim=-1, norm="ortho") * self.H, dim=-1, norm="ortho")  # 回折
        return torch.fft.fft(u[:, :self.n], dim=-1, norm="ortho").abs() ** 2                        # レンズ → 明るさ


def rows(mat, k, rank):
    """語 x ごとに、正の PMI の上位 k 語 [(y, pmi)] と、その和 S_x。"""
    by = defaultdict(list)
    for (x, y), c in mat.cpair.items():
        p = mat.pmi(x, y)
        if p > 0:
            by[x].append((p * c if rank == "weight" else p, y, p))
    out = {}
    for x, v in by.items():
        v.sort(reverse=True)
        out[x] = [(y, p) for _, y, p in v[:k]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="light", choices=["table", "light"])
    ap.add_argument("--rank", default="weight", choices=["pmi", "weight"])
    ap.add_argument("--n", type=int, default=128, help="器の縦の画素（上位 k = n − 3 語）")
    ap.add_argument("--k", type=int, default=0, help="table のとき上位 k 語（0 = n − 3）")
    ap.add_argument("--plates", type=int, default=16)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--chunk", type=int, default=4096, help="一度にまとめて学習する器の数")
    ap.add_argument("--n_q", type=int, default=3000, help="試しの問の数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/exp09.txt")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    t0 = time.perf_counter()
    mat, tg = K.load_tables()
    convs = K.load_conversations()
    k = a.k or a.n - 3
    R = rows(mat, k, a.rank)
    words = sorted(R)
    print(f"語板 {len(words)} 台・上位 {k} 語（{a.rank}）・{time.perf_counter() - t0:.0f} 秒", flush=True)
    S = {x: sum(p for _, p in R[x]) for x in words}
    rec = dict(exp="09", **vars(a), k_used=k, n_grains=len(words), runs=[], version=K.__version__, torch=torch.__version__)
    lookup = {}                                                     # 語 x → {y: 光（または表）の PMI}
    if a.mode == "table":
        for x in words:
            lookup[x] = dict(R[x])
    else:
        val0 = 3                                                    # 観点 1 つ: 値は周波数 3 から（sincnn3.Grain の val0 = 観点の数 + 2）
        tv_all, t1 = [], time.perf_counter()
        for c0 in range(0, len(words), a.chunk):
            ws = words[c0:c0 + a.chunk]
            D = Devices(len(ws), a.n, a.plates, seed=a.seed * 1000 + c0).to(dev)
            q = torch.zeros(len(ws), a.n - val0 + 1)
            for r, x in enumerate(ws):
                for j, (y, p) in enumerate(R[x]):
                    q[r, j] = p / S[x]
            q = q.to(dev)
            idx = torch.arange(len(ws), device=dev)
            opt = torch.optim.Adam(D.parameters(), lr=0.01)
            for _ in range(a.steps):                                 # sincnn3.Grain.fit と同じ学び方
                Wb = D.brightness(idx)[:, val0:]
                p = torch.cat([Wb, (1 - Wb.sum(1)).clamp(min=1e-9)[:, None]], 1)
                loss = -(q * torch.log(p + 1e-9)).sum(1).mean() * len(ws)   # 器ごとに独立なので、和で学ぶ（1 台あたりの速さは Grain.fit と同じ）
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                Wb = D.brightness(idx)[:, val0:].cpu()
            qc = q[:, :-1].cpu()
            tv_all += (0.5 * (Wb - qc).abs().sum(1) + 0.5 * (1 - Wb.sum(1)).clamp(min=0)).tolist()   # 表との差（全変動距離）
            for r, x in enumerate(ws):
                lookup[x] = {y: float(Wb[r, j]) * S[x] for j, (y, _) in enumerate(R[x])}
            print(f"器 {c0 + len(ws)}/{len(words)} 損失 {loss.item() / len(ws):.3f} 表との差 {sum(tv_all) / len(tv_all):.4f} {time.perf_counter() - t1:.0f} 秒", flush=True)
        rec.update(train_s=time.perf_counter() - t1, table_tv=sum(tv_all) / len(tv_all),
                   params_light=len(words) * a.n * a.plates, params_digital=len(words), wiring=sum(len(R[x]) for x in words))

    def grain_res(uw, cw_):
        """語板の共鳴 = 前の発言の語ごとの器の明るさ（× S_x）を、候補の語で拾って足す（融合 = 明るさの足し算）。"""
        if not uw or not cw_:
            return 0.0
        return sum(lookup.get(x, {}).get(y, 0.0) for x in uw for y in cw_) / math.sqrt(len(uw) * len(cw_))

    # 試し: 実験06・07 と同じ作り方の 10 択
    rng = random.Random(a.seed)
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
    cases = cases[:a.n_q]
    an = {}

    def az(t):
        if t not in an:
            an[t] = K.analyze(t)[0]
        return an[t]

    def score(ctx, cand, how):
        uw, cw_ = az(ctx[-1]), az(cand)
        sc = 0.0
        if "A" in how:
            sc += K.overlap(uw, cw_)
        if "B" in how:
            sc += mat.resonance(uw, cw_)
        if "G" in how:
            sc += grain_res(uw, cw_)
        return sc
    gname = "語板の共鳴（光）" if a.mode == "light" else f"上位 {k} 語に切った PMI（表）"
    for hard in (False, True):
        for how, name in (("R", "でたらめ"), ("A", "語の重なり"), ("B", "数え上げの共鳴（PMI）"), ("AB", "重なり + PMI（0.1.0 の芯）"),
                          ("G", gname), ("AG", "重なり + " + gname)):
            top1, by = 0, {}
            for src, ctx, gold, negs, same in cases:
                cands = [gold] + (same if hard else negs)
                scs = [rng.random() for _ in cands] if how == "R" else [score(ctx, c, how) + 1e-6 * rng.random() for c in cands]
                ok = all(scs[0] > s for s in scs[1:])
                top1 += ok
                b = by.setdefault(src, [0, 0])
                b[0] += ok
                b[1] += 1
            r = dict(hard=hard, how=how, name=name, top1=top1 / len(cases), by_source={kk: v[0] / v[1] for kk, v in by.items()})
            rec["runs"].append(r)
            print("同じ会話から" if hard else "ほかの会話から", name, round(r["top1"], 3), {kk: round(v, 3) for kk, v in r["by_source"].items()}, flush=True)
    rec["total_s"] = time.perf_counter() - t0
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
