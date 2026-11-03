"""実験07: 「次の発言の共鳴」を光のグレインにする — 返事選びの点を、数え上げから光へ置き換える最初の 1 つ

着想: 持ち主の方針（2026-11-02「選ぶ部分を、少しずつ光のグレインに置き換える」）。kotoyui.py 0.1.0 の返事選び（Sin-CNN3 から取り込み）の芯は、
      「ある発言に x があると、次の発言に y が出やすい」を数えた PMI（共鳴、実験06 で 10 択の 56〜60%）。これを光の器にする。
方法: 前の発言の中身の語（よく出る 4,095 語）を、64 × 64 の場のマス 1 つずつに置いた光（明るさ = あるか）→ 4f 系 4 段 × 幅 4 + 平方根 → レンズの後の明るさを
      ハートレー変換（実験04 の方法 3、持ち主の案「スクリーンの明るさを 2 次元フーリエ変換して照らし合わせる」）→ 係数 1 つ = 語 1 つの点数。
      符号の表は持たない（デジタルは倍率と語ごとの背景光だけ）。--read codes で、窓 256 個 + 学習する符号（語 × 窓）の読み方と比べる。答えは「次の発言に出た語」の分布（交差エントロピー）。
      返事の候補の点 = 候補の語の「光が出した次の語の確率 ÷ ふだんの出やすさ」の対数の平均（0 未満は 0、PMI と同じ形）。
比べる（同じ 10 択、実験06 と同じ問の作り方）: でたらめ、語の重なり、数え上げの共鳴（PMI）、重なり + PMI（kotoyui 0.1.0 の芯）。
      光は単独・重なり + 光・重なり + PMI + 光（足し合わせ）で測る。
材料: RealPersonaChat・JMultiWOZ（CC BY-SA 4.0）、会話ごとに教え 9 : 試し 1（kotoyui.load_conversations）。学習 9000 ステップ。
結果: results/exp07.txt
実行: python experiments/exp07_light_resonance.py --seed 0 --device cuda --out results/exp07.txt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import kotoyui as K  # noqa: E402
import textlight  # noqa: E402

L = 64
NV = L * L - 1                     # 語の数（マス 0 = 直流 は使わない）


class ResonanceGrain(torch.nn.Module):
    """前の発言の語の袋（64 × 64 のマス）→ 4f 系 + 平方根 → ハートレー係数 1 つ = 次に出やすい語 1 つ（符号の表なし）。"""

    def __init__(self, stages=4, width=4, seed=0, read="hartley", K=256):
        super().__init__()
        self.read = read
        self.body = textlight.Multi4f(stages, width, seed)
        if read == "codes":                                        # 比べる: 窓 K 個 + 学習する符号（語 4096 × 窓 K、デジタル）
            self.register_buffer("det", textlight.p2.window_grid(L, L, K, 4))
            self.codes = torch.nn.Parameter(torch.randn(L * L, K, generator=torch.Generator().manual_seed(seed)))
        self.gain = torch.nn.Parameter(torch.tensor(3.0))
        self.bias = torch.nn.Parameter(torch.zeros(L * L))

    def forward(self, bag):
        """bag (B, 4096)（0/1）→ 語の対数確率 (B, 4096)。"""
        a = bag / (bag.pow(2).mean(1, keepdim=True).sqrt() + 1e-6)
        I = self.body.screen(a.view(-1, L, L).to(torch.complex64))
        if self.read == "codes":
            v = textlight.p2.code_scores(textlight.p2.read_windows(I, self.det), self.codes, 1.0)
        else:
            Fq = torch.fft.fft2(I, norm="ortho")
            v = (Fq.real - Fq.imag).flatten(1)
        v = (v - v.mean(1, keepdim=True)) / (v.std(1, keepdim=True) + 1e-6)
        z = self.gain * v + self.bias
        z[:, 0] = -1e4                                              # マス 0（直流）は語に使わない
        return F.log_softmax(z, 1)

    def params(self):
        return sum(q.numel() for q in self.body.parameters()), self.gain.numel() + self.bias.numel() + (self.codes.numel() if self.read == "codes" else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", type=int, default=4)
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--steps", type=int, default=9000)
    ap.add_argument("--read", default="hartley", choices=["hartley", "codes"], help="hartley = フーリエの係数で読む（符号なし）、codes = 窓 256 + 学習する符号（比べる）")
    ap.add_argument("--n", type=int, default=3000, help="試しの問の数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/exp07.txt")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    t0 = time.perf_counter()
    mat, tg = K.load_tables()
    convs = K.load_conversations()
    # 語の表（よく出る中身の語 4,095）と、教えの組（前の発言の語 → 次の発言の語）
    freq = Counter()
    for _, turns in convs["train"]:
        for t in turns:
            freq.update(K.analyze(t)[0])
    vocab = [w for w, _ in freq.most_common(NV)]
    wid = {w: i + 1 for i, w in enumerate(vocab)}
    ids = lambda t: [wid[w] for w in K.analyze(t)[0] if w in wid]
    P, Q = [], []
    for _, turns in convs["train"]:
        A = [ids(t) for t in turns]
        for i in range(len(A) - 1):
            if A[i] and A[i + 1]:
                P.append(A[i])
                Q.append(A[i + 1])
    uni = torch.zeros(L * L)
    for q in Q:
        uni[q] += 1.0 / len(q)
    log_uni = torch.log((uni + 1) / (uni + 1).sum())
    print(f"語 {len(vocab)}・教えの組 {len(P)}（{time.perf_counter() - t0:.0f} 秒）", flush=True)

    def dense(lists, normalize=False):
        x = torch.zeros(len(lists), L * L)
        for r, l in enumerate(lists):
            x[r, l] = 1.0 / len(l) if normalize else 1.0
        return x
    m = ResonanceGrain(a.stages, a.width, a.seed, a.read).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    g = torch.Generator().manual_seed(a.seed)
    t1 = time.perf_counter()
    for k in range(a.steps):
        idx = torch.randint(len(P), (256,), generator=g).tolist()
        x = dense([P[i] for i in idx]).to(dev)
        y = dense([Q[i] for i in idx], True).to(dev)
        loss = -(y * m(x)).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()
        if (k + 1) % 1000 == 0:
            print(f"{k + 1}/{a.steps} 損失 {loss.item():.3f} {time.perf_counter() - t1:.0f} 秒", flush=True)
    train_s = time.perf_counter() - t1
    m.eval()
    # 試し: 実験06 と同じ作り方の 10 択
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
    cases = cases[:a.n]
    lp_cache = {}

    @torch.no_grad()
    def light_lp(u):
        if u not in lp_cache:
            l = ids(u)
            lp_cache[u] = (m(dense([l]).to(dev))[0].cpu() - log_uni) if l else None
        return lp_cache[u]

    def light(u, cand):
        lift = light_lp(u)
        cl = ids(cand)
        if lift is None or not cl:
            return 0.0
        return float(lift[cl].clamp(min=0).mean())

    def score(ctx, cand, how):
        u = ctx[-1]
        uw, _ = K.analyze(u)
        cw_, _ = K.analyze(cand)
        sc = 0.0
        if "A" in how:
            sc += K.overlap(uw, cw_)
        if "B" in how:
            sc += mat.resonance(uw, cw_)
        if "L" in how:
            sc += light(u, cand)
        return sc
    rec = dict(exp="07", **vars(a), n_vocab=len(vocab), n_pairs=len(P), train_s=train_s, train_loss=loss.item(),
               params_light=m.params()[0], params_digital=m.params()[1], runs=[], version=K.__version__, torch=torch.__version__)
    for hard in (False, True):
        for how, name in (("R", "でたらめ"), ("A", "語の重なり"), ("B", "数え上げの共鳴（PMI）"), ("AB", "重なり + PMI（0.1.0 の芯）"),
                          ("L", "光の共鳴"), ("AL", "重なり + 光の共鳴"), ("ABL", "重なり + PMI + 光の共鳴")):
            top1 = 0
            by = {}
            for src, ctx, gold, negs, same in cases:
                cands = [gold] + (same if hard else negs)
                scs = [rng.random() for _ in cands] if how == "R" else [score(ctx, c, how) + 1e-6 * rng.random() for c in cands]
                ok = all(scs[0] > s for s in scs[1:])
                top1 += ok
                b = by.setdefault(src, [0, 0])
                b[0] += ok
                b[1] += 1
            r = dict(hard=hard, how=how, name=name, top1=top1 / len(cases), by_source={k: v[0] / v[1] for k, v in by.items()})
            rec["runs"].append(r)
            print("同じ会話から" if hard else "ほかの会話から", name, round(r["top1"], 3), {k: round(v, 3) for k, v in r["by_source"].items()}, flush=True)
    rec["total_s"] = time.perf_counter() - t0
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
