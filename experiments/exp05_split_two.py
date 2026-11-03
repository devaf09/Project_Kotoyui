"""実験05: 4f 系の分担の 2DTG に「2 段で読む」を組み合わせる — 一番良い光の形で、デジタルの符号をなくせるか（3 seed）

着想: 実験03 で一番良かった光は 4f 系の分担の 2DTG（担当 4、27.1%）だが、学習するデジタルの符号が 419 万（学ぶ数の 89%）。
      実験04 で、光路 1 本なら「2 段で読む」でデジタルを 105 万 → 130 個にしても −0.9 点だった（持ち主の依頼の方法 1）。分担の形でも同じか。
方法: 担当（直前の単位の番号の余り 4 つ）ごとに、光路 A（文脈 → 64 組、窓 64 個）と光路 B（文脈 + 組 → 組の中の 64 番、窓 64 個）を持つ。
      中身はどちらも 4f 系 4 段 × 幅 4 + 平方根（実験03 の分担と同じ）、各 9000 ステップ。組の分け方は「前に来る単位が似たもの」（実験04 で良かった方）。
      各担当は、自分の担当の文脈だけで学ぶ。デジタルは担当ごとに 130 個（倍率と窓の背景光）。
比べる: 実験03 の 4f 系の分担（学習する符号、seed 0〜2 は sweep05.sh で exp03_4f.py を回す）、GRU 28.33%（3 seed）、3-gram 30.88%。
データ・テスト: 実験02〜04 と同じ（現代の雑談、口調のスイッチなし、テスト 3 万位置）。
結果: results/exp05.txt
実行: python experiments/exp05_split_two.py --seed 0 --device cuda --out results/exp05.txt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp02_split_style as e2  # noqa: E402
import exp03_4f as e3  # noqa: E402
import exp04_codes as e4  # noqa: E402

NG = e4.NG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--G", type=int, default=4)
    ap.add_argument("--stages", type=int, default=4)
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--group", default="clust", choices=["clust", "freq"])
    ap.add_argument("--steps", type=int, default=9000)
    ap.add_argument("--max_train", type=int, default=2000000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/exp05.txt")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    t0 = time.perf_counter()
    (X, y, s), (Xt, yt, st), sp, tr_u = e2.build(a.max_train, a.seed)
    V = sp.get_piece_size()
    n = e2.N_CTX
    grp, idx = e4.grouping(a.group, X, y, V, a.seed)
    withg = lambda Z, gg: torch.cat([(V + 2 + gg)[:, None], Z], 1)
    ti = torch.randperm(len(y), generator=torch.Generator().manual_seed(1))[:20000]
    LP = torch.empty(len(yt), V)
    LPtr = torch.empty(len(ti), V)
    light = digital = 0
    train_s = 0.0
    per = []
    for gi in range(a.G):
        own = (X[:, -1] % a.G == gi).nonzero().squeeze(1)
        ownt = (Xt[:, -1] % a.G == gi).nonzero().squeeze(1)
        owntr = (X[ti, -1] % a.G == gi).nonzero().squeeze(1)
        sd = a.seed * 100 + gi
        mA = e4.WindowNet(n, V + 2, NG, a.stages, a.width, sd).to(dev)
        train_s += e3.fit(mA, X[own], grp[y[own]], a.steps, dev, sd)
        mB = e4.WindowNet(n + 1, V + 2 + NG, NG, a.stages, a.width, sd + 50).to(dev)
        train_s += e3.fit(mB, withg(X[own], grp[y[own]]), idx[y[own]], a.steps, dev, sd + 50)
        mA.eval()
        mB.eval()

        def full_logp(Z):
            LA = e3.logp_of(mA, Z, dev)
            out = torch.empty(len(Z), V)
            for g in range(NG):
                LB = e3.logp_of(mB, withg(Z, torch.full((len(Z),), g)), dev)
                members = (grp == g).nonzero().squeeze(1)
                out[:, members] = LA[:, g:g + 1] + LB[:, idx[members]]
            return out
        LP[ownt] = full_logp(Xt[ownt])
        LPtr[owntr] = full_logp(X[ti][owntr])
        accA, _ = e3.evals(e3.logp_of(mA, Xt[ownt], dev), grp[yt[ownt]])
        per.append(dict(g=gi, n_train=len(own), n_test=len(ownt), group_acc=accA))
        for m in (mA, mB):
            l, d = m.params()
            light += l
            digital += d
        print(f"担当 {gi + 1}/{a.G} 組の正解率 {accA:.1f}% {train_s:.0f} 秒", flush=True)
    acc, ce = e3.evals(LP, yt)
    tr_acc, _ = e3.evals(LPtr, y[ti])
    rec = dict(exp="05", **vars(a), n_train=len(y), n_test=len(yt), test_acc=acc, test_ce=ce, train_acc=tr_acc,
               params_light=light, params_digital_learned=digital, table_stored=0, per_grain=per, train_s=train_s,
               total_s=time.perf_counter() - t0, version=e4.p2.__version__, torch=torch.__version__)
    print(json.dumps(rec, ensure_ascii=False), flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
