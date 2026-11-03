"""実験01: 光の 2DWIG — 語尾・言い回しの袋を 64 × 64 の光の場に入れて、話者の類・丁寧さを当てる

着想: 持ち主の案（2DWIG、2026-10-22。SN2-Parts-2D 実験71・72）と、見積もり E2・E5・E6（2026-10-24〜26）。
      SN2 の光の 2DWIG は末尾 12 文字を位置ごとに並べて 43.1%、数えた表（袋）は 52.7%。袋を光に入れる形にする。
      E2 で、袋を 4096 個に散らしても回帰はほとんど落ちない（51.1 → 49.6%）ので、64 × 64 = 4096 画素に 1 つずつ置く。
      持ち主の判断（2026-10-26）: 隠し味は予測に使わない。2DWIG は口調（丁寧さ・一人称）の見分けに使う（M1 の組み替え）。
方法:
  入力: 会話文の一人称を「＊」で隠し、語尾 1〜3 文字 + 文字の 2 つ組（--feats tri で 3 つ組も）を crc32 で 4096 個に散らして数え、
        画素ごとの数の平方根を振幅にして 64 × 64 の場に載せる（明るさの平均を 1 に）。画像の分類と同じ入れ方（振幅、SN2 実験02）。
  plates : 光路 1 本（vendor の Net2D、板 ① × --layers 枚、区画 --patches、窓 = 類 1 つに 1 窓。デジタルは窓の背景光と温度だけ）
  lr     : 比べる相手。同じ 4096 個の袋のロジスティック回帰（学習の長さは光と同じ。E2・E5 の 2000 ステップより長い）
  nb     : 比べる相手。同じ 4096 個の袋の数えた表（ナイーブベイズ、なめらかさ 0.5）
課題（--task）:
  aozora12 : 青空文庫 全 4,477 作品の一人称つき会話文（12 類、作品の 15% がテスト、aozora.load("all")）。--spread で交互の会話で広げたラベルを訓練に足す
  polite   : 現代の雑談（RealPersonaChat・名大会話コーパス）の丁寧さ（です・ます → 丁寧）。知らない話者 51 人でテスト（est_modern.split）
指標: 正解率、類ごとの正解率の平均、CE、訓練の正解率（5,000 件、覚えすぎの目安）、いつも一番多い類。
      学習する数は光の部分（板の A・φ）とデジタルの部分（窓の背景光・温度、回帰の重み）に分けて数える。
結果: results/exp01.txt（図は tools/fig_exp01.py）
実行: python experiments/exp01_wig_bag.py --task aozora12 --method plates --seed 0 --device cuda --out results/exp01.txt
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time
import zlib
from collections import Counter

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
import aozora  # noqa: E402

_spec = importlib.util.spec_from_file_location("sn2_parts_2d_v0_11_0", os.path.join(ROOT, "vendor", "sn2_parts_2d_v0_11_0.py"))
p2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p2)
L = 64


def feats(u, tri=False):
    """袋の中身: 語尾 1〜3 文字 + 文字の 2 つ組（tri なら 3 つ組も）。"""
    f = ["E" + u[-k:] for k in (1, 2, 3)] + [u[i:i + 2] for i in range(len(u) - 1)]
    if tri:
        f += [u[i:i + 3] for i in range(len(u) - 2)]
    return f


def bag(texts, tri=False):
    """会話文の並び → 4096 個に散らした数 (N, 4096)（float16 で CPU に置く）。"""
    X = torch.zeros(len(texts), L * L, dtype=torch.float16)
    for i, u in enumerate(texts):
        c = Counter(zlib.crc32(f.encode("utf-8")) % (L * L) for f in feats(u, tri))
        X[i, list(c)] = torch.tensor(list(c.values()), dtype=torch.float16)
    return X


def to_field(v):
    """数 (B, 4096) → 光の場 (B, 64, 64)。数の平方根を振幅にし、明るさの平均を 1 にする。"""
    a = v.float().sqrt()
    a = a / (a.pow(2).mean(1, keepdim=True).sqrt() + 1e-6)
    return a.view(-1, L, L).to(torch.complex64)


def data(task, spread=False, max_train=100000, max_test=20000, seed=0):
    """(訓練の会話文, 訓練の類, テストの会話文, テストの類, 類の並び)。"""
    if task == "aozora12":
        d = aozora.load("all")
        tr = aozora.labeled(d, "train")
        if spread:
            import est_wig_bag
            tr = tr + est_wig_bag.spread_items(d)
        te = aozora.labeled(d, "test")
        trx, try_ = [u for _, u, _ in tr], [g for _, _, g in tr]
        tex, tey = [u for _, u, _ in te], [g for _, _, g in te]
    elif task == "polite":
        import est_modern
        us = est_modern.load()
        tr_d, _, te_d, test_spk = est_modern.split(us, 0)
        tr = [(aozora.mask(u["text"]), u["polite"]) for u in us if (u["src"], u["dialog"]) in tr_d]
        te = [(aozora.mask(u["text"]), u["polite"]) for u in us if (u["src"], u["dialog"]) in te_d and u["spk"] in test_spk]
        trx, try_ = [t for t, _ in tr], [c for _, c in tr]
        tex, tey = [t for t, _ in te], [c for _, c in te]
    else:
        raise ValueError(task)
    rng = random.Random(seed)
    if len(trx) > max_train:
        ids = rng.sample(range(len(trx)), max_train)
        trx, try_ = [trx[i] for i in ids], [try_[i] for i in ids]
    if len(tex) > max_test:
        ids = random.Random(0).sample(range(len(tex)), max_test)    # テストは seed によらず同じ
        tex, tey = [tex[i] for i in ids], [tey[i] for i in ids]
    classes = [c for c, _ in Counter(try_).most_common()]
    classes += [c for c in sorted(set(tey)) if c not in classes]   # 訓練に少なくて抜けた類（窓は持つ）
    return trx, try_, tex, tey, classes


def scores(pred, gold, C):
    acc = 100.0 * (pred == gold).float().mean().item()
    rec = [100.0 * (pred[gold == c] == c).float().mean().item() for c in range(C) if (gold == c).any()]
    return acc, sum(rec) / len(rec), rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="aozora12", choices=["aozora12", "polite"])
    ap.add_argument("--method", default="plates", choices=["plates", "lr", "nb"])
    ap.add_argument("--spread", action="store_true", help="交互の会話で広げたラベルを訓練に足す（aozora12 だけ）")
    ap.add_argument("--feats", default="bi", choices=["bi", "tri"], help="袋の中身: bi = 語尾 + 2 つ組、tri = + 3 つ組")
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--patches", type=int, default=16)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--max_train", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/exp01.txt")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    t0 = time.perf_counter()
    trx, try_, tex, tey, classes = data(a.task, a.spread, a.max_train, seed=a.seed)
    ci = {c: i for i, c in enumerate(classes)}
    C = len(classes)
    X, y = bag(trx, a.feats == "tri"), torch.tensor([ci[c] for c in try_])
    Xt, yt = bag(tex, a.feats == "tri"), torch.tensor([ci[c] for c in tey])
    t_data = time.perf_counter() - t0
    majority = 100.0 * (yt == 0).float().mean().item()
    sub = torch.randperm(len(y), generator=torch.Generator().manual_seed(a.seed))[:5000]
    dev = torch.device(a.device)
    light = digital = 0
    t1 = time.perf_counter()
    if a.method == "nb":
        cnt = torch.zeros(C, L * L)
        cnt.index_add_(0, y, X.float())
        logp = torch.log((cnt + 0.5) / (cnt.sum(1, keepdim=True) + 0.5 * L * L))
        prior = torch.log(torch.bincount(y, minlength=C).float() + 1e-9)
        S = lambda Z: Z.float() @ logp.T + prior
        St, Str = S(Xt), S(X[sub])
        digital = C * L * L
    else:
        if a.method == "plates":
            net = p2.Net2D(L, C, ["xy"] * a.layers, patches=a.patches, seed=a.seed).to(dev)
            p2.train(net, X, y, steps=a.steps, lr=a.lr, batch_size=256, seed=a.seed, to_input=to_field, schedule="cosine")
            digital = net.gain.numel() + net.bias.numel()
            light = net.n_params() - digital
            fwd = lambda Z: net(to_field(Z.to(dev)))
        else:
            lin = torch.nn.Linear(L * L, C).to(dev)
            opt = torch.optim.Adam(lin.parameters(), lr=0.003, weight_decay=1e-5)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
            g = torch.Generator().manual_seed(a.seed)
            for _ in range(a.steps):
                i = torch.randint(len(y), (256,), generator=g)
                xb = X[i].float().sqrt().to(dev)
                loss = F.cross_entropy(lin(xb / (xb.pow(2).mean(1, keepdim=True).sqrt() + 1e-6)), y[i].to(dev))
                opt.zero_grad()
                loss.backward()
                opt.step()
                sch.step()
            digital = sum(p.numel() for p in lin.parameters())
            fwd = lambda Z: lin((lambda v: v / (v.pow(2).mean(1, keepdim=True).sqrt() + 1e-6))(Z.float().sqrt().to(dev)))
        with torch.no_grad():
            St = torch.cat([fwd(Xt[i:i + 1000]).cpu() for i in range(0, len(Xt), 1000)])
            Str = torch.cat([fwd(X[sub][i:i + 1000]).cpu() for i in range(0, len(sub), 1000)])
    train_s = time.perf_counter() - t1
    acc, macro, per = scores(St.argmax(1), yt, C)
    rec = dict(exp="01", **vars(a), classes=classes, n_train=len(y), n_test=len(yt), majority=majority,
               test_acc=acc, test_macro=macro, per_class=dict(zip(classes, per)),
               test_ce=F.cross_entropy(St.float(), yt).item(), train_acc=100.0 * (Str.argmax(1) == y[sub]).float().mean().item(),
               params_light=light, params_digital=digital, data_s=t_data, train_s=train_s,
               version=p2.__version__, torch=torch.__version__)
    print(json.dumps({k: v for k, v in rec.items() if k != "per_class"}, ensure_ascii=False), flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
