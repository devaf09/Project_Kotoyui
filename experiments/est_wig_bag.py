"""見積もり: 2DWIG の「語尾・言い回しの袋」の中身と大きさ、話者のラベルを広げた訓練データ（M1 T1 の前、2026-10-24）

着想: SN2-Parts-2D 実験71・72 で、数えた表（語尾 1〜3 文字 + 文字の 2 つ組の袋のナイーブベイズ）が 52.7% と一番良く、光の位置並べは 43.1%。
      光に袋を入れる前に、袋の中身（語尾の長さ・2 つ組・3 つ組）と、番号で散らして大きさを決めたとき（光の場に置ける数）にどこまで落ちるかを測る。
      あわせて、交互の会話で広げたラベル（est_label_spread.py で一致 93.6%）を訓練に足すと良くなるかを測る（Q4）。
方法: 特徴 feats（下の FEATS）× 大きさ H（0 = 散らさない、256・1024・4096 = 特徴の番号を crc32 で H に散らす）×
      訓練 train（real = 一人称つき 19,324 件、spread = それに 2 つ先・前の同じ列の会話文の類を写したラベルを足す）。
      分類器: nb（ナイーブベイズ、なめらかさ 0.5、SN2 と同じ）、lr（袋のロジスティック回帰、EmbeddingBag の平均、Adam 2000 ステップ）。
      テストは本物の一人称つき会話文だけ（作品の 15%、4,101 件、12 類）。
指標: 正解率、類ごとの正解率の平均（少ない類が当たっているか）、いつも一番多い類。
結果: results/est_wig_bag.txt（1 条件 = 1 行の JSON）
実行: python experiments/est_wig_bag.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import zlib
from collections import Counter

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aozora  # noqa: E402

FEATS = {
    "end123+bi": lambda u: ["E" + u[-k:] for k in (1, 2, 3)] + [u[i:i + 2] for i in range(len(u) - 1)],   # SN2 の実験71・72 と同じ
    "end123": lambda u: ["E" + u[-k:] for k in (1, 2, 3)],
    "bi": lambda u: [u[i:i + 2] for i in range(len(u) - 1)],
    "end12345+bi": lambda u: ["E" + u[-k:] for k in (1, 2, 3, 4, 5)] + [u[i:i + 2] for i in range(len(u) - 1)],
    "end123+bi+tri": lambda u: ["E" + u[-k:] for k in (1, 2, 3)] + [u[i:i + 2] for i in range(len(u) - 1)] + [u[i:i + 3] for i in range(len(u) - 2)],
}


def spread_items(d, k=2):
    """訓練の作品で、ラベルのない会話文に、同じ列の k の倍数だけ離れた一番近いラベルつきの会話文の類を写す。"""
    out = []
    for wi, w in enumerate(d["works"]):
        if wi in d["test_works"]:
            continue
        for r in w["runs"]:
            c = [aozora.cls(x["fp"]) for x in r]
            for i, x in enumerate(r):
                if c[i] is not None:
                    continue
                cand = [j for j in range(i % k, len(r), k) if j != i and c[j]]
                if cand:
                    j = min(cand, key=lambda j: abs(j - i))
                    out.append((wi, aozora.mask(x["u"]), c[j]))
    return out


def scores(pred, gold, classes):
    acc = 100.0 * sum(p == g for p, g in zip(pred, gold)) / len(gold)
    rec = [100.0 * sum(p == g for p, g in zip(pred, gold) if g == c) / max(1, sum(g == c for g in gold)) for c in classes]
    return acc, sum(rec) / len(rec), dict(zip(classes, rec))


def hashed(fs, H):
    return [f if H == 0 else zlib.crc32(f.encode("utf-8")) % H for f in fs]


def nb(tr, te, feats, H, classes):
    """ナイーブベイズ（数えた表）。"""
    cnt = Counter(g for _, _, g in tr)
    Fq = {g: Counter() for g in classes}
    for _, u, g in tr:
        Fq[g].update(hashed(feats(u), H))
    tot = {g: sum(Fq[g].values()) for g in classes}
    Vf = len({f for g in classes for f in Fq[g]})
    pred = []
    for _, u, _ in te:
        fs = hashed(feats(u), H)
        pred.append(max(classes, key=lambda g: math.log(cnt[g] + 1e-9) + sum(math.log((Fq[g][f] + 0.5) / (tot[g] + 0.5 * Vf)) for f in fs)))
    return pred


def lr(tr, te, feats, H, classes, seed, steps=2000):
    """袋のロジスティック回帰（特徴の重みの平均 → 類のスコア）。H = 0 のときは訓練に出た特徴に番号を振る。"""
    torch.manual_seed(seed)
    if H == 0:
        vocab = {}
        for _, u, _ in tr:
            for f in feats(u):
                vocab.setdefault(f, len(vocab))
        idx = lambda u: [vocab[f] for f in feats(u) if f in vocab] or [len(vocab)]
        n = len(vocab) + 1
    else:
        idx = lambda u: hashed(feats(u), H)
        n = H
    ci = {g: i for i, g in enumerate(classes)}

    def pack(ds):
        ii = [idx(u) for _, u, _ in ds]
        off = torch.tensor([0] + [len(x) for x in ii[:-1]]).cumsum(0)
        return torch.tensor([j for x in ii for j in x]), off, torch.tensor([ci[g] for _, _, g in ds]), ii
    _, _, y, ii = pack(tr)
    emb = torch.nn.EmbeddingBag(n, len(classes), mode="mean")
    torch.nn.init.zeros_(emb.weight)
    b = torch.nn.Parameter(torch.zeros(len(classes)))
    opt = torch.optim.Adam(list(emb.parameters()) + [b], lr=0.05)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        sel = torch.randint(len(y), (256,), generator=g).tolist()
        flat = torch.tensor([j for s in sel for j in ii[s]])
        off = torch.tensor([0] + [len(ii[s]) for s in sel[:-1]]).cumsum(0)
        loss = F.cross_entropy(emb(flat, off) + b, y[sel]) + 1e-5 * emb.weight.pow(2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()
    flat, off, _, _ = pack(te)
    with torch.no_grad():
        p = (emb(flat, off) + b).argmax(1).tolist()
    return [classes[i] for i in p], n * len(classes) + len(classes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="1000", choices=["1000", "all"], help="青空文庫の作品の組（1000 = 最初の 1,000 作品、all = 全 4,477 作品）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=3, help="lr の seed の数")
    ap.add_argument("--device", default="cpu", help="使わない（約束の引数）")
    ap.add_argument("--out", default="results/est_wig_bag.txt")
    a = ap.parse_args()
    torch.set_num_threads(4)
    d = aozora.load(a.set)
    real = aozora.labeled(d, "train")
    te = aozora.labeled(d, "test")
    sp = spread_items(d)
    classes = [g for g, _ in Counter(g for _, _, g in real).most_common()]
    gold = [g for _, _, g in te]
    maj = 100.0 * sum(g == classes[0] for g in gold) / len(gold)
    print(f"訓練 {len(real)} 件（広げたラベル +{len(sp)}）、テスト {len(te)} 件、いつも一番多い類 {maj:.1f}%")
    conds = [("nb", f, 0, t) for f in FEATS for t in ("real", "spread")]
    conds += [("nb", "end123+bi", H, "real") for H in (256, 1024, 4096)]
    conds += [("lr", f, H, t) for f in ("end123+bi", "end123+bi+tri") for H in (0, 1024, 4096) for t in ("real", "spread")]
    for method, fname, H, tname in conds:
        tr = real + sp if tname == "spread" else real
        t0 = time.perf_counter()
        runs = []
        for s in (range(a.seed, a.seed + a.seeds) if method == "lr" else [a.seed]):
            if method == "nb":
                pred, params = nb(tr, te, FEATS[fname], H, classes), None
            else:
                pred, params = lr(tr, te, FEATS[fname], H, classes, s)
            runs.append(scores(pred, gold, classes))
        accs = [r[0] for r in runs]
        macs = [r[1] for r in runs]
        se = lambda v: (sum((x - sum(v) / len(v)) ** 2 for x in v) / (len(v) - 1)) ** 0.5 / len(v) ** 0.5 if len(v) > 1 else None
        rec = dict(exp="est_wig_bag", set=a.set, method=method, feats=fname, H=H, train=tname, seed=a.seed, n_seeds=len(runs), out=a.out,
                   n_train=len(tr), n_test=len(te), majority=maj, test_acc=sum(accs) / len(accs), test_acc_se=se(accs),
                   macro_recall=sum(macs) / len(macs), macro_recall_se=se(macs), per_class=runs[0][2], params=params,
                   time_s=time.perf_counter() - t0, torch=torch.__version__)
        print(f"{method:3s} {fname:15s} H={H:5d} {tname:6s}  正解率 {rec['test_acc']:.1f}  類ごとの平均 {rec['macro_recall']:.1f}  ({rec['time_s']:.0f} 秒)", flush=True)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
