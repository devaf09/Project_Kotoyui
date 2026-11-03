"""実験12: 似た語の器を融合して、表に無い組を補う — 推し量る力の 3 つ目の案（実験11 の続き）

着想: 持ち主の問い（2026-11-03「推し量る力をもっと詳しく、介入の実験が参考にならないか」）。実験11 で、表に無い組だけの問は
      どの方法もでたらめ並み（PMI 1.1%、語板 0.9%、1 本の光路 9.0%、でたらめ 10%）だった。Sin-CNN3 では「似た物の器から写すと、見ていない属性も受け継ぐ」（実験51）、
      融合 = 出口の明るさの重み付きの足し算（sincnn3.FusedGrain）。そこで、語 x の共鳴に、似た語 x' の器の明るさを重み付きで足す。
方法: 語 x の共鳴 = 自分の器 + β × Σ_{x' = 似た語の上位 k} 近さ(x, x') × x' の器 ÷ Σ 近さ。
      似た語 = 次の発言に出る語の正の PMI の行（よく出る次の語 4,096 個の上）どうしのコサインが近い語（自分は除く）。
      器は 2 通り: table = 上位 61 語に切った数えた表、light = 語板の光（kotoyui 0.2.0 の LightResonance、実験09 の形）。
測る: 10 択（実験06〜11 と同じ問の作り方・同じ seed で同じ問）の全体と、表に無い組だけの問（実験11 の unseen）。
材料: RealPersonaChat・JMultiWOZ（CC BY-SA 4.0）。GPU は使わない（CPU で数分）。
      --fill missing（実験12b、実験13 の後）: 自分の器に無い y だけを補う。補い方は似た語の融合（nb）と 2 段たどり（two: x の行の語 z の器を z の割合で足す）。
結果: results/exp12.txt
実行: CUDA_VISIBLE_DEVICES= python experiments/exp12_fused_neighbors.py --seed 0 --out results/exp12.txt
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import kotoyui as K  # noqa: E402


def neighbors(mat, words, k, dims=4096):
    """語ごとに、次の発言に出る語の正の PMI の行が近い語 k 個と近さ（コサイン）。"""
    ys = [y for y, _ in Counter(mat.cnext).most_common(dims)]
    yid = {y: j for j, y in enumerate(ys)}
    row = {x: i for i, x in enumerate(words)}
    X = torch.zeros(len(words), dims)
    for (x, y), c in mat.cpair.items():
        if x in row and y in yid:
            X[row[x], yid[y]] = mat.pmi(x, y)
    X = X / (X.norm(dim=1, keepdim=True) + 1e-9)
    out = {}
    for c0 in range(0, len(words), 2048):
        S = X[c0:c0 + 2048] @ X.T
        S[torch.arange(len(S)), torch.arange(c0, c0 + len(S))] = -1     # 自分は除く
        v, i = S.topk(k, dim=1)
        for r in range(len(S)):
            out[words[c0 + r]] = [(words[j], s) for j, s in zip(i[r].tolist(), v[r].tolist()) if s > 0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--fill", default="add", choices=["add", "missing"], help="add = 全部の組に足す（実験12）、missing = 表に無い組だけを補う、似た語の融合と 2 段たどり（実験12b）")
    ap.add_argument("--n_q", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/exp12.txt")
    a = ap.parse_args()
    t0 = time.perf_counter()
    mat, tg = K.load_tables()
    convs = K.load_conversations()
    G = K.load_light_resonance(mat)
    words = G.words
    NB = neighbors(mat, words, a.k)
    print(f"似た語 {len(NB)} 語（{time.perf_counter() - t0:.0f} 秒）。例:", [(w, [x for x, _ in NB[w][:3]]) for w in ("犬", "映画", "ラーメン", "仕事", "旅行") if w in NB], flush=True)
    table = {x: dict(zip(G.ys[x], (mat.pmi(x, y) for y in G.ys[x]))) for x in words}
    rows = {"table": lambda x: table.get(x, {}), "light": G.lift}
    fused = {}

    def fuse(kind, x, beta, k, via="nb", fill="add"):
        """語 x の共鳴の行。via = nb: 似た語 k の器を近さで重み付けして足す（融合）、two: x の行の語 z の器を、z の割合で重み付けして足す（2 段たどり）。
        fill = add: 全部の y に足す（実験12）、missing: 自分の器に無い y だけに足す（実験13 の後、表に無い組だけを補う）。"""
        key = (kind, x, beta, k, via, fill)
        if key not in fused:
            base = rows[kind](x)
            own = dict(base)
            if via == "nb":
                nb = NB.get(x, [])[:k]
            else:
                z0 = sum(base.values())
                nb = [(x2, v / z0) for x2, v in base.items() if x2 != x] if z0 > 0 else []
            z = sum(s_ for _, s_ in nb)
            if beta > 0 and z > 0:
                for x2, s_ in nb:                                    # 融合: 器の明るさを重み付けして足す
                    for y, v in rows[kind](x2).items():
                        if fill == "add" or y not in base:
                            own[y] = own.get(y, 0.0) + beta * s_ / z * v
            fused[key] = own
        return fused[key]

    def res(uw, cw_, kind, beta, k, via="nb", fill="add"):
        if not uw or not cw_:
            return 0.0
        return sum(fuse(kind, x, beta, k, via, fill).get(y, 0.0) for x in uw for y in cw_) / math.sqrt(len(uw) * len(cw_))

    rng = random.Random(a.seed)                                      # 実験06〜11 と同じ問の作り方
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
            an[t] = set(K.analyze(t)[0])
        return an[t]
    qs = [(az(ctx[-1]), [az(t) for t in [gold] + negs], [az(t) for t in [gold] + same]) for _, ctx, gold, negs, same in cases]
    unseen = [i for i, (uw, c, _) in enumerate(qs) if mat.resonance(uw, c[0]) == 0]
    rec = dict(exp="12", **vars(a), n_unseen=len(unseen), runs=[], version=K.__version__)
    if a.fill == "add":
        configs = [("B", None, 0.0, 0, "nb")] + [(kind, kind, beta, k, "nb") for kind in ("table", "light") for beta in (0.0, 0.3, 1.0, 3.0) for k in (5, a.k)]
    else:                                                            # 表に無い組だけを補う（実験12b）
        configs = [("B", None, 0.0, 0, "nb"), ("table", "table", 0.0, 0, "nb")] + [(kind, kind, beta, a.k, via) for kind in ("table", "light")
                                                                                  for via in ("nb", "two") for beta in (0.1, 0.3, 1.0)]
    seen = set()
    for name, kind, beta, k, via in configs:
        if beta == 0 and (name, k) in seen:
            continue
        seen.add((name, k if beta else 0))
        f = (lambda uw, c: mat.resonance(uw, c)) if name == "B" else (lambda uw, c, kind=kind, beta=beta, k=k, via=via: res(uw, c, kind, beta, k, via, a.fill))
        r = dict(kind=name, beta=beta, k=k if beta else 0, via=via, fill=a.fill)
        for tag, idx, hard in (("all", range(len(qs)), False), ("hard", range(len(qs)), True), ("unseen", unseen, False)):
            ok = 0
            for i in idx:
                uw, cn, cs = qs[i]
                cands = cs if hard else cn
                sc_ = [f(uw, c) + 1e-6 * rng.random() for c in cands]
                ok += all(sc_[0] > v for v in sc_[1:])
            r[tag] = ok / max(1, len(idx))
        rec["runs"].append(r)
        print(name, via, "β", beta, "k", r["k"], {t: round(r[t], 3) for t in ("all", "hard", "unseen")}, flush=True)
    rec["total_s"] = time.perf_counter() - t0
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
