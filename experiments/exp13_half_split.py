"""実験13: 推し量る力を、教えの会話を半分に分けて測り直す（実験11・12 の続き）

着想: 持ち主の問い（2026-11-03「推し量る力をもっと詳しく」）と、実験12 で分かったこと（「表に無い組の問」は話の形でつながる短い返事で、語の推し量りの物差しに外れていた）。
方法: 教えの会話を、会話ごとに 2 つに分ける（前半 A・後半 B、crc32 で決める）。A だけで表・器を作り、
      「B に 3 回以上出る組なのに、A には 1 回も出ない組（x → y）」= 新しい組を、どれだけ高く点付けできるかを測る。
      比べる相手（負の組）: 同じ x と、A・B のどちらでも x の次に 1 度も出ない y（B の次の発言での出やすさに合わせて選ぶ）。
      物差しは AUC（新しい組の点 > 負の組の点 となる割合。0.5 = 見分けられない、1.0 = 完全）。点が同じなら半分と数える。
方法（点の付け方、どれも A だけから）:
      PMI           数えた表（新しい組は定義から 0 → AUC 0.5 付近になるはず。目安）
      出やすさだけ  A の次の発言での y の回数（x を見ない。y がよく出る語かどうかだけで見分けられる分の目安）
      融合          似た語（A の次の語の正の PMI の行が近い 20 語）の表を近さで重み付けして足す（実験12 の形）
      2 段          x → z → y（x の次に出やすい z の、さらに次に出やすい y。表を 2 回たどる、sincnn3 の RelationChain の考え方）
      光の光路      実験08 の最良の形（マス + 像の面の板 + 像の面で読む）を A の組で学習。語 x 1 つを入れた明るさの、ふだんの出やすさとの比（対数）
      低ランク      光を使わない学習する低ランク（同じ学び方）
語の範囲: よく出る中身の語 4,095（全体の教えで数える、実験07・08 と同じ）。
材料: RealPersonaChat・JMultiWOZ（CC BY-SA 4.0）。
結果: results/exp13.txt
実行: python experiments/exp13_half_split.py --seed 0 --device cuda --out results/exp13.txt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import zlib
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import kotoyui as K  # noqa: E402
from exp08_resonance_fix import L, NV, ResonanceGrain  # noqa: E402


def count(convs, ids):
    """会話 → 語の組の回数（前の発言の語 x → 次の発言の語 y）・x の回数・y の回数・発言の組の数、と教えの組（P, Q）。"""
    cp, cx, cy, n, P, Q = Counter(), Counter(), Counter(), 0, [], []
    for _, turns in convs:
        A = [ids(t) for t in turns]
        for i in range(len(A) - 1):
            if A[i] and A[i + 1]:
                n += 1
                cx.update(A[i])
                cy.update(A[i + 1])
                for x in A[i]:
                    for y in A[i + 1]:
                        cp[(x, y)] += 1
                P.append(A[i])
                Q.append(A[i + 1])
    return cp, cx, cy, n, P, Q


def auc(pos, neg):
    """pos・neg の点（同じ長さで組になっている）→ pos > neg の割合（同点は 0.5）。"""
    return sum(1.0 if p > q else 0.5 if p == q else 0.0 for p, q in zip(pos, neg)) / len(pos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=9000)
    ap.add_argument("--n_pairs", type=int, default=20000, help="測る新しい組の数（上限）")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/exp13.txt")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    t0 = time.perf_counter()
    convs = K.load_conversations()["train"]
    freq = Counter()
    for _, turns in convs:
        for t in turns:
            freq.update(K.analyze(t)[0])
    vocab = [w for w, _ in freq.most_common(NV)]
    wid = {w: i + 1 for i, w in enumerate(vocab)}
    ids = lambda t: [wid[w] for w in K.analyze(t)[0] if w in wid]
    half = lambda i: zlib.crc32(f"half{a.seed}:{i}".encode()) % 2              # 会話ごとに A（0）・B（1）
    cA = [c for i, c in enumerate(convs) if half(i) == 0]
    cB = [c for i, c in enumerate(convs) if half(i) == 1]
    pA, xA, yA, nA, P, Q = count(cA, ids)
    pB, _, yB, _, _, _ = count(cB, ids)
    pmiA = lambda x, y: max(0.0, math.log(pA[(x, y)] * nA / (xA[x] * yA[y]))) if pA.get((x, y), 0) >= 2 else 0.0
    # 新しい組と、負の組
    rng = random.Random(a.seed)
    new = [(x, y) for (x, y), c in pB.items() if c >= 3 and pA.get((x, y), 0) == 0 and xA[x] >= 5]
    rng.shuffle(new)
    new = new[:a.n_pairs]
    ylist, yw = zip(*yB.items())
    pos, negs = [], []
    for x, y in new:
        for _ in range(50):
            y2 = rng.choices(ylist, yw)[0]
            if pA.get((x, y2), 0) == 0 and pB.get((x, y2), 0) == 0:
                pos.append((x, y))
                negs.append((x, y2))
                break
    print(f"会話 A {len(cA)}・B {len(cB)}、新しい組 {len(pos)}（{time.perf_counter() - t0:.0f} 秒）", flush=True)
    rec = dict(exp="13", **vars(a), n_A=len(cA), n_B=len(cB), n_new=len(pos), runs=[], version=K.__version__, torch=torch.__version__)

    def report(name, f):
        r = dict(name=name, auc=auc([f(x, y) for x, y in pos], [f(x, y) for x, y in negs]))
        r["pos_nonzero"] = sum(f(x, y) != 0 for x, y in pos[:2000]) / min(2000, len(pos))
        rec["runs"].append(r)
        print(name, {k: round(v, 3) for k, v in r.items() if k != "name"}, flush=True)
    report("PMI（A の表）", pmiA)
    report("ふだんの出やすさだけ（x を見ない、目安）", lambda x, y: yA.get(y, 0))
    # 融合（似た語の表）
    rows = defaultdict(dict)
    for (x, y), c in pA.items():
        p = pmiA(x, y)
        if p > 0:
            rows[x][y] = p
    X = torch.zeros(L * L, L * L)
    for x, d in rows.items():
        for y, p in d.items():
            X[x, y] = p
    Xn = X / (X.norm(dim=1, keepdim=True) + 1e-9)
    S = Xn @ Xn.T
    S.fill_diagonal_(0)
    v, i = S.topk(a.k, dim=1)
    W = torch.zeros_like(S).scatter_(1, i, v.clamp(min=0))
    W = W / (W.sum(1, keepdim=True) + 1e-9)
    fused = W @ X                                                     # 似た語の表の重み付きの足し算
    report("融合（似た語 20 の表）", lambda x, y: float(fused[x, y]))
    Xp = X / (X.sum(1, keepdim=True) + 1e-9)
    two = Xp @ X                                                      # 2 段: x → z（割合）→ y（PMI）
    report("2 段（x → z → y）", lambda x, y: float(two[x, y]))
    # 光の光路・低ランク（A の組で学習）
    uni = torch.zeros(L * L)
    for q in Q:
        uni[q] += 1.0 / len(q)
    log_uni = torch.log((uni + 1) / (uni + 1).sum())

    def dense(lists, normalize=False):
        z = torch.zeros(len(lists), L * L)
        for r, l in enumerate(lists):
            z[r, l] = 1.0 / len(l) if normalize else 1.0
        return z
    for name, kw in (("光の光路（実験08 の最良の形）", dict(read="image", input="pixel", plates="xy")), ("低ランク（デジタル）", dict(read="lowrank"))):
        m = ResonanceGrain(4, 4, a.seed, **kw).to(dev)
        opt = torch.optim.Adam(m.parameters(), lr=0.01)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
        g = torch.Generator().manual_seed(a.seed)
        t1 = time.perf_counter()
        for k in range(a.steps):
            idx = torch.randint(len(P), (256,), generator=g).tolist()
            loss = -(dense([Q[j] for j in idx], True).to(dev) * m(dense([P[j] for j in idx]).to(dev))).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
        m.eval()
        with torch.no_grad():
            xs = sorted({x for x, _ in pos})
            lift = {}
            for c0 in range(0, len(xs), 256):
                b = xs[c0:c0 + 256]
                out = m(dense([[x] for x in b]).to(dev)).cpu() - log_uni
                for r, x in enumerate(b):
                    lift[x] = out[r]
        report(name, lambda x, y: float(lift[x][y]))
        rec["runs"][-1]["train_s"] = time.perf_counter() - t1
        rec["runs"][-1]["train_loss"] = loss.item()
    rec["total_s"] = time.perf_counter() - t0
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as fo:
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
