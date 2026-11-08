"""介入の研究 実験I27u: 小説をまたいで共通のキャラクター像を作り、作品の中の名前に当てて育てる

持ち主（2026-11-08）: 「今小説全部で共通するキャラクター像で最初固められないかなと思ってる」「２は特にいい案だね」
I27t（語尾から学び、介入で広げる）は、1 作品の正解が 20 台詞ほどで、人ごとの板が薄かった。上からの流れを足す:
  1. 像を作る: I27t で育てた人ごとの板（語尾・一人称の割合）を、ほかの作品の人どうしで重ね、似たものを束ねる（k 平均）。
     束の平均が「像」（すかして浮き上がる形）。自分の作品の人は使わない。
  2. 当てる: 作品の中の名前ごとに、いまの板と像の重なり（cos）で、像の混ぜ方を決める。
  3. 育てる: 語尾の差分で、その人の台詞が少ないときは、作品全体の割合ではなく、像の割合に寄せる（I27t の ending_score の back）。
     介入で広げるたびに、名前の板が変わるので、当て直す。
物差しは I27t と同じ（正解を交互に半分隠し、隠した台詞の話し手を当てる）。
Web 小説は _wn。結果は非公開の置き場だけ。本文はコードに書かない。
実行: python intervention/i27u_character_images.py --cache <解析済み JSON> --stats <jsonl> --out <像の md>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i27t_first_person as T  # noqa: E402
from board_ops import light  # noqa: E402

USE = {"語尾", "流れ", "多い人", "一人称", "広げる"}


def person_vecs(P, min_n=3.0):
    """人ごとの板 → {人: (重み, {特徴: 割合})}。"""
    out = {}
    people = {k[0] for k in P.B if k[0] != "全" and k[1] == "数"}
    for c in people:
        n = light(P.B, (c, "数"))
        if n < min_n:
            continue
        sh = {k[1]: v / n for k, v in P.B.items() if k[0] == c and k[1] != "数"}
        out[c] = (n, sh)
    return out


def kmeans(X, W, K, seed=0, iters=30):
    rng = np.random.default_rng(seed)
    n = len(X)
    K = min(K, n)
    idx = [int(rng.choice(n, p=W / W.sum()))]
    for _ in range(1, K):
        d = 1 - (X @ X[idx].T).max(1)
        p = np.maximum(d, 0) * W
        idx.append(int(rng.choice(n, p=p / p.sum())) if p.sum() > 0 else int(rng.integers(n)))
    C = X[idx].copy()
    for _ in range(iters):
        a = (X @ C.T).argmax(1)
        for j in range(K):
            m = a == j
            if m.any():
                c = (X[m] * W[m, None]).sum(0)
                C[j] = c / (np.linalg.norm(c) + 1e-9)
    return C, (X @ C.T).argmax(1)


class Images:
    """ほかの作品の人の板から作った像。"""

    def __init__(self, vecs_by_work, K, beta):
        self.K, self.beta = K, beta
        feats = Counter()
        for vs in vecs_by_work:
            for c, (n, sh) in vs.items():
                for f in sh:
                    feats[f] += 1
        self.vocab = [f for f, m in feats.most_common() if m >= 2][:600]
        self.ix = {f: i for i, f in enumerate(self.vocab)}
        self.rows = []
        for wi, vs in enumerate(vecs_by_work):
            for c, (n, sh) in vs.items():
                self.rows.append((wi, c, n, sh))
        self.cache = {}

    def vec(self, sh):
        v = np.zeros(len(self.vocab))
        for f, x in sh.items():
            if f in self.ix:
                v[self.ix[f]] = x
        return v

    def for_work(self, wi):
        """作品 wi を除いた人で像を作る（作品ごとに 1 回）。像 = (cos 用の向き, 割合の平均)。"""
        if wi in self.cache:
            return self.cache[wi]
        rs = [r for r in self.rows if r[0] != wi]
        S = np.array([self.vec(r[3]) for r in rs])
        X = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-9)
        W = np.array([math.log1p(r[2]) for r in rs])
        C, a = kmeans(X, W, self.K, seed=wi + 1)
        M = np.array([(S[a == j] * W[a == j, None]).sum(0) / max(1e-9, W[a == j].sum()) for j in range(len(C))])
        self.cache[wi] = (C, M, a, rs)
        return self.cache[wi]

    def back(self, wi, P, cands):
        C, M, _, _ = self.for_work(wi)
        vs = person_vecs(P, min_n=0.0)
        out = {}
        for c in cands:
            if c not in vs:
                continue
            v = self.vec(vs[c][1])
            nv = np.linalg.norm(v)
            if nv == 0:
                continue
            sim = C @ (v / nv)
            w = np.exp(self.beta * (sim - sim.max()))
            w /= w.sum()
            m = w @ M
            out[c] = {f: float(m[i]) for i, f in enumerate(self.vocab) if m[i] > 0}
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    works = json.load(open(a.cache, encoding="utf-8"))
    works_q = [T.quotes_of(w) for w in works]
    narr = [T.narr_class(w) for w in works]
    rec = dict(exp="I27u", 結果={})
    base = Counter()
    base_n = 0
    pass1 = {}
    for fold in (0, 1):
        h, n, boards, _ = T.run_fold(works_q, fold, USE, a.rounds, narr)
        base.update(h)
        base_n += n
        pass1[fold] = [person_vecs(P) for P in boards]

    def summ(h, n):
        return dict(当たり=round(h["当たり"] / max(1, n), 4), 数=n,
                    当たり_話す動詞=round(h[("当", "話す動詞")] / max(1, h[("数", "話す動詞")]), 4))
    rec["結果"]["I27t（像なし）"] = summ(base, base_n)
    print("I27t", rec["結果"]["I27t（像なし）"], flush=True)
    for K in (4, 8, 16):
        for beta in (5.0, 20.0):
            h2, n2 = Counter(), 0
            for fold in (0, 1):
                im = Images(pass1[fold], K, beta)
                h, n, _, _ = T.run_fold(works_q, fold, USE, a.rounds, narr, image=im.back)
                h2.update(h)
                n2 += n
            name = f"像 K={K} β={int(beta)}"
            rec["結果"][name] = summ(h2, n2)
            print(name, rec["結果"][name], flush=True)
    # 像の中身（全作品の人、fold 0 の板から。報告用）
    im = Images(pass1[0], 8, 20.0)
    C, M, asg, rs = im.for_work(-1)
    lines = ["# I27u 小説をまたいで共通のキャラクター像（非公開）", "",
             "I27t で育てた人ごとの板（台詞の最後の 1 語の割合と、一人称の語）を、50 作品の人どうしで束ねた（k 平均、K=8）。",
             "その像らしい語尾 = 像の割合 − 全員の平均の割合 が大きい順。", ""]
    allm = M.mean(0)
    for j in range(len(C)):
        mem = [rs[i] for i in range(len(rs)) if asg[i] == j]
        if not mem:
            continue
        d = M[j] - allm
        top = [im.vocab[i] for i in np.argsort(-d)[:8] if d[i] > 0]
        ends = [f[1] for f in top if f[0] == "尾1"][:6]
        fps = sorted(((M[j][im.ix[f]], f[1]) for f in im.vocab if f[0] == "一"), reverse=True)[:3]
        narr_share = sum(1 for r in mem if r[1] == T.NARR) / len(mem)
        lines.append(f"## 像 {j + 1}（{len(mem)} 人、うち語り手 {narr_share:.0%}）")
        lines.append(f"- らしい語尾: {'、'.join(ends)}")
        lines.append(f"- 一人称: {'、'.join(f'{w} {v:.2f}' for v, w in fps if v > 0) or '—'}")
        lines.append(f"- 人（作品の番号の後ろ 4 けた）: {'、'.join(f'{r[1]}（{works[r[0]]["name"][-8:-4]}）' for r in sorted(mem, key=lambda r: -r[2])[:8])}")
        lines.append("")
        rec.setdefault("像", []).append(dict(人数=len(mem), 語尾=ends, 一人称=[w for v, w in fps if v > 0], 語り手の割合=round(narr_share, 3)))
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    with open(a.stats, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    random.seed(0)
    main()
