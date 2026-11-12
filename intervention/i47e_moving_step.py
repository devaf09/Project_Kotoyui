"""I47e: 「動かす段」の試し。語の波を、その場の語との共鳴で書き換えてから重ねる。

I47a〜d の形は、語の波がいつも同じまま足されるだけで、場面に応じて動く所が無い。
ここでは 1 手だけ足す: 行（と前後の行）の語どうしを共鳴させ、強く響いた相手の波を自分に混ぜる。
重みは置かず、I47b のそろえ方（強い所の平均を 1 にし、1 に近い所だけ残す）で決める。

  動かす前 : 語の波 w
  動かす   : w' = w + Σ（そろえて残った相手 j）共鳴の強さ × w_j     … これを何段か重ねられる

測ること:
  sense : 2 つの語を 1 つの語にまとめた「にせの多義語」（例: 電話|橋）を作って学ばせ、出てきた場面ごとに
          動かした波が、元の語ごとに分かれるか（同じ元どうしの波が、違う元どうしより似ている割合。0.5 = 分かれない）。
          比べる相手: まわりの語を同じ重さで混ぜるだけ（共鳴で選ばない）。
  scoop : 動かした波で行の波を作り、次の行の語の当たりが変わるか（I47a と同じ測り方）。

使い方:
  python intervention/i47e_moving_step.py --cache wn50x5_parsed.json --out i47e_wn.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter

import numpy as np

from i47b_layers_negation import Waves, content
from i47d_negation_band import auc


def evenv(x, times=1):
    y = np.maximum(0.0, x).copy()
    for _ in range(times):
        nz = y[y > 0]
        if not nz.size:
            break
        s = nz[nz >= nz.mean()].mean()
        y = np.where(y / (s + 1e-9) > 0.5, y / (s + 1e-9), 0.0)
    return y


def move(wv, ids, mode="res", steps=1, times=1):
    """ids の語の波を、互いの共鳴で書き換えて返す（語 × D）。mode: res = 共鳴でそろえて選ぶ、uni = 同じ重さ、none = 動かさない。"""
    X = wv.W[ids].copy()
    if mode == "none" or len(ids) < 2:
        return X
    for _ in range(steps):
        if mode == "uni":
            A = np.ones((len(ids), len(ids)))
        else:
            Cn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
            A = np.abs(Cn.conj() @ Cn.T)
        np.fill_diagonal(A, 0.0)
        if mode == "res":
            A = np.stack([evenv(r, times) for r in A])
        A = A / (A.sum(1, keepdims=True) + 1e-9)
        X = X + A @ X
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) / math.sqrt(wv.D) + 1e-9)
    return X


def part_sense(seq, a, res):
    rng = random.Random(a.seed)
    df = Counter(w for ws in seq for w in set(ws))
    mid = sorted(w for w, c in df.items() if a.lo <= c <= a.hi)
    rng.shuffle(mid)
    pairs = [(mid[2 * i], mid[2 * i + 1]) for i in range(min(a.pairs, len(mid) // 2))]
    mp = {}
    for x, y in pairs:
        mp[x] = mp[y] = x + "|" + y
    seq2 = [[mp.get(w, w) for w in ws] for ws in seq]
    orig = [[w if w in mp else None for w in ws] for ws in seq]
    df2 = Counter(w for ws in seq2 for w in set(ws))
    wv = Waves(sorted(w for w, c in df2.items() if c >= 2), D=a.D, seed=a.seed)
    wv.learn(seq2, mix=a.mix)
    modes = {"まわりを同じ重さで混ぜる": ("uni", 1, 1), "共鳴で 1 回そろえて混ぜる（1 段）": ("res", 1, 1),
             "共鳴で 4 回そろえて混ぜる（1 段）": ("res", 1, 4), "共鳴で 1 回そろえて混ぜる（2 段）": ("res", 2, 1),
             "共鳴で 4 回そろえて混ぜる（2 段）": ("res", 2, 4)}
    occ = {}
    for k, ws in enumerate(seq2):
        for i, w in enumerate(ws):
            if "|" in w and orig[k][i]:
                occ.setdefault(w, []).append((k, orig[k][i]))
    tot = {m: [] for m in modes}
    for w, oc in occ.items():
        rng.shuffle(oc)
        oc = oc[: a.occ]
        if len({o for _, o in oc}) < 2:
            continue
        vec = {m: [] for m in modes}
        lab = []
        for k, o in oc:
            ctx = list(dict.fromkeys(x for kk in (k - 1, k, k + 1) if 0 <= kk < len(seq2) for x in seq2[kk] if x in wv.ix and x != w))
            if not ctx:
                continue
            ids = [wv.ix[w]] + [wv.ix[x] for x in ctx]
            lab.append(o)
            for m, (md, st, tm) in modes.items():
                v = move(wv, ids, md, st, tm)[0] - wv.W[ids[0]]      # 動いた分だけを見る
                vec[m].append(v / (np.linalg.norm(v) + 1e-9))
        lab = np.array(lab)
        same = lab[:, None] == lab[None]
        iu = np.triu_indices(len(lab), 1)
        for m in modes:
            V = np.stack(vec[m])
            S = np.abs(V.conj() @ V.T)[iu]
            tot[m].append(auc(S[same[iu]], S[~same[iu]]))
    res["にせの多義語"] = dict(組=len(tot[next(iter(tot))]), 例=[x + "|" + y for x, y in pairs[:8]],
                         元の語ごとに分かれる割合={m: round(float(np.nanmean(v)), 3) for m, v in tot.items()})


def part_scoop(works, a, res):
    tr_w, te_w = works[a.test_works:], works[:a.test_works]

    def scenes(ws_):
        out = []
        for wk in ws_:
            L = wk["lines"]
            cw = [content(ln["toks"]) for ln in L]
            dl = [ln["text"].startswith("「") for ln in L]
            for i in range(1, len(L)):
                if dl[i - 1] and dl[i] and cw[i - 1] and cw[i]:
                    out.append((cw[i - 1], cw[i]))
        return out
    train, test = scenes(tr_w), scenes(te_w)
    seq = [content(ln["toks"]) for wk in tr_w for ln in wk["lines"]]
    df = Counter(w for ws in seq for w in set(ws))
    idf = {w: math.log((len(seq) + 1) / (c + 0.5)) for w, c in df.items()}
    wv = Waves(sorted(df), D=a.D, seed=a.seed)
    wv.learn(seq, mix=a.mix)
    freq = Counter(w for _, b in train for w in b)
    prior = [w for w, _ in freq.most_common(200)]
    common = set(prior[:50])
    z = sum(freq.values())

    def line(ws, md, st, tm):
        ws = [w for w in dict.fromkeys(ws) if w in wv.ix]
        if not ws:
            return None
        X = move(wv, [wv.ix[w] for w in ws], md, st, tm)
        v = sum(X[i] * idf.get(w, 1.0) for i, w in enumerate(ws))
        return v / (np.linalg.norm(v) + 1e-9)
    out = {}
    Z = np.zeros(a.D, complex)
    for name, (md, st, tm) in {"動かさない（I47a）": ("none", 0, 0), "同じ重さで混ぜる": ("uni", 1, 1), "共鳴で 1 回そろえて混ぜる（1 段）": ("res", 1, 1),
                               "共鳴で 4 回そろえて混ぜる（1 段）": ("res", 1, 4), "共鳴で 4 回そろえて混ぜる（2 段）": ("res", 2, 4)}.items():
        K = np.stack([x if (x := line(q, md, st, tm)) is not None else Z for q, _ in train])
        t = [0.0, 0, 0.0, 0]
        for q, b in test:
            v = line(q, md, st, tm)
            c = Counter()
            if v is not None:
                r = np.abs(K.conj() @ v)
                for j in np.argsort(-r)[: a.top]:
                    for w in set(train[j][1]):
                        c[w] += r[j] ** 2
            s = sum(c.values()) or 1.0
            sc = Counter({w: (1 - a.lam) * x / s for w, x in c.items()})
            for w in prior:
                sc[w] += a.lam * freq[w] / z
            top = {w for w, _ in sc.most_common(10)}
            B, Bi = set(b), set(b) - common
            t[0] += len(B & top) / len(B)
            t[1] += bool(B & top)
            if Bi:
                t[2] += len(Bi & top) / len(Bi)
                t[3] += 1
        n = len(test)
        out[name] = dict(再現率=round(t[0] / n, 4), 当たり=round(t[1] / n, 4), 多い語を除いた再現率=round(t[2] / max(1, t[3]), 4))
    res["すくい上げ"] = dict(train=len(train), test=len(test), 結果=out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--test_works", type=int, default=10)
    ap.add_argument("--D", type=int, default=512)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--mix", type=float, default=0.6)
    ap.add_argument("--pairs", type=int, default=60)
    ap.add_argument("--occ", type=int, default=80)
    ap.add_argument("--lo", type=int, default=30)
    ap.add_argument("--hi", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    works = json.load(open(a.cache, encoding="utf-8"))
    works = works["works"] if isinstance(works, dict) else works
    random.Random(a.seed).shuffle(works)
    seq = [content(ln["toks"]) for wk in works for ln in wk["lines"]]
    res = dict(行=len(seq))
    part_sense(seq, a, res)
    part_scoop(works, a, res)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
