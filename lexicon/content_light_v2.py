"""語板 v2 / v2f を光の器に学ばせ、v1 と同じ確かめ + 文章の確かめをする（2026-11-05）

器: --device pixel = v2（v1 と同じ: 画素ごとの位相、板 4 枚、600 回）、--device fourier = v2f（v1f と同じ: 周波数成分 512、設計を写した初めの板、300 回）。
確かめ（v1 と同じ）: 学びの正確さ、逆回し、融合、消去、崩した文（役の入れ替え・名詞の入れ替え）。
新しい確かめ:
  1. 崩した文を親の語板で補う: 名詞の語板に「動詞/役」が無いとき、その名詞のクラスの親（子の明るさの和 = 光の和の融合）の値で補う。
  2. 的はずれな文を見分ける（前後の文を使う）: 試しの文書（学びに使っていない）で、真ん中の文を
       (a) 別の文書の文、(b) 名詞を 1 つ別の語に替えた文 に入れ替えて、本当の文と見分ける（AUROC）。
     文の光 = 文の語の語板の明るさ（帯ごとに割合にしたもの）の和。文脈の光 = 前後 2 文の光の和（どちらも光の和の融合）。
     手がかり: 帯ごとの重なり（cos）、文脈の話題 G と候補のクラス D の重なり、候補の語のうちいちばん文脈に合わない語の重なり。
     学び: 手がかりを足す重み（10 個前後のデジタルの重み、ロジスティック回帰）を、試しの文書の半分で決めて残り半分で測る（2 回入れ替えて平均）。
     比べ: v1 の語板（数えた割合、G なし）、v2 の数えた割合、v2 / v2f の光。
結果: <dir>/light_<v2|v2f>_s<seed>.json, light_<v2|v2f>_bright_s<seed>.npy
実行: python lexicon/content_light_v2.py --dir <v2 の置き場> --v1_dir <v1 の置き場> --device pixel|fourier
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import board_light as BL  # noqa: E402
from content_light import PRIV, auroc, band_ids, band_prob, odd_sentence_test, train_dev  # noqa: E402
from content_light_v1f import designed_device, fit, train_v1f  # noqa: E402

BANDS = ["A", "B", "C", "D", "E", "G"]


def load(d, n_boards):
    boards = json.load(open(os.path.join(d, "content_boards.json"), encoding="utf-8"))[:n_boards]
    boards = [b for b in boards if b["attrs"]]
    for b in boards:
        dd = b["attrs"].get("D")
        b["func"] = max(dd, key=dd.get).split(":", 1)[1] if dd else "クラスなし"
    S, SLOT = BL.slots_of(boards)
    Q = np.stack([BL.target(b, SLOT, len(S)) for b in boards]).astype(np.float32)
    return boards, S, SLOT, Q


def band_norm(W, S):
    """帯ごとに割合にした明るさ（語ごと）。帯の中の光が無い語は 0。"""
    ids = band_ids(S)
    out = np.zeros_like(W, dtype=np.float32)
    for b, ix in ids.items():
        x = W[:, ix].astype(np.float32)
        t = x.sum(1, keepdims=True)
        out[:, ix] = np.where(t > 1e-6, x / np.maximum(t, 1e-9), 0)
    return out, ids


def odd_with_parent(W, boards, S, SLOT, triples, cls_of, rng, lam=1.0):
    """崩した文の見分け。名詞の「動詞/役」が 0 のとき、名詞のクラスの親（子の明るさの和）で補う。"""
    idx = {b["name"]: i for i, b in enumerate(boards)}
    ids = band_ids(S)
    groups = defaultdict(list)
    for i, b in enumerate(boards):
        groups[b["func"]].append(i)
    parent = {k: W[v].sum(0) for k, v in groups.items()}               # 光の和の融合
    nouns = [b["name"] for b in boards if b["pos"] == "名詞"]

    def score(noun, verb, role):
        s = 0.0
        j = SLOT.get(("C", f"{verb}/{role}"), -1)
        if noun in idx:
            p = band_prob(W[idx[noun]], ids, "C", j)
            if p == 0 or p < 1e-4:
                p = lam * band_prob(parent[boards[idx[noun]]["func"]], ids, "C", j)
            s += p
        if verb in idx and noun in cls_of:
            s += band_prob(W[idx[verb]], ids, "C", SLOT.get(("C", f"{'主語' if role == 'する' else '目的語'}:{cls_of[noun]}"), -1))
        return s

    T = [t for t in triples if t["noun"] in idx and t["verb"] in idx]
    pos = [score(t["noun"], t["verb"], t["role"]) for t in T]
    swap = [score(t["noun"], t["verb"], "される" if t["role"] == "する" else "する") for t in T]
    repl = [score(rng.choice(nouns), t["verb"], t["role"]) for t in T]
    return dict(組=len(T), 役の入れ替え=round(auroc(pos, swap), 3), 名詞の入れ替え=round(auroc(pos, repl), 3),
                点が0の割合=round(float(np.mean(np.array(pos) == 0)), 3))


def make_items(tdocs, nouns, seed, vocab):
    """文書の真ん中の文ごとに (文脈の語, 本当の文の語, 別の文書の文の語, 名詞を替えた文の語)。
    名詞を替えるのは語板のある名詞だけ（語板の無い語を語板のある語に替えると、語の数の違いで見分けられてしまうため）。"""
    rng = random.Random(seed)
    items = []
    for di, d in enumerate(tdocs):
        ss = d["sents"]
        if len(ss) < 3:
            continue
        for i in range(1, len(ss) - 1):
            ctx = [t["w"] for k in range(max(0, i - 2), min(len(ss), i + 3)) if k != i for t in ss[k]]
            cand = [t["w"] for t in ss[i]]
            other = rng.choice([x for x in range(len(tdocs)) if x != di])
            neg = [t["w"] for t in rng.choice(tdocs[other]["sents"])]
            npos = [k for k, t in enumerate(ss[i]) if t["pos"] == "名詞" and t["w"] in vocab]
            rep = None
            if npos:
                rep = list(cand)
                rep[rng.choice(npos)] = rng.choice(nouns)
            items.append(dict(doc=di, ctx=ctx, cand=cand, neg=neg, rep=rep))
    return items


def features(Wn, ids, idx, ctx, cand, bands):
    """文脈と候補の文の手がかり。Wn: 帯ごとに割合にした明るさ。"""
    ci = [idx[w] for w in ctx if w in idx]
    ki = [idx[w] for w in cand if w in idx]
    f = []
    if not ci or not ki:
        return None
    C, K = Wn[ci].sum(0), Wn[ki].sum(0)

    def cos(x, y):
        return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))
    for b in bands:
        f.append(cos(C[ids[b]], K[ids[b]]))
    if "G" in bands:
        # 文脈の話題 G と、候補の語のクラス D（同じクラスの名前どうしを並べる）
        f.append(cos(C[ids["Gm"]], K[ids["Dm"]]))
        per = [cos(C[ids["G"]], Wn[k][ids["G"]]) for k in ki]
        f.append(min(per))
    per = [cos(C[ids["D"]], Wn[k][ids["D"]]) for k in ki]
    f.append(min(per))
    per = [cos(C[ids["C"]], Wn[k][ids["C"]]) for k in ki]
    f.append(min(per))
    return f


def logreg(X, y, l2=1e-2, it=500, lr=0.5):
    X = np.asarray(X, dtype=np.float64)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[(X - mu) / sd, np.ones(len(X))]
    w = np.zeros(Z.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-Z @ w))
        g = Z.T @ (p - y) / len(y) + l2 * np.r_[w[:-1], 0]
        w -= lr * g
    return lambda Xt: np.c_[(np.asarray(Xt) - mu) / sd, np.ones(len(Xt))] @ w, w


def context_test(W, boards, S, items, bands):
    idx = {b["name"]: i for i, b in enumerate(boards)}
    Wn, ids = band_norm(W, S)
    ids = dict(ids)
    if "G" in bands and "D" in ids:
        # G の「話題:X」と D の「クラス:X」の同じ名前どうしを並べた番号（文脈の話題と候補のクラスの重なりを測る）
        gname = {S[j][1].split(":", 1)[1]: j for j in ids["G"]}
        dname = {S[j][1].split(":", 1)[1]: j for j in ids["D"]}
        common = sorted(set(gname) & set(dname))
        ids["Gm"], ids["Dm"] = [gname[c] for c in common], [dname[c] for c in common]
    out = {}
    for kind in ("neg", "rep"):
        X, y, g = [], [], []
        for it in items:
            if it[kind] is None:
                continue
            fp = features(Wn, ids, idx, it["ctx"], it["cand"], bands)
            fn = features(Wn, ids, idx, it["ctx"], it[kind], bands)
            if fp is None or fn is None:
                continue
            X += [fp, fn]
            y += [1, 0]
            g += [it["doc"], it["doc"]]
        X, y, g = np.array(X), np.array(y), np.array(g)
        res = {}
        names = [f"重なり_{b}" for b in bands] + (["話題G×クラスD", "いちばん合わない語_G"] if "G" in bands else []) + ["いちばん合わない語_D", "いちばん合わない語_C"]
        for k, nm in enumerate(names):
            res[nm] = round(auroc(X[y == 1, k], X[y == 0, k]), 3)
        half = (g % 2 == 0)
        sc = np.zeros(len(y))
        for tr in (half, ~half):
            f, _ = logreg(X[tr], y[tr])
            sc[~tr] = f(X[~tr])
        res["学んだ重みで足す"] = round(auroc(sc[y == 1], sc[y == 0]), 3)
        res["対"] = int(len(y) // 2)
        out["別の文書の文" if kind == "neg" else "名詞を1つ替えた文"] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(PRIV, "data", "boards", "v2"))
    ap.add_argument("--v1_dir", default=os.path.join(PRIV, "data", "boards", "v1"))
    ap.add_argument("--device", choices=["pixel", "fourier"], default="pixel")
    ap.add_argument("--n_boards", type=int, default=5000)
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--plates", type=int, default=4)
    ap.add_argument("--m", type=int, default=512)
    ap.add_argument("--steps", type=int, default=0, help="0 = pixel 600 / fourier 300")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_fusion", action="store_true")
    a = ap.parse_args()
    steps = a.steps or (600 if a.device == "pixel" else 300)
    tag = "v2" if a.device == "pixel" else "v2f"
    torch.set_num_threads(4)
    rng = random.Random(a.seed)
    boards, S, SLOT, Q = load(a.dir, a.n_boards)
    A = len(S)
    assert BL.VAL0 + A <= a.n, f"画素が足りない（属性 {A}）"
    triples = json.load(open(os.path.join(a.dir, "test_triples.json"), encoding="utf-8"))
    tdocs = json.load(open(os.path.join(a.dir, "test_docs.json"), encoding="utf-8"))
    cls_of = {b["name"]: b["func"] for b in boards if b["pos"] == "名詞" and b["func"] != "クラスなし"}
    print(tag, "語板", len(boards), "属性", A, flush=True)
    res = dict(設定=dict(vars(a), steps=steps), 語板=len(boards), 属性=A)
    t0 = time.perf_counter()
    if a.device == "pixel":
        W, _ = train_dev(Q, a.n, a.plates, steps, a.seed)
        res["器の重み_1枚"] = a.plates * a.n
    else:
        W = train_v1f(Q, a.n, a.plates, a.m, steps, a.seed)
        res["器の重み_1枚"] = a.plates * 2 * a.m
    res["学ぶ秒"] = round(time.perf_counter() - t0, 1)
    res["学びの正確さ_全変動"] = round(float(np.mean([BL.tv(W[i], b, S, SLOT) for i, b in enumerate(boards)])), 4)
    res["光の総量"] = round(float(W.sum(1).mean()), 3)
    print("学んだ", res["学ぶ秒"], res["学びの正確さ_全変動"], flush=True)
    np.save(os.path.join(a.dir, f"light_{tag}_bright_s{a.seed}.npy"), W.astype(np.float16))
    res["逆回し"] = {q: BL.reverse_test(W, boards, S, SLOT, qv, rng) for q, qv in
                    [("C（動的）", ["C"]), ("A+C", ["A", "C"]), ("B+C", ["B", "C"]), ("G（話題）", ["G"]), ("C+G", ["C", "G"])]}
    keep = defaultdict(list)
    for i, b in enumerate(boards):
        keep[b["func"]].append(i)
    sub = sorted(i for k, v in keep.items() if len(v) >= 5 and k != "クラスなし" for i in v[:40])
    sb = [boards[i] for i in sub]
    if not a.skip_fusion:
        if a.device == "pixel":
            dev_sub = BL.BatchDevice(len(sub), n=a.n, plates=a.plates)
            q = torch.tensor(Q[sub])
            opt = torch.optim.Adam(dev_sub.parameters(), lr=0.03)
            for _ in range(steps):
                Wd = dev_sub.bright(A)
                p = torch.cat([Wd, (1 - Wd.sum(1, keepdim=True)).clamp(min=1e-9)], 1)
                loss = -(torch.cat([q, torch.zeros(len(q), 1)], 1) * torch.log(p + 1e-9)).sum(1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        else:
            dev_sub = fit(designed_device(Q[sub], a.n, a.plates, a.m, a.seed), Q[sub], steps)
        res["融合"] = BL.fusion_test(dev_sub, W[sub], sb, S, SLOT, Q[sub], a.n, steps, a.seed)
        if a.device == "fourier":
            groups = defaultdict(list)
            for i, b in enumerate(sb):
                groups[BL.label_of(b)].append(i)
            names = [k for k, v in groups.items() if len(v) >= 2]
            truth = np.stack([Q[sub][groups[k]].mean(0) for k in names]).astype(np.float32)
            for st in (100, steps):
                dp = fit(designed_device(truth, a.n, a.plates, a.m, a.seed + 9), truth, st)
                with torch.no_grad():
                    Wp = dp.bright(A).numpy()
                res["融合"][f"周波数で融合（設計 + {st} 回）"] = round(float(np.mean(
                    [0.5 * np.abs(Wp[j] / Wp[j].sum() - truth[j] / truth[j].sum()).sum() for j in range(len(names))])), 3)
    res["消去"] = BL.erase_test(W[sub], sb, S)
    res["崩した文_光"] = odd_sentence_test(W, boards, S, SLOT, triples, cls_of, random.Random(1))
    res["崩した文_数えた割合"] = odd_sentence_test(Q, boards, S, SLOT, triples, cls_of, random.Random(1))
    res["崩した文_光_親で補う"] = odd_with_parent(W, boards, S, SLOT, triples, cls_of, random.Random(1))
    res["崩した文_数えた割合_親で補う"] = odd_with_parent(Q, boards, S, SLOT, triples, cls_of, random.Random(1))
    vocab = {b["name"] for b in boards}
    v1p = os.path.join(a.v1_dir, "content_boards.json")
    if os.path.exists(v1p):
        b1, S1, SLOT1, Q1 = load(a.v1_dir, a.n_boards)
        vocab &= {b["name"] for b in b1}
    nouns = sorted(b["name"] for b in boards if b["pos"] == "名詞" and b["name"] in vocab)
    items = make_items(tdocs, nouns, a.seed, vocab)
    res["的はずれな文_光"] = context_test(W, boards, S, items, BANDS)
    res["的はずれな文_数えた割合"] = context_test(Q, boards, S, items, BANDS)
    res["的はずれな文_v2でGなし_数えた割合"] = context_test(Q, boards, S, items, [b for b in BANDS if b != "G"])
    if os.path.exists(v1p):
        res["的はずれな文_v1_数えた割合"] = context_test(Q1, b1, S1, items, ["A", "B", "C", "D", "E"])
        tr1 = os.path.join(a.v1_dir, "test_triples.json")
        if os.path.exists(tr1):
            cls1 = {b["name"]: b["func"] for b in b1 if b["pos"] == "名詞" and b["func"] != "クラスなし"}
            res["崩した文_v1_数えた割合_親で補う"] = odd_with_parent(Q1, b1, S1, SLOT1, json.load(open(tr1, encoding="utf-8")), cls1, random.Random(1))
    json.dump(res, open(os.path.join(a.dir, f"light_{tag}_s{a.seed}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "設定"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
