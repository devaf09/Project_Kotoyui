"""介入の研究 実験I16: 本物の文で「注意の経験」— 崩した文の見分けで間違えた場面を覚え、似た場面で光らせて、答えを直す

持ち主の案（2026-11-05）: 「1 − 誤った経験で注意の経験を作る。差分を作ってそれを比較する」。
材料: 語板 v1 の崩した文の試し（学びに使っていない文の「名詞・動詞・役」8,445 組、../Project_Kotoyui_private/data/boards/v1/test_triples.json）。
もとの点（語板のスレッドと同じ）: 名詞の語板の「動詞/役」の明るさ（C 帯の中の割合）+ 動詞の語板の「主語:クラス」または「目的語:クラス」の明るさ。
崩し方: 役の入れ替え（する ↔ される）、名詞の入れ替え（別の名詞に）。見分け = 本物の組の点が崩した組より高いか（AUROC）。
方法:
  1. 組を半分に分ける（経験用・試し用）。経験用で、もとの点の誤り（本物なのに点が低い、崩したのに点が高い。区切り = 点の中央値）を集める。
  2. 経験 = 場面の記号の組: {動詞:…, 役:…, クラス:（名詞のクラス）, 点:（4 段階）, 崩し:（本物か崩したかは使わない）} + 結論:誤り / 結論:正しい。
  3. 経験の器 = 口 2 つ（誤り・正しい）の光の器（I8b の labeled）。注意の明るさ = 誤りの口の起こりやすさ − 正しい口の起こりやすさ（差分の比べ、持ち主の案）を 0〜1 にしたもの。
  4. 試し用で、注意の明るさで点を直す: 注意が光るほど、もとの点の代わりに「クラスの親の点」（同じクラスの知っている名詞の明るさの和）を使う。
       直した点 = (1 − 注意) × もとの点 + 注意 × 親の点。
  5. 注意が光る組は答えない（「わからない」）: 注意の低い順に答えたときの、もとの点の正解率（答える割合 100・70・50・30%）。
  物差し: 注意の明るさが、もとの点の誤りをどれだけ見分けるか（AUROC）。直した点での崩した文の見分け（AUROC）。比べ: もとの点、親の点だけ。
結果: ../Project_Kotoyui_private/intervention/results/i16.jsonl
実行: python intervention/i16_attention_real.py --seed 0
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

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from i15_real_hierarchy import DATA, load  # noqa: E402
from i02_reverse import PortDevice  # noqa: E402

VAL0 = 8


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    allv = np.concatenate([pos, neg])
    r = allv.argsort().argsort().astype(float) + 1
    # 同点は平均の順位
    order = np.argsort(allv)
    vals = allv[order]
    ranks = np.empty(len(allv))
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


class Memory2:
    """口 2 つ（誤り・正しい）の経験の器。経験の記号の組を、その結論の口に学ばせる。"""

    def __init__(self, eps, labels, seed, steps=400):
        self.vocab = sorted({t for e in eps for t in e})
        self.idx = {t: i for i, t in enumerate(self.vocab)}
        n = 1 << int(math.ceil(math.log2(len(self.vocab) + VAL0 + 1)))
        X = torch.zeros(len(eps), len(self.vocab))
        for i, e in enumerate(eps):
            for t in e:
                X[i, self.idx[t]] = 1
        y = torch.tensor(labels)
        cnt = torch.bincount(y, minlength=2).float()
        w = (1.0 / cnt)[y]
        w = w / w.sum() * len(eps)
        torch.manual_seed(seed)
        dev = PortDevice(2, n=n, plates=4, seed=seed)
        opt = torch.optim.Adam(dev.parameters(), lr=0.03)
        for _ in range(steps):
            W = dev.forward_field(dev.ports).abs()[:, VAL0:VAL0 + len(self.vocab)] ** 2
            ll = X @ torch.log(W + 1e-9).T
            loss = -(ll[torch.arange(len(eps)), y] * w).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            W = (dev.forward_field(dev.ports).abs()[:, VAL0:VAL0 + len(self.vocab)] ** 2).numpy()
        self.W = W / W.sum(1, keepdims=True)
        self.n = n

    def attention(self, scene):
        ids = [self.idx[t] for t in scene if t in self.idx]
        ll = np.log(self.W[:, ids] + 1e-6).sum(1)
        return float(1 / (1 + math.exp(-(ll[1] - ll[0]))))         # 口 1 = 誤り。誤りの口と正しい口の起こりやすさの差（差分の比べ）


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="../Project_Kotoyui_private/intervention/results/i16.jsonl")
    a = ap.parse_args()
    torch.set_num_threads(4)
    t0 = time.perf_counter()
    rng = random.Random(a.seed)
    boards, S, L = load()
    SLOT = {s: i for i, s in enumerate(S)}
    idx = {b["name"]: i for i, b in enumerate(boards)}
    cids = [i for i, (v, _) in enumerate(S) if v == "C"]
    Ctot = L[:, cids].sum(1) + 1e-12
    cls_of = {}
    for b in boards:
        d = b["attrs"].get("D")
        if b["pos"] == "名詞" and d:
            cls_of[b["name"]] = max(d, key=d.get).split(":", 1)[1]
    by_cls = defaultdict(list)
    for nme, c in cls_of.items():
        by_cls[c].append(idx[nme])
    nouns = [b["name"] for b in boards if b["pos"] == "名詞"]
    triples = [t for t in json.load(open(os.path.join(DATA, "test_triples.json"), encoding="utf-8")) if t["noun"] in idx and t["verb"] in idx]
    rng.shuffle(triples)
    half = len(triples) // 2
    exp_set, test_set = triples[:half], triples[half:]

    def bp(row, key):
        j = SLOT.get(("C", key), -1)
        return float(L[row, j] / Ctot[row]) if j >= 0 else 0.0

    def score(noun, verb, role):
        s = bp(idx[noun], f"{verb}/{role}") if noun in idx else 0.0
        if noun in cls_of:
            s += bp(idx[verb], f"{'主語' if role == 'する' else '目的語'}:{cls_of[noun]}")
        return s

    def parent_score(noun, verb, role, exclude=True):
        """クラスの親の点: 同じクラスの名詞の明るさの和（光の融合）で、名詞の側の点を出す（その語自身は除く）。"""
        c = cls_of.get(noun)
        if not c:
            return score(noun, verb, role)
        rows = [r for r in by_cls[c] if not (exclude and r == idx.get(noun))]
        if not rows:
            return score(noun, verb, role)
        j = SLOT.get(("C", f"{verb}/{role}"), -1)
        s = float(L[rows, j].sum() / L[np.ix_(rows, cids)].sum()) if j >= 0 else 0.0
        s += bp(idx[verb], f"{'主語' if role == 'する' else '目的語'}:{c}")
        return s

    def make(ts):
        """本物・役の入れ替え・名詞の入れ替えの組。(名詞, 動詞, 役, 本物か, 崩し方)"""
        out = []
        for t in ts:
            out.append((t["noun"], t["verb"], t["role"], 1, "本物"))
            out.append((t["noun"], t["verb"], "される" if t["role"] == "する" else "する", 0, "役"))
            out.append((rng.choice(nouns), t["verb"], t["role"], 0, "名詞"))
        return out

    E, T = make(exp_set), make(test_set)
    sE = np.array([score(n, v, r) for n, v, r, _, _ in E])
    th = float(np.median(sE))
    qs = np.quantile(sE[sE > 0], [0.33, 0.66]) if (sE > 0).any() else [0, 0]

    def bucket(s):
        return "0" if s == 0 else ("低" if s < qs[0] else ("中" if s < qs[1] else "高"))

    def scene(n, v, r, s):
        return [f"動詞:{v}", f"役:{r}", f"クラス:{cls_of.get(n, 'なし')}", f"点:{bucket(s)}"]

    eps, labs = [], []
    for (n, v, r, y, k), s in zip(E, sE):
        wrong = (y == 1 and s < th) or (y == 0 and s >= th)
        eps.append(scene(n, v, r, s))
        labs.append(1 if wrong else 0)
    mem = Memory2(eps, labs, a.seed)
    sT = np.array([score(n, v, r) for n, v, r, _, _ in T])
    pT = np.array([parent_score(n, v, r) for n, v, r, _, _ in T])
    att = np.array([mem.attention(scene(n, v, r, s)) for (n, v, r, _, _), s in zip(T, sT)])
    yT = np.array([y for _, _, _, y, _ in T])
    kT = np.array([k for _, _, _, _, k in T])
    wrongT = ((yT == 1) & (sT < th)) | ((yT == 0) & (sT >= th))
    rec = dict(exp="I16", seed=a.seed, n_exp=len(E), n_test=len(T), vocab=len(mem.vocab), n_pixels=mem.n, err_rate_exp=float(np.mean(labs)),
               attention_auroc_for_errors=auroc(att[wrongT], att[~wrongT]))
    adj = (1 - att) * sT + att * pT
    for name, sc in (("もとの点", sT), ("親の点だけ", pT), ("注意で直した点", adj), ("半分ずつ（注意なし）", 0.5 * sT + 0.5 * pT)):
        rec[name] = {k: auroc(sc[(kT == "本物")], sc[(kT == k)]) for k in ("役", "名詞")}
    # 注意が光る組は「わからない」として答えない（自覚）: 注意の低い順に答えたときの、もとの点の正解率（区切り = 中央値）
    correct = ~wrongT
    order = np.argsort(att)
    rec["答える割合ごとの正解率"] = {f"{int(c * 100)}%": float(correct[order[:int(len(order) * c)]].mean()) for c in (1.0, 0.7, 0.5, 0.3)}
    rnd = np.random.default_rng(a.seed).permutation(len(order))
    rec["答える割合ごとの正解率（でたらめに選ぶ）"] = {f"{int(c * 100)}%": float(correct[rnd[:int(len(rnd) * c)]].mean()) for c in (1.0, 0.7, 0.5, 0.3)}
    rec["注意の明るさ（平均）"] = {"もとの点が誤り": float(att[wrongT].mean()), "もとの点が正しい": float(att[~wrongT].mean())}
    rec["total_s"] = time.perf_counter() - t0
    print(json.dumps(rec, ensure_ascii=False), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
