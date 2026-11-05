"""介入の研究 実験I13b: 消去 — 誤った経験を経験の器から消して、悪い知見の広がりを止められるか

持ち主の方針（2026-11-05）: 光の消去の特性を優先する。I5 では、見かけの反例で熟し具合を誤って否定すると、その誤りが物の数だけ答えを壊した。
方法:
  - 経験: I8 で集めた経験（規則で結論を付けたもの、seed ごとに保存済み）。器は I8b の labeled（口ごとに 1 種類）。
  - 汚す: 「観点:状態」の場面を「結論:見かけ」にした誤った経験を足す（正しい「観点:状態 → 本物」の経験の 1.5 倍の数）。見かけの反例を何度も見た、という想定。
  - 消し方:
      汚れたまま: 何もしない。
      学び直し: 誤った経験を除いて器を学び直す（理想。どれが誤りかを全部覚えている必要がある）。
      注意として残す（I13c）: 誤りの跡を「注意」の口として器に足し、思い出すときに注意の口に寄る分を結論から外す。注意の明るさも記録する。
      **明るさの引き算（光の消去）**: 誤った経験だけを別の小さな器に学ばせ、その明るさを「見かけ」の口から、数の割合ぶん差し引く（打ち消す）。
        元の器は学び直さない。
  - 物差し: 世界 B（I8 と同じ、ウメ型なし、文 10・50）の塗る・熟す・見たままの正解率。6 seed。
結果: ../Project_Kotoyui_private/intervention/results/i13b.jsonl
実行: CUDA_VISIBLE_DEVICES= python intervention/i13b_erase.py --seed 0
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i01_small_world as I1  # noqa: E402
import i03_reverse_split as I3  # noqa: E402
import i08_light_judge as I8  # noqa: E402
from i05_more_words import NAMES  # noqa: E402
from i07_experience import fruits, all_pairs  # noqa: E402


def evaluate(mem, seed, KB, nB):
    rng = random.Random(5000 + seed * 10 + 0 + nB)                 # I8 と同じ世界 B（ウメ型なし）
    S = fruits(NAMES[12:12 + KB], rng)
    I8.MEM = mem
    I8.SPLIT_LOG = []
    B, found = I8.build_world(S, NAMES[12:12 + KB], seed + 200, nB)
    I8.MEM = None
    Q = I1.questions()
    per = defaultdict(lambda: [0, 0])
    effs, labs = {}, []
    for q in Q:
        if q["word"] not in effs:
            effs[q["word"]] = I8.light_effects(all_pairs(B, q["word"]), mem, labs)
        per[q["kind"]][0] += I1.answer(B, q, effs[q["word"]]) == q["ans"]
        per[q["kind"]][1] += 1
    return {k: v[0] / v[1] for k, v in per.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--KB", type=int, default=12)
    ap.add_argument("--bad_ratio", type=float, default=1.5)
    ap.add_argument("--out", default="../Project_Kotoyui_private/intervention/results/i13b.jsonl")
    a = ap.parse_args()
    torch.set_num_threads(2)
    I3.MIN_REF, I3.LOOSE = 5.0, True
    t0 = time.perf_counter()
    eps = json.load(open(f"../Project_Kotoyui_private/intervention/results/i08_episodes_s{a.seed}.json", encoding="utf-8"))
    good_state = [e for e in eps if "観点:状態" in e and "結論:本物" in e]
    rng = random.Random(a.seed)
    bad = [[t for t in e if not t.startswith("結論:")] + ["結論:見かけ"] for e in rng.choices(good_state, k=int(a.bad_ratio * len(good_state)))]
    mem_clean = I8.Memory(eps, a.seed, how="labeled")
    mem_dirty = I8.Memory(eps + bad, a.seed, how="labeled")
    # 光の消去: 誤った経験だけの器を作り、その明るさを「見かけ」の口から数の割合ぶん差し引く
    mem_erase = copy.copy(mem_dirty)
    W = mem_dirty.W.copy()
    Wn = W / W.sum(1, keepdims=True)
    labels = sorted({next(t for t in e if t.startswith("結論:")) for e in eps + bad})
    port = labels.index("結論:見かけ")
    bad_mem = I8.Memory(bad, a.seed + 1, M=1, how="mix")
    bad_W = np.zeros(len(mem_dirty.vocab))
    for t, i in bad_mem.idx.items():
        bad_W[mem_dirty.idx[t]] = bad_mem.W[0, i]
    bad_W = bad_W / bad_W.sum()
    n_port = sum(1 for e in eps + bad if "結論:見かけ" in e)
    frac = len(bad) / n_port
    Wn[port] = np.clip(Wn[port] - frac * bad_W, 0, None)
    Wn[port] /= Wn[port].sum()
    mem_erase.W = Wn
    # 注意の経験（I13c、持ち主 2026-11-05「1 − 誤った経験で注意の経験を作る」）: 誤りの跡（誤った経験だけの器の明るさ）を消さずに「注意」の口として足す。
    # 思い出すとき、注意の口に寄る分は結論から外し（重ねない）、注意の明るさ（その場面が前の誤りに似ている度合い）を記録する。
    class Attention:
        def __init__(self, base, bad_W, frac):
            self.vocab, self.idx = base.vocab, base.idx
            Wb = base.W / base.W.sum(1, keepdims=True)
            self.W = np.vstack([Wb, bad_W[None]])
            self.prior = np.array([1.0] * len(Wb) + [frac * len(Wb)])  # 注意の口の重み = 誤った経験の数の割合
            self.n_pass, self.flags = 0, []

        def recall(self, scene, labels):
            self.n_pass += 1
            ids = [self.idx[t] for t in scene if t in self.idx]
            ll = np.log(self.W[:, ids] + 1e-6).sum(1) + np.log(self.prior)
            post = np.exp(ll - ll.max())
            post /= post.sum()
            att = float(post[-1])
            self.flags.append((next((t for t in scene if t.startswith("観点:")), "?"), att))
            out = post[:-1] @ self.W[:-1] / max(post[:-1].sum(), 1e-9)
            return {l: float(out[self.idx[f"結論:{l}"]]) if f"結論:{l}" in self.idx else 0.0 for l in labels}

    mem_att = Attention(mem_dirty, bad_W, frac)
    rec = dict(exp="I13b", seed=a.seed, bad_ratio=a.bad_ratio, n_bad=len(bad), n_good_state=len(good_state), frac=frac)
    for nB in (10, 50):
        for name, mem in (("きれい（学び直し）", mem_clean), ("汚れたまま", mem_dirty), ("光の消去（引き算）", mem_erase), ("注意として残す", mem_att)):
            if name == "注意として残す":
                mem.flags = []
            rec[f"{name}|{nB}"] = evaluate(mem, a.seed, a.KB, nB)
            if name == "注意として残す":
                by = defaultdict(list)
                for v, att in mem.flags:
                    by[v].append(att)
                rec[f"{name}|{nB}"]["注意の明るさ"] = {v: float(np.mean(x)) for v, x in by.items()}
            print(name, nB, rec[f"{name}|{nB}"], flush=True)
    rec["total_s"] = time.perf_counter() - t0
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
