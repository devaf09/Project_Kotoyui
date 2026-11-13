"""頻度の少ない語を、クラスの親の明るさで補う（介入の研究からの注文 2、2026-11-06）

帯ごとに、語の数えの回数 n（語板の n_obs）に応じて親と混ぜる:
  補った割合 = (n × 語の割合 + k × 親の割合) / (n + k)
  親 = 同じクラス（帯 D のいちばん明るいクラス）の語の割合（帯ごとに割合にしたもの）の平均 = 光の和を割合にしたもの。
光の器では学び直さない: 語の器の明るさと親の光（子の明るさの和）を、帯ごとに (n : k) で混ぜて足すだけ（光の和の融合と同じ操作）。
確かめ: 崩した文（役・名詞の入れ替え）、的はずれな文（別の文書の文・名詞を 1 つ替えた文）を、k = 0（補わない）と k を変えて比べる。
  数えた割合（content_boards.json）と、学んだ明るさ（light_v2_bright_s0.npy）の両方で。
出す物: <dir>/parent_smooth_s0.json（結果）、--save_k を付けると補った割合 content_boards_smooth_k<k>.json（先頭 --n_boards 語を補った数えた割合。残りの語はそのまま）。
実行: python lexicon/parent_smooth.py --dir <v2 の置き場> [--ks 0 2 5 10 20 50] [--save_k 10]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import board_light as BL  # noqa: E402
from content_light import odd_sentence_test  # noqa: E402
from content_light_v2 import band_norm, context_test, load, make_items, BANDS, PRIV  # noqa: E402


def smooth(Wn, ids, boards, k, bands=("B", "C")):
    """Wn: 帯ごとに割合にした明るさ。bands の帯だけ、回数で親と混ぜる。"""
    if k <= 0:
        return Wn
    groups = defaultdict(list)
    for i, b in enumerate(boards):
        if b["func"] != "クラスなし":
            groups[b["func"]].append(i)
    out = Wn.copy()
    for b in bands:
        ix = ids[b]
        for c, mem in groups.items():
            P = Wn[mem][:, ix].mean(0)
            if P.sum() <= 0:
                continue
            P = P / P.sum()
            for i in mem:
                n = boards[i].get("n_obs", {}).get(b, 0.0)
                out[i, ix] = (n * Wn[i, ix] + k * P) / (n + k)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(PRIV, "data", "boards", "v2"))
    ap.add_argument("--n_boards", type=int, default=5000)
    ap.add_argument("--bright", default="light_v2_bright_s0.npy")
    ap.add_argument("--ks", type=float, nargs="+", default=[0, 2, 5, 10, 20, 50])
    ap.add_argument("--bands", nargs="+", default=["B", "C"])
    ap.add_argument("--save_k", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    boards, S, SLOT, Q = load(a.dir, a.n_boards)
    triples = json.load(open(os.path.join(a.dir, "test_triples.json"), encoding="utf-8"))
    tdocs = json.load(open(os.path.join(a.dir, "test_docs.json"), encoding="utf-8"))
    cls_of = {b["name"]: b["func"] for b in boards if b["pos"] == "名詞" and b["func"] != "クラスなし"}
    vocab = {b["name"] for b in boards}
    nouns = sorted(b["name"] for b in boards if b["pos"] == "名詞")
    items = make_items(tdocs, nouns, a.seed, vocab)
    srcs = {"数えた割合": Q}
    bp = os.path.join(a.dir, a.bright)
    if os.path.exists(bp):
        Wl = np.load(bp).astype(np.float32)
        if Wl.shape == Q.shape:                                      # 同じ語板・同じ語数で学んだ明るさだけ使う
            srcs["光"] = Wl
    # 回数の少ない語の目安
    nC = np.array([b.get("n_obs", {}).get("C", 0) for b in boards])
    res = dict(設定=vars(a), 語板=len(boards), 帯Cの回数=dict(中央値=float(np.median(nC)), 二十未満の割合=round(float(np.mean(nC < 20)), 3)))
    for name, W in srcs.items():
        Wn, ids = band_norm(W, S)
        res[name] = {}
        for k in a.ks:
            Ws = smooth(Wn, ids, boards, k, a.bands)
            r = dict(崩した文=odd_sentence_test(Ws, boards, S, SLOT, triples, cls_of, random.Random(1)))
            ct = context_test(Ws, boards, S, items, BANDS)
            r["的はずれな文"] = {kk: v["学んだ重みで足す"] for kk, v in ct.items()}
            # 回数の少ない名詞だけの崩した文（帯 C が 20 回未満）
            rare = {b["name"] for b in boards if b["pos"] == "名詞" and b.get("n_obs", {}).get("C", 0) < 20}
            rt = [t for t in triples if t["noun"] in rare]
            r["崩した文_回数の少ない名詞"] = odd_sentence_test(Ws, boards, S, SLOT, rt, cls_of, random.Random(1))
            res[name][f"k={k:g}"] = r
            print(name, k, json.dumps(r, ensure_ascii=False), flush=True)
    if a.save_k is not None:
        Wn, ids = band_norm(Q, S)
        Ws = smooth(Wn, ids, boards, a.save_k, a.bands)
        full = json.load(open(os.path.join(a.dir, "content_boards.json"), encoding="utf-8"))
        for i, b in enumerate(boards):
            for bd in a.bands:
                v = {S[j][1]: round(float(Ws[i, j]), 4) for j in ids[bd] if Ws[i, j] >= 5e-5}
                if v:
                    full[i]["attrs"][bd] = dict(sorted(v.items(), key=lambda kv: -kv[1]))
        json.dump(full, open(os.path.join(a.dir, f"content_boards_smooth_k{a.save_k:g}.json"), "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(res, open(os.path.join(a.dir, f"parent_smooth_s{a.seed}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
