"""見積もり: 交互の会話で話者のラベルを広げたとき、どのくらい正しいか（Q4、持ち主の判断 2026-10-24「8 割を超えたら広げる」）

方法: 地の文をはさまずに続く会話文の列（aozora.runs）の中で、k 個先の会話文を同じ話者とみなす（k = 2 なら A・B・A・B の交互）。
      両方に一人称の類（12 類）がついている組で、類が同じ割合を「広げ方の正しさ」とする。
      比べる: k = 1（すぐ次 = 別の話者のはず。低いほど交互の見立てが正しい）、同じ作品のでたらめな組（偶然の一致）。
      列の長さ（2〜3・4〜6・7 以上）ごとにも分ける（人が増えると交互が崩れる見込み）。
      広げられる数: ラベルのない会話文のうち、同じ列の ±k・±2k … にラベルつきの会話文があるもの。
指標: 同じ類の割合（%）、組の数、ラベルを広げられる会話文の数。
結果: results/est_label_spread.txt（1 行の JSON）
実行: python experiments/est_label_spread.py
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aozora  # noqa: E402


def agree(works, k, min_len=0, max_len=10**9):
    """列の中で k 個離れた 2 つの会話文が、どちらもラベルつきのときの一致率（%）と組の数。"""
    same = tot = 0
    for w in works:
        for r in w["runs"]:
            if not (min_len <= len(r) <= max_len):
                continue
            c = [aozora.cls(x["fp"]) for x in r]
            for i in range(len(r) - k):
                if c[i] and c[i + k]:
                    tot += 1
                    same += c[i] == c[i + k]
    return (100.0 * same / tot if tot else None), tot


def chance(works, seed=0, n=200000):
    """同じ作品の中のでたらめな 2 つのラベルつき会話文が同じ類の割合（%）。"""
    rng = random.Random(seed)
    per = [[aozora.cls(x["fp"]) for r in w["runs"] for x in r if aozora.cls(x["fp"])] for w in works]
    per = [p for p in per if len(p) >= 2]
    same = 0
    for _ in range(n):
        p = rng.choice(per)
        a, b = rng.sample(range(len(p)), 2)
        same += p[a] == p[b]
    return 100.0 * same / n


def spreadable(works, k):
    """ラベルのない会話文のうち、同じ列の k の倍数だけ離れた所にラベルつきの会話文があるものの数（一番近いものを使う想定）。"""
    n = 0
    for w in works:
        for r in w["runs"]:
            c = [aozora.cls(x["fp"]) for x in r]
            for i in range(len(r)):
                if c[i] is None and any(c[j] for j in range(i % k, len(r), k) if j != i):
                    n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="1000", choices=["1000", "all"], help="青空文庫の作品の組（1000 = 最初の 1,000 作品、all = 全 4,477 作品）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="使わない（約束の引数）")
    ap.add_argument("--out", default="results/est_label_spread.txt")
    a = ap.parse_args()
    t0 = time.perf_counter()
    d = aozora.load(a.set)
    W = d["works"]
    n_utt = sum(len(r) for w in W for r in w["runs"])
    n_lab = sum(1 for w in W for r in w["runs"] for x in r if aozora.cls(x["fp"]))
    rec = dict(exp="est_label_spread", **vars(a), n_works=len(W), n_utt=n_utt, n_labeled=n_lab,
               n_runs=sum(len(w["runs"]) for w in W), chance=chance(W, a.seed))
    for k in (1, 2, 3, 4):
        rec[f"agree_k{k}"], rec[f"pairs_k{k}"] = agree(W, k)
    for lo, hi in ((2, 3), (4, 6), (7, 10**9)):
        for k in (1, 2):
            rec[f"agree_k{k}_len{lo}_{hi if hi < 10**9 else 'up'}"], rec[f"pairs_k{k}_len{lo}_{hi if hi < 10**9 else 'up'}"] = agree(W, k, lo, hi)
    rec["spreadable_k2"] = spreadable(W, 2)
    rec["time_s"] = time.perf_counter() - t0
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
