"""介入の研究 実験I6: 誤った否定が広がるのを防ぐ（否定の見直し・最小二乗）と、型の違う物（熟しても酸っぱいウメ）を混ぜる

I5 で、否定 1 つの誤り（熟し具合が見かけの反例で否定された）が、物の数だけ答えを壊した（熟すの問い 72 問中 56 問）。
また本当の世界には、熟しても味の変わらない物（ウメ: 未熟 = 緑・酸っぱい、完熟 = 黄・酸っぱい）がある。これは「熟し具合 → 味」の本物の反例で、
1 つの反例で否定する規則だと、ほかの全部の果物の「熟させたら」の答えを壊す。
  - 世界: I5 と同じ（反例のリンゴ + 交絡した果物 K 個）に、ウメ型の果物を P 個（0・2・6）足す。
  - 原因の効き目の出し方:
      deny   = 1 つでも反例があれば否定（I3 の最終形）
      revise = 否定の見直し: 否定した観点 v について、「v でしか説明できない支持の組」（ほかの否定されていない観点では説明できない、味の変わった組）が
               反例の組より多ければ、否定を取り消す。効き目 = 支持 / (支持 + 反例)
      lsq    = 最小二乗（味の差 ≈ Σ 効き目 × 観点の差）
      corr   = 相関だけ
  - 正解: 塗る = 味は変わらない。熟させる = その物の完熟の味になる（ウメは酸っぱいまま）。
結果: results/i06.jsonl
実行: CUDA_VISIBLE_DEVICES= python intervention/i06_revise.py --K 12 --P 2 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i03_reverse_split as I3  # noqa: E402
from i05_more_words import NAMES, make_senses, use_senses  # noqa: E402

PLUMS = ["ウメ", "カリン", "シークワーサー", "ライム", "スグリ", "クランベリー"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--P", type=int, default=2)
    ap.add_argument("--n_train", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="intervention/results/i06.jsonl")
    a = ap.parse_args()
    S = make_senses(a.K, a.seed)
    rng = random.Random(2000 + a.seed)
    for name in PLUMS[:a.P]:
        shape = rng.choice(["丸い", "長い"])
        S[f"{name}/完熟"] = (name, {"種類": ["果物"], "色": ["黄"], "形": [shape], "味": ["酸っぱい"], "状態": ["完熟"], "される": ["切る"]})
        S[f"{name}/未熟"] = (name, {"種類": ["果物"], "色": ["緑"], "形": [shape], "味": ["酸っぱい"], "状態": ["未熟"], "される": ["切る"]})
    use_senses(S, ["リンゴ"] + NAMES[:a.K] + PLUMS[:a.P])
    I3.MIN_REF, I3.LOOSE = 5.0, True
    import torch
    torch.set_num_threads(2)
    t0 = time.perf_counter()
    rec = I3.run(argparse.Namespace(seed=a.seed, n_train=a.n_train, min_support=0.0, min_ref=5.0, loose=1, steps=800, max_depth=3))
    out = dict(exp="I6", K=a.K, P=a.P, n_train_per_sense=a.n_train, seed=a.seed, n_senses=len(S),
               senses_found=rec["senses_found"], purity=rec["purity"],
               qa={k: v for k, v in rec["qa"].items() if k in ("deny|全部（読み直し）", "revise|全部（読み直し）", "lsq|全部（読み直し）", "corr|全部")},
               items=rec["qa_items"], total_s=time.perf_counter() - t0)
    print(json.dumps({k: v for k, v in out.items() if k != "items"}, ensure_ascii=False), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
