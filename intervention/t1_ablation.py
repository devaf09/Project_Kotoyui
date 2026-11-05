"""ツッコミどころ T1: 手で決めたもの（動詞の規則・区切り）を 1 つずつ外す・動かすと、どこまで崩れるか

持ち主への報告（2026-11-05）のツッコミどころ 1「規則や記号を私が手で決めている部分が多い。うまくいった結果の一部は、器ではなく私の決めた型が効いているかもしれない」を確かめる。
I3 の最終形（小さな世界、意味 1 つあたり 10・50 文）で、次を 1 つずつ変える:
  規則なし: 「塗る = 色だけ、熟させる = 状態だけ」を外す（動詞がどの観点を変えるか知らない = 元と変えた後で違う観点は全部変わるとみなす）。
  変わった（CHG 0.5 → 0.3・0.7）: 組の観点の差がこれ以上なら「その観点が違う」。
  変わらない（SAME 0.2 → 0.1・0.35）: 味の差がこれ未満なら「味は同じ」（反例）。
  競合（COMP 0.5 → 0.3・0.7）: 分けるかの判断で、口どうしの値が食い違うとみなす差。
  暗い（DARK 0.2 → 0.1・0.35）: 違い（自覚）で、観点の最大の明るさのこの割合未満を「暗い」とする。
結果: ../Project_Kotoyui_private/intervention/results/t1.jsonl
実行: CUDA_VISIBLE_DEVICES= python intervention/t1_ablation.py --seed 0 --n_train 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i01_small_world as I1  # noqa: E402
import i03_reverse_split as I3  # noqa: E402

CONDS = [("既定", {}), ("規則なし", {"VERB_RULE": False}),
         ("変わった 0.3", {"CHG": 0.3}), ("変わった 0.7", {"CHG": 0.7}),
         ("変わらない 0.1", {"SAME": 0.1}), ("変わらない 0.35", {"SAME": 0.35}),
         ("競合 0.3", {"COMP": 0.3}), ("競合 0.7", {"COMP": 0.7}),
         ("暗い 0.1", {"DARK": 0.1}), ("暗い 0.35", {"DARK": 0.35})]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_train", type=int, default=10)
    ap.add_argument("--out", default="../Project_Kotoyui_private/intervention/results/t1.jsonl")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(2)
    for name, ch in CONDS:
        t0 = time.perf_counter()
        I1.VERB_RULE = ch.get("VERB_RULE", True)
        I3.CHG, I3.SAME, I3.COMP, I3.DARK = ch.get("CHG", 0.5), ch.get("SAME", 0.2), ch.get("COMP", 0.5), ch.get("DARK", 0.2)
        I3.MIN_REF, I3.LOOSE = 5.0, True
        rec = I3.run(argparse.Namespace(seed=a.seed, n_train=a.n_train, min_support=0.0, min_ref=5.0, loose=1, steps=800, max_depth=3))
        out = dict(exp="T1", cond=name, seed=a.seed, n_train=a.n_train, senses_found=rec["senses_found"], n_boards=rec["n_boards"], purity=rec["purity"],
                   qa=rec["qa"]["deny|全部（読み直し）"], corr=rec["qa"]["corr|全部"], total_s=time.perf_counter() - t0)
        print(json.dumps(out, ensure_ascii=False), flush=True)
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
