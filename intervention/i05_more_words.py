"""介入の研究 実験I5: 反例の文を固定したまま、ほかの語だけを増やすと、介入で得る知見は増えるか

I4 では文の総数と世界の大きさを一緒に増やしたため、「同じ反例のまま、語が増えると知見が増えるか」を切り分けられなかった。
  - 固定: 反例を作る語（リンゴ: 赤と緑で味は同じ、人のリンゴ）、意味 1 つあたり --n_train 文。
  - 増やす: 熟すと色と味が一緒に変わる果物（交絡した語）を K 個（3・6・12・24）。名前は実在の果物の名前を借りるが、属性は作り物（乱数）:
      完熟 = 赤・黄・橙・紫・白のどれか・甘い（たまに甘くて酸っぱい）、未熟 = 緑・渋いか酸っぱい。形は丸いか長い。
  - 方法: I3 の最終形。比べ: 相関だけ。
  - 測る: 介入で得た知見（相関だけでは間違え、介入で正しく答えた問いの数）。K に比例して増えれば、1 つの反例が語の数だけ効いている。
結果: results/i05.jsonl
実行: CUDA_VISIBLE_DEVICES= python intervention/i05_more_words.py --K 6 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i01_small_world as I1  # noqa: E402
import i02_reverse as I2  # noqa: E402
import i03_reverse_split as I3  # noqa: E402
from i04_scale import LARGE_COLORS  # noqa: E402

NAMES = ["アケビ", "スモモ", "ビワ", "ザクロ", "キウイ", "マンゴー", "パパイヤ", "ライチ", "スイカ", "ナシ", "イチジク", "ユズ",
         "スダチ", "カボス", "グアバ", "ドリアン", "ザボン", "ハッサク", "デコポン", "ポンカン", "キンカン", "ヤマモモ", "アンズ", "サクランボ"]
FIXED = {
    "リンゴ/果物": ("リンゴ", {"種類": ["果物"], "色": ["赤", "緑"], "形": ["丸い"], "味": ["甘い"], "状態": ["完熟"], "される": ["食べる", "切る"]}),
    "リンゴ/人": ("リンゴ", {"種類": ["人"], "する": ["手を振る", "話す", "歩く"]}),
}


def make_senses(K, seed):
    rng = random.Random(1000 + seed)
    S = dict(FIXED)
    for name in NAMES[:K]:
        shape = rng.choice(["丸い", "長い"])
        ripe_c = rng.choice(["赤", "黄", "橙", "紫", "白"])
        ripe_t = ["甘い", "酸っぱい"] if rng.random() < 0.25 else ["甘い"]
        un_t = [rng.choice(["渋い", "酸っぱい"])]
        if un_t == ["酸っぱい"] and "酸っぱい" in ripe_t:
            un_t = ["渋い"]
        S[f"{name}/完熟"] = (name, {"種類": ["果物"], "色": [ripe_c], "形": [shape], "味": ripe_t, "状態": ["完熟"], "される": ["食べる", "切る"]})
        S[f"{name}/未熟"] = (name, {"種類": ["果物"], "色": ["緑"], "形": [shape], "味": un_t, "状態": ["未熟"], "される": ["切る"]})
    return S


def use_senses(S, qwords):
    I1.SENSES = S
    I1.VIEWS["色"] = list(LARGE_COLORS)
    I1.COLOR_TOK.clear(); I1.PAINT_TOK.clear()
    for c, (t, p) in LARGE_COLORS.items():
        I1.COLOR_TOK[c], I1.PAINT_TOK[c] = t, p
        I1.LEX[t] = [("色", c)]
    I1.SLOTS = [(v, x) for v, xs in I1.VIEWS.items() for x in xs]
    I1.SLOT = {s: i for i, s in enumerate(I1.SLOTS)}
    I1.WORDS = sorted({w for w, _ in S.values()})
    I1.FRUIT_WORDS = qwords
    for m in (I2, I3):
        for k in ("SENSES", "SLOTS", "SLOT", "VIEWS", "WORDS"):
            if hasattr(m, k):
                setattr(m, k, getattr(I1, k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--n_train", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="../Project_Kotoyui_private/intervention/results/i05.jsonl")
    a = ap.parse_args()
    S = make_senses(a.K, a.seed)
    use_senses(S, ["リンゴ"] + NAMES[:a.K])
    I3.MIN_REF, I3.LOOSE = 5.0, True
    import torch
    torch.set_num_threads(2)
    t0 = time.perf_counter()
    rec = I3.run(argparse.Namespace(seed=a.seed, n_train=a.n_train, min_support=0.0, min_ref=5.0, loose=1, steps=800, max_depth=3))
    it = rec["qa_items"]
    gain = sum(1 for d, c in zip(it["deny"], it["corr"]) if d and not c)
    lost = sum(1 for d, c in zip(it["deny"], it["corr"]) if c and not d)
    out = dict(exp="I5", K=a.K, n_train_per_sense=a.n_train, seed=a.seed, n_senses=len(S),
               n_sent_total=a.n_train * len(S), n_sent_counterexample=a.n_train,
               n_q={k: it["kind"].count(k) for k in ("塗る", "熟す", "見たまま")},
               correct=sum(it["deny"]), correct_corr=sum(it["corr"]), gain=gain, lost=lost,
               gain_by_kind={k: sum(1 for kd, d, c in zip(it["kind"], it["deny"], it["corr"]) if kd == k and d and not c) for k in ("塗る", "熟す", "見たまま")},
               senses_found=rec["senses_found"], purity=rec["purity"], qa=rec["qa"], total_s=time.perf_counter() - t0)
    print(json.dumps({k: v for k, v in out.items() if k != "qa"}, ensure_ascii=False), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
