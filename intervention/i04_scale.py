"""介入の研究 実験I4: 文の数と世界の大きさを変えて、正しく答えられる問いの数がどう増えるかを測る

持ち主（2026-11-04）: 「原理的には、関連する分野では資料に対して指数関数的に知見が得られるかもしれない」。
1 つの反例（色違いのリンゴ）から出た否定（色は味の原因ではない）は、ほかの全部の語の「塗る」問いに効く。
世界を大きくすると問いの数は組み合わせで増えるが、手がかり（反例）は数個のまま。そこで:
  - 世界: small（I1 と同じ、5 語・9 つの意味）と large（10 語・18 の意味、色 6 つ。関連する分野 = 同じ果物の世界を広げた）。
    large では、色違いでも味が同じ物が 2 つ（リンゴ: 赤・緑、ブドウ: 紫・緑）、人の名前が 2 つ（リンゴ・モモ）。
  - 文の数: 意味 1 つあたり 2・3・5・10・20・50。
  - 方法: I3 の最終形（逆からの判断で分ける、否定 + 見かけの反例を外す + loose）。比べ: 相関だけ。
  - 測る: 正しく答えた問いの数（塗る・熟す・見たまま）、相関だけでは間違える問いのうち正しく答えた数（= 介入で得た知見）。
結果: results/i04.jsonl
実行: CUDA_VISIBLE_DEVICES= python intervention/i04_scale.py --world large --n_train 10 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i01_small_world as I1  # noqa: E402
import i02_reverse as I2  # noqa: E402
import i03_reverse_split as I3  # noqa: E402

LARGE = {
    "リンゴ/果物": ("リンゴ", {"種類": ["果物"], "色": ["赤", "緑"], "形": ["丸い"], "味": ["甘い"], "状態": ["完熟"], "される": ["食べる", "切る"]}),
    "リンゴ/人": ("リンゴ", {"種類": ["人"], "する": ["手を振る", "話す", "歩く"]}),
    "トマト/完熟": ("トマト", {"種類": ["野菜"], "色": ["赤"], "形": ["丸い"], "味": ["甘い", "酸っぱい"], "状態": ["完熟"], "される": ["食べる", "切る"]}),
    "トマト/未熟": ("トマト", {"種類": ["野菜"], "色": ["緑"], "形": ["丸い"], "味": ["渋い"], "状態": ["未熟"], "される": ["切る"]}),
    "イチゴ/完熟": ("イチゴ", {"種類": ["果物"], "色": ["赤"], "形": ["丸い"], "味": ["甘い"], "状態": ["完熟"], "される": ["食べる"]}),
    "イチゴ/未熟": ("イチゴ", {"種類": ["果物"], "色": ["白"], "形": ["丸い"], "味": ["酸っぱい"], "状態": ["未熟"], "される": ["切る"]}),
    "バナナ/完熟": ("バナナ", {"種類": ["果物"], "色": ["黄"], "形": ["長い"], "味": ["甘い"], "状態": ["完熟"], "される": ["食べる", "切る"]}),
    "バナナ/未熟": ("バナナ", {"種類": ["果物"], "色": ["緑"], "形": ["長い"], "味": ["渋い"], "状態": ["未熟"], "される": ["切る"]}),
    "レモン": ("レモン", {"種類": ["果物"], "色": ["黄"], "形": ["丸い"], "味": ["酸っぱい"], "状態": ["完熟"], "される": ["切る"]}),
    "ブドウ": ("ブドウ", {"種類": ["果物"], "色": ["紫", "緑"], "形": ["丸い"], "味": ["甘い"], "状態": ["完熟"], "される": ["食べる"]}),
    "ミカン/完熟": ("ミカン", {"種類": ["果物"], "色": ["橙"], "形": ["丸い"], "味": ["甘い"], "状態": ["完熟"], "される": ["食べる"]}),
    "ミカン/未熟": ("ミカン", {"種類": ["果物"], "色": ["緑"], "形": ["丸い"], "味": ["酸っぱい"], "状態": ["未熟"], "される": ["切る"]}),
    "カキ/完熟": ("カキ", {"種類": ["果物"], "色": ["橙"], "形": ["丸い"], "味": ["甘い"], "状態": ["完熟"], "される": ["食べる", "切る"]}),
    "カキ/未熟": ("カキ", {"種類": ["果物"], "色": ["緑"], "形": ["丸い"], "味": ["渋い"], "状態": ["未熟"], "される": ["切る"]}),
    "メロン": ("メロン", {"種類": ["果物"], "色": ["緑"], "形": ["丸い"], "味": ["甘い"], "状態": ["完熟"], "される": ["切る"]}),
    "モモ/完熟": ("モモ", {"種類": ["果物"], "色": ["白"], "形": ["丸い"], "味": ["甘い"], "状態": ["完熟"], "される": ["食べる"]}),
    "モモ/未熟": ("モモ", {"種類": ["果物"], "色": ["緑"], "形": ["丸い"], "味": ["酸っぱい"], "状態": ["未熟"], "される": ["切る"]}),
    "モモ/人": ("モモ", {"種類": ["人"], "する": ["手を振る", "話す", "歩く"]}),
}
LARGE_COLORS = {"赤": ("赤い", "赤に"), "緑": ("緑の", "緑に"), "黄": ("黄色い", "黄色に"), "白": ("白い", "白に"),
                "紫": ("紫の", "紫に"), "橙": ("橙の", "橙に")}
LARGE_Q_WORDS = ["イチゴ", "カキ", "トマト", "バナナ", "ブドウ", "ミカン", "モモ", "リンゴ"]


def use_world(name):
    """I1・I2・I3 の世界（意味・観点・語の表）を入れ替える。"""
    if name == "small":
        return
    I1.SENSES = LARGE
    I1.VIEWS["色"] = list(LARGE_COLORS)
    I1.COLOR_TOK.clear(); I1.PAINT_TOK.clear()
    for c, (t, p) in LARGE_COLORS.items():
        I1.COLOR_TOK[c], I1.PAINT_TOK[c] = t, p
        I1.LEX[t] = [("色", c)]
    I1.SLOTS = [(v, x) for v, xs in I1.VIEWS.items() for x in xs]
    I1.SLOT = {s: i for i, s in enumerate(I1.SLOTS)}
    I1.WORDS = sorted({w for w, _ in LARGE.values()})
    I1.FRUIT_WORDS = LARGE_Q_WORDS
    for m in (I2, I3):
        for k in ("SENSES", "SLOTS", "SLOT", "VIEWS", "WORDS"):
            if hasattr(m, k):
                setattr(m, k, getattr(I1, k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="small", choices=["small", "large"])
    ap.add_argument("--n_train", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="intervention/results/i04.jsonl")
    a = ap.parse_args()
    use_world(a.world)
    I3.MIN_REF, I3.LOOSE = 5.0, True
    import torch
    torch.set_num_threads(2)
    t0 = time.perf_counter()
    args = argparse.Namespace(seed=a.seed, n_train=a.n_train, min_support=0.0, min_ref=5.0, loose=1, steps=800, max_depth=3)
    rec = I3.run(args)
    # 問いごとの正誤（最終形と相関だけ）
    B = None
    Q = I1.questions()
    out = dict(exp="I4", world=a.world, n_train_per_sense=a.n_train, seed=a.seed, n_senses=len(I1.SENSES), n_words=len(I1.WORDS),
               n_q={k: sum(q["kind"] == k for q in Q) for k in ("塗る", "熟す", "見たまま")},
               senses_found=rec["senses_found"], n_boards=rec["n_boards"], purity=rec["purity"], route_acc=rec["route_acc"],
               anomaly_auroc=rec["anomaly_auroc"], qa=rec["qa"], correct_pairs=rec.get("qa_items"), total_s=time.perf_counter() - t0)
    print(json.dumps({k: v for k, v in out.items() if k not in ("qa", "correct_pairs")}, ensure_ascii=False), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
