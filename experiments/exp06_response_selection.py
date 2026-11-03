"""実験06: 10 択の返事選び — Sin-CNN3 から取り込んだ返事の選び方（kotoyui.py 0.1.0）が、元と同じ成績を出すか

着想: 持ち主の判断（2026-11-02）で、Sin-CNN3 のテストボット02 を kotoyui.py にまとめ直した。語の解析を Sudachi から fugashi（UniDic）に替え、
      材料は RealPersonaChat だけ（JMultiWOZ は許可待ち）にしたので、Sin-CNN3 の実験115 と同じ物差しで測り直す。
物差し（Sin-CNN3 実験115 と同じ）: 本物の会話を途中で止め、正しい次の発言を 10 個の候補から選べるか（1 位の正答・MRR。でたらめ 10%）。
  やさしい版: ほかの会話の発言 9 個が外れ。むずかしい版: 同じ会話のほかの発言 9 個が外れ（話題は同じで、流れだけが違う）。
選び方: R でたらめ、A 語の重なり、B 次の発言の共鳴、C A + B、D C + 文脈 3 発言（古いほど 0.5 倍）、E D + 口調の合い（直前の相手の発言と）、
        F C + 状態遷移グレイン（0.5 × log P(候補の状態 | 直前の状態)、Sin-CNN3 実験120）。
比べる: Sin-CNN3 実験115（全体: やさしい版 C 60.8%・E 65.5%、むずかしい版 C 38.2%。雑談の分 C 56.8%・E 62.9%、むずかしい版 C 34.5%）。
材料: RealPersonaChat 1.0.0・JMultiWOZ 1.0（CC BY-SA 4.0。JMultiWOZ は 2026-11-02 から）。会話ごとに教え 9 : 試し 1（kotoyui.load_conversations）。
結果: results/exp06.txt
実行: python experiments/exp06_response_selection.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import kotoyui as K  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000, help="試しの問の数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="使わない（約束の引数）")
    ap.add_argument("--out", default="results/exp06.txt")
    a = ap.parse_args()
    t0 = time.perf_counter()
    mat, tg = K.load_tables()
    convs = K.load_conversations()
    rng = random.Random(a.seed)
    pool = [t for _, turns in convs["test"] for t in turns]
    cases = []
    for src, turns in convs["test"]:
        for i in range(3, len(turns), 4):
            negs = []
            while len(negs) < 9:
                t2 = rng.choice(pool)
                if t2 != turns[i] and t2 not in turns:
                    negs.append(t2)
            same = [t for j, t in enumerate(turns) if abs(j - i) > 1 and t != turns[i]]
            cases.append((src, turns[max(0, i - 3):i], turns[i], negs, rng.sample(same, min(9, len(same)))))
    rng.shuffle(cases)
    cases = cases[:a.n]

    def score(ctx, cand, how):
        cw_, cp = K.analyze(cand)
        sc = 0.0
        hist = list(reversed(ctx)) if how in ("D", "E") else [ctx[-1]]
        for k, u in enumerate(hist):
            uw, up = K.analyze(u)
            wk = 0.5 ** k
            if how in ("A", "C", "D", "E", "F"):
                sc += wk * K.overlap(uw, cw_)
            if how in ("B", "C", "D", "E", "F"):
                sc += wk * mat.resonance(uw, cw_)
        if how == "E":
            sc += 0.3 if K.analyze(ctx[-1])[1] == cp else 0.0
        if how == "F":
            s_prev = tg.state(ctx[-2] if len(ctx) >= 2 else "", ctx[-1])
            sc += 0.5 * tg.score(s_prev, ctx[-1], cand)
        return sc

    rec = dict(exp="06", **vars(a), n_cases=len(cases), version=K.__version__, runs=[])
    for hard in (False, True):
        for how, name in (("R", "でたらめ"), ("A", "語の重なり"), ("B", "次の発言の共鳴"), ("C", "重なり + 共鳴"), ("D", "+ 文脈 3 発言"),
                          ("E", "+ 口調の合い"), ("F", "重なり + 共鳴 + 状態遷移")):
            top1 = rr = 0.0
            by = {}
            for src, ctx, gold, negs, same in cases:
                cands = [gold] + (same if hard else negs)
                scs = [rng.random() for _ in cands] if how == "R" else [score(ctx, c, how) + 1e-6 * rng.random() for c in cands]
                rank = 1 + sum(s > scs[0] for s in scs[1:])
                top1 += rank == 1
                rr += 1 / rank
                b = by.setdefault(src, [0, 0])
                b[0] += rank == 1
                b[1] += 1
            r = dict(hard=hard, how=how, name=name, top1=top1 / len(cases), mrr=rr / len(cases), by_source={k: v[0] / v[1] for k, v in by.items()})
            rec["runs"].append(r)
            print("同じ会話から" if hard else "ほかの会話から", name, "1 位", round(r["top1"], 3), "MRR", round(r["mrr"], 3),
                  {k: round(v, 3) for k, v in r["by_source"].items()}, flush=True)
    rec["time_s"] = time.perf_counter() - t0
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
