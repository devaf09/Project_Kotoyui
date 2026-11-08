"""介入の研究 I27w の確かめ: 像の数 K と k 平均の種を変えても、同じ像（とくに息づかいの像）が出るか

持ち主（2026-11-08）: 息づかいの像は種を選べば出るだけかもしれない → 種選びはやめ、K=12/16 で安定して分かれるかを見る。
安定しなければ、息づかいは像ではなく語尾そのものの属性にする（3 つ目の表: 目印の語尾と一緒に出る語の偏り）。
  1. I27t（fold=-1）で人の板を 1 回だけ育てる（i27w と同じ）。
  2. K × 種 ごとに像を作り、像のらしい語尾の上位 4 つに目印が 2 つ以上ある像を「息づかい」と数える。
     種どうしの像の重なり（重心の cos の最良一致の平均）も出す。
  3. 語尾が目印の台詞にだけ出る語の偏り（像を通さない）を、作品を半分に分けて相関を見る。切れ端（〜っ の語）は外す。
結果は非公開の置き場だけ。
実行: python intervention/i27w_k_stability.py --quotes <quotes_words_wn.jsonl> --stats <jsonl> --out <md> --jobs 10
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i27t_first_person as T  # noqa: E402
import i27u_character_images as U  # noqa: E402
import i27v_images_1808 as V  # noqa: E402
import i27w_word_images as W  # noqa: E402

MARK = ("っ", "いっ", "すっ", "んっ", "ょ")


def top_ends(im, C, M, j, n=4):
    d = M[j] - M.mean(0)
    return [im.vocab[i][1] for i in np.argsort(-d)[:12] if d[i] > 0 and im.vocab[i][0] == "尾1"][:n]


def is_breath(ends):
    return sum(1 for e in ends if e in MARK) >= 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quotes", required=True)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--Ks", default="8,12,16")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--beta", type=float, default=20.0)
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--min_count", type=int, default=20)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    works = V.load(a.quotes)
    allq = [w[1] for w in works]
    rates = T.flow_rates(allq, T.known_of(allq, -1))
    from multiprocessing import Pool
    idx = list(range(len(works)))
    parts = [p for p in (idx[i::a.jobs * 4] for i in range(a.jobs * 4)) if p]
    out = {}
    with Pool(a.jobs) as pool:
        tasks = [([works[i][1] for i in p], [works[i][2] for i in p], a.rounds, rates) for p in parts]
        for p, res in zip(parts, pool.map(W.job, tasks)):
            for i, r in zip(p, res):
                out[i] = r
    print("広げた", f"{time.time() - t0:.0f}s", flush=True)
    vecs3 = [{c: v for c, v in out[i][1].items() if v[0] >= 3.0} for i in idx]
    rec = dict(exp="I27w_K", 作品=len(works), 目印=MARK, K={})
    lines = ["# I27w の確かめ: K と種を変えた像（非公開）", "",
             f"息づかい = 像のらしい語尾の上位 4 つに目印（{'・'.join(MARK)}）が 2 つ以上。", ""]
    for K in [int(x) for x in a.Ks.split(",")]:
        cents, breath, rows = [], [], []
        for s in range(a.seeds):
            im = U.Images(vecs3, K, a.beta, seed=s)
            C, M, asg, rs = im.for_work(-1)
            names = [top_ends(im, C, M, j) for j in range(len(C))]
            wsum = np.array([sum(np.log1p(rs[i][2]) for i in range(len(rs)) if asg[i] == j) for j in range(len(C))])
            share = wsum / wsum.sum()
            br = [(j, round(float(share[j]), 4)) for j in range(len(C)) if is_breath(names[j])]
            breath.append(br)
            cents.append(C)
            rows.append(f"- 種 {s}: " + " / ".join(f"{'★' if is_breath(n) else ''}{'・'.join(n)}（{share[j]:.1%}）" for j, n in enumerate(names)))
        # 種どうしの重なり: 種 0 の各像に、ほかの種でいちばん近い像の cos
        over = []
        for s in range(1, a.seeds):
            S = cents[0] @ cents[s].T
            over.append(float(S.max(1).mean()))
        nb = sum(1 for b in breath if b)
        rec["K"][K] = dict(息づかいが出た種=nb, 種=a.seeds, 息づかいの割合=[b for b in breath],
                           種0との重なり=round(float(np.mean(over)), 3) if over else None)
        lines += [f"## K={K}: 息づかいが出た種 {nb}/{a.seeds}、種 0 との像の重なり（cos の平均）{rec['K'][K]['種0との重なり']}", ""] + rows + [""]
        print("K", K, rec["K"][K], f"{time.time() - t0:.0f}s", flush=True)
    # 3: 像を通さない、目印の語尾の台詞での語の偏り
    rng = random.Random(0)
    half = {i: rng.random() < 0.5 for i in idx}
    cnt = [[defaultdict(float), defaultdict(float)] for _ in (0, 1)]   # [半分][目印か]
    tot = [[0.0, 0.0] for _ in (0, 1)]
    for i in idx:
        h = int(half[i])
        for q in works[i][1]:
            if "w" not in q:
                continue
            m = int(any(k[0] == "尾1" and k[1] in MARK for k in q["f"]))
            for x, v in q["w"].items():
                if x[1].endswith("っ"):
                    continue
                cnt[h][m][x] += v
                tot[h][m] += v

    def bias(h):
        base = tot[h][1] / (tot[h][0] + tot[h][1])
        o = {}
        for x in set(cnt[h][0]) | set(cnt[h][1]):
            n1, n0 = cnt[h][1].get(x, 0.0), cnt[h][0].get(x, 0.0)
            o[x] = (n0 + n1, float(np.log((n1 + a.alpha * base) / (n0 + n1 + a.alpha) / base)))
        return o
    BA, BB = bias(0), bias(1)
    common = sorted(x for x in BA if x in BB and BA[x][0] >= a.min_count and BB[x][0] >= a.min_count)
    r = float(np.corrcoef([BA[x][1] for x in common], [BB[x][1] for x in common])[0, 1]) if len(common) > 2 else None
    share = (tot[0][1] + tot[1][1]) / sum(map(sum, tot))
    rec["目印の語尾の属性"] = dict(語=len(common), 半分どうしの相関=None if r is None else round(r, 3), 目印の台詞の語の割合=round(share, 4))
    top = sorted(common, key=lambda x: -(BA[x][1] + BB[x][1]))[:30]
    lines += ["## 像を通さない属性: 目印の語尾の台詞に偏る語（〜っ の切れ端は外した）", "",
              f"{len(common):,} 語、半分どうしの相関 {rec['目印の語尾の属性']['半分どうしの相関']}、目印の台詞の語の割合 {share:.1%}。", "",
              "- " + "、".join(f"{x[1]}（{x[0]}、{(BA[x][1] + BB[x][1]) / 2:+.2f}）" for x in top), ""]
    rec["秒"] = round(time.time() - t0)
    print(json.dumps(rec["目印の語尾の属性"], ensure_ascii=False), flush=True)
    with open(a.stats, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
