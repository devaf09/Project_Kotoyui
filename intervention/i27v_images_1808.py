"""介入の研究 実験I27v: 1,808 作品の 8 割で共通のキャラクター像を学び、残り 2 割で話し手を当てる

持ち主（2026-11-08）: 「とりあえず環境がOKなら1808作品の８割くらいで学習してくれる？」
材料: i27v_extract_quotes.py の JSONL（作品ごとの台詞の語尾・一人称・話し手の正解）。
  学ぶ 8 割: I27t（語尾から学び、介入で広げる）で人ごとの板を育て、I27u と同じく k 平均で束ねて像にする。
  確かめる 2 割: 正解を交互に半分隠し、I27t（像なし）と、8 割の像を使った I27u で、隠した台詞の話し手を当てる。
流れの割合（何個前の台詞と同じ人か）も 8 割から数える。作品ごとに独立なので、並べて回す（--jobs）。
結果と像の中身は非公開の置き場だけ。本文はコードに書かない。
実行: python intervention/i27v_images_1808.py --quotes <quotes_1808_wn.jsonl> --stats <jsonl> --out <像の md> --jobs 10
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i27t_first_person as T  # noqa: E402
import i27u_character_images as U  # noqa: E402
from board_ops import Board  # noqa: E402


def load(path):
    works = []
    with open(path, encoding="utf-8") as f:
        for x in f:
            r = json.loads(x)
            qs = [dict(li=q["li"], f=Board({(a, b): v for a, b, v in q["f"]}), fp=Board({(a, b): v for a, b, v in q["fp"]}),
                       gold=q["gold"], kind=q["kind"], **({"w": Counter({(a, b): v for a, b, v in q["w"]})} if "w" in q else {}))
                  for q in r["quotes"]]
            works.append((r["work_id"], qs, Counter(r["narr"])))
    works.sort()
    return works


class ImgFn:
    """作品の外（8 割）で作った像を、どの作品にも使う。"""

    def __init__(self, im):
        self.im = im

    def __call__(self, wi, P, cands):
        return self.im.back(-1, P, cands)


def job(args):
    qs_list, narr, fold, rounds, rates, image = args
    h, n, boards, _ = T.run_fold(qs_list, fold, U.USE, rounds, narr, image=image, rates=rates)
    return h, n, [U.person_vecs(P) for P in boards]


def chunks(xs, k):
    return [xs[i::k] for i in range(k)]


def run_par(pool, ws, fold, rounds, rates, image, jobs):
    parts = chunks(ws, jobs * 4)
    tasks = [([w[1] for w in p], [w[2] for w in p], fold, rounds, rates, image) for p in parts if p]
    H, N, V = Counter(), 0, []
    for h, n, vs in pool.imap_unordered(job, tasks):
        H.update(h)
        N += n
        V += vs
    return H, N, V


def summ(h, n):
    return dict(当たり=round(h["当たり"] / max(1, n), 4), 数=n,
                当たり_話す動詞=round(h[("当", "話す動詞")] / max(1, h[("数", "話す動詞")]), 4),
                数_話す動詞=h[("数", "話す動詞")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quotes", required=True)
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--Ks", default="8,16,32")
    ap.add_argument("--stats", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    works = load(a.quotes)
    rng = random.Random(0)
    order = list(range(len(works)))
    rng.shuffle(order)
    ntr = int(len(works) * a.train)
    train = [works[i] for i in sorted(order[:ntr])]
    test = [works[i] for i in sorted(order[ntr:])]
    gold = Counter(q["kind"] for w in works for q in w[1] if q["gold"] is not None)
    rec = dict(exp="I27v", 作品=len(works), 学ぶ作品=len(train), 確かめる作品=len(test),
               台詞=sum(len(w[1]) for w in works), 正解=dict(gold), 結果={})
    print(json.dumps({k: v for k, v in rec.items() if k != "結果"}, ensure_ascii=False), flush=True)
    from multiprocessing import Pool
    with Pool(a.jobs) as pool:
        res = {"I27t（像なし）": (Counter(), 0)}
        Ks = [int(x) for x in a.Ks.split(",")]
        for K in Ks:
            res[f"像 K={K}（8 割から）"] = (Counter(), 0)
        n_people = 0
        for fold in (0, 1):
            tq = [w[1] for w in train]
            rates = T.flow_rates(tq, T.known_of(tq, fold))
            _, _, V = run_par(pool, train, fold, a.rounds, rates, None, a.jobs)    # 学ぶ 8 割で人の板を育てる
            n_people = max(n_people, sum(len(v) for v in V))
            print("fold", fold, "学ぶ 8 割の人", sum(len(v) for v in V), f"{time.time() - t0:.0f}s", flush=True)
            h, n, _ = run_par(pool, test, fold, a.rounds, rates, None, a.jobs)
            res["I27t（像なし）"][0].update(h)
            res["I27t（像なし）"] = (res["I27t（像なし）"][0], res["I27t（像なし）"][1] + n)
            for K in Ks:
                im = U.Images(V, K, 20.0)
                im.for_work(-1)
                h, n, _ = run_par(pool, test, fold, a.rounds, rates, ImgFn(im), a.jobs)
                nm = f"像 K={K}（8 割から）"
                res[nm][0].update(h)
                res[nm] = (res[nm][0], res[nm][1] + n)
                print("fold", fold, nm, summ(h, n), f"{time.time() - t0:.0f}s", flush=True)
                if fold == 0 and K == Ks[0]:
                    report = (im, K)
    for nm, (h, n) in res.items():
        rec["結果"][nm] = summ(h, n)
    rec["学ぶ 8 割の人（台詞の重み 3 以上）"] = n_people
    im, K = report
    C, M, asg, rs = im.for_work(-1)
    allm = M.mean(0)
    lines = [f"# I27v 1,808 作品の 8 割で学んだキャラクター像（K={K}、非公開）", ""]
    import numpy as np
    for j in range(len(C)):
        mem = [rs[i] for i in range(len(rs)) if asg[i] == j]
        if not mem:
            continue
        d = M[j] - allm
        top = [im.vocab[i] for i in np.argsort(-d)[:12] if d[i] > 0]
        ends = [f[1] for f in top if f[0] == "尾1"][:8]
        fps = sorted(((M[j][im.ix[f]], f[1]) for f in im.vocab if f[0] == "一"), reverse=True)[:3]
        narr_share = sum(1 for r in mem if r[1] == T.NARR) / len(mem)
        lines += [f"## 像 {j + 1}（{len(mem)} 人、うち語り手 {narr_share:.0%}）",
                  f"- らしい語尾: {'、'.join(ends)}",
                  f"- 一人称: {'、'.join(f'{w} {v:.2f}' for v, w in fps if v > 0) or '—'}",
                  f"- 人（台詞の多い順）: {'、'.join(r[1] for r in sorted(mem, key=lambda r: -r[2])[:10])}", ""]
        rec.setdefault("像", []).append(dict(人数=len(mem), 語尾=ends, 一人称=[w for v, w in fps if v > 0], 語り手の割合=round(narr_share, 3)))
    rec["秒"] = round(time.time() - t0)
    print(json.dumps(rec["結果"], ensure_ascii=False, indent=1), flush=True)
    with open(a.stats, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
