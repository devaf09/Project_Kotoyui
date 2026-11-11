"""I27u: 板の 4 つの操作の行列の版（board_ops_xp）が、辞書の版（board_ops）と同じ答えになるかと、速さを比べる

I27s と同じ 2,000 か所で、作品全体を直線の坂で束ねた板（束ねる）と、その照らした値（照らす）を両方で作って比べる。
実行: python intervention/i27u_bench_xp.py --cache <解析済み JSON> [--backend numpy|torch]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i27q_book_insert as Q  # noqa: E402
import i27s_fused_generate as S  # noqa: E402
import board_ops as B  # noqa: E402
import board_ops_xp as X  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--backend", default="numpy")
    ap.add_argument("--n_stats", type=int, default=2000)
    a = ap.parse_args()
    works = json.load(open(a.cache, encoding="utf-8"))
    Q.prep(works)
    rng = random.Random(0)
    sites = [(wi, li) for wi, w in enumerate(works) for li in range(10, len(w["lines"]) - 10)
             if any(t["dep"] == "nsubj" for t in w["lines"][li]["toks"])]
    rng.shuffle(sites)
    sites = sites[:a.n_stats]
    rows = [S.line_rows(w) for w in works]
    # 辞書の版
    t0 = time.perf_counter()
    dict_boards = {}
    for wi, li in sites:
        n = len(works[wi]["lines"])
        bb = B.bundle([(lj, rows[wi][lj]) for lj in range(n) if lj != li], weight=lambda lj: Q.slope("直線", abs(lj - li), n))
        vals = [v for k, v in bb.items() if k[0] == "板"]
        sat = sum(vals) / len(vals)
        dict_boards[(wi, li)] = {k: B.light(bb, k, sat=sat) for k in bb if k[0] == "板"}
    t_dict = time.perf_counter() - t0
    # 行列の版
    xp = X.backend(a.backend)
    t0 = time.perf_counter()
    by_work = defaultdict(list)
    for wi, li in sites:
        by_work[wi].append(li)
    mat_boards = {}
    t_build = 0.0
    for wi, lis in by_work.items():
        tb = time.perf_counter()
        idx = X.KeyIndex()
        R = X.rows_matrix(xp, rows[wi], idx)
        n = len(works[wi]["lines"])
        Wt = X.line_slope(xp, lis, n)
        t_build += time.perf_counter() - tb
        BBm = X.bundle_many(R, Wt)
        cols = [i for k, i in idx.items() if k[0] == "板"]
        keys = [k for k, i in idx.items() if k[0] == "板"]
        P = BBm[:, cols]
        nz = (P > 0)
        sat = (P.sum(1) / nz.sum(1))[:, None]
        L = X.light_many(xp, P, sat=sat)
        Ll = xp.tolist(L)
        for r, li in enumerate(lis):
            mat_boards[(wi, li)] = (keys, Ll[r])
    t_mat = time.perf_counter() - t0
    # 比べる
    maxd, n = 0.0, 0
    for s_, d in dict_boards.items():
        keys, vals = mat_boards[s_]
        for k, v in zip(keys, vals):
            dv = d.get(k, 0.0)
            maxd = max(maxd, abs(dv - v))
            n += 1
    print(json.dumps(dict(か所=len(sites), 辞書の版_秒=round(t_dict, 2), 行列の版_秒=round(t_mat, 2),
                          うち行列を作る_秒=round(t_build, 2), 速さの比=round(t_dict / t_mat, 1),
                          比べた値=n, いちばん大きい差=maxd, backend=a.backend, device=xp.device), ensure_ascii=False))


if __name__ == "__main__":
    main()
