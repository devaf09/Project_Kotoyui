"""I49zzm: 前の文の終わりを席にして、続く状況を数える（未確定の試し。I49zzl の続き）。

問い: 時制や丁寧さのような状況は、会話の中でしばらく続く。なら、前の文の終わりと今の文の終わりは結び付くはず。
  時の語は使わない。手で打つ語も無い。全部の文の組から学ぶ。
組の作り方（2 通りを別々に数える）:
  同じ行の中: 1 つの発話の中で続いた 2 文（同じ話し手）。
  行をまたぐ: 前の行の最後の文と、次の行の初めの文（多くは別の話し手。作品の切れ目も混ざる）。
作り方: 今の文の終わりの切り出し（1 文字から TAIL 文字まで全部）ごとに、前の文の終わりの切り出し（多い SEATS 個が席）を数え、平均を引いて波にする。
数える物:
  1. 2 つから選ぶ: 前の文の終わりを見て、本当に続いた文の終わりと、別の組の終わりのどちらかを選ぶ（当てずっぽう .5）。
     並べ替えた前半で学び、後半で試す。比べ役は「前の文の終わりと同じ切り出しが多い方を選ぶ」（学ばない、真似るだけ）。
  2. 続き方: 手で決めた 3 つの印（丁寧・過去・問い）で、前の文に印が有る時に今の文にも有る割合を、全体の割合と比べる（読むための物で、学習には使わない）。
  3. 独立に動く向きの取り分と、強い向きの両側の終わり。
手で置いた物: TAIL、SEATS、MIN、同じ組を 1 回だけ数える事、3 つの印の決め方、別の組のずらし方。
材料の文は出さない。出すのは数と、文の終わりの短い切り出し（語尾）だけ。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "intervention")]

import i49_situation as SIT
import i49zzl_tense_text_run as ZL
from i49_situation import Assoc, marks, overlap, pairs_of, runs_of  # noqa: F401  波の部品は i49_situation に有る


def two_choice(a, held):
    """本当に続いた終わりと、別の組（半分ずらした所）の終わりから選ぶ。同点は半分。"""
    n = len(held)
    if n < 2:
        return dict(問い=0)
    shift = n // 2 + 1
    ok = cp = 0.0
    for i, (p, c) in enumerate(held):
        other = held[(i + shift) % n][1]
        if other == c:
            ok += .5
            cp += .5
            continue
        s1, s0 = a.score(p, c), a.score(p, other)
        ok += 1.0 if s1 > s0 else (.5 if s1 == s0 else 0.0)
        o1, o0 = overlap(p, c), overlap(p, other)
        cp += 1.0 if o1 > o0 else (.5 if o1 == o0 else 0.0)
    return dict(問い=n, 学んだ波で選ぶ=round(ok / n, 4), 同じ切り出しが多い方を選ぶ=round(cp / n, 4))


def carry(pairs):
    out = {}
    for k in ("丁寧", "過去", "問い"):
        both = prev = cur = 0
        for p, c in pairs:
            a, b = marks(p)[k], marks(c)[k]
            prev += a
            cur += b
            both += a and b
        n = max(len(pairs), 1)
        out[k] = dict(全体の割合=round(cur / n, 4), 前の文に有る時の割合=round(both / max(prev, 1), 4),
                      倍=round((both / max(prev, 1)) / max(cur / n, 1e-9), 2))
    return out


def directions(a, k=3, n=8):
    if a.w.size == 0:
        return [], []
    u, s, vt = np.linalg.svd(a.w, full_matrices=False)
    share = s / s.sum() if s.sum() > 0 else s
    out = []
    for d in range(min(k, len(s))):
        col = u[:, d]
        out.append(dict(取り分=round(float(share[d]), 4),
                        今の文の正の側=[a.units[i] for i in np.argsort(-col, kind="stable")[:n]],
                        今の文の負の側=[a.units[i] for i in np.argsort(col, kind="stable")[:n]],
                        前の文の正の側=[a.seats[i] for i in np.argsort(-vt[d], kind="stable")[:n]],
                        前の文の負の側=[a.seats[i] for i in np.argsort(vt[d], kind="stable")[:n]]))
    return [round(float(x), 4) for x in share[:10]], out


def run_one(pairs):
    order = ZL.golden_order(len(pairs))
    half = len(order) // 2
    train = [pairs[i] for i in order[:half]]
    held = [pairs[i] for i in order[half:]]
    a = Assoc(train)
    share, dirs = directions(a)
    return dict(組=len(pairs), 席=len(a.seats), 数えた切り出し=len(a.units), 二つから選ぶ=two_choice(a, held),
                続き方=carry(pairs), 独立に動く向き=share, 強い向き=dirs)


def run_lags(runs, max_lag=4):
    """何個前の発話まで見るかを変えて数える（持ち主の問い: 弱くても重ねれば強くなるか、丁寧さは相手との関係で決まるか）。
    組: lag 個前の発話の最後の文の終わり → 今の発話の初めの文の終わり。続きの切れ目はまたがない。
    1. 続き方: lag ごとに、3 つの印が続く倍率。2 人の会話なら、lag 2 は同じ話し手に戻る事が多い。
    2. 重ねる: 続きを並べ替えて半分で学び、残りで、1 個前だけ・2 個前まで・… と足して 2 つから選ぶ。lag ごとに別の波を持ち、点を足すだけ。"""
    carry_by = {}
    for lag in range(1, max_lag + 1):
        ps = [(r[i - lag][1], r[i][0]) for r in runs for i in range(lag, len(r))]
        carry_by[str(lag)] = dict(組=len(ps), **carry(ps))
    order = ZL.golden_order(len(runs))
    half = len(order) // 2
    train = [runs[i] for i in order[:half]]
    held = [runs[i] for i in order[half:]]
    assoc = {}
    for lag in range(1, max_lag + 1):
        seen = {}
        for r in train:
            for i in range(lag, len(r)):
                seen.setdefault((r[i - lag][1], r[i][0]), None)
        assoc[lag] = Assoc(list(seen))
    items = [([r[i - lag][1] for lag in range(1, max_lag + 1)], r[i][0]) for r in held for i in range(max_lag, len(r))]
    n = len(items)
    stack = {}
    alone = {}
    if n >= 2:
        shift = n // 2 + 1
        ok = [0.0] * max_lag
        one = [0.0] * max_lag
        for i, (prevs, c) in enumerate(items):
            other = items[(i + shift) % n][1]
            if other == c:
                ok = [x + .5 for x in ok]
                one = [x + .5 for x in one]
                continue
            d = [assoc[lag].score(prevs[lag - 1], c) - assoc[lag].score(prevs[lag - 1], other) for lag in range(1, max_lag + 1)]
            tot = 0.0
            for k in range(max_lag):
                tot += d[k]
                ok[k] += 1.0 if tot > 0 else (.5 if tot == 0 else 0.0)
                one[k] += 1.0 if d[k] > 0 else (.5 if d[k] == 0 else 0.0)
        stack = {"%d 個前まで" % (k + 1): round(ok[k] / n, 4) for k in range(max_lag)}
        alone = {"%d 個前だけ" % (k + 1): round(one[k] / n, 4) for k in range(max_lag)}
    return dict(続き=len(runs), 発話=sum(len(r) for r in runs), 続き方=carry_by, 二つから選ぶ=dict(問い=n, 重ねる=stack, 一つだけ=alone))


def run_breaks(runs):
    """状況波が合わなくなった所を、場の切れ目として読めるかを数える（持ち主:「作品の切れ目シチュエーションが変わるからだね。」）。
    続きを並びのまま前半と後半に分け、前半で 1 個前・2 個前の波を学ぶ。後半で、発言ごとに「前の 2 つの発言との合い方」（点の足し算）を出す。
    切れ目の直後の発言（前の 2 つは別の続き）と、続きの中の発言（3 つ目以降）で、合い方を比べる。
    出す物: 続きの中の方が合い方が高い割合（順位で数える。.5 = 見分けられない）。比べ役は、同じ切り出しの数（真似るだけ）。
    確かめ: 切れ目の直後の発言は「場の初めの発言」でもあり、それだけで終わり方が違うかもしれない。
      なので、続きの中の発言はそのままに、前の発言だけ別の続きの物に替えた時と比べた割合も出す（位置は同じで、場だけ違う）。"""
    import i49zzf_attr_bands as ZF
    half = len(runs) // 2
    train, held = runs[:half], runs[half:]
    assoc = {}
    for lag in (1, 2):
        seen = {}
        for r in train:
            for i in range(lag, len(r)):
                seen.setdefault((r[i - lag][1], r[i][0]), None)
        assoc[lag] = Assoc(list(seen))
    fit, copy, inside = [], [], []
    kept = []

    def add(p1, p2, c, is_inside):
        if is_inside:
            kept.append((p1, p2, c))
        fit.append(assoc[1].score(p1, c) + assoc[2].score(p2, c))
        copy.append(overlap(p1, c) + overlap(p2, c))
        inside.append(is_inside)

    for k, r in enumerate(held):
        if k and len(held[k - 1]) >= 2:
            add(held[k - 1][-1][1], held[k - 1][-2][1], r[0][0], False)
        for i in range(2, len(r)):
            add(r[i - 1][1], r[i - 2][1], r[i][0], True)
    f, c, y = np.array(fit, dtype=np.float64), np.array(copy, dtype=np.float64), np.array(inside, dtype=bool)
    # 確かめ: 続きの中の発言に、別の続きの中の前の発言を当てた時（発言の位置は同じで、場だけ違う）と比べる
    n = len(kept)
    swap = [assoc[1].score(kept[(i + n // 2 + 1) % n][0], kept[i][2]) + assoc[2].score(kept[(i + n // 2 + 1) % n][1], kept[i][2]) for i in range(n)]
    ctrl = ZF.rank_share(np.array([x for x in f[y]] + swap, dtype=np.float64), np.array([True] * n + [False] * n)) if n else None
    lo = float(np.median(f[~y])) if (~y).any() else None
    hi = float(np.median(f[y])) if y.any() else None
    return dict(学ぶ続き=len(train), 試す続き=len(held), 切れ目の直後の発言=int((~y).sum()), 続きの中の発言=int(y.sum()),
                続きの中の方が合う割合=dict(学んだ波=ZF.rank_share(f, y), 同じ切り出しの数=ZF.rank_share(c, y)),
                前の発言だけ別の場の物に替えた時より合う割合=ctrl,
                合い方の真ん中の値=dict(切れ目の直後=None if lo is None else round(lo, 4), 続きの中=None if hi is None else round(hi, 4)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--breaks", action="store_true")  # 状況波が合わなくなった所を、場の切れ目として読めるか
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lags", action="store_true")   # 空の行を続きの切れ目として、何個前の発話まで見るかを変えて数える
    a = ap.parse_args()
    if a.lags or a.breaks:
        t0 = time.time()
        lines = []
        for f in a.text.split(","):
            lines += open(f, encoding="utf-8", errors="replace").read().split("\n")
        res = dict(行=len(lines), 手で置いた値=dict(TAIL=ZL.TAIL, SEATS=SIT.SEATS, MIN=SIT.MIN), **(run_breaks if a.breaks else run_lags)(runs_of(lines)))
        res["秒"] = round(time.time() - t0)
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        return
    t0 = time.time()
    lines = []
    for f in a.text.split(","):
        lines += open(f, encoding="utf-8", errors="replace").read().splitlines()
    inside, across = pairs_of(lines)
    res = dict(行=len(lines), 手で置いた値=dict(TAIL=ZL.TAIL, SEATS=SIT.SEATS, MIN=SIT.MIN), 同じ行の中=run_one(inside), 行をまたぐ=run_one(across))
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
