"""I49zzs: 掛け合わせた波と、外れた時だけ位相を動かす学びを、小さい世界で比べる道具（`intervention/i49zzs_bind.py`。持ち主の案、未確定の試し）。

世界（どれも、1 つ目の語と答え、2 つ目の語と答えの回数が均一。2 つずつの数えでは答えが決まらない）:
  輪の足し算: a ＋ b の 1 の位（BASE 進）。答えに決まり（足し算）が有る。
  でたらめ:   同じ組に、答えを黄金比 × 2 乗の並びで入れ替えて付けた物。答えに決まりが無い（比べ役）。
  猫と犬:     BASE = 2 の輪（持ち主の例「猫が・魚を・食べる」「犬が・骨を・かじる」と同じ形）。
見る物:
  覚えた組を引けるか（聞いた組での当たり）と、見ていない組が当たるか（外して置いた組での当たり）。
  外す組は、a と b を入れ替えた組も一緒に外す（入れ替えの組が残っていると、同じ表を使う形ではそれだけで当たるため）。
  語の類の世界: 「誰が・何を・どうする」のように、席ごとに語が違う世界。誰が の語も 何を の語も、G 個の類に M 語ずつ分かれ（類は見せない）、
              どうする は 2 つの類の組で決まる（類の番号の足し算の輪。周りの回数は均一）。同じ類の語は同じ振る舞いをするので、
              聞いていない組（その 誰が と 何を の組は聞いていないが、それぞれの語は別の組で聞いた）に届くかを見る。表は席ごと。
手で置いた物: BASES、HOLDS、DIMS、動かす大きさ、回す上限、類の数と語の数（CLASSES）、隠す割合（CLASS_HOLDS）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "intervention")]

import numpy as np  # noqa: E402

import i49_situation as SIT  # noqa: E402
import i49zzs_bind as B  # noqa: E402

BASES = (10, 16)
HOLDS = (0.0, 0.2, 0.5)
DIMS = (256, 1024, 4096)
ETAS = (0.03, 0.1, 0.3)


def scramble(n):
    return sorted(range(n), key=lambda i: (((i + 1) ** 2) * SIT.PHI) % 1.0)


def world(base, hold, null=False):
    """→ (聞く組, 外して置く組)。組は (a, b, 答え)。"""
    half = [(a, b) for a in range(base) for b in range(a, base)]
    ans = {(a, b): (a + b) % base for a, b in half}
    if null:
        order = scramble(len(half))
        ans = {half[i]: (half[j][0] + half[j][1]) % base for i, j in enumerate(order)}
    k = int(len(half) * hold)
    out = [half[i] for i in SIT.golden_order(len(half))]
    both = lambda ps: [(a, b, ans[(a, b)]) for a, b in ps] + [(b, a, ans[(a, b)]) for a, b in ps if a != b]
    return both(out[k:]), both(out[:k])


CLASSES = ((2, 4), (4, 5), (4, 10), (8, 5))   # (類の数, 類ごとの語の数)
CLASS_HOLDS = (0.2, 0.5, 0.8)


def class_world(g, m, hold, null=False):
    """→ (聞く組, 外して置く組, 席ごとの語の数)。組は (誰が, 何を, どうする)。語の番号と類は、黄金比 × 2 乗の並びで混ぜる。"""
    n = g * m
    mix = scramble(n)
    gs = [mix.index(i) % g for i in range(n)]
    go = [mix[i] % g for i in range(n)]
    pairs = [(s, o) for s in range(n) for o in range(n)]
    ans = {(s, o): (gs[s] + go[o]) % g for s, o in pairs}
    if null:
        ans = {pairs[i]: (gs[pairs[j][0]] + go[pairs[j][1]]) % g for i, j in enumerate(scramble(len(pairs)))}
    order = [pairs[i] for i in SIT.golden_order(len(pairs))]
    k = int(len(pairs) * hold)
    f = lambda ps: [(s, o, ans[(s, o)]) for s, o in ps]
    return f(order[k:]), f(order[:k]), n


def class_one(g, m, hold, dims, null=False):
    tr, te, n = class_world(g, m, hold, null)
    row = dict(聞く組=len(tr), 外した組=len(te), 当てずっぽう=round(1 / g, 4))
    for name, mk in (("2 つずつの数え", lambda: B.PairSum(n, tr, g)), ("掛け合わせて数える", lambda: B.BindCount(n, tr, dims, g))):
        mod = mk()
        row[name] = [round(B.accuracy(mod, tr), 4), round(B.accuracy(mod, te), 4)]
    for eta in ETAS:
        mod = B.BindNudge(n, dims, False, g)
        miss = mod.learn(tr, eta=eta)
        row["外れた時だけ動かす（席ごとの表）、動かす大きさ %s" % eta] = dict(後=[round(B.accuracy(mod, tr), 4), round(B.accuracy(mod, te), 4)], 回した数=len(miss))
    return row


def ladder(m, base):
    """同じ表の位相が、足し算の形（θ(a＋1) − θ(a) が、どの a でも θ(1) と同じ）にどれだけ揃っているか。次元ごとの揃い方の平均（1 で全部揃う）。"""
    t = m.a.astype(np.float64)
    step = np.roll(t, -1, axis=0) - t - t[1][None, :]
    return float(np.abs(np.exp(1j * step).mean(axis=0)).mean())


def one(base, hold, dims, null=False, eta=None):
    tr, te = world(base, hold, null)
    row = dict(聞く組=len(tr), 外した組=len(te), 当てずっぽう=round(1 / base, 4))
    m = B.PairSum(base, tr)
    row["2 つずつの数え"] = [round(B.accuracy(m, tr), 4), round(B.accuracy(m, te), 4)]
    m = B.BindCount(base, tr, dims)
    row["掛け合わせて数える"] = [round(B.accuracy(m, tr), 4), round(B.accuracy(m, te), 4)]
    for name, shared in (("外れた時だけ動かす（同じ表）", True), ("外れた時だけ動かす（席ごとの表）", False)):
        m = B.BindNudge(base, dims, shared)
        first = [round(B.accuracy(m, tr), 4), round(B.accuracy(m, te), 4)]
        lad = ladder(m, base) if shared and base > 2 else None
        miss = m.learn(tr, eta=eta)
        row[name] = dict(動かす前=first, 後=[round(B.accuracy(m, tr), 4), round(B.accuracy(m, te), 4)], 回した数=len(miss), 外れの数=miss[:8])
        if lad is not None:
            row[name]["足し算の形への揃い方"] = [round(lad, 4), round(ladder(m, base), 4)]
    return row


def run():
    res = dict(手で置いた値=dict(BASES=list(BASES), HOLDS=list(HOLDS), DIMS=list(DIMS), ETA=B.ETA, EPOCHS=B.EPOCHS, ETAS=list(ETAS)))
    res["猫と犬"] = {str(d): one(2, 0.0, d) for d in DIMS}
    for name, null in (("輪の足し算", False), ("でたらめ", True)):
        res[name] = {"%d 進、外す割合 %s、%d 次元" % (b, h, d): one(b, h, d, null) for b in BASES for h in HOLDS for d in DIMS}
    res["動かす大きさを替える"] = {"%d 進、外す割合 %s、動かす大きさ %s" % (b, h, e): one(b, h, 4096, null, eta=e)
                           for null in (False,) for b in BASES for h in HOLDS[1:] for e in ETAS}
    res["動かす大きさを替える（でたらめ）"] = {"%d 進、外す割合 %s、動かす大きさ %s" % (b, h, e): one(b, h, 4096, True, eta=e)
                                  for b in BASES for h in HOLDS[1:] for e in ETAS}
    res["手で置いた値"].update(CLASSES=[list(x) for x in CLASSES], CLASS_HOLDS=list(CLASS_HOLDS))
    for name, null in (("語の類の世界", False), ("語の類の世界（でたらめ）", True)):
        res[name] = {"類 %d × %d 語、外す割合 %s" % (g, m, h): class_one(g, m, h, 4096, null) for g, m in CLASSES for h in CLASS_HOLDS}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    res = run()
    res["秒"] = round(time.time() - t0, 1)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
