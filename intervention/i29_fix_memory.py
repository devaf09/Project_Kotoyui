"""介入の研究 実験I29: 間違った計算の記憶を介入で直す — 0〜9 の板の演算の帯に誤りを混ぜ、2 桁の式の答えとの食い違いから見つけて直す
持ち主（2026-11-06）: 「間違った記憶の計算違いを介入で直すことはできるかな？」
  - 0〜9 の板の演算の帯（I22）を、文 300 個（10 × 10 × 3 演算）で教える。そのうち割合 q の文は、結果を間違えて覚える
    （「7 に 5 を足すと 3、桁上り 0」のような誤った文。1 文なので、7 の板の（左, ＋, 5）と 5 の板の（右, ＋, 7）の両方に誤りが入る）。
    q = 0.05・0.1・0.2・0.4（でたらめな誤り）と、系統の誤り「桁上り忘れ」（＋ の桁上りをいつも 0 と覚えた 45 文）。
  - 計算の手順は I28 で学べた手の手順（つなぎの板）を使う。2 桁の数と演算と、答えの数だけが返ってくる。照らすときは入れ替えの気付きを使わない
    （使うと、誤った 7 + 5 と正しい 5 + 7 の明るさが同じになり、どちらが光るかが並びで決まってしまうため）。
  - **判断の板（介入）**: 答えが違ったら、その式で照らした帯の組（照射ごとの (a, 演算, b)）を 1 つずつ取り、その結果だけを別の値に替えて解き直す。
    答えが合う替え方を「直す候補」とする（候補が k 個なら、それぞれに 1 / k 票）。
    同じ帯の組・同じ値に票が V 票たまり、その組の中でいちばん票が多ければ、その文を正しい結果で覚え直す（いまの明るさより強く書く）。
    候補が 1 つも無い（誤りが 2 つ以上重なった）式は見送る。
  - **入れ替えの気付きで決める**（比べ）: 候補の中で、入れ替えた組（b, 演算, a）の結果と同じ値になるものに、票を足す（＋ と × だけ）。
  - **入れ替えの差分だけ**（比べ、式を使わない）: 同じ板の（左, op, y）と（右, op, y）が食い違う組を「怪しい」とする。どちらが正しいかは決められないので、数えるだけ。
I29b（--exp b）: 返ってくる答えのうち割合 e（0・5・10・20 %）を誤らせる（桁を 1 つ別の数字に）。票のしきい値 V = 2・3・5 で比べる。
  「確かめ」: 覚え直す前に、答えが合った最近の式（40 個）を新しい値で解き直し、1 つでも壊れたら覚え直さない（その票は捨てる）。
物差し: 誤った文のうち直った数・正しい文を誤って替えた数、0〜99 の組の正解率（直す前と後）、見送った式の数。
結果: ../Project_Kotoyui_private/intervention/results/i29.jsonl
実行: python intervention/i29_fix_memory.py --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i19_grain_arith as G  # noqa: E402
from i21_board_alu import OpBands  # noqa: E402
from i22_carry import digit_fact  # noqa: E402
from i28_route_board import N_STEP, OPS, OUT_ROW, OUTS, PAIRS, hand_choice, order_of  # noqa: E402

FACTS = [(a, op, b) for op in OPS for a in range(10) for b in range(10)]
CARRY_MAX = {"+": 1, "-": 1, "*": 8}


def wrong_result(key, rng):
    a, op, b = key
    right = digit_fact(a, op, b)
    while True:
        r = (rng.randrange(10), rng.randrange(CARRY_MAX[op] + 1))
        if r != right:
            return r


def lookup(bands, key):
    a, op, b = key
    return bands.light(a, op, b, False)


def solve(op, A, B, tab):
    """手の手順で照らす。答えと、照らした帯の組（照射ごと）を返す。"""
    ch = hand_choice(op)
    order = order_of(ch)
    p = [A % 10, A // 10, B % 10, B // 10] + [0] * (2 * N_STEP)
    used = []
    for k in order:
        l, r, o = ch[3 * k], ch[3 * k + 1], ch[3 * k + 2]
        key = (p[l], OPS[o], p[r])
        used.append(key)
        p[4 + 2 * k], p[5 + 2 * k] = tab[key]
    h, t, d = (p[ch[OUT_ROW[x]]] for x in OUTS)
    return h * 100 + t * 10 + d, used


def accuracy(tab):
    out = {}
    for op in OPS:
        out[op] = sum(solve(op, x, y, tab)[0] == G.FN[op](x, y) for x, y in PAIRS[op]) / len(PAIRS[op])
    return out


def alternatives(key):
    a, op, b = key
    return [(d, c) for d in range(10) for c in range(CARRY_MAX[op] + 1)]


def noisy(v, rng):
    """返ってくる答えの誤り: 桁を 1 つ選んで別の数字にする（0〜199 に収まるもの）。"""
    while True:
        pos = rng.choice((1, 10, 100))
        d = v // pos % 10
        w = v + (rng.randrange(10) - d) * pos
        if w != v and 0 <= w <= 199:
            return w


def run(seed, q, method, n_ex, votes_need, noise=0.0):
    rng = random.Random(seed * 101 + int(q * 1000) if q != "桁上り忘れ" else seed * 101 + 7)
    if q == "桁上り忘れ":                                          # 系統の誤り: ＋ の桁上りをいつも 0 と覚えた（45 文）
        bad = {k for k in FACTS if k[1] == "+" and digit_fact(*k)[1] == 1}
        wrong = {k: (digit_fact(*k)[0], 0) for k in bad}
    else:
        bad = set(rng.sample(FACTS, int(round(q * len(FACTS)))))
        wrong = {k: wrong_result(k, rng) for k in sorted(bad, key=str)}
    bands = OpBands()
    for key in FACTS:
        a, op, b = key
        bands.teach(a, op, b, wrong[key] if key in bad else digit_fact(a, op, b))
    tab = {key: lookup(bands, key) for key in FACTS}
    # 入れ替えの差分だけ（式を使わない）: 同じ板の（左, op, y）と（右, op, y）の食い違い
    suspect = {(a, op, b) for a, op, b in FACTS if op in ("+", "*") and tab[(a, op, b)] != tab[(b, op, a)]}
    rec = dict(q=q, method=method, noise=noise, votes=votes_need, n_bad=len(bad), acc_before=accuracy(tab),
               swap_suspect=len(suspect), swap_suspect_bad=len(suspect & bad))
    if method == "なし":
        rec.update(acc_after=rec["acc_before"], fixed=0, broke=0, skipped=0, solves=0, examples=0)
        return rec
    ex_rng = random.Random(seed * 7 + 3)
    votes = defaultdict(Counter)
    fixed_keys, broke, skipped, solves, rewrites = set(), 0, 0, 0, 0
    curve, memory, vetoed = [], [], 0
    for n in range(1, n_ex + 1):
        op = ex_rng.choice(OPS)
        x, y = ex_rng.choice(PAIRS[op])
        target = G.FN[op](x, y)
        if noise and ex_rng.random() < noise:
            target = noisy(target, ex_rng)                         # I29b: 返ってくる答えが誤っている
        got, used = solve(op, x, y, tab)
        solves += 1
        if got == target:
            memory.append((op, x, y, target))                      # 合った式は覚えの例（新しいものから 40 個）
            del memory[:-40]
        if got != target:
            cands = []
            for key in dict.fromkeys(used):                        # 照らした帯の組を 1 つずつ（同じ組は 1 回）
                keep = tab[key]
                for r in alternatives(key):
                    if r == keep:
                        continue
                    tab[key] = r
                    solves += 1
                    if solve(op, x, y, tab)[0] == target:
                        cands.append((key, r))
                tab[key] = keep
            if not cands:
                skipped += 1
            for key, r in cands:
                w = 1.0 / len(cands)
                a, o, b = key
                if "入れ替え" in method and o in ("+", "*") and tab[(b, o, a)] == r:
                    w += 1.0                                       # 入れ替えた組が同じ結果を言っている
                votes[key][r] += w
            for key, r in cands:
                if not votes[key]:
                    continue                                       # この式で、もう覚え直した組
                (best, v), = votes[key].most_common(1)
                if best == r and v >= votes_need and tab[key] != r:
                    if "確かめ" in method:                           # 覚え直す前に、覚えの例を壊さないかを確かめる
                        keep = tab[key]
                        tab[key] = r
                        bad_mem = 0
                        for o2, x2, y2, t2 in memory:
                            solves += 1
                            if solve(o2, x2, y2, tab)[0] != t2:
                                bad_mem += 1
                                break
                        tab[key] = keep
                        if bad_mem:
                            vetoed += 1
                            votes[key][r] = 0.0
                            continue
                    a, o, b = key
                    cur = sum(bands.band[a][("左", o, b)].values()) + sum(bands.band[b][("右", o, a)].values())
                    bands.teach(a, o, b, r, n=cur + 1)              # 覚え直す: いまの明るさより強く書く
                    tab[key] = lookup(bands, key)
                    rewrites += 1
                    votes[key].clear()
                    if r == digit_fact(a, o, b):
                        fixed_keys.add(key)
                    else:
                        broke += 1
        if n in (100, 300, 1000, 3000, 10000):
            curve.append((n, sum(tab[k] != digit_fact(*k) for k in FACTS)))
    wrong_now = {k for k in FACTS if tab[k] != digit_fact(*k)}
    rec.update(acc_after=accuracy(tab), fixed=len(fixed_keys & bad), still_wrong=len(wrong_now), still_wrong_bad=len(wrong_now & bad),
               newly_wrong=len(wrong_now - bad), broke=broke, vetoed=vetoed, rewrites=rewrites, skipped=skipped, solves=solves, examples=n_ex,
               curve=curve)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_ex", type=int, default=3000)
    ap.add_argument("--votes", type=float, default=2.0)
    ap.add_argument("--exp", default="a", help="a: 記憶の誤り（I29）、b: 返ってくる答えにも誤り（I29b）")
    ap.add_argument("--out", default="../Project_Kotoyui_private/intervention/results/i29.jsonl")
    a = ap.parse_args()
    t0 = time.perf_counter()
    out = dict(exp="I29" if a.exp == "a" else "I29b", seed=a.seed, n_ex=a.n_ex, votes=a.votes, runs=[])
    if a.exp == "b":
        for q in (0.1, "桁上り忘れ"):
            for noise in (0.0, 0.05, 0.1, 0.2):
                for v in (2.0, 3.0, 5.0):
                    out["runs"].append(run(a.seed, q, "判断 + 入れ替え", a.n_ex, v, noise))
                out["runs"].append(run(a.seed, q, "判断 + 入れ替え + 確かめ", a.n_ex, 2.0, noise))
    for q in ((0.05, 0.1, 0.2, 0.4, "桁上り忘れ") if a.exp == "a" else ()):
        for method in ("なし", "判断の板", "判断 + 入れ替え"):
            out["runs"].append(run(a.seed, q, method, a.n_ex, a.votes))
    out["total_s"] = time.perf_counter() - t0
    print(json.dumps(out, ensure_ascii=False), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
