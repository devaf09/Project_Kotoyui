"""介入の研究 実験I21: 数の板が計算する — 4 の板に「3 と ＋」が来たら 7 が光る。新しいバッファ板 = 一般常識の 7 の板 + ネスト（3・＋・4）

持ち主（2026-11-06）: 「数字の算板には、4 の板には、3 と ＋ に入力があったら 7 が光って、新しいバッファ板 7 は一般常識からの 7 と、3 と 4 の板を記録するネストが入ってる？」
I19・I20 では、値の計算は手で書いた演算器だった。この実験で、演算器を数の板そのものに置き換える。
  - 一般常識の数の板: 0〜99 の語板（I17 と同じ光の器。属性 = 一の位・十の位・偶奇）、文から学ぶ。負の数は符号の帯を足す。
  - 演算の帯: 1 桁の数の板（−9〜9）ごとに、{役（左・右）, 演算, 相手} → 結果 の経験（数え上げの明るさ、デジタル）。
    文「3 に 4 を足すと 7」は、3 の板の（左, ＋, 4）と 4 の板の（右, ＋, 3）の両方に 7 を書く。
    教えるのは、すべての組（19 × 19 × 3 演算）の一部（割合 p）だけ。
  - 計算 = 2 枚の板を同時に照らし、演算の帯の明るさを足して（融合）、いちばん明るい結果。経験に無ければ「わからない」。
  - **気付き（差分）**: 演算ごとに、同じ板の（左, 演算, y）と（右, 演算, y）の結果を比べる（両方とも経験にあるものだけ）。
    ほぼ一致すれば「この演算は左右を入れ替えても同じ」と気付き、照らすときに入れ替えた組も使う（＋ と × は一致、− は一致しないはず）。
  - **逆回し**: − を 1 文も教えないとき、a − b を「z + b = a になる z」として ＋ の帯から探す（I17 の引き算と同じ）。
  - バッファ板 = 一般常識の |v| の板の明るさ + 符号 + 参照の帯（左の板・演算・右の板）。値はバッファ板の明るさから読む（一の位・十の位・符号）。
  - 式全体: I20 の文法解析グレイン（判断の板、窓 前後 2）を、学んだ計算だけで動かす。数 2〜4 個、途中の値がすべて −9〜9 に収まる式（1 桁の板で計算できるもの）。
    計算が「わからない」とき、その式は止まる（答えが返らない）。判断の板は、文法の誤りなら一手替えで直せるが、計算の誤りは直せない（見送り）。
物差し: 2 つの数の計算の正解率・「わからない」の割合（p・気付き・逆回し別）、バッファ板から読んだ値の正しさ、ネストからの復元、式全体の正解率。
結果: ../Project_Kotoyui_private/intervention/results/i21.jsonl
実行: python intervention/i21_board_alu.py --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i17_arithmetic as M  # noqa: E402
import i19_grain_arith as G  # noqa: E402
import i20_judge_board as J  # noqa: E402

DIG = list(range(-9, 10))


class OpBands:
    """1 桁の数の板ごとの演算の帯。band[x][(役, 演算, 相手)] = Counter(結果)。"""

    def __init__(self):
        self.band = defaultdict(lambda: defaultdict(Counter))
        self.swap = {}

    def teach(self, a, op, b, r, n=1):
        self.band[a][("左", op, b)][r] += n
        self.band[b][("右", op, a)][r] += n

    def notice(self, thr=0.9):
        """気付き: 同じ板の（左, op, y）と（右, op, y）の結果を比べる（差分）。"""
        out = {}
        for op in G.OPS:
            same = n = 0
            for x, d in self.band.items():
                for (role, o, y), c in d.items():
                    if role != "左" or o != op:
                        continue
                    c2 = d.get(("右", op, y))
                    if c2:
                        n += 1
                        same += c.most_common(1)[0][0] == c2.most_common(1)[0][0]
            out[op] = (same / n if n else None, n)
            self.swap[op] = bool(n) and same / n >= thr
        return out

    def light(self, a, op, b, use_swap):
        c = Counter()
        c.update(self.band[a].get(("左", op, b), {}))
        c.update(self.band[b].get(("右", op, a), {}))
        if use_swap and self.swap.get(op):
            c.update(self.band[a].get(("右", op, b), {}))
            c.update(self.band[b].get(("左", op, a), {}))
        return c.most_common(1)[0][0] if c else None

    def compute(self, a, op, b, use_swap=True, reverse=False):
        if a not in self.band and b not in self.band:
            return None
        r = self.light(a, op, b, use_swap)
        if r is None and reverse and op == "-":                    # 逆回し: z + b = a になる z
            for z in DIG:
                if self.light(z, "+", b, use_swap) == a:
                    return z
        return r


class BoardArr(G.Arr):
    """数式アレイ。縮めるとき、演算の帯で計算し、バッファ板 = 一般常識の板の明るさ + 符号 + 参照。"""
    ops_bands = None
    general = None
    use_swap = True
    reverse = False

    def apply(self, i):
        if self.kind(i) == "(":
            return super().apply(i)
        a, o, b = self.seq[i - 1], self.seq[i], self.seq[i + 1]
        va, vb = self.boards[a]["val"], self.boards[b]["val"]
        op = self.boards[o]["op"]
        v = self.ops_bands.compute(va, op, vb, self.use_swap, self.reverse) if va in DIG and vb in DIG else None
        if v is None:
            raise ValueError("わからない")
        light = self.general[abs(v)] if abs(v) < 100 else None
        rv = None
        if light is not None:                                        # 値はバッファ板の明るさから読む
            rv = (M.top(light, "十の位") * 10 + M.top(light, "一の位")) * (1 if v >= 0 else -1)
        nid = self.new(dict(kind="num", val=rv, sign=1 if v >= 0 else -1, light=light, ref=(a, o, b)))
        self.seq = self.seq[:i - 1] + [nid] + self.seq[i + 2:]

    def clone(self):
        c = BoardArr([])
        c.boards, c.seq = self.boards[:], self.seq[:]
        return c


def solve(dec, toks, forced=None, max_steps=40):
    arr = BoardArr(toks)
    path = []
    for step in range(max_steps):
        if len(arr.seq) == 1:
            break
        i = forced[step] if forced and step in forced else dec.choose(arr)
        if i is None or not arr.legal(i):
            return None, path + [(arr.clone(), i)]
        path.append((arr.clone(), i))
        try:
            arr.apply(i)
        except ValueError:
            return "わからない", path
    if len(arr.seq) == 1 and arr.kind(0) == "数":
        return arr.boards[arr.seq[0]]["val"], path
    return None, path


def learn_one(dec, toks, v, stats):
    ans, path = solve(dec, toks)
    stats["solves"] += 1
    if ans == v:
        for arr, i in path:
            dec.add(arr, i, "縮める")
        return
    for s, (arr, i) in enumerate(path):
        for j in arr.candidates():
            if j == i or not arr.legal(j):
                continue
            a2, p2 = solve(dec, toks, forced={s: j})
            stats["solves"] += 1
            if a2 == v:
                dec.add(arr, i, "待つ")
                dec.add(arr, j, "縮める")
                for arr2, i2 in p2:
                    dec.add(arr2, i2, "縮める")
                stats["found"] += 1
                return
    stats["given_up_unknown" if ans == "わからない" else "given_up"] += 1


def small_exprs(rng, n_lo, n_hi, count):
    """途中の値がすべて −9〜9 の式（正しい順番で縮めたとき）。"""
    out, seen = [], set()
    tries = 0
    while len(out) < count and tries < 200000:
        tries += 1
        t = G.rand_tree(rng.randint(n_lo, n_hi), rng)

        def ok(t):
            if isinstance(t, int):
                return True
            return ok(t[1]) and ok(t[2]) and -9 <= G.value(t) <= 9
        if not ok(t):
            continue
        toks = G.render(t, rng, 0.2)
        if tuple(toks) in seen:
            continue
        seen.add(tuple(toks))
        out.append((t, toks, G.value(t)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_sent", type=int, default=10)
    ap.add_argument("--out", default="../Project_Kotoyui_private/intervention/results/i21.jsonl")
    a = ap.parse_args()
    t0 = time.perf_counter()
    rng = random.Random(a.seed)
    general = M.learn_boards(list(range(100)), a.n_sent, a.seed)    # 一般常識の数の板（光の器）
    rec = dict(exp="I21", seed=a.seed, n_sent=a.n_sent,
               general_read=float(np.mean([M.top(general[x], "一の位") == x % 10 and M.top(general[x], "十の位") == x // 10 for x in range(100)])),
               pair={}, notice={}, expr={})
    triples = [(x, op, y) for op in G.OPS for x in DIG for y in DIG]
    for p in (0.3, 0.5, 0.7, 1.0):
        for no_minus in (False, True):
            ob = OpBands()
            r2 = random.Random(a.seed * 7 + int(p * 10))
            for x, op, y in triples:
                if no_minus and op == "-":
                    continue
                if r2.random() < p:
                    ob.teach(x, op, y, G.FN[op](x, y))
            note = ob.notice()
            rec["notice"][f"p={p}|{'− なし' if no_minus else '− あり'}"] = {op: v for op, v in note.items()}
            for cond, sw, rev in (("帯だけ", False, False), ("気付き", True, False), ("気付き+逆回し", True, True)):
                res = {}
                for op in G.OPS:
                    ok = unk = 0
                    for x, o, y in triples:
                        if o != op:
                            continue
                        r = ob.compute(x, op, y, sw, rev)
                        ok += r == G.FN[op](x, y)
                        unk += r is None
                    res[op] = dict(acc=ok / 361, unknown=unk / 361)
                rec["pair"][f"p={p}|{'− なし' if no_minus else '− あり'}|{cond}"] = res
            if not no_minus and p in (0.5, 1.0):                     # 式全体（I20 の判断の板で文法を学ぶ。計算は学んだ帯だけ）
                BoardArr.ops_bands, BoardArr.general, BoardArr.use_swap, BoardArr.reverse = ob, general, True, True
                er = random.Random(a.seed + 11)
                train = small_exprs(er, 2, 3, 300) + small_exprs(er, 4, 4, 200)
                tests = {n: small_exprs(random.Random(a.seed * 100 + n), n, n, 150) for n in (2, 3, 4, 5)}
                dec = J.Decoder(2)
                st = dict(solves=0, found=0, given_up=0, given_up_unknown=0)
                for t, toks, v in train:
                    learn_one(dec, toks, v, st)
                acc, unk, rest = {}, {}, {}
                for n, ex in tests.items():
                    ok = un = rs = 0
                    for t, toks, v in ex:
                        ans, _ = solve(dec, toks)
                        ok += ans == v
                        un += ans == "わからない"
                        arr = BoardArr(toks)                         # 復元: 正しい順番で縮めたバッファ板から
                        try:
                            while len(arr.seq) > 1:
                                arr.apply(G.teacher_choice(arr))
                            rs += G.restore(arr, arr.seq[0], 0.0, np.random.default_rng(0), len(arr.boards)) == t
                        except ValueError:
                            pass
                    acc[n], unk[n], rest[n] = ok / len(ex), un / len(ex), rs / len(ex)
                rec["expr"][f"p={p}"] = dict(acc=acc, unknown=unk, restore=rest, **st)
    rec["total_s"] = time.perf_counter() - t0
    print(json.dumps(rec, ensure_ascii=False), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
