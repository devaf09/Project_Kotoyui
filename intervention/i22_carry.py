"""介入の研究 実験I22: 算板に桁上りを足す — 0〜9 の板の表だけで、2 桁の足し算・引き算・（2 桁 × 1 桁）の掛け算

持ち主（2026-11-06）: 「算板の数に 0〜9 のあとに桁上りを追加してくれる？」「板にできる操作（照射・融合・差分）だけで計算できているなら、その判断もやらせられるか」。
  - 0〜9 の板の演算の帯の出口を {一の位の数, 桁上り（＋・×）/ 桁下がり（−）} にする。文「7 に 5 を足すと 2、桁上り 1」を、7 の板（左, ＋, 5）と 5 の板（右, ＋, 7）に書く。
    桁上りは 0〜9 の板で表す（＋ は 0 か 1、× は 0〜8、− の桁下がりは 0 か 1）。
  - **桁ごとの照射（条件分岐なし）**: 2 桁の数 A = (A十, A一)、B = (B十, B一) の足し算は
      1. A一 と B一 を照らす → (一, 桁上り c1)
      2. A十 と B十 を照らす → (t, 桁上り c2)
      3. t と c1 を ＋ で照らす → (十, 桁上り c3)        ← 「桁上りがあれば足す」ではなく、いつも桁上りの板（0 か 1）を足す照射
      4. c2 と c3 を ＋ で照らす → 百
    引き算も同じ形（桁下がりを − で引く照射。答えが負になる組は扱わない）。掛け算は 2 桁 × 1 桁（A一 × b → (一, c1)、A十 × b → (t, c2)、t ＋ c1 → (十, c3)、c2 ＋ c3 → 百）。
    桁の手順（1〜4）は手で与えた（文法解析グレインに学ばせるのは次の段）。
  - 教えるのは 0〜9 の組（10 × 10 × 3 演算）の割合 p。I21 と同じ気付き（左右の入れ替え）を使う。
  - 比べ: I21 の形（数そのものの表）で、同じ文の数（10 × 10 × 3 × p 文）を 0〜99 の組に配ったもの。
  - 式全体: I20 の判断の板（窓 前後 2）で文法を学ぶ。入力の数は 0〜9、途中の値は 0〜99 に収まる式（I21 は −9〜9 だけだった）。バッファ板 = 一般常識の板 + ネスト。
物差し: 2 つの数（0〜99）の計算の正解率と「わからない」、式全体の正解率と復元。
結果: ../Project_Kotoyui_private/intervention/results/i22.jsonl
実行: python intervention/i22_carry.py --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i17_arithmetic as M  # noqa: E402
import i19_grain_arith as G  # noqa: E402
import i20_judge_board as J  # noqa: E402
from i21_board_alu import OpBands  # noqa: E402

D10 = list(range(10))


def digit_fact(a, op, b):
    if op == "+":
        return ((a + b) % 10, (a + b) // 10)
    if op == "-":
        return ((a - b) % 10, 1 if a < b else 0)
    return ((a * b) % 10, (a * b) // 10)


class CarryALU:
    def __init__(self, bands):
        self.b = bands

    def L(self, a, op, b):
        r = self.b.compute(a, op, b, True, False)
        if r is None:
            raise ValueError("わからない")
        return r

    def compute(self, A, op, B):
        """桁ごとの照射。A, B は 0〜99。答えが負・3 桁を越える・2 桁 × 2 桁は扱わない（None）。"""
        try:
            if op == "*":
                if B >= 10 and A >= 10:
                    return None
                if B >= 10:
                    A, B = B, A                                        # 入れ替えの気付き（× は左右を入れ替えても同じ）があるときだけ
                    if not self.b.swap.get("*"):
                        return None
                d1, c1 = self.L(A % 10, "*", B)
                t, c2 = self.L(A // 10, "*", B)
                d10, c3 = self.L(t, "+", c1)
                h, _ = self.L(c2, "+", c3)
                return h * 100 + d10 * 10 + d1
            if op == "+":
                d1, c1 = self.L(A % 10, "+", B % 10)
                t, c2 = self.L(A // 10, "+", B // 10)
                d10, c3 = self.L(t, "+", c1)                          # いつも桁上りの板（0 か 1）を足す
                h, _ = self.L(c2, "+", c3)
                return h * 100 + d10 * 10 + d1
            d1, w1 = self.L(A % 10, "-", B % 10)
            t, w2 = self.L(A // 10, "-", B // 10)
            d10, w3 = self.L(t, "-", w1)                              # いつも桁下がりの板（0 か 1）を引く
            if w2 or w3:
                return None                                           # 答えが負（扱わない）
            return d10 * 10 + d1
        except ValueError:
            return "わからない"


class CarryArr(G.Arr):
    alu = None
    general = None

    def apply(self, i):
        if self.kind(i) == "(":
            return super().apply(i)
        a, o, b = self.seq[i - 1], self.seq[i], self.seq[i + 1]
        va, vb, op = self.boards[a]["val"], self.boards[b]["val"], self.boards[o]["op"]
        v = self.alu.compute(va, op, vb) if 0 <= va <= 99 and 0 <= vb <= 99 else None
        if v is None or v == "わからない" or not (0 <= v <= 99):
            raise ValueError("わからない")
        light = self.general[v]
        rv = M.top(light, "十の位") * 10 + M.top(light, "一の位")      # 値はバッファ板（一般常識の板の明るさ）から読む
        nid = self.new(dict(kind="num", val=rv, light=light, ref=(a, o, b)))
        self.seq = self.seq[:i - 1] + [nid] + self.seq[i + 2:]

    def clone(self):
        c = CarryArr([])
        c.boards, c.seq = self.boards[:], self.seq[:]
        return c


def solve(dec, toks, forced=None, max_steps=40):
    arr = CarryArr(toks)
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


def learn_one(dec, toks, v, st):
    ans, path = solve(dec, toks)
    st["solves"] += 1
    if ans == v:
        for arr, i in path:
            dec.add(arr, i, "縮める")
        return
    for s, (arr, i) in enumerate(path):
        for j in arr.candidates():
            if j == i or not arr.legal(j):
                continue
            a2, p2 = solve(dec, toks, forced={s: j})
            st["solves"] += 1
            if a2 == v:
                dec.add(arr, i, "待つ")
                dec.add(arr, j, "縮める")
                for arr2, i2 in p2:
                    dec.add(arr2, i2, "縮める")
                st["found"] += 1
                return
    st["given_up_unknown" if ans == "わからない" else "given_up"] += 1


def exprs_0_99(rng, n_lo, n_hi, count):
    """入力 0〜9、正しい順番で縮めたときの途中の値がすべて 0〜99、掛け算は片方が 1 桁。"""
    out, seen, tries = [], set(), 0
    while len(out) < count and tries < 300000:
        tries += 1
        t = G.rand_tree(rng.randint(n_lo, n_hi), rng)

        def ok(t):
            if isinstance(t, int):
                return True
            if not (ok(t[1]) and ok(t[2])):
                return False
            a, b = G.value(t[1]), G.value(t[2])
            if t[0] == "*" and a >= 10 and b >= 10:
                return False
            return 0 <= G.value(t) <= 99
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
    ap.add_argument("--out", default="../Project_Kotoyui_private/intervention/results/i22.jsonl")
    a = ap.parse_args()
    t0 = time.perf_counter()
    general = M.learn_boards(list(range(100)), a.n_sent, a.seed)
    rec = dict(exp="I22", seed=a.seed, pair={}, table={}, expr={}, notice={})
    pairs = {"+": [(x, y) for x in range(100) for y in range(100) if x + y <= 199],
             "-": [(x, y) for x in range(100) for y in range(100) if x >= y],
             "*": [(x, y) for x in range(100) for y in range(10)]}
    for p in (0.5, 0.7, 1.0):
        r2 = random.Random(a.seed * 7 + int(p * 10))
        ob = OpBands()
        n_facts = 0
        for op in G.OPS:
            for x in D10:
                for y in D10:
                    if r2.random() < p:
                        ob.teach(x, op, y, digit_fact(x, op, y))
                        n_facts += 1
        rec["notice"][f"p={p}"] = {op: v[0] for op, v in ob.notice().items()}
        alu = CarryALU(ob)
        res = {}
        for op, ps in pairs.items():
            ok = unk = 0
            for x, y in ps:
                r = alu.compute(x, op, y)
                ok += r == G.FN[op](x, y)
                unk += r == "わからない"
            res[op] = dict(acc=ok / len(ps), unknown=unk / len(ps), n=len(ps))
        rec["pair"][f"p={p}"] = res
        # 比べ: 同じ文の数を、数そのものの表（I21 の形、0〜99 の組）に配る
        tb = OpBands()
        r3 = random.Random(a.seed * 13 + int(p * 10))
        for op, ps in pairs.items():
            for x, y in r3.sample(ps, n_facts // 3):
                tb.teach(x, op, y, G.FN[op](x, y))
        tb.notice()
        res = {}
        for op, ps in pairs.items():
            ok = 0
            for x, y in ps:
                ok += tb.light(x, op, y, True) == G.FN[op](x, y)
            res[op] = ok / len(ps)
        rec["table"][f"p={p}"] = dict(acc=res, n_facts=n_facts)
        if p in (0.7, 1.0):
            CarryArr.alu, CarryArr.general = alu, general
            er = random.Random(a.seed + 11)
            train = exprs_0_99(er, 2, 3, 300) + exprs_0_99(er, 4, 4, 200)
            tests = {n: exprs_0_99(random.Random(a.seed * 100 + n), n, n, 150) for n in (2, 3, 4, 5)}
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
                    arr = CarryArr(toks)
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
