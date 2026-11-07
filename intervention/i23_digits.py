"""介入の研究 実験I23: 数字が続いたら桁を判断する — 数字 1 文字ずつのアレイから、十の位・一の位の板にまとめることを答えだけから学ぶ

持ち主（2026-11-06）: 「桁上りも学習できるといいんだけどな、桁フラグを先読みするか、数字が続いた場合桁を判断できるとか」「先に算数を収めとく？」
この実験は後者（数字が続いた場合に桁を判断する）。
  - 式を数字 1 文字ずつの板（字の板）で渡す: 「4 7 ＋ 3 5」。演算は数の板（十の位・一の位を持つ一般常識の板）どうしでしか実行できない。
  - 文法解析グレイン（命令デコーダ、I20 の判断の板）の命令を 4 つに: 縮める（数 演算 数）、外す（（ 数 ））、
    **まとめる**（字 字 → 2 桁の数の板。前の字が十の位、後の字が一の位）、**読む**（字 → 1 桁の数の板）。
    字の所では「まとめる」と「読む」のどちらも出せる。どちらを出すかを、前後 2 つの板の種類と命令の名前から決める（経験の器、I18 の組）。
    まちがえると: 「4 7」を 4 と 7 に別々に読むと、数 数 が並び、7 ＋ 3 を先に縮めてしまうなどして答えが合わない。
  - 計算は I22 の 0〜9 の板 + 桁上りの照射（表は全部教えた）。新しい数の板 = 一般常識の板の明るさ（+ 縮めた板はネスト）。
  - 学び: 答えだけ（I20 と同じ判断の板: 違ったら一手ずつ替えて解き直す）。教え: 数 2〜3 個の式 300 + 数 4 個 200。試し: 数 2〜5 個 各 150。
    数は 0〜99（2 桁の数が半分ほど）、途中の値も 0〜99、× は片方が 1 桁。
物差し: 正解率、判断の板が見つけた分かれ目の種類（まとめる / 読む / 縮める）、復元。
結果: ../Project_Kotoyui_private/intervention/results/i23.jsonl
実行: python intervention/i23_digits.py --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i17_arithmetic as M  # noqa: E402
import i19_grain_arith as G  # noqa: E402
from i18_combo import ComboExp  # noqa: E402
from i21_board_alu import OpBands  # noqa: E402
from i22_carry import CarryALU, CarryArr, digit_fact, D10  # noqa: E402


class DigitArr(CarryArr):
    def __init__(self, toks):
        super().__init__([])
        for t in toks:
            if isinstance(t, tuple):                                  # ("字", d)
                self.seq.append(self.new(dict(kind="digit", val=t[1])))
            elif isinstance(t, int):
                self.seq.append(self.new(dict(kind="num", val=t)))
            else:
                self.seq.append(self.new(dict(kind="op", op=t) if t in G.OPS else dict(kind=t)))

    def kind(self, i):
        if 0 <= i < len(self.seq) and self.boards[self.seq[i]]["kind"] == "digit":
            return "字"
        return super().kind(i)

    def moves(self):
        out = []
        for i in range(len(self.seq)):
            k = self.kind(i)
            if k in G.OPS:
                out.append((i, "縮める"))
            elif k == "(":
                out.append((i, "外す"))
            elif k == "字":
                out.append((i, "まとめる"))
                out.append((i, "読む"))
        return out

    def legal_move(self, m):
        i, ins = m
        if ins == "まとめる":
            return self.kind(i) == "字" and self.kind(i + 1) == "字"
        if ins == "読む":
            return self.kind(i) == "字"
        return self.legal(i)

    def do(self, m):
        i, ins = m
        if ins in ("縮める", "外す"):
            return self.apply(i)
        if ins == "まとめる":
            v = self.boards[self.seq[i]]["val"] * 10 + self.boards[self.seq[i + 1]]["val"]
            rest = self.seq[i + 2:]
        else:
            v = self.boards[self.seq[i]]["val"]
            rest = self.seq[i + 1:]
        light = self.general[v]
        rv = M.top(light, "十の位") * 10 + M.top(light, "一の位")      # 一般常識の板の明るさから読む
        nid = self.new(dict(kind="num", val=rv, light=light))
        self.seq = self.seq[:i] + [nid] + rest

    def clone(self):
        c = DigitArr([])
        c.boards, c.seq = self.boards[:], self.seq[:]
        return c


def feats(arr, m, k=2):
    i, ins = m
    w = [f"{d}:{arr.kind(i + d)}" for d in range(-k, k + 1)]
    return w + [f"操作:{ins}"]


LEARN_REFUSAL = False                                               # 学びの間だけ True: 実行できない手を断られたら覚える
ORDER = "操作ごと"                                                   # 命令ごとに器を分ける（I18 の操作ごと）。"組" は I20 と同じ器


class Decoder:
    def __init__(self):
        self.e = ComboExp(ORDER)

    def add(self, arr, m, lab):
        self.e.add(feats(arr, m), "命令", lab)

    def score(self, arr, m):
        s = self.e.scores(feats(arr, m), "命令")
        if not s:
            return 0.0
        return s.get("縮める", -50.0) - s.get("待つ", -50.0)

    def choose(self, arr):
        ms = arr.moves()
        if not ms:
            return None
        sc = [self.score(arr, m) for m in ms]
        return ms[int(np.argmax(sc))]                                 # 同じなら左（先に並ぶ手）


def solve(dec, toks, forced=None, max_steps=60):
    arr = DigitArr(toks)
    path = []
    for step in range(max_steps):
        if len(arr.seq) == 1 and arr.kind(0) == "数":
            break
        m = forced[step] if forced and step in forced else dec.choose(arr)
        tries = 0
        while LEARN_REFUSAL and m is not None and not arr.legal_move(m) and tries < 20:
            dec.add(arr, m, "待つ")                                   # 実行できない手は、その場で断られる（機械からの返事）ので「待つ」と覚える
            m, tries = dec.choose(arr), tries + 1
        if m is None or not arr.legal_move(m):
            return None, path + [(arr.clone(), m)]
        path.append((arr.clone(), m))
        try:
            arr.do(m)
        except ValueError:
            return "わからない", path
    if len(arr.seq) == 1 and arr.kind(0) == "数":
        return arr.boards[arr.seq[0]]["val"], path
    return None, path


def learn_one(dec, toks, v, st, log):
    ans, path = solve(dec, toks)
    st["solves"] += 1
    if ans == v:
        for arr, m in path:
            dec.add(arr, m, "縮める")
        return
    for s, (arr, m) in enumerate(path):
        for m2 in arr.moves():
            if m2 == m or not arr.legal_move(m2):
                continue
            a2, p2 = solve(dec, toks, forced={s: m2})
            st["solves"] += 1
            if a2 == v:
                dec.add(arr, m, "待つ")
                dec.add(arr, m2, "縮める")
                for arr2, mm in p2:
                    dec.add(arr2, mm, "縮める")
                st["found"] += 1
                log[f"{m[1] if m else '止まる'} → {m2[1]}"] += 1
                return
    st["given_up"] += 1


def digit_toks(toks):
    out = []
    for t in toks:
        if isinstance(t, int):
            out += [("字", int(c)) for c in str(t)]
        else:
            out.append(t)
    return out


def make(rng, n_lo, n_hi, count):
    out, seen, tries = [], set(), 0
    while len(out) < count and tries < 300000:
        tries += 1
        n = rng.randint(n_lo, n_hi)

        def tree(n):
            if n == 1:
                return rng.randint(0, 9) if rng.random() < 0.5 else rng.randint(10, 99)
            k = rng.randint(1, n - 1)
            return (rng.choice(G.OPS), tree(k), tree(n - k))
        t = tree(n)

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
        toks = digit_toks(G.render(t, rng, 0.2))
        if tuple(toks) in seen:
            continue
        seen.add(tuple(toks))
        out.append((t, toks, G.value(t)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--order", default="操作ごと")
    ap.add_argument("--n_long", type=int, default=200)
    ap.add_argument("--out", default="../Project_Kotoyui_private/intervention/results/i23.jsonl")
    a = ap.parse_args()
    t0 = time.perf_counter()
    global ORDER
    ORDER = a.order
    general = M.learn_boards(list(range(100)), 10, a.seed)
    ob = OpBands()
    for op in G.OPS:
        for x in D10:
            for y in D10:
                ob.teach(x, op, y, digit_fact(x, op, y))
    ob.notice()
    DigitArr.alu, DigitArr.general = CarryALU(ob), general
    rng = random.Random(a.seed)
    train = make(rng, 2, 3, 300) + make(rng, 4, 4, a.n_long)
    tests = {n: make(random.Random(a.seed * 100 + n), n, n, 150) for n in (2, 3, 4, 5)}
    dec = Decoder()
    st, log = dict(solves=0, found=0, given_up=0), Counter()
    curve = {}
    global LEARN_REFUSAL
    for m, (t, toks, v) in enumerate(train, 1):
        LEARN_REFUSAL = True
        learn_one(dec, toks, v, st, log)
        LEARN_REFUSAL = False
        if m in (30, 100, 300, 500, 1000, 1500):
            acc = {}
            for n, ex in tests.items():
                acc[n] = float(np.mean([solve(dec, toks)[0] == v for t, toks, v in ex]))
            curve[m] = dict(acc=acc, **st)
    rest = {}
    for n, ex in tests.items():
        ok = 0
        for t, toks, v in ex:
            ans, path = solve(dec, toks)
            if ans != v:
                continue
            arr = path[-1][0].clone()
            arr.do(path[-1][1])
            ok += G.restore(arr, arr.seq[0], 0.0, np.random.default_rng(0), len(arr.boards)) == t
        rest[n] = ok / len(ex)
    rec = dict(exp="I23", seed=a.seed, n_long=a.n_long, order=a.order, curve=curve, divergences=dict(log), restore=rest,
               two_digit_share=float(np.mean([sum(1 for x in toks if isinstance(x, tuple)) for t, toks, v in train])),
               total_s=time.perf_counter() - t0)
    print(json.dumps(rec, ensure_ascii=False), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
