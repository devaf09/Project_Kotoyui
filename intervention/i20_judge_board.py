"""介入の研究 実験I20: 判断の板 — 総当たりの代わりに、答えが違ったとき「どの一手で道が分かれたか」を差分で見つけて覚える

I19 の「答えだけ」の学びは、短い式で縮め方を全部試していた（総当たり）。持ち主（2026-11-06）:
  「気付きとか判断の板がないと、介入で総当たりする何かになっちゃう」。
この実験:
  - 文法解析グレイン（命令デコーダ、I19 と同じ。経験の器 = I18 の組）を空の状態から始め、式を 1 つずつ解く（一括: いちばん明るい所を縮める。
    明るさが同じなら左の所）。最後の答えだけが返ってくる。
  - 答えが合ったら: その道の一手ずつを「縮める」の経験として足す。
  - **答えが違ったら（判断の板）**: 自分の道の一手ずつについて、前から順に「この一手だけを別の手に替え、あとはいまのデコーダのまま解き直す」を試す（介入）。
    答えが合った所が「道が分かれた一手」。そこで選んだ手を「待つ」、替えた手を「縮める」として、注意つきの経験に足す。解き直しの回数を数える。
    見つからなければ（2 手以上まちがえた）、その式は見送る。
  - 比べ: I19 の総当たり（短い式で縮め方を全部試す）。試した状態の数を数える。
  - 窓の大きさ（前後 1・2・3）を変える（I19 で手で決めた 3 番目のもの）。
  - 教え: 数 2〜3 個の式 300 個のあと、数 4〜5 個の式 300 個（途中 10・30・100・300・400・600 個で測る）。
  - 注意の重み: 道が分かれた一手の経験を 1 回ぶん / 3 回ぶん足す。試し: 数 2〜6 個、各 200 個（I19 と同じ作り方）。
手で決めたもの（I19 と同じ）: 命令の種類と形・命令を出せる位置・演算器・一括のきまり。判断の板で替える手は「実行できる手」（legal）だけ。
結果: ../Project_Kotoyui_private/intervention/results/i20.jsonl
実行: python intervention/i20_judge_board.py --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i19_grain_arith as G  # noqa: E402
from i18_combo import ComboExp  # noqa: E402


def window(arr, i, k):
    return [f"{d}:{arr.kind(i + d)}" for d in range(-k, k + 1)]


class Decoder:
    def __init__(self, k):
        self.k, self.e = k, ComboExp("組")

    def add(self, arr, i, lab):
        self.e.add(window(arr, i, self.k), "命令", lab)

    def score(self, arr, i):
        s = self.e.scores(window(arr, i, self.k), "命令")
        if not s:
            return 0.0
        return s.get("縮める", -50.0) - s.get("待つ", -50.0)

    def choose(self, arr):
        cand = arr.candidates()
        if not cand:
            return None
        sc = {i: self.score(arr, i) for i in cand}
        best = max(sc.values())
        return next(i for i in cand if sc[i] == best)               # 同じ明るさなら左


def solve(dec, toks, forced=None, max_steps=40):
    """デコーダで解く。forced = {手の番号: 位置} はその一手だけ替える（介入）。返り値: (答え or None, 道 [(状態, 位置)])。"""
    arr = G.Arr(toks)
    path = []
    for step in range(max_steps):
        if len(arr.seq) == 1:
            break
        i = forced[step] if forced and step in forced else dec.choose(arr)
        if i is None or not arr.legal(i):
            return None, path + [(arr.clone(), i)]
        path.append((arr.clone(), i))
        arr.apply(i)
    if len(arr.seq) == 1 and arr.kind(0) == "数":
        return arr.boards[arr.seq[0]]["val"], path
    return None, path


ATT = 3                                                             # 注意の重み: 道が分かれた一手の経験を何回ぶん足すか


def learn_one(dec, toks, v, stats):
    ans, path = solve(dec, toks)
    stats["solves"] += 1
    if ans == v:
        for arr, i in path:
            dec.add(arr, i, "縮める")
        return True
    for s, (arr, i) in enumerate(path):                             # 判断の板: 前から一手ずつ替えて解き直す
        for j in arr.candidates():
            if j == i or not arr.legal(j):
                continue
            a2, p2 = solve(dec, toks, forced={s: j})
            stats["solves"] += 1
            if a2 == v:
                for _ in range(ATT):
                    dec.add(arr, i, "待つ")                         # 道が分かれた一手（注意）
                    dec.add(arr, j, "縮める")
                for arr2, i2 in p2:
                    dec.add(arr2, i2, "縮める")
                stats["found"] += 1
                return True
    stats["given_up"] += 1
    return False


def evaluate(dec, tests):
    out = {}
    for n, ex in tests.items():
        ok = 0
        for t, toks, v in ex:
            a, _ = solve(dec, toks)
            ok += a == v
        out[n] = ok / len(ex)
    return out


def exhaustive_cost(train):
    states = 0
    for t, toks, v in train:
        memo = {}
        arr = G.Arr(toks)
        G.exists_correct(arr, v, memo)
        states += len(memo)
    return states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--out", default="../Project_Kotoyui_private/intervention/results/i20.jsonl")
    a = ap.parse_args()
    t0 = time.perf_counter()
    rng = random.Random(a.seed)
    train = G.make_exprs(rng, 2, 3, 300) + G.make_exprs(rng, 4, 5, 300)     # 短い式 300 のあと、長い式 300
    tests = {n: G.make_exprs(random.Random(a.seed * 100 + n), n, n, a.n_test) for n in range(2, 7)}
    rec = dict(exp="I20", seed=a.seed, res={})
    global ATT
    for k, att in ((1, 3), (2, 1), (2, 3), (3, 3)):
        ATT = att
        dec = Decoder(k)
        stats = dict(solves=0, found=0, given_up=0)
        curve = {}
        for m, (t, toks, v) in enumerate(train, 1):
            learn_one(dec, toks, v, stats)
            if m in (10, 30, 100, 300, 400, 600):
                curve[m] = dict(acc=evaluate(dec, tests), **stats)
        rec["res"][f"窓 {k}|注意 {att}"] = curve
    # I19 の総当たりの手間（同じ 300 式）
    rec["exhaustive_states"] = dict(short=exhaustive_cost(train[:300]), long=exhaustive_cost(train[300:]))
    rec["total_s"] = time.perf_counter() - t0
    print(json.dumps(rec, ensure_ascii=False), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
