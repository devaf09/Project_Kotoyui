"""介入の研究 実験I19: グレインによる算術 — 数式アレイを、文法解析グレイン（命令デコーダ）が縮めていき、答えから式を戻す

持ち主の構想（2026-11-06、構想メモ「算術・文章圧縮・全体把握」）:
  算板（数・桁・符号 / ＋ − × / ＝ （ ））、数式アレイ（式を算板で並べたもの）、一時保存グレイン（バッファ板、縮めるたびに作る）、
  ネストグレイン（バッファ板の元を参照周波数で持つ）、文法解析グレイン（CPU の制御装置・命令デコーダに近い。前後の板から「ここを縮める」を指示する）、
  カウンタ 3 つ（アレイの位置・バッファ板の指定・ネストの昇り降り）、総合板 z（全バッファ板の和）。
  例: a + (b + c) → a + x → y。y からネストとバッファ板で a + (b + c) に戻せることが大事。最初はデジタルの重みで良い。
この実験（1 回目）:
  - 世界: 0〜9 の数 2〜6 個と ＋ − × の式。括弧は要るところに付け、ときどき余計な括弧も付ける（2 割）。答えは −99〜99 に収まるものだけ。
  - 算板: 数の板 = {種類:数, 符号, 十の位, 一の位}、演算の板 = {種類:演算, 演算:＋/−/×}、括弧の板。バッファ板も数の板と同じ帯（値の帯）に、
    参照の帯（左の子・演算・右の子の参照周波数。周波数 = 板の番号）を足したもの。
  - 計算（演算器）: 2 つの数の板と演算から値を出すところは、手で与えた（正確な計算）。I17・I18 で、+1〜+3 と繰り上がりは経験で見つけられると分かっているので、
    ここでは「どこから縮めるか」（文法）だけを学ぶ。
  - 文法解析グレイン = 命令デコーダ: アレイの位置ごとに、前後 2 つずつの板の種類（数・＋・−・×・（・）・端）を見て、「縮める」か「待つ」かを出す。
    命令は 2 種類: 演算を縮める（数 演算 数 → バッファ板）、括弧を外す（（ 数 ） → 数）。
    デコーダの中身は 2 通り（どちらもデジタルの重み）:
      経験の器（I18 の組。前後の板の種類の組を数え上げた明るさ）／ロジスティック回帰（前後の板の種類と、その組の one-hot）。
  - 動かし方 2 通り:
      一括: すべての位置を一度に照らし、いちばん「縮める」の明るい所を縮める（光の一括処理）。
      走査: カウンタ 1（アレイの位置）を左から 1 つずつ進め、デコーダが「縮める」と言った所で縮め、カウンタを 2 つ戻す（前後に動く）。端まで行ったら、いちばん明るい所。
  - 学び 2 通り:
      手順つき: 正しい順番（括弧の中、× を先、同じ強さなら左から）を教える。
      答えだけ: 短い式（数 2〜3 個）で、縮め方を全部試し、最後の答えが合う縮め方がある手を「縮める」、無い手を「待つ」とする（I9 と同じ「答えが合ったか」）。
  - 教えは数 2〜3 個の式、試しは数 2〜6 個（教えより長い式に広がるか）。
  - 復元: 最後のバッファ板 y から、参照の帯をたどって式の木を戻し、もとの木と同じかを見る。参照の帯は光の明るさとして持ち、融合のたびの劣化（雑音 σ）を入れる。
    比べ: 総合板 z（全バッファ板の明るさの和）だけから戻す（参照の帯の和から、演算の数と答えが読めるか）。
物差し: 最後の答えの正解率、一手ごとに「値が保たれたか」（縮めた後のアレイを正しい順番で計算して答えと同じか）、復元率。
結果: ../Project_Kotoyui_private/intervention/results/i19.jsonl
実行: python intervention/i19_grain_arith.py --seed 0
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from i18_combo import ComboExp  # noqa: E402

OPS = ["+", "-", "*"]
PREC = {"+": 1, "-": 1, "*": 2}
FN = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b}


# ---------------------------------------------------------------- 式の世界
def rand_tree(n, rng):
    if n == 1:
        return rng.randint(0, 9)
    k = rng.randint(1, n - 1)
    return (rng.choice(OPS), rand_tree(k, rng), rand_tree(n - k, rng))


def value(t):
    return t if isinstance(t, int) else FN[t[0]](value(t[1]), value(t[2]))


def render(t, rng=None, extra=0.0):
    """木 → トークン列。括弧は要るところ（優先順位・左結合）に付け、rng があれば extra の割合で余計な括弧も付ける。"""
    if isinstance(t, int):
        return [t]
    op, l, r = t
    L, R = render(l, rng, extra), render(r, rng, extra)
    if not isinstance(l, int) and (PREC[l[0]] < PREC[op] or (rng and rng.random() < extra)):
        L = ["("] + L + [")"]
    if not isinstance(r, int) and (PREC[r[0]] <= PREC[op] or (rng and rng.random() < extra)):   # 右は同じ強さでも括弧（木がそのまま戻るように）
        R = ["("] + R + [")"]
    return L + [op] + R


def make_exprs(rng, n_lo, n_hi, count):
    out, seen = [], set()
    while len(out) < count:
        t = rand_tree(rng.randint(n_lo, n_hi), rng)
        v = value(t)
        if abs(v) > 99:
            continue
        toks = render(t, rng, 0.2)
        key = tuple(toks)
        if key in seen:
            continue
        seen.add(key)
        out.append((t, toks, v))
    return out


def parse_value(toks):
    """正しい順番でアレイを計算（物差し用。数は int、バッファ板も int として入れる）。"""
    pos = 0

    def expr():
        nonlocal pos
        v = term()
        while pos < len(toks) and toks[pos] in ("+", "-"):
            op = toks[pos]
            pos += 1
            v = FN[op](v, term())
        return v

    def term():
        nonlocal pos
        v = atom()
        while pos < len(toks) and toks[pos] == "*":
            pos += 1
            v = v * atom()
        return v

    def atom():
        nonlocal pos
        t = toks[pos]
        pos += 1
        if t == "(":
            v = expr()
            pos += 1
            return v
        return t

    try:
        v = expr()
        return v if pos == len(toks) else None
    except (IndexError, TypeError):
        return None


# ---------------------------------------------------------------- アレイ（板の並び）と命令
class Arr:
    """数式アレイ。要素は板の番号。板の中身は boards[id] = {"kind", "val"/"op", "ref"}。"""

    def __init__(self, toks):
        self.boards, self.seq = [], []
        for t in toks:
            self.seq.append(self.new(dict(kind="num", val=t) if isinstance(t, int) else dict(kind="op", op=t) if t in OPS else dict(kind=t)))

    def new(self, b):
        self.boards.append(b)
        return len(self.boards) - 1

    def kind(self, i):
        if i < 0 or i >= len(self.seq):
            return "端"
        b = self.boards[self.seq[i]]
        return "数" if b["kind"] == "num" else b.get("op", b["kind"])

    def candidates(self):
        """命令を出せる位置: 演算の板（縮める）と左括弧（外す）。中身が整っているかはデコーダが判断する。"""
        return [i for i in range(len(self.seq)) if self.kind(i) in OPS or self.kind(i) == "("]

    def legal(self, i):
        k = self.kind(i)
        if k in OPS:
            return self.kind(i - 1) == "数" and self.kind(i + 1) == "数"
        return k == "(" and self.kind(i + 1) == "数" and self.kind(i + 2) == ")"

    def apply(self, i):
        """命令を実行。縮める: 数 演算 数 → バッファ板（値の帯 + 参照の帯）。外す: （ 数 ） → 数。"""
        if self.kind(i) == "(":
            self.seq = self.seq[:i] + [self.seq[i + 1]] + self.seq[i + 3:]
            return
        a, o, b = self.seq[i - 1], self.seq[i], self.seq[i + 1]
        op = self.boards[o]["op"]
        v = FN[op](self.boards[a]["val"], self.boards[b]["val"])      # 演算器（手で与えた）
        nid = self.new(dict(kind="num", val=v, ref=(a, o, b)))
        self.seq = self.seq[:i - 1] + [nid] + self.seq[i + 2:]

    def toks(self):
        out = []
        for i in range(len(self.seq)):
            b = self.boards[self.seq[i]]
            out.append(b["val"] if b["kind"] == "num" else b.get("op", b["kind"]))
        return out

    def clone(self):
        c = Arr([])
        c.boards, c.seq = self.boards[:], self.seq[:]
        return c


def window(arr, i):
    return [f"{d}:{arr.kind(i + d)}" for d in (-2, -1, 0, 1, 2)]


def teacher_choice(arr):
    """正しい順番: いちばん内側の括弧の組（無ければ全体）の中で、（ 数 ） なら外す、× を先、同じ強さなら左から。"""
    k = [arr.kind(i) for i in range(len(arr.seq))]
    # いちばん深い括弧の範囲を探す
    depth, maxd, span = 0, 0, (0, len(k))
    stack = []
    for i, t in enumerate(k):
        if t == "(":
            stack.append(i)
            if len(stack) > maxd:
                maxd = len(stack)
                span = None
        elif t == ")":
            s = stack.pop()
            if len(stack) + 1 == maxd and span is None:
                span = (s + 1, i)
    if maxd == 0:
        span = (0, len(k))
    a, b = span
    if maxd > 0 and b - a == 1:
        return a - 1                                                  # （ 数 ） を外す
    for ops in (("*",), ("+", "-")):
        for i in range(a, b):
            if k[i] in ops and arr.legal(i):
                return i
    return None


def exists_correct(arr, target, memo):
    """答えだけの学び用: この状態から、合法な手で最後の答えが target になる縮め方があるか（全部試す）。"""
    key = tuple(arr.toks())
    if key in memo:
        return memo[key]
    if len(arr.seq) == 1:
        r = arr.kind(0) == "数" and arr.boards[arr.seq[0]]["val"] == target
    else:
        r = False
        for i in arr.candidates():
            if arr.legal(i):
                c = arr.clone()
                c.apply(i)
                if exists_correct(c, target, memo):
                    r = True
                    break
    memo[key] = r
    return r


# ---------------------------------------------------------------- 命令デコーダ（文法解析グレイン）
class ExpDecoder:
    def __init__(self):
        self.e = ComboExp("組")

    def fit(self, X, y):
        for f, lab in zip(X, y):
            self.e.add(f, "命令", lab)

    def score(self, f):
        s = self.e.scores(f, "命令")
        if "縮める" not in s:
            return -1e9
        return s["縮める"] - s.get("待つ", -1e9)


class LRDecoder:
    def __init__(self, seed):
        self.seed = seed

    def feats(self, f):
        return f + [a + "&" + b for a, b in itertools.combinations(f, 2)]

    def fit(self, X, y, steps=300, lr=0.5, l2=1e-3):
        vocab = sorted({t for f in X for t in self.feats(f)})
        self.idx = {t: i for i, t in enumerate(vocab)}
        A = np.zeros((len(X), len(vocab)))
        for r, f in enumerate(X):
            for t in self.feats(f):
                A[r, self.idx[t]] = 1
        t = np.array([1.0 if lab == "縮める" else 0.0 for lab in y])
        self.w, self.b = np.zeros(len(vocab)), 0.0
        for _ in range(steps):
            p = 1 / (1 + np.exp(-(A @ self.w + self.b)))
            g = p - t
            self.w -= lr * (A.T @ g / len(X) + l2 * self.w)
            self.b -= lr * g.mean()

    def score(self, f):
        return self.b + sum(self.w[self.idx[t]] for t in self.feats(f) if t in self.idx)


def train_data(exprs, how):
    X, y = [], []
    for t, toks, v in exprs:
        arr = Arr(toks)
        memo = {}
        while len(arr.seq) > 1:
            cand = arr.candidates()
            good = teacher_choice(arr) if how == "手順つき" else None
            labs = {}
            for i in cand:
                if how == "手順つき":
                    labs[i] = i == good
                else:
                    ok = arr.legal(i)
                    if ok:
                        c = arr.clone()
                        c.apply(i)
                        ok = exists_correct(c, v, memo)
                    labs[i] = ok
            for i in cand:
                X.append(window(arr, i))
                y.append("縮める" if labs[i] else "待つ")
            nxt = good if how == "手順つき" else next((i for i in cand if labs[i]), None)
            if nxt is None:
                break
            arr.apply(nxt)
    return X, y


def run_expr(dec, toks, mode, max_steps=40):
    """デコーダで式を縮める。返り値: (アレイ, 一手ごとに値が保たれたか, 止まったか)。"""
    arr = Arr(toks)
    v0 = parse_value(toks)
    kept = []
    for _ in range(max_steps):
        if len(arr.seq) == 1:
            break
        cand = arr.candidates()
        if not cand:
            break
        sc = {i: dec.score(window(arr, i)) for i in cand}
        if mode == "一括":
            i = max(cand, key=sc.get)
        else:                                                         # 走査: カウンタを左から進め、縮めると言った最初の所。無ければいちばん明るい所
            i = next((j for j in cand if sc[j] > 0), max(cand, key=sc.get))
        if not arr.legal(i):
            return arr, kept, True                                    # 縮められない所を指した = 止まる
        arr.apply(i)
        kept.append(parse_value(arr.toks()) == v0)
    return arr, kept, False


# ---------------------------------------------------------------- 復元（ネストの逆引き）と総合板
def restore(arr, root, sigma, rng, n_ids):
    """参照の帯を光の明るさ（板の番号の周波数に 1、ほかは 0）として持ち、雑音 σ を足してから、いちばん明るい周波数を読む。"""
    def dec(bid, depth=0):
        b = arr.boards[bid]
        if "ref" not in b:
            return b.get("val")                                       # 数でない板（演算・括弧）を指したら None（戻せない）
        if depth > 20:
            return None
        out = []
        for child in b["ref"]:
            vec = np.zeros(n_ids)
            vec[child] = 1.0
            vec += rng.normal(0, sigma, n_ids)
            out.append(int(np.argmax(vec)))
        a, o, c = out
        if arr.boards[o].get("kind") != "op" or a == bid or c == bid:
            return None
        return (arr.boards[o]["op"], dec(a, depth + 1), dec(c, depth + 1))
    return dec(root)


def total_board(arr):
    """総合板 z = すべてのバッファ板の明るさの和: 値の帯（符号・十・一）と、参照の帯のうち演算の部分（＋ − × の数）。"""
    ops = Counter()
    vals = np.zeros(21)
    for b in arr.boards:
        if "ref" in b:
            ops[arr.boards[b["ref"][1]]["op"]] += 1
            v = b["val"]
            vals[0 if v >= 0 else 1] += 1
            vals[2 + (abs(v) // 10) % 10] += 1
            vals[11 + abs(v) % 10] += 1
    return ops, vals


def count_ops(t):
    if isinstance(t, int):
        return Counter()
    c = Counter([t[0]])
    c.update(count_ops(t[1]))
    c.update(count_ops(t[2]))
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_train", type=int, default=300)
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--out", default="../Project_Kotoyui_private/intervention/results/i19.jsonl")
    a = ap.parse_args()
    t0 = time.perf_counter()
    rng = random.Random(a.seed)
    train = make_exprs(rng, 2, 3, a.n_train)
    tests = {n: make_exprs(random.Random(a.seed * 100 + n), n, n, a.n_test) for n in range(2, 7)}
    rec = dict(exp="I19", seed=a.seed, n_train=a.n_train, n_test=a.n_test, res={})
    for how in ("手順つき", "答えだけ"):
        X, y = train_data(train, how)
        for dname in ("経験の器", "ロジスティック回帰"):
            dec = ExpDecoder() if dname == "経験の器" else LRDecoder(a.seed)
            dec.fit(X, y)
            for mode in ("一括", "走査"):
                r = {}
                for n, ex in tests.items():
                    ok = kept = steps = stuck = 0
                    for t, toks, v in ex:
                        arr, kp, st = run_expr(dec, toks, mode)
                        fin = len(arr.seq) == 1 and arr.kind(0) == "数" and arr.boards[arr.seq[0]]["val"] == v
                        ok += fin
                        kept += sum(kp)
                        steps += len(kp)
                        stuck += st
                    r[n] = dict(acc=ok / len(ex), step_kept=kept / max(steps, 1), stuck=stuck / len(ex))
                rec["res"][f"{how}|{dname}|{mode}"] = r
        rec[f"n_train_rows|{how}"] = len(X)
    # 復元: 手順どおりに縮めた最後のバッファ板 y から、ネストの逆引きで木を戻す
    rr = random.Random(a.seed + 5)
    nrng = np.random.default_rng(a.seed)
    rest = {}
    for sigma in (0.0, 0.2, 0.35, 0.5):
        for n, ex in tests.items():
            ok = zok = zans = 0
            for t, toks, v in ex:
                arr = Arr(toks)
                while len(arr.seq) > 1:
                    arr.apply(teacher_choice(arr))
                root = arr.seq[0]
                back = restore(arr, root, sigma, nrng, len(arr.boards))
                ok += back == t
                ops, vals = total_board(arr)
                zok += ops == count_ops(t)                             # z から演算の数は読める
                # z から答えを読む: 値の帯の和で、いちばん明るい符号・十・一（全バッファの和なので、最後の板だけを取り出せない）
                zv = (int(np.argmax(vals[2:11])) * 10 + int(np.argmax(vals[11:21]))) * (1 if vals[0] >= vals[1] else -1)
                zans += zv == v
            rest[f"σ={sigma}|{n}"] = dict(restore=ok / len(ex), z_ops=zok / len(ex), z_answer=zans / len(ex))
    rec["restore"] = rest
    rec["total_s"] = time.perf_counter() - t0
    print(json.dumps(rec, ensure_ascii=False), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
