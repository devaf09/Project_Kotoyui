"""介入の研究 実験I28: 板の操作を光で指定する — 桁上りの照射の手順（I22 の ①〜④）を「つなぎの板」（光のクロスバー）にして、答えだけから学ぶ
持ち主（2026-11-06）: 「板の操作も光で指定できるようにならないかな？」→「うん、ぜひお願い」。
I22 では、桁ごとの照射の手順（どの出口を次のどの入口へ送るか・どの演算で照らすか・答えの各桁をどの出口から読むか）を手で書いた。
この実験で、それを光で指定する板に置き換える。
  - 口（ポート）: 入口の数の板 A一・A十・B一・B十 と、照射 k（k = 1〜6。I22 の手の手順は 4 つなので 2 つは空き）の出口 k.一（一の位）・k.上（桁上り / 桁下がり）。どれも 0〜9 の板の光。
    照射の番号は順番ではない。照射の入口にはどの口でもつなげ、入口が光った照射から照らす（輪になるつなぎは照らせない = 答えが出ない）。
  - つなぎの板 = 行 × 列 の明るさ。行は「照射 k の左の入口」「照射 k の右の入口」「照射 k の演算」「答えの百・十・一の位」。
    列は、その行に送れる口か、演算（＋・−・×）。
    照らすとき: 行ごとにいちばん明るい画素だけを通すマスク（勝者総取り）にし、口の光（0〜9 の板の one-hot）をマスクと掛けて入口へ送る（クロスバーの行列 × ベクトル）。
  - 式の演算（＋・−・×）の光が、つなぎの板の層を選ぶ（演算ごとに 1 層）。
  - 照射そのものは I22 と同じ 0〜9 の板の演算の帯（表は全部教えた、入れ替えの気付きあり）。条件分岐なし。
  - 学び（答えだけ、判断の板）: 2 つの数と演算と、答えの数だけが返ってくる。
      合った: 使った画素を少し明るくする。合った例は「覚えの例」として持つ（新しいものから最大 M 個）。
      違った: 答えの板と、桁ごと（百・十・一の位の観点）に差分を取り、違った桁の答えの行と、照射の行だけを替える候補にする。
        替える画素の数を 1 → 2 と増やしながら、「いまの例が合い、覚えの例も壊れない」替え方を探す（介入）。
        それで直らなければ、照射 1 つ（入口 2 つと演算）を決め直し、違った桁の答えの行をその出口へつなぐ（空いている照射から、次に使っている照射の組み直し）。
        見つかれば、替えた画素を行のいちばん明るい画素より明るくする。
        見つからなければ見送る。解き直した回数を数える（総当たりの数と比べる）。
  - 教えの順: でたらめ / 易しい順（1 桁どうし → 2 桁を含む）。
物差し: 0〜99 の組（I22 と同じ範囲）の正解率、学んだ手順が I22 の手の手順と同じ働きか、解き直しの回数、見送りの数。
結果: ../Project_Kotoyui_private/intervention/results/i28.jsonl
実行: python intervention/i28_route_board.py --seed 0
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i19_grain_arith as G  # noqa: E402
from i21_board_alu import OpBands  # noqa: E402
from i22_carry import CarryALU, digit_fact  # noqa: E402

N_STEP = 6                                                          # 照射の数（I22 の手の手順は 4 つ。空きを 2 つ）
PORTS = ["A一", "A十", "B一", "B十"] + [f"{k}.{x}" for k in range(1, N_STEP + 1) for x in ("一", "上")]
OPS = ["+", "-", "*"]
OUTS = ["百", "十", "一"]
PAIRS = {"+": [(x, y) for x in range(100) for y in range(100)],
         "-": [(x, y) for x in range(100) for y in range(100) if x >= y],
         "*": [(x, y) for x in range(100) for y in range(10)]}


def rows():
    """行の名前と、その行の列の数。"""
    r = []
    for k in range(N_STEP):
        n_in = len(PORTS)                                             # どの口からでも入れられる（光のクロスバー）。輪になるつなぎは照らせない
        r += [(f"{k + 1}.左", n_in), (f"{k + 1}.右", n_in), (f"{k + 1}.演算", len(OPS))]
    r += [(o, len(PORTS)) for o in OUTS]
    return r


ROWS = rows()
STEP_ROWS = [i for i, (n, _) in enumerate(ROWS) if "." in n]
OUT_ROW = {o: i for i, (n, _) in enumerate(ROWS) if n in OUTS for o in [n]}


class RouteBoard:
    """つなぎの板（演算ごとに 1 層）。b[行] = 列ごとの明るさ。"""

    def __init__(self, rng):
        self.b = [rng.uniform(0, 0.01, n) for _, n in ROWS]          # 初めはかすかな雑音だけ

    def mask(self):
        return [int(np.argmax(x)) for x in self.b]                    # 行ごとの勝者総取り


def table_of(bands):
    """演算の帯の照射の答えを先に引いておく（帯の中身は I22 と同じ）。"""
    t = {}
    for op in OPS:
        for a in range(10):
            for b in range(10):
                t[(a, op, b)] = bands.compute(a, op, b, True, False)
    return t


def order_of(ch):
    """答えに要る照射を、照らす順に並べる（入口が先に光っている順）。輪になっていたら None。"""
    order, state = [], {}

    def visit(k):
        if state.get(k) == 2:
            return True
        if state.get(k) == 1:
            return False                                              # 輪
        state[k] = 1
        for p in (ch[3 * k], ch[3 * k + 1]):
            if p >= 4 and not visit((p - 4) // 2):
                return False
        state[k] = 2
        order.append(k)
        return True

    for o in OUTS:
        p = ch[OUT_ROW[o]]
        if p >= 4 and not visit((p - 4) // 2):
            return None
    return order


def run(choice, A, B, tab, order=None):
    """選んだ画素（行ごとの列の番号）で照らす。答えの数を返す（照らせないつなぎは -1）。"""
    order = order_of(choice) if order is None else order
    if order is None:
        return -1
    p = [A % 10, A // 10, B % 10, B // 10] + [0] * (2 * N_STEP)
    for k in order:
        l, r, o = choice[3 * k], choice[3 * k + 1], choice[3 * k + 2]
        p[4 + 2 * k], p[5 + 2 * k] = tab[(p[l], OPS[o], p[r])]
    h, t, d = (p[choice[OUT_ROW[x]]] for x in OUTS)
    return h * 100 + t * 10 + d


def run_light(board, A, B, tab):
    """run と同じものを、クロスバーの行列 × 口の光（one-hot）で照らす。確かめ用。"""
    masks = []
    for x in board.b:
        m = np.zeros(len(x))
        m[int(np.argmax(x))] = 1.0
        masks.append(m)
    ch = board.mask()
    order = order_of(ch)
    if order is None:
        return -1
    P = np.zeros((len(PORTS), 10))                                    # 口 × 0〜9 の光
    for i, v in enumerate((A % 10, A // 10, B % 10, B // 10)):
        P[i, v] = 1.0
    for k in order:
        a = int(np.argmax(masks[3 * k] @ P))
        b = int(np.argmax(masks[3 * k + 1] @ P))
        op = OPS[int(np.argmax(masks[3 * k + 2]))]
        d, c = tab[(a, op, b)]
        P[4 + 2 * k, d] = P[5 + 2 * k, c] = 1.0
    h, t, d = (int(np.argmax(masks[OUT_ROW[x]] @ P)) for x in OUTS)
    return h * 100 + t * 10 + d


def digits(v):
    return (v // 100 % 10, v // 10 % 10, v % 10)


def used_steps(ch):
    """答えの行から参照をたどって、使われている照射（0 始まり）。"""
    used, todo = set(), [ch[OUT_ROW[o]] for o in OUTS]
    while todo:
        p = todo.pop()
        if p < 4:
            continue
        k = (p - 4) // 2
        if k in used:
            continue
        used.add(k)
        todo += [ch[3 * k], ch[3 * k + 1]]
    return used


def accept(ch, c2, A, B, target, memory, tab, st):
    st["solves"] += 1
    order = order_of(c2)
    if order is None or run(c2, A, B, tab, order) != target:
        return False
    for a2, b2, t2 in memory:
        st["solves"] += 1
        if run(c2, a2, b2, tab, order) != t2:
            return False
    return True


def commit(board, ch, c2, memory, A, B, target, mem_max):
    for i, (j0, j) in enumerate(zip(ch, c2)):
        if j != j0:
            board.b[i][j] = board.b[i].max() + 1.0                    # 注意つき: 行のいちばん明るい画素より明るく
    memory.append((A, B, target))
    del memory[:-mem_max]


def learn_one(board, A, B, target, memory, tab, st, max_flips=2, mem_max=40, use_free=True):
    ch = board.mask()
    st["solves"] += 1
    got = run(ch, A, B, tab)
    if got == target:
        for i, j in enumerate(ch):
            board.b[i][j] += 0.1                                      # 合った: 使った画素を少し明るく
        memory.append((A, B, target))
        del memory[:-mem_max]
        return "合った"
    # 判断の板: 桁ごとの差分で、替える候補の行をしぼる（違った桁の答えの行 + 使われている照射の行）
    wrong = [o for o, g, w in zip(OUTS, digits(got), digits(target)) if g != w]
    used = used_steps(ch)
    cand_rows = [3 * k + d for k in sorted(used) for d in range(3)] + [OUT_ROW[o] for o in wrong]
    alts = [(i, j) for i in cand_rows for j in range(ROWS[i][1]) if j != ch[i]]
    for nf in range(1, max_flips + 1):
        for flips in itertools.combinations(alts, nf):
            if len({i for i, _ in flips}) < nf:
                continue
            c2 = ch[:]
            for i, j in flips:
                c2[i] = j
            if accept(ch, c2, A, B, target, memory, tab, st):
                commit(board, ch, c2, memory, A, B, target, mem_max)
                st[f"替えた{nf}"] += 1
                return f"替えた{nf}"
    # 空いている照射を使う: 使われていない照射 1 つの入口と演算を決め、違った桁の答えの行をその出口へつなぐ。
    # 空きで直らなければ、使っている照射 1 つを組み直す（入口 2 つと演算を替える）
    if use_free:
        for k in sorted(range(N_STEP), key=lambda k: (k in used, k)):    # 空いている照射から。それでもだめなら、使っている照射を 1 つ組み直す
            n_in = len(PORTS)
            outs = [(OUT_ROW[o], (4 + 2 * k, 5 + 2 * k)) for o in wrong]
            for l, r, o in itertools.product(range(n_in), range(n_in), range(len(OPS))):
                for pick in itertools.product(*[(None,) + p for _, p in outs]):
                    if all(x is None for x in pick):
                        continue
                    c2 = ch[:]
                    c2[3 * k], c2[3 * k + 1], c2[3 * k + 2] = l, r, o
                    for (row, _), x in zip(outs, pick):
                        if x is not None:
                            c2[row] = x
                    if accept(ch, c2, A, B, target, memory, tab, st):
                        commit(board, ch, c2, memory, A, B, target, mem_max)
                        name = "組み直し" if k in used else "空き照射"
                        st[name] += 1
                        return name
    st["見送り"] += 1
    return "見送り"


def describe(board, op):
    ch = board.mask()
    s = []
    for k in range(N_STEP):
        s.append(f"{k + 1}: {PORTS[ch[3 * k]]} {OPS[ch[3 * k + 2]]} {PORTS[ch[3 * k + 1]]}")
    s.append("答え: " + " ".join(f"{o}={PORTS[ch[OUT_ROW[o]]]}" for o in OUTS))
    return s


HAND = {  # I22 の手の手順（比べ用）: (左, 右, 演算) × 4 と 答えの口
    "+": [("A一", "B一", "+"), ("A十", "B十", "+"), ("2.一", "1.上", "+"), ("2.上", "3.上", "+"), ("4.一", "3.一", "1.一")],
    "-": [("A一", "B一", "-"), ("A十", "B十", "-"), ("2.一", "1.上", "-"), ("2.上", "3.上", "+"), ("4.一", "3.一", "1.一")],
    "*": [("A一", "B一", "*"), ("A十", "B一", "*"), ("2.一", "1.上", "+"), ("2.上", "3.上", "+"), ("4.一", "3.一", "1.一")],
}


def hand_choice(op):
    h = HAND[op]
    c = []
    for l, r, o in h[:4] + [("A一", "A一", "+")] * (N_STEP - 4):
        c += [PORTS.index(l), PORTS.index(r), OPS.index(o)]
    c += [PORTS.index(x) for x in h[4]]
    return c


def easy_key(xy, op):
    x, y = xy
    carry = int(digit_fact(x % 10, op, y % 10)[1] > 0)
    return ((x >= 10) + (y >= 10), carry)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_train", type=int, default=300)
    ap.add_argument("--max_flips", type=int, default=2)
    ap.add_argument("--no_free", action="store_true", help="空いている照射を使う探し方を外す（比べ）")
    ap.add_argument("--out", default="../Project_Kotoyui_private/intervention/results/i28.jsonl")
    a = ap.parse_args()
    t0 = time.perf_counter()
    ob = OpBands()
    for op in OPS:
        for x in range(10):
            for y in range(10):
                ob.teach(x, op, y, digit_fact(x, op, y))
    ob.notice()
    tab = table_of(ob)
    alu = CarryALU(ob)
    rec = dict(exp="I28", seed=a.seed, n_train=a.n_train, max_flips=a.max_flips, free=not a.no_free, n_step=N_STEP, runs={})
    # 手の手順（I22）が、この口と照射の形で全部合うかを先に確かめる
    rec["hand"] = {op: sum(run(hand_choice(op), x, y, tab) == G.FN[op](x, y) for x, y in PAIRS[op]) / len(PAIRS[op]) for op in OPS}
    rec["i22"] = {op: sum(alu.compute(x, op, y) == G.FN[op](x, y) for x, y in PAIRS[op]) / len(PAIRS[op]) for op in OPS}
    space = 1
    for _, n in ROWS:
        space *= n
    rec["space_per_op"] = space
    for order in ("でたらめ", "易しい順"):
        res = {}
        for op in OPS:
            rng = random.Random(a.seed * 31 + OPS.index(op))
            board = RouteBoard(np.random.default_rng(a.seed * 31 + OPS.index(op)))
            train = rng.sample(PAIRS[op], a.n_train)
            if order == "易しい順":
                train.sort(key=lambda xy: easy_key(xy, op))
            st = {"solves": 0, "見送り": 0, "空き照射": 0, "組み直し": 0, **{f"替えた{k}": 0 for k in range(1, a.max_flips + 1)}}
            memory, curve = [], []
            for n, (x, y) in enumerate(train, 1):
                learn_one(board, x, y, G.FN[op](x, y), memory, tab, st, a.max_flips, use_free=not a.no_free)
                if n in (10, 30, 100, 300, 1000):
                    ch = board.mask()
                    curve.append((n, sum(run(ch, u, v, tab) == G.FN[op](u, v) for u, v in PAIRS[op]) / len(PAIRS[op])))
            ch = board.mask()
            acc = sum(run(ch, x, y, tab) == G.FN[op](x, y) for x, y in PAIRS[op]) / len(PAIRS[op])
            sample = random.Random(1).sample(PAIRS[op], 200)
            light_same = all(run_light(board, x, y, tab) == run(ch, x, y, tab) for x, y in sample)
            res[op] = dict(acc=acc, curve=curve, light_same=light_same, program=describe(board, op),
                           same_as_hand=ch == hand_choice(op), used_steps=len(used_steps(ch)), **st)
        rec["runs"][order] = res
    rec["total_s"] = time.perf_counter() - t0
    print(json.dumps(rec, ensure_ascii=False, indent=1), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
