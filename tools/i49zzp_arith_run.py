"""I49zzp: 算数の文字の波を数える道具（`intervention/i49zzp_arith_wave.py`。持ち主の案、未確定の試し）。

世界は 2 つ:
  小さい世界: 答えが 1 文字で書ける式の全部。足し算 a+b=c（a+b が 9 以下、55 問）と、引き算 a-b=c（a が b 以上、55 問）。
  全部の世界: 0〜9 どうしの式の全部。足し算 100 問（答えは 0〜18。"12" は 2 文字）と、引き算 100 問（答えは -9〜9。"-3" は 2 文字）。
  何も関係の無い世界（確かめ用）: 同じ式の答えだけを、別の式の答えと入れ替える（並べ替えは黄金比 × 番号の 2 乗。乱数は使わない）。
    小さい世界と全部の世界では、答えを入れ替えても「どの数字が何回、どこに出るか」が残る（小さい数ほど答えに多い、など）。
  1 の位だけの世界: 答えを 1 の位だけにした式の全部（7+5=2、3-5=8。200 問）。どの数字もどの場所にも同じ回数だけ出るので、回数の偏りが無い。
    この世界の「何も関係の無い世界」が、本当に何も無い確かめになる。
数える物:
  1. 数字どうしの波の近さを、数の差ごとに平均する（差が小さいほど近ければ、数の並びが波に出ている）。
  2. 数字の波の上の 2 つの向きの面で、位相（角度）が数の順に並ぶか。位相の足し算 φ(a)+φ(b) が φ(c) に合うか。
     数字の波の散らばり（数字どうしの波の違いの大きさ ÷ 波の大きさ）が 0 なら、数字の波は全部同じで、位相は読めない。
  3. ＋ と − の波の近さ。
  4. 見せていない式の答えを当てる（黄金比の並びで 5 つに 1 つを取り分ける）。
     波で選ぶ（候補は覚えた式に出た答えの全部）、一番多い答えを言い続ける。外れの大きさ（数の差の平均）も出す。
     位相の足し算で選ぶ形は、答えが 1 文字の式だけで数える。
  5. ずれを運ぶ（shift_test）: 見せていない式を、片方の数だけが 1 違う覚えた式から運んで答える（2+3=5 から 3+3= を答える）。
`--base 16` で、0〜F の 16 進の世界で同じ事を数える（他は同じ）。
手で置いた物: 世界の作り方、取り分ける割合、確かめ用の並べ替えの作り方、位相の足し算で選ぶ時の決まり（φ(0) を 0 とし、＋は足し、−は引き、一番近い数字を取る）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))

from i49zzp_arith_wave import DIGITS, CharWave  # noqa: E402

PHI = 0.6180339887498949
HOLD = 5   # 何個に 1 個を取り分けるか（手で置いた値）


HEX = "0123456789ABCDEF"


def fmt(n, digits=DIGITS):
    """数を、digits の進み方で文字の並びにする（16 進なら 26 → "1A"、-3 → "-3"）。"""
    b, sign, n = len(digits), "-" if n < 0 else "", abs(n)
    out = digits[n % b]
    while n >= b:
        n //= b
        out = digits[n % b] + out
    return sign + out


def num(text, digits=DIGITS):
    neg = text.startswith("-")
    v = 0
    for c in text.lstrip("-"):
        v = v * len(digits) + digits.index(c)
    return -v if neg else v


def world(full=False, digits=DIGITS):
    r = range(len(digits))
    add = [digits[a] + "+" + digits[b] + "=" + fmt(a + b, digits) for a in r for b in r if full or a + b < len(digits)]
    sub = [digits[a] + "-" + digits[b] + "=" + fmt(a - b, digits) for a in r for b in r if full or a >= b]
    return add, sub


def ring_world(digits=DIGITS):
    """1 の位だけの世界: 答えを 1 の位だけにする（10 進なら 7+5=2、3-5=8）。どの数字も、どの場所にも同じ回数だけ出る。
    回数の偏りが無いので、数の並びが出るなら、式の関係から出た物。"""
    r, n = range(len(digits)), len(digits)
    add = [digits[a] + "+" + digits[b] + "=" + digits[(a + b) % n] for a in r for b in r]
    sub = [digits[a] + "-" + digits[b] + "=" + digits[(a - b) % n] for a in r for b in r]
    return add, sub


def ans(line):
    return line.split("=")[1]


def head(line):
    return line.split("=")[0] + "="


def golden_order(n):
    return sorted(range(n), key=lambda i: ((i + 1) * PHI) % 1.0)


def scramble_order(n):
    """確かめ用の並べ替え。黄金比の並び（i に比例）は、隣の式が決まった歩幅で動くので、表の形に並んだ式では数の並びが残る。
    ここでは i の 2 乗に比例させて、歩幅が決まらないようにする（乱数は使わない）。"""
    return sorted(range(n), key=lambda i: (((i + 1) * (i + 1) * PHI) % 1.0, i))


def null_world(lines):
    """答えだけを、別の式の答えと入れ替える（答えの数の出る回数は同じ）。"""
    order = scramble_order(len(lines))
    return [head(lines[i]) + ans(lines[order[i]]) for i in range(len(lines))]


def by_diff(w, digits=DIGITS):
    units, m = w.waves()
    n = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)
    out = {}
    for d in range(1, len(digits)):
        v = [float(n[units.index(digits[i])] @ n[units.index(digits[i + d])]) for i in range(len(digits) - d)]
        out[str(d)] = round(sum(v) / len(v), 4)
    return out


def rank_corr(x, y):
    rx, ry = np.argsort(np.argsort(x)).astype(np.float64), np.argsort(np.argsort(y)).astype(np.float64)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    d = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    return float(rx @ ry) / d if d > 0 else 0.0


def phase_report(w, digits=DIGITS):
    have, p, share = w.plane(digits)
    _, ph = w.phase(digits)
    vals = np.array([digits.index(c) for c in have], dtype=np.float64)
    turns = np.diff(ph)
    same = int(max((turns > 0).sum(), (turns < 0).sum()))
    units, m = w.waves()
    x = m[[units.index(c) for c in have]]
    spread = float(np.linalg.norm(x - x.mean(axis=0, keepdims=True)) / max(float(np.linalg.norm(x)), 1e-12))
    return dict(数字の波の散らばり=round(spread, 4), 位相を読める=bool(spread > 1e-3),
                向きの取り分=[round(float(s), 4) for s in share[:4]],
                位相=dict(zip(have, [round(float(x), 3) for x in ph])),
                座標=dict(zip(have, [[round(float(a), 4), round(float(b), 4)] for a, b in p])),
                位相と数の順の合い方=round(abs(rank_corr(ph, vals)), 4),
                隣へ同じ向きに回った数=same, 隣の数=len(turns),
                一番強い向きと数の順の合い方=round(abs(rank_corr(p[:, 0], vals)), 4))


def phase_answer(w, line, digits=DIGITS):
    have, ph = w.phase(digits)
    f = dict(zip(have, [float(x) for x in ph]))
    a, op, b = line[0], line[1], line[2]
    if a not in f or b not in f:
        return None
    base = f.get("0", 0.0)
    t = (f[a] - base) + (f[b] - base) * (1 if op == "+" else -1) + base
    return min(have, key=lambda c: (abs(f[c] - t), c))


def one_digit(lines):
    return [x for x in lines if len(ans(x)) == 1]


def held_out(lines, offsets=True, digits=DIGITS):
    order = golden_order(len(lines))
    test = [lines[i] for k, i in enumerate(order) if k % HOLD == 0]
    hold = set(test)
    train = [x for x in lines if x not in hold]
    w = CharWave(offsets=offsets)
    for x in train:
        w.learn(x)
    cnt = {}
    for x in train:
        cnt[ans(x)] = cnt.get(ans(x), 0) + 1
    cands = sorted(cnt)
    top = sorted(cnt, key=lambda c: (-cnt[c], c))[0]
    res = {k: dict(当たり=0.0, 外れの大きさ=0.0) for k in ("波で選ぶ", "一番多い答えを言い続ける")}
    for x in test:
        got = {"波で選ぶ": w.answer(head(x), cands)[0][0], "一番多い答えを言い続ける": top}
        for k, g in got.items():
            res[k]["当たり"] += g == ans(x)
            res[k]["外れの大きさ"] += abs(num(g, digits) - num(ans(x), digits))
    n = max(len(test), 1)
    out = dict(覚えた式=len(train), 試した式=len(test), **{k: {a: round(b / n, 4) for a, b in v.items()} for k, v in res.items()})
    t1 = one_digit(test)
    if t1:
        got = [phase_answer(w, x, digits) for x in t1]
        out["位相の足し算で選ぶ_答えが 1 文字の式"] = dict(試した式=len(t1), 当たり=round(sum(g == ans(x) for g, x in zip(got, t1)) / len(t1), 4),
                                         外れの大きさ=round(sum(abs(num(g, digits) - num(ans(x), digits)) for g, x in zip(got, t1)) / len(t1), 4))
    return out


def neighbours(x, train, digits, step=1):
    """x と、片方の数だけが step 違う、覚えた式。→ (覚えた式, どちらの数が違うか 0/2, 違う数の元, 違う数の今)。"""
    out = []
    for y in train:
        if y[1] != x[1] or len(ans(y)) != 1:
            continue
        for pos, other in ((0, 2), (2, 0)):
            if y[other] == x[other] and abs(digits.index(y[pos]) - digits.index(x[pos])) == step:
                out.append((y, pos, y[pos], x[pos]))
    return out


def learn_signs(train, f):
    """片方の数だけが違う覚えた式の組で、数の位相が増えた時に答えの位相が増えたか減ったかを数える。→ (＋か−, どちらの数) ごとの向き。
    「引く数が増えると答えは減る」を手で与えないための数え。"""
    cnt = {}
    one = [y for y in train if len(ans(y)) == 1]
    for i, y in enumerate(one):
        for z in one[i + 1:]:
            if y[1] != z[1]:
                continue
            for pos, other in ((0, 2), (2, 0)):
                if y[other] == z[other] and y[pos] != z[pos] and ans(y) != ans(z):
                    same = (f[y[pos]] - f[z[pos]]) * (f[ans(y)] - f[ans(z)]) > 0
                    c = cnt.setdefault((y[1], pos), [0, 0])
                    c[0 if same else 1] += 1
    return {k: (1 if v[0] >= v[1] else -1) for k, v in cnt.items()}


def vote(preds):
    c = {}
    for p in preds:
        c[p] = c.get(p, 0) + 1
    return sorted(c, key=lambda k: (-c[k], k))[0] if c else None


def shift_test(lines, offsets=True, digits=DIGITS, step=1):
    """持ち主の頼み:「２＋３＝５の位相を３＋３＝に合わせると６にはなれるか」。
    見せていない式（答えが 1 文字）を、片方の数だけが step 違う覚えた式から運んで答える。3 通りを並べる:
      そのまま借りる: 覚えた式の答えをそのまま言う（表の借り方。必ず外れる）。
      位相で運ぶ: 答えの位相 + 向き ×（今の数の位相 − 元の数の位相）に一番近い数字。位相は強い 2 つの向きの面の角度。
      位置で運ぶ: 同じ事を、一番強い向きの上の座標（グラデーションそのもの）でやる。
      順で運ぶ: 位置の順に数字を並べ、その並びの上で同じ歩数だけ動く（間隔を等しいとみなす。ここは手で置いた物）。
    向き（増えるか減るか）は learn_signs で数える。位相も向きも、覚えた式だけから作る。"""
    order = golden_order(len(lines))
    test = [lines[i] for k, i in enumerate(order) if k % HOLD == 0]
    hold = set(test)
    train = [x for x in lines if x not in hold]
    w = CharWave(offsets=offsets)
    for x in train:
        w.learn(x)
    have, ph = w.phase(digits)
    _, p, _ = w.plane(digits)
    f = dict(zip(have, [float(v) for v in ph]))            # 位相（強い 2 つの向きの面の角度）
    g = dict(zip(have, [float(v) for v in p[:, 0]]))       # 位置（一番強い向きの上の座標）
    ranked = sorted(have, key=lambda c: (g[c], c))
    rk = {c: float(i) for i, c in enumerate(ranked)}       # 順（位置の順に並べた時の番号）
    reads = {"位相で運ぶ": f, "位置で運ぶ": g, "順で運ぶ": rk}
    signs = {k: learn_signs(train, v) for k, v in reads.items()}
    keys = ("そのまま借りる",) + tuple(reads)
    res = {k: dict(当たり=0.0, 外れの大きさ=0.0) for k in keys}
    n = 0
    for x in test:
        if len(ans(x)) != 1 or any(c not in f for c in (x[0], x[2])):
            continue
        nb = [t for t in neighbours(x, train, digits, step) if ans(t[0]) in f]
        if not nb:
            continue
        n += 1
        preds = {k: [] for k in keys}
        for y, pos, old, new in nb:
            preds["そのまま借りる"].append(ans(y))
            for k, v in reads.items():
                t = v[ans(y)] + signs[k].get((x[1], pos), 1) * (v[new] - v[old])
                preds[k].append(min(have, key=lambda c: (abs(v[c] - t), c)))
        for k in keys:
            got = vote(preds[k])
            res[k]["当たり"] += got == ans(x)
            res[k]["外れの大きさ"] += abs(num(got, digits) - num(ans(x), digits))
    return dict(試した式=n, 位置の順が数の順と同じ=bool(ranked == list(have) or ranked == list(have)[::-1]),
                数えた向き={"%s の %s" % (op, "前の数" if pos == 0 else "後ろの数"): v for (op, pos), v in sorted(signs["位置で運ぶ"].items())},
                **{k: {a: round(b / max(n, 1), 4) for a, b in v.items()} for k, v in res.items()})


def run(lines, offsets=True, digits=DIGITS):
    w = CharWave(offsets=offsets)
    for x in lines:
        w.learn(x)
    l1 = one_digit(lines)
    seen = [phase_answer(w, x, digits) == ans(x) for x in l1]
    return dict(式=len(lines), 席=len(w.seat), 数の差ごとの近さ=by_diff(w, digits), 位相=phase_report(w, digits),
                足すと引くの近さ=round(w.near("+", "-"), 4), 等しいと足すの近さ=round(w.near("=", "+"), 4),
                位相の足し算が合う式の割合_全部覚えた時=round(sum(seen) / max(len(seen), 1), 4),
                見せていない式=held_out(lines, offsets, digits),
                ずれを運ぶ=shift_test(lines, offsets, digits))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", type=int, default=10)   # 何進で数えるか（2〜16。16 なら 0〜F）
    a = ap.parse_args()
    t0 = time.time()
    digits = HEX[:a.base]
    res = dict(手で置いた値=dict(HOLD=HOLD), 数字=digits)
    worlds = (("小さい世界", world(False, digits)), ("全部の世界", world(True, digits)), ("1 の位だけの世界", ring_world(digits)))
    for wname, (add, sub) in worlds:
        lines = add + sub
        res[wname] = {}
        for name, ls in (("算数", lines), ("何も関係の無い世界", null_world(lines)), ("足し算だけ", add), ("引き算だけ", sub)):
            res[wname][name] = dict(離れ方を見る=run(ls, True, digits), 離れ方を見ない=run(ls, False, digits))
    res["秒"] = round(time.time() - t0, 2)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
