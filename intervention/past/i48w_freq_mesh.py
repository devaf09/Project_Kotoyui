"""I48w: 周波数の網の小さな数値の試し（ロードマップ 50 の下調べ）。数は float32／complex64。

感情の帯 8 本と型の帯 6 本を、別々の周波数の余弦（位相は乱数。位相は出どころの印に使う想定）で 1 本の波に重ね、
非線形（2 乗）を掛ける。2 乗の中の掛け合わせ a_e b_t cos(f_e) cos(f_t) は、和 f_e + f_t と差 |f_e − f_t| の周波数に a_e b_t / 4 の振幅を作る。
その振幅から「どの感情とどの型が同時に立っているか」（a_e b_t）を読み出せるか、ほかの組（感情どうし・型どうし・2 倍）が同じ周波数に落ちて
混ざる（にせの信号）か、を数える。

周波数の組 3 通り:
  primes    : 感情 = 3 から 8 つの奇数の素数、型 = その次の 6 つの奇数の素数。
  primes_mix: 感情 = 奇数の素数 8 つ、型 = 偶数（2 × 素数）6 つ。和・差が奇数になり、感情どうし・型どうしの和・差（偶数）と分かれる。
  sidon     : 14 本の和・差・2 倍がすべて別の周波数になるように、小さい方から選んだ組（ほかの帯とも重ならない）。
読み出し: 和の周波数の振幅 × 4。結びつきの重み: 目標 Σ W[e,t] a_e b_t を、読み出しから最小二乗で学べるか（W の誤差）。
"""
from __future__ import annotations
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする

import argparse
import itertools
import json
import math

import numpy as np

N_E, N_T = 8, 6


def odd_primes(n, start=3):
    out, k = [], start
    while len(out) < n:
        if k % 2 and all(k % p for p in range(3, int(math.isqrt(k)) + 1, 2)):
            out.append(k)
        k += 1
    return out


def sidon(n, lo=3):
    """和 f_i + f_j（i ≤ j）と差 |f_i − f_j|（i < j）が互いに、また帯の周波数とも重ならない n 本。"""
    out, used = [], set()
    k = lo
    while len(out) < n:
        new = {k + f for f in out} | {abs(k - f) for f in out} | {2 * k, k}
        if not (new & used) and len(new) == len(out) * 2 + 2 and not ({k + f for f in out} & {abs(k - f) for f in out}):
            out.append(k)
            used |= new | set(out)
        k += 1
    return out


def bands(kind):
    if kind == "primes":
        p = odd_primes(N_E + N_T)
        return p[:N_E], p[N_E:]
    if kind == "primes_mix":
        p = odd_primes(N_E + N_T)
        return p[:N_E], [2 * x for x in p[N_E:]]
    s = sidon(N_E + N_T)
    return s[:N_E], s[N_E:]


def collisions(fe, ft):
    """和の周波数 f_e + f_t に、ほかの組（別の感情 × 型、感情どうし、型どうし、2 倍、差）が落ちる数。"""
    fs = fe + ft
    terms = Counter_like()
    for i, j in itertools.combinations_with_replacement(range(len(fs)), 2):
        terms.add(fs[i] + fs[j], (i, j))
        if i != j:
            terms.add(abs(fs[i] - fs[j]), (i, j, "差"))
    bad = 0
    for e in range(len(fe)):
        for t in range(len(ft)):
            bad += len(terms.get(fe[e] + ft[t])) > 1
    return bad


class Counter_like(dict):
    def add(self, k, v):
        self.setdefault(k, []).append(v)

    def get(self, k, d=None):
        return dict.get(self, k, d or [])


def run(kind, trials=400, N=1024, seed=0):
    fe, ft = bands(kind)
    rng = np.random.default_rng(seed)
    n = np.arange(N, dtype=np.float32)
    K = N // 2
    F = np.exp(-2j * np.pi * np.outer(np.arange(K), n) / N).astype(np.complex64)   # DFT の行列（complex64）
    ce = np.cos(2 * np.pi * np.outer(fe, n) / N).astype(np.float32)
    ct = np.cos(2 * np.pi * np.outer(ft, n) / N).astype(np.float32)
    W = rng.normal(size=(N_E, N_T)).astype(np.float32)
    true, read, read_d, tgt = [], [], [], []
    for _ in range(trials):
        a = rng.uniform(0, 1, N_E).astype(np.float32) * (rng.uniform(size=N_E) < 0.5)   # 感情はいくつかが同時に立つ（両側も）
        b = rng.uniform(0, 1, N_T).astype(np.float32) * (rng.uniform(size=N_T) < 0.5)
        pe = rng.uniform(0, 2 * np.pi, N_E).astype(np.float32)
        pt = rng.uniform(0, 2 * np.pi, N_T).astype(np.float32)
        x = (a[:, None] * np.cos(2 * np.pi * np.outer(fe, n) / N + pe[:, None])).sum(0) + \
            (b[:, None] * np.cos(2 * np.pi * np.outer(ft, n) / N + pt[:, None])).sum(0)
        y = (x.astype(np.float32) ** 2)
        Y = (F @ y.astype(np.complex64)) / N
        r = np.array([[4 * abs(Y[fe[e] + ft[t]]) for t in range(N_T)] for e in range(N_E)], np.float32)
        rd = np.array([[4 * abs(Y[abs(fe[e] - ft[t])]) for t in range(N_T)] for e in range(N_E)], np.float32)
        ab = np.outer(a, b)
        true.append(ab.ravel()), read.append(r.ravel()), read_d.append(rd.ravel())
        tgt.append(float((W * ab).sum()))
    T, Rr, Rd = np.array(true), np.array(read), np.array(read_d)
    err = lambda R: float(np.abs(R - T).mean())
    leak = lambda R: float(np.abs(R - T)[T == 0].mean())                # 立っていない組に出た読み（にせの信号）
    corr = lambda R: float(np.corrcoef(R.ravel(), T.ravel())[0, 1])
    # 結びつきの重みを読み出しから学ぶ（前半で学び、後半で確かめる）
    h = trials // 2
    Wh, *_ = np.linalg.lstsq(Rr[:h], np.array(tgt[:h], np.float32), rcond=None)
    pred = Rr[h:] @ Wh
    rel = float(np.abs(pred - np.array(tgt[h:])).mean() / (np.abs(np.array(tgt[h:])).mean() + 1e-9))
    return dict(kind=kind, 感情の帯=fe, 型の帯=ft, 重なる和の帯=collisions(fe, ft), 帯の上限=max(fe + ft) * 2,
                和の読み_相関=round(corr(Rr), 4), 和の読み_平均誤差=round(err(Rr), 4), 和の読み_にせ=round(leak(Rr), 4),
                差の読み_相関=round(corr(Rd), 4), 差の読み_にせ=round(leak(Rd), 4),
                重みを学んだ時の誤差=round(rel, 4), W_の相関=round(float(np.corrcoef(Wh, W.ravel())[0, 1]), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    a = ap.parse_args()
    res = [run(k) for k in ("primes", "primes_mix", "sidon")]
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()


# ---- 複素の波（語板の波は complex64 の長さ D）での置き方: 2 乗は和の周波数だけを作る（差は出ない）。和は D で回る ----

def place_complex(n_e, n_t, D, seed=0, tries=200):
    """感情 × 型の和 (f_e + f_t) mod D が、ほかのどの組の和（感情どうし・型どうし・2 倍・別の感情 × 型）とも、帯の周波数とも重ならない置き方を、
    乱数の順で小さい方から詰めて探す。見つからなければ None。"""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(tries):
        fe, ft = [], []
        order = ["e"] * n_e + ["t"] * n_t
        rng.shuffle(order)
        ok = True
        for side in order:
            for k in range(1, D):
                e2, t2 = (fe + [k], ft) if side == "e" else (fe, ft + [k])
                if k in fe or k in ft:
                    continue
                if _good(e2, t2, D):
                    fe, ft = e2, t2
                    break
            else:
                ok = False
                break
        if ok and (best is None or max(fe + ft) < max(best[0] + best[1])):
            best = (sorted(fe), sorted(ft))
    return best


def _good(fe, ft, D):
    base = set(fe) | set(ft)
    allf = fe + ft
    others = Counter_like()
    for i, j in itertools.combinations_with_replacement(range(len(allf)), 2):
        others.add((allf[i] + allf[j]) % D, (i, j))
    for e in fe:
        for t in ft:
            s = (e + t) % D
            if len(others.get(s)) > 1 or s in base:
                return False
    return True


def run_complex(fe, ft, D, trials=300, seed=0):
    rng = np.random.default_rng(seed)
    n = np.arange(D, dtype=np.float32)
    F = np.exp(-2j * np.pi * np.outer(np.arange(D), n) / D).astype(np.complex64)
    ne, nt = len(fe), len(ft)
    true, read = [], []
    for _ in range(trials):
        a = rng.uniform(0, 1, ne).astype(np.float32) * (rng.uniform(size=ne) < 0.5)
        b = rng.uniform(0, 1, nt).astype(np.float32) * (rng.uniform(size=nt) < 0.5)
        pa = rng.uniform(0, 2 * np.pi, ne)
        pb = rng.uniform(0, 2 * np.pi, nt)
        x = (a[:, None] * np.exp(1j * (2 * np.pi * np.outer(fe, n) / D + pa[:, None]))).sum(0) + \
            (b[:, None] * np.exp(1j * (2 * np.pi * np.outer(ft, n) / D + pb[:, None]))).sum(0)
        y = (x.astype(np.complex64) ** 2)
        Y = (F @ y) / D
        r = np.array([[abs(Y[(e + t) % D]) / 2 for t in ft] for e in fe], np.float32)   # 2 a b の和の項
        true.append(np.outer(a, b).ravel()), read.append(r.ravel())
    T, R = np.array(true), np.array(read)
    return dict(相関=round(float(np.corrcoef(R.ravel(), T.ravel())[0, 1]), 4), にせ=round(float(np.abs(R - T)[T == 0].mean()), 5),
                平均誤差=round(float(np.abs(R - T).mean()), 5))
