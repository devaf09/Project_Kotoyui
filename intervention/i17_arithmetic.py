"""介入の研究 実験I17: 足し算・引き算を、介入の学習（差分と経験）で見つける

持ち主（2026-11-05）: 「計算はロジック的には光の性質（位相の坂で位置をずらす）だが、今回の介入による学習では、
  数の語板と差分・経験から規則を見つける形。特に桁や別のものへの理解を考えると」。
世界:
  - 数 0〜99。数の語板の属性 = 一の位（0〜9）・十の位（0〜9）・偶奇。語板は光の器（I1 の器、縦 32 画素）で、数を説明する文（1 文に属性 1〜2 個）から学ぶ。
  - 操作の文: 「23 に 1 を 足す と 24」。教えに使う数は 6 割（残り 4 割の数を足す前の数として試す）。操作は +1・+2・+3。
仕組み:
  1. 差分検証（光の相関器）: 足す前と後の語板の、観点ごとの明るさを巡回の相関で比べ、いちばん明るいずらし量（その観点が何個ずれたか）を読む。
     （4f 系の相関器で、明るさの並びどうしの「ずれ」を光の位置として読むのと同じ。）
  2. 経験: 場面 = {操作:+k, 一の位:d, 十の位:t}（足す前の数の語板でいちばん明るい値）、結論 = {観点:ずらし量}。観点ごとに、結論の種類ごとに口を 1 つ持つ光の器（I8b）。
  3. 答える: 試しの数の場面を経験の器に通し、観点ごとにいちばん明るいずらし量で、足す前の語板の明るさを巡回でずらす（= 位相の坂の板を通すのと同じ）。
     ずらした明るさから一の位・十の位を読み、答えの数にする。
  比べ:
    規則だけ: 操作ごとに、いちばん多いずらし量を観点ごとに 1 つだけ覚える（条件なし。繰り上がりを知らない）。
    丸暗記: (足す前の数, 操作) → 答え の表。教えに無い数には答えられない。
  引き算（逆回し）: −1・−2 は教えない。足し算の経験を逆に使う: 答えの一の位を仮に置き、「その数に +k すると今の数になるか」を経験に照らして、合う答えを選ぶ。
  別の物への移り（時計の分 0〜59）: 数で学んだ +1 の経験を、そのまま分に使う。59 分 + 1 = 0 分（十の位が 5 → 0 に回る）だけは数と違う。
    さらに、59 → 0 の文を 1 つだけ「例外の経験」（場面の記号が全部そろったときだけ光る注意の口）として足したとき直るかを見る。
    （ふつうの経験に混ぜると、「物:時計」の記号がその 1 文にしか無いため、時計の場面ぜんぶがその 1 文に引っ張られた。1 回目の試しで分かった。）
結果: ../Project_Kotoyui_private/intervention/results/i17.jsonl
実行: CUDA_VISIBLE_DEVICES= python intervention/i17_arithmetic.py --seed 0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from i01_small_world import Devices  # noqa: E402

VIEWS = {"一の位": 10, "十の位": 10, "偶奇": 2}
OFF = {"一の位": 0, "十の位": 10, "偶奇": 20}
A = 22
VAL0 = 3


def attrs(x, tens_mod=10):
    return {"一の位": x % 10, "十の位": (x // 10) % tens_mod, "偶奇": x % 2}


def learn_boards(numbers, n_sent, seed, tens_mod=10, steps=300):
    rng = random.Random(seed)
    Q = np.zeros((len(numbers), A))
    for r, x in enumerate(numbers):
        at = attrs(x, tens_mod)
        for _ in range(n_sent):
            for v in rng.sample(list(VIEWS), rng.randint(1, 2)):
                Q[r, OFF[v] + at[v]] += 1
    Q = Q / Q.sum(1, keepdims=True)
    D = Devices(len(numbers), n=32, plates=8, seed=seed)
    q = torch.tensor(np.concatenate([Q, np.zeros((len(numbers), 32 - VAL0 - A + 1))], 1), dtype=torch.float32)
    opt = torch.optim.Adam(D.parameters(), lr=0.02)
    idx = torch.arange(len(numbers))
    for _ in range(steps):
        Wb = D.brightness(idx)[:, VAL0:]
        p = torch.cat([Wb, (1 - Wb.sum(1)).clamp(min=1e-9)[:, None]], 1)
        loss = -(q * torch.log(p + 1e-9)).sum(1).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        L = D.brightness(idx)[:, VAL0:VAL0 + A].numpy()
    return {x: L[r] for r, x in enumerate(numbers)}


def view(b, v):
    s = b[OFF[v]:OFF[v] + VIEWS[v]]
    return s / max(s.sum(), 1e-12)


def shift_of(b0, b1, v):
    """巡回の相関で、観点 v の明るさが何個ずれたか（光の相関器）。"""
    x, y = view(b0, v), view(b1, v)
    corr = np.real(np.fft.ifft(np.conj(np.fft.fft(x)) * np.fft.fft(y)))
    return int(np.argmax(corr))


def top(b, v):
    return int(np.argmax(view(b, v)))


class Exp:
    """経験の器（観点ごと、結論 = ずらし量の種類ごとに口 1 つ）。I8b の labeled と同じ形を、記号の数え上げの明るさで持つ。"""

    def __init__(self):
        self.ep = defaultdict(list)                                 # 観点 → [(場面の記号, ずらし量)]
        self.exc = defaultdict(list)                                # 例外の経験（注意の口）: 場面の記号が全部そろったときだけ光る

    def add(self, scene, v, s):
        self.ep[v].append((scene, s))

    def add_exception(self, scene, v, s):
        self.exc[v].append((scene, s))

    def recall(self, scene, v):
        for sc, s in self.exc.get(v, []):                          # 例外の経験に、いまの場面がすっかり重なれば、そちらを使う
            if all(t in scene for t in sc):
                return s
        known = {t for sc, _ in self.ep[v] for t in sc}
        scene = [t for t in scene if t in known]                    # 経験に一度も出ていない記号は、どの口も同じに照らす（光らない）
        labs = Counter(s for _, s in self.ep[v])
        if not labs:
            return 0
        best, bsc = None, -1e9
        for lab, n in labs.items():                                 # 口ごとの起こりやすさ（記号の明るさの積、加算の平滑化）
            rows = [sc for sc, s in self.ep[v] if s == lab]
            sc = math.log(n)
            for t in scene:
                sc += math.log((sum(t in r for r in rows) + 0.1) / (len(rows) + 0.2))
            if sc > bsc:
                best, bsc = lab, sc
        return best


def apply(b, shifts, tens_mod=10):
    """足す前の語板の明るさを、観点ごとにずらす（位相の坂の板と同じ）。答えの数を返す。"""
    o = (top(b, "一の位") + shifts["一の位"]) % 10
    t = (top(b, "十の位") + shifts["十の位"]) % 10
    return t * 10 + o, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_sent", type=int, default=10)
    ap.add_argument("--out", default="../Project_Kotoyui_private/intervention/results/i17.jsonl")
    a = ap.parse_args()
    torch.set_num_threads(2)
    t0 = time.perf_counter()
    rng = random.Random(a.seed)
    nums = list(range(100))
    B = learn_boards(nums, a.n_sent, a.seed)
    train = set(rng.sample(nums, 60))
    test = [x for x in nums if x not in train]
    ops = [1, 2, 3]
    exp, glob, memo = Exp(), defaultdict(Counter), {}
    for k in ops:
        for x in sorted(train):
            y = x + k
            if y > 99:
                continue
            scene = [f"操作:+{k}", f"一の位:{top(B[x], '一の位')}", f"十の位:{top(B[x], '十の位')}"]
            for v in ("一の位", "十の位"):
                s = shift_of(B[x], B[y], v)
                exp.add(scene, v, s)
                glob[(k, v)][s] += 1
            memo[(x, k)] = y
    res = defaultdict(lambda: [0, 0])
    carry = defaultdict(lambda: [0, 0])
    for k in ops:
        for x in test:
            y = x + k
            if y > 99:
                continue
            scene = [f"操作:+{k}", f"一の位:{top(B[x], '一の位')}", f"十の位:{top(B[x], '十の位')}"]
            is_carry = (x % 10) + k >= 10
            pe, _ = apply(B[x], {v: exp.recall(scene, v) for v in ("一の位", "十の位")})
            pg, _ = apply(B[x], {v: glob[(k, v)].most_common(1)[0][0] for v in ("一の位", "十の位")})
            pm = memo.get((x, k))
            for name, p in (("経験（条件つき）", pe), ("規則だけ（条件なし）", pg), ("丸暗記", pm)):
                res[name][0] += p == y
                res[name][1] += 1
                c = carry[(name, is_carry)]
                c[0] += p == y
                c[1] += 1
    # 引き算（逆回し）: 足し算の経験だけで、x − k を求める。答えの候補 z ごとに「z + k = x になるか」を経験で確かめる
    sub = defaultdict(lambda: [0, 0])
    for k in ops:
        for x in test:
            y = x - k
            if y < 0:
                continue
            ok_z = None
            for z in range(100):
                scene = [f"操作:+{k}", f"一の位:{z % 10}", f"十の位:{z // 10}"]
                p, _ = apply(B[z], {v: exp.recall(scene, v) for v in ("一の位", "十の位")})
                if p == x:
                    ok_z = z
                    break
            borrow = (x % 10) - k < 0
            for name, p in (("逆回し（足し算の経験）", ok_z), ("規則だけの逆（条件なし）", None)):
                if name.startswith("規則"):
                    o = (top(B[x], "一の位") - glob[(k, "一の位")].most_common(1)[0][0]) % 10
                    t = (top(B[x], "十の位") - glob[(k, "十の位")].most_common(1)[0][0]) % 10
                    p = t * 10 + o
                sub[(name, borrow)][0] += p == y
                sub[(name, borrow)][1] += 1
    # 時計の分（0〜59）への移り: 数で学んだ +1 の経験をそのまま使う
    Bc = learn_boards(list(range(60)), a.n_sent, a.seed + 7, tens_mod=6)
    clock = defaultdict(lambda: [0, 0])
    for add59 in (False, True):
        e2 = Exp()
        e2.ep = defaultdict(list, {v: list(r) for v, r in exp.ep.items()})
        if add59:                                                   # 「59 分の 1 分後は 0 分」の文を 1 つだけ、例外の経験（注意の口）として足す
            sc = ["操作:+1", "一の位:9", "十の位:5", "物:時計"]
            e2.add_exception(sc, "一の位", 1)
            e2.add_exception(sc, "十の位", 5)                        # 十の位 5 → 0（6 で回る）= 10 の巡回では +5
        for x in range(60):
            y = (x + 1) % 60
            scene = ["操作:+1", f"一の位:{top(Bc[x], '一の位')}", f"十の位:{top(Bc[x], '十の位')}"] + (["物:時計"] if add59 else [])
            sh = {v: e2.recall(scene, v) for v in ("一の位", "十の位")}
            o = (top(Bc[x], "一の位") + sh["一の位"]) % 10
            t = (top(Bc[x], "十の位") + sh["十の位"]) % 10
            p = t * 10 + o
            key = ("59 → 0 を 1 文足す" if add59 else "数の経験のまま", x == 59)
            clock[key][0] += p == y
            clock[key][1] += 1
    rec = dict(exp="I17", seed=a.seed, n_sent=a.n_sent,
               add={k: v[0] / v[1] for k, v in res.items()},
               add_by_carry={f"{k[0]}|{'繰り上がり' if k[1] else 'なし'}": (v[0] / v[1], v[1]) for k, v in carry.items()},
               sub_by_borrow={f"{k[0]}|{'繰り下がり' if k[1] else 'なし'}": (v[0] / v[1], v[1]) for k, v in sub.items()},
               clock={f"{k[0]}|{'59 分' if k[1] else 'ほか'}": (v[0] / v[1], v[1]) for k, v in clock.items()},
               board_read_acc=float(np.mean([top(B[x], "一の位") == x % 10 and top(B[x], "十の位") == x // 10 for x in nums])),
               total_s=time.perf_counter() - t0)
    print(json.dumps(rec, ensure_ascii=False), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
