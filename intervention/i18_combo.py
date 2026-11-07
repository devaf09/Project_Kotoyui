"""介入の研究 実験I18: 組み合わせを持つ経験の器 と、「動詞が何を変えるか」を操作の文の差分から学ぶ

I17 の限界: 経験の器（記号ごとの明るさの積）は「一の位:8 × 操作:+1」のような記号の組み合わせを持てず、
  繰り上がりの取りこぼし（+1 の 8 を繰り上がると答える、など）が出た。
T1 の限界: 介入の得は、手で与えた動詞の規則（塗る = 色だけ、熟させる = 状態だけ）から来ていた。外すと塗るが 1.00 → 0.81。
この実験:
  A. 経験の器に「組の記号」を持たせる。場面の記号の並び s を、縦と横に同じ s を置いた面（s ⊗ s、2 枚の板を交差させて通した光の明るさ）
     に広げ、面の 1 つの画素 = 記号の組 1 つ。口ごとに、経験の面の明るさの和（融合）を持つ。思い出すときは、いまの場面の面を当てて口ごとの明るさを読む。
     （ここでは面の明るさを数え上げで持つ。I17 の器と同じく、光の器としての学びはまだ。）
       単独: I17 の器（記号 1 つずつ）。組: 面の画素（記号の組）だけ。単独 + 組: 両方。
     一度も経験に出ていない記号・組は、どの口も同じに照らす（光らない）。
     確かめ: I17 の足し算（繰り上がり）・引き算の逆回し・時計の分。
  B. 小さな世界（I3 の最終形 = T1 の既定）で、「動詞が変える観点」の手の規則を外し、操作の文から学ぶ。
     操作の文（例）:「緑の 未熟な トマト を 赤に 塗った。 赤い 未熟な トマト は 渋い まま だった」「白い イチゴ を 熟させた。 赤い 甘い イチゴ に なった」。
       前と後の文に出た観点ごとに、値が違えば「変わった」、同じなら「変わらない」（差分）。1 つの文に出る観点は 2〜4 個（全部は言わない）。
       熟させる は 完熟の物にもする（何も変わらない）。
     経験 = 場面 {動詞, 色, 状態}（前の文の値）→ 観点ごとの結論 {変わる, 変わらない}。A の器（単独 / 組）で持つ。
     問いに答えるとき（I1 の answer）: 学んだ器で「この動詞がこの場面で変える観点」を思い出し、そのうち原因の観点（色・状態・形）の効き目 × 差 の最大を使う。
     1 語抜き: 聞かれた語の操作の文は使わない。
     比べ: 手の規則（T1 の既定）、規則なし（T1）、動詞だけの多数決（場面を見ない）。
結果: ../Project_Kotoyui_private/intervention/results/i18.jsonl
実行: CUDA_VISIBLE_DEVICES= python intervention/i18_combo.py --seed 0 --part A
"""
from __future__ import annotations

import argparse
import itertools
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
import i17_arithmetic as M  # noqa: E402


# ---------------------------------------------------------------- 組み合わせを持つ経験の器
class ComboExp:
    """観点ごと、結論ごとに口 1 つ。場面の面 s ⊗ s の画素（記号 1 つ = 対角、記号の組 = 対角の外）の明るさを口ごとに足して持つ。"""

    def __init__(self, order="単独"):
        self.order = order
        self.ep = defaultdict(list)
        self.exc = defaultdict(list)

    def feats(self, scene):
        sc = sorted(set(scene))
        f = []
        if self.order in ("単独", "単独+組"):
            f += sc                                                # 面の対角
        if self.order in ("組", "単独+組"):
            f += [a + "&" + b for a, b in itertools.combinations(sc, 2)]   # 面の対角の外（上半分だけ、対称なので）
        if self.order == "操作ごと":                               # 操作（動詞）の記号との組だけ = 操作ごとに器を分け、その中は記号 1 つずつ
            op = [t for t in sc if t.startswith(("動詞:", "操作:"))]
            f += op + [o + "&" + t for o in op for t in sc if t not in op]
        if self.order == "段々":                                   # 記号 1 つ・2 つ・… 全部の組（面を重ねた立体の画素）。思い出すときは細かい組から
            f += ["&".join(c) for k in range(1, len(sc) + 1) for c in itertools.combinations(sc, k)]
        return f

    def add(self, scene, v, s):
        self.ep[v].append((set(self.feats(scene)), s))

    def add_exception(self, scene, v, s):
        self.exc[v].append((scene, s))

    def scores(self, scene, v):
        for sc, s in self.exc.get(v, []):
            if all(t in scene for t in sc):
                return {s: 0.0}
        rows = self.ep.get(v, [])
        if not rows:
            return {}
        known = set().union(*(r for r, _ in rows))
        f = [t for t in self.feats(scene) if t in known]
        if self.order == "段々" and f:                             # いちばん細かい（記号の多い）組で、経験に出たものだけを使う
            top_k = max(t.count("&") for t in f)
            f = [t for t in f if t.count("&") == top_k]
        if self.order == "操作ごと":                               # 同じ操作の経験だけを照らす（ほかの操作の経験は、この器に無い）
            op = [t for t in scene if t.startswith(("動詞:", "操作:"))]
            rows = [(r, s) for r, s in rows if all(o in r for o in op)]
            if not rows:
                return {}
            known = set().union(*(r for r, _ in rows))
            f = [t for t in self.feats(scene) if t in known]
        labs = Counter(s for _, s in rows)
        out = {}
        for lab, n in labs.items():
            rr = [r for r, s in rows if s == lab]
            sc = math.log(n)
            for t in f:
                sc += math.log((sum(t in r for r in rr) + 0.1) / (len(rr) + 0.2))
            out[lab] = sc
        return out

    def recall(self, scene, v):
        sc = self.scores(scene, v)
        return max(sc, key=sc.get) if sc else 0


# ---------------------------------------------------------------- A. 足し算・引き算・時計（I17 と同じ世界）
HOLE = {(3, 7), (2, 8)}                                            # 穴: 教えに入れない（操作, 一の位）の組。どちらも繰り上がる


def part_a(seed, n_sent, order, hole=False):
    rng = random.Random(seed)
    nums = list(range(100))
    B = M.learn_boards(nums, n_sent, seed)
    train = set(rng.sample(nums, 60))
    test = [x for x in nums if x not in train]
    ops = [1, 2, 3]
    exp = ComboExp(order)
    for k in ops:
        for x in sorted(train):
            y = x + k
            if y > 99 or (hole and (k, x % 10) in HOLE):
                continue
            scene = [f"操作:+{k}", f"一の位:{M.top(B[x], '一の位')}", f"十の位:{M.top(B[x], '十の位')}"]
            for v in ("一の位", "十の位"):
                exp.add(scene, v, M.shift_of(B[x], B[y], v))
    add = defaultdict(lambda: [0, 0])
    for k in ops:
        for x in test:
            y = x + k
            if y > 99:
                continue
            scene = [f"操作:+{k}", f"一の位:{M.top(B[x], '一の位')}", f"十の位:{M.top(B[x], '十の位')}"]
            p, _ = M.apply(B[x], {v: exp.recall(scene, v) for v in ("一の位", "十の位")})
            c = add["穴" if (k, x % 10) in HOLE and hole else ("繰り上がり" if (x % 10) + k >= 10 else "なし")]
            c[0] += p == y
            c[1] += 1
    sub = defaultdict(lambda: [0, 0])
    for k in ops:
        for x in test:
            y = x - k
            if y < 0:
                continue
            ok_z = None
            for z in range(100):
                scene = [f"操作:+{k}", f"一の位:{z % 10}", f"十の位:{z // 10}"]
                p, _ = M.apply(B[z], {v: exp.recall(scene, v) for v in ("一の位", "十の位")})
                if p == x:
                    ok_z = z
                    break
            c = sub["繰り下がり" if (x % 10) - k < 0 else "なし"]
            c[0] += ok_z == y
            c[1] += 1
    if hole:
        return dict(add={k: (v[0] / v[1], v[1]) for k, v in add.items()})
    Bc = M.learn_boards(list(range(60)), n_sent, seed + 7, tens_mod=6)
    clock = defaultdict(lambda: [0, 0])
    for add59 in (False, True):
        e2 = ComboExp(order)
        e2.ep = defaultdict(list, {v: list(r) for v, r in exp.ep.items()})
        if add59:
            sc = ["操作:+1", "一の位:9", "十の位:5", "物:時計"]
            e2.add_exception(sc, "一の位", 1)
            e2.add_exception(sc, "十の位", 5)
        for x in range(60):
            y = (x + 1) % 60
            scene = ["操作:+1", f"一の位:{M.top(Bc[x], '一の位')}", f"十の位:{M.top(Bc[x], '十の位')}"] + (["物:時計"] if add59 else [])
            sh = {v: e2.recall(scene, v) for v in ("一の位", "十の位")}
            p = ((M.top(Bc[x], "十の位") + sh["十の位"]) % 10) * 10 + (M.top(Bc[x], "一の位") + sh["一の位"]) % 10
            c = clock[("59 → 0 を 1 文足す" if add59 else "数の経験のまま") + "|" + ("59 分" if x == 59 else "ほか")]
            c[0] += p == y
            c[1] += 1
    f = lambda d: {k: (v[0] / v[1], v[1]) for k, v in d.items()}
    return dict(add=f(add), sub=f(sub), clock=f(clock),
                board_read_acc=float(np.mean([M.top(B[x], "一の位") == x % 10 and M.top(B[x], "十の位") == x // 10 for x in nums])))


# ---------------------------------------------------------------- B. 小さな世界: 動詞が変える観点を操作の文から学ぶ
def op_sentences(I1, seed, n_op):
    """操作の文: (語, 動詞, 前の観点 {観点: 値}, 後の観点 {観点: 値})。前後とも、言うのは観点の一部だけ。"""
    rng = random.Random(seed + 1234)
    out = []
    food = [s for s, (w, at) in I1.SENSES.items() if "味" in at]
    views = ["色", "状態", "形", "味", "種類"]

    def say(at, must):
        vs = [v for v in views if v in at and v != must]
        pick = [must] + rng.sample(vs, rng.randint(1, min(3, len(vs))))
        return {v: rng.choice(at[v]) if v != "味" else "+".join(at["味"]) for v in pick}

    for s in food:
        w, at = I1.SENSES[s]
        for _ in range(n_op):                                       # 塗る: 色だけ変わる
            c0 = rng.choice(at["色"])
            c1 = rng.choice([c for c in I1.VIEWS["色"] if c != c0])
            pre = say(at, "色")
            pre["色"] = c0
            post = say(at, "色")
            post["色"] = c1
            out.append((w, "塗る", pre, post))
        ri = [r for r, (ww, a2) in I1.SENSES.items() if ww == w and a2.get("状態") == ["完熟"]]
        for _ in range(n_op):                                       # 熟させる: 未熟なら完熟の姿に、完熟なら変わらない
            a1 = I1.SENSES[ri[0]][1] if at["状態"] == ["未熟"] and ri else at
            pre = say(at, "状態")
            post = say(a1, "状態")
            if "色" in pre and "色" in post and len(at["色"]) > 1 and a1 is at:   # 同じ物の 2 色（リンゴ）は、同じ色で言う
                post["色"] = pre["色"]
            out.append((w, "熟させる", pre, post))
    return out


def verb_learner(I1, ops, order):
    """語ごとに（1 語抜き）経験の器を作り、I1.VERB_VIEWS の関数を返す。order = 単独 / 組 / 単独+組 / 動詞だけ。"""
    cache = {}
    views = ["色", "状態", "形", "味"]

    def build(word):
        e = ComboExp("単独" if order == "動詞だけ" else order)
        for w, verb, pre, post in ops:
            if w == word:
                continue
            scene = [f"動詞:{verb}"] + ([] if order == "動詞だけ" else [f"{v}:{pre[v]}" for v in ("色", "状態") if v in pre])
            for v in views:
                if v in pre and v in post:
                    e.add(scene, v, "変わる" if pre[v] != post[v] else "変わらない")
        return e

    def f(q, B, r0, L):
        if q["word"] not in cache:
            cache[q["word"]] = build(q["word"])
        e = cache[q["word"]]
        verb = "塗る" if q["kind"] == "塗る" else "熟させる"
        scene = [f"動詞:{verb}"]
        if order != "動詞だけ":
            scene.append(f"色:{q['src'][0][1]}")
            st = I1.view_norm(B, r0, "状態", L) if B.known(r0, "状態", L) else None
            if st is not None:
                scene.append(f"状態:{I1.VIEWS['状態'][int(np.argmax(st))]}")
        out = {v for v in views if e.recall(scene, v) == "変わる"}          # 経験が無い観点は「変わる」と言わない
        f.log[f"{q['word']}|{' '.join(scene)}"] = sorted(out)
        return out

    f.cache, f.log = cache, {}
    return f


def part_b(seed, n_train, n_op, conds):
    import i01_small_world as I1
    import i03_reverse_split as I3
    ops = op_sentences(I1, seed, n_op)
    out = {}
    for name in conds:
        I1.VERB_RULE, I1.VERB_VIEWS = name != "規則なし", None
        f = None
        if name.startswith("学ぶ:"):
            f = verb_learner(I1, ops, name.split(":")[1])
            I1.VERB_VIEWS = f
        I3.CHG, I3.SAME, I3.COMP, I3.DARK = 0.5, 0.2, 0.5, 0.2
        I3.MIN_REF, I3.LOOSE = 5.0, True
        rec = I3.run(argparse.Namespace(seed=seed, n_train=n_train, min_support=0.0, min_ref=5.0, loose=1, steps=800, max_depth=3))
        o = dict(qa=rec["qa"]["deny|全部（読み直し）"], senses_found=rec["senses_found"])
        if f is not None:                                           # 学んだ規則の中身（語・場面 → 変わると思い出した観点）
            o["rules"] = f.log
        out[name] = o
        I1.VERB_VIEWS = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--part", default="A")
    ap.add_argument("--n_sent", type=int, default=10)
    ap.add_argument("--n_train", type=int, default=10)
    ap.add_argument("--n_op", type=int, default=3)
    ap.add_argument("--conds", default="手の規則,規則なし,学ぶ:動詞だけ,学ぶ:単独,学ぶ:組,学ぶ:単独+組,学ぶ:段々,学ぶ:操作ごと")
    ap.add_argument("--out", default="../Project_Kotoyui_private/intervention/results/i18.jsonl")
    a = ap.parse_args()
    torch.set_num_threads(1)
    t0 = time.perf_counter()
    if a.part == "A":
        rec = dict(exp="I18A", seed=a.seed, n_sent=a.n_sent, res={o: part_a(a.seed, a.n_sent, o) for o in ("単独", "組", "単独+組", "段々", "操作ごと")},
                   hole={o: part_a(a.seed, a.n_sent, o, hole=True) for o in ("単独", "組", "単独+組", "段々", "操作ごと")})
    else:
        rec = dict(exp="I18B", seed=a.seed, n_train=a.n_train, n_op=a.n_op, res=part_b(a.seed, a.n_train, a.n_op, a.conds.split(",")))
    rec["total_s"] = time.perf_counter() - t0
    print(json.dumps(rec, ensure_ascii=False), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
