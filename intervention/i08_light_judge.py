"""介入の研究 実験I8: 判断（原因か・分けるか）を、経験の器に 1 回通す光の判断に置き換える

持ち主（2026-11-04）: 「経験の器は失敗や反映でどんどん種類が増える、むしろ増やしてほしい。判断の多くはデジタル重みで持っているが、
  光に一度通すだけのきっかけになれば判断の重みを減らせる。同じ属性に拘る経験は光で融合できれば、判断にかかる処理を低減できないか」。
方法:
  1. 経験を集める（世界 A を 6 つ: 果物 6・ウメ型 0/2/6・文 10/50。I3 の最終形 + I6 の見直しで学び、判断のたびに経験を 1 つ足す）。
     経験の種類:
       原因の判断: 場面 = {観点:v, 支持:なし/少し/多い, 反例:なし/1/多い, 共に変わる:あり/なし, 相手の反例:なし/あり, 割合:低/中/高}
                    結論 = {結論:本物 / 結論:例外あり / 結論:見かけ / 結論:不明}（I6 の見直しの規則が出した答え）
       分けるかの判断: 場面 = {競合:0/1/2+, 片方だけ:0/1+, 文:少/多} 結論 = {結論:分ける / 結論:分けない}（I3 の判断の規則が出した答え）
  2. 経験の器 = 口 8 つの光の器（I2 の mix）。経験の組を文と同じように学ぶ。同じ観点・同じ場面の経験は同じ口に融合するかを見る。
  3. 世界 B（A に出てこない果物 12・ウメ型 0/2。文 10/50、6 seed）で、判断を全部「場面を器に 1 回通して、いちばん明るい結論」にする:
       分けるか → 器、原因か → 器（本物 = 1、例外あり = 支持の割合、見かけ・不明 = 0）。
     比べ: デジタルの規則（I3 の判断 + I6 の見直し）。
  物差し: 問いの正解率、規則との一致率、判断 1 回あたりのデジタルの比べの数（規則）と光を通す回数（器）。
結果: results/i08.jsonl
実行: CUDA_VISIBLE_DEVICES= python intervention/i08_light_judge.py --seed 0
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
import i01_small_world as I1  # noqa: E402
import i03_reverse_split as I3  # noqa: E402
from i02_reverse import PortDevice  # noqa: E402
from i05_more_words import NAMES, FIXED, use_senses  # noqa: E402
from i06_revise import PLUMS  # noqa: E402
from i07_experience import fruits, plums, all_pairs  # noqa: E402

VAL0 = 8
SPLIT_LOG = []                                                      # 分けるかの判断の記録（経験集めのとき）
MEM = None                                                          # 世界 B で判断に使う器


def b3(x, cuts, names):
    for c, n in zip(cuts, names):
        if x <= c:
            return n
    return names[-1]


def cause_scene(P, v):
    V = I1.CAUSE_VIEWS
    sup = [d for _, d, dt, _, _ in P if d[v] >= 0.5 and dt >= 0.5]
    ref = [d for _, d, dt, _, _ in P if d[v] >= 0.5 and dt < 0.2]
    co = Counter(u for d in sup for u in V if u != v and d[u] >= 0.5)
    partner = co.most_common(1)[0][0] if co and co.most_common(1)[0][1] >= 0.5 * len(sup) else None
    pref = sum(1 for _, d, dt, _, _ in P if partner and d[partner] >= 0.5 and dt < 0.2) if partner else 0
    ratio = len(sup) / max(len(sup) + len(ref), 1)
    return [f"観点:{v}", "支持:" + b3(len(sup), [0, 2], ["なし", "少し", "多い"]), "反例:" + b3(len(ref), [0, 1], ["なし", "1", "多い"]),
            "共に変わる:" + ("あり" if partner else "なし"), "相手の反例:" + ("あり" if pref else "なし"),
            "割合:" + b3(ratio, [0.5, 0.8], ["低", "中", "高"])], len(sup), len(ref)


def cause_label(e_rev, v, n_sup, n_ref):
    if n_sup == 0:
        return "不明"
    if e_rev[v] == 0:
        return "見かけ"
    return "例外あり" if n_ref else "本物"


def split_scene(comp, excl, n):
    return ["競合:" + b3(len(comp), [0, 1], ["0", "1", "2+"]), "片方だけ:" + ("0" if not excl else "1+"), "文:" + ("少" if n < 30 else "多")]


# I3 の判断を差し替える（経験集めでは記録、世界 B では器で判断）
_orig_judge = I3.judge


def judge_hook(W, post, sents, min_support):
    comp, excl, sp = _orig_judge(W, post, sents, min_support)
    scene = split_scene(comp, excl, len(sents))
    if MEM is not None:
        concl = MEM.recall(scene, ["分ける", "分けない"])
        light_sp = concl["分ける"] > concl["分けない"]
        SPLIT_LOG.append(dict(rule=bool(sp), light=bool(light_sp)))
        return comp, excl, light_sp
    SPLIT_LOG.append(dict(scene=scene, rule=bool(sp)))
    return comp, excl, sp


I3.judge = judge_hook


class Memory:
    def __init__(self, eps, seed, M=8, steps=1000, restarts=3):
        self.vocab = sorted({t for e in eps for t in e})
        self.idx = {t: i for i, t in enumerate(self.vocab)}
        X = torch.zeros(len(eps), len(self.vocab))
        for i, e in enumerate(eps):
            for t in e:
                X[i, self.idx[t]] = 1
        best = None
        for r in range(restarts):                                   # I7 の「口に分かれず混ざる」対策: 数回学んで、起こりやすさの最も高いものを使う
            torch.manual_seed(seed * 10 + r)
            dev = PortDevice(M, n=64, seed=seed * 10 + r)
            opt = torch.optim.Adam(dev.parameters(), lr=0.02)
            for _ in range(steps):
                W = dev.forward_field(dev.ports).abs()[:, VAL0:VAL0 + len(self.vocab)] ** 2
                loss = -(torch.logsumexp(X @ torch.log(W + 1e-9).T, 1) - math.log(M)).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            if best is None or float(loss.detach()) < best[0]:
                with torch.no_grad():
                    best = (float(loss.detach()), (dev.forward_field(dev.ports).abs()[:, VAL0:VAL0 + len(self.vocab)] ** 2).numpy())
        self.loss, self.W = best
        self.n_pass = 0

    def recall(self, scene, labels):
        """場面を 1 回通す: 口ごとの起こりやすさで重ねた明るさから、結論の明るさを読む。"""
        self.n_pass += 1
        ids = [self.idx[t] for t in scene if t in self.idx]
        Wn = self.W / self.W.sum(1, keepdims=True)
        ll = np.log(Wn[:, ids] + 1e-6).sum(1)
        post = np.exp(ll - ll.max())
        post /= post.sum()
        out = post @ Wn
        return {l: float(out[self.idx[f"結論:{l}"]]) if f"結論:{l}" in self.idx else 0.0 for l in labels}

    def ports_summary(self):
        """口ごとに明るい記号（融合の様子）。"""
        Wn = self.W / self.W.sum(1, keepdims=True)
        return [[self.vocab[j] for j in np.argsort(-Wn[m])[:6]] for m in range(len(Wn))]


def light_effects(P, mem, stats):
    e = {}
    for v in I1.CAUSE_VIEWS:
        scene, ns, nr = cause_scene(P, v)
        c = mem.recall(scene, ["本物", "例外あり", "見かけ", "不明"])
        lab = max(c, key=c.get)
        e[v] = 1.0 if lab == "本物" else (ns / max(ns + nr, 1) if lab == "例外あり" else 0.0)
        stats.append(lab)
    return e


def build_world(S, qwords, seed, n_train):
    use_senses(S, qwords)
    a = argparse.Namespace(seed=seed, n_train=n_train, min_support=0.0, min_ref=5.0, loose=1, steps=800, max_depth=3)
    train, test = I1.make_world(seed, n_train=n_train)
    leaves = []
    for w in I1.WORDS:
        ss = [(s, toks) for s, toks in train if I1.SENSES[s][0] == w]
        sents = [sorted(I1.SLOT[x] for x in I1.attrs_of(toks, w)) for _, toks in ss]
        leaves += I3.build(w, sents, [s for s, _ in ss], a)
    B = I3.LeafBoards(leaves)
    return B, len({m.most_common(1)[0][0] for m in B.members})


def collect(seed):
    """世界 A を 6 つ学んで経験を集める。"""
    global SPLIT_LOG
    eps = []
    for i, (PA, nA) in enumerate([(0, 50), (2, 50), (6, 50), (0, 10), (2, 10), (6, 10)]):
        rng = random.Random(4000 + seed * 10 + i)
        S = dict(FIXED)
        S.update(fruits(NAMES[:6], rng))
        S.update(plums(PLUMS[:PA], rng))
        SPLIT_LOG = []
        B, _ = build_world(S, ["リンゴ"] + NAMES[:6] + PLUMS[:PA], 100 + seed * 10 + i, nA)
        for r in SPLIT_LOG:
            eps.append(r["scene"] + ["結論:" + ("分ける" if r["rule"] else "分けない")])
        for key in [None] + sorted(set(B.word)):                    # 1 語抜きの場面も経験にする（世界 B でも 1 語抜きで判断するため）
            P = all_pairs(B, key)
            rev = I3.effects(P, "revise")
            for v in I1.CAUSE_VIEWS:
                scene, ns, nr = cause_scene(P, v)
                eps.append(scene + ["結論:" + cause_label(rev, v, ns, nr)])
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--KB", type=int, default=12)
    ap.add_argument("--out", default="intervention/results/i08.jsonl")
    a = ap.parse_args()
    global MEM, SPLIT_LOG
    torch.set_num_threads(2)
    I3.MIN_REF, I3.LOOSE = 5.0, True
    t0 = time.perf_counter()
    eps = collect(a.seed)
    mem = Memory(eps, a.seed)
    kinds = Counter(" ".join(t for t in e if t.startswith(("観点:", "競合:"))) or "?" for e in eps)
    t_mem = time.perf_counter() - t0
    for PB in (0, 2):
        for nB in (10, 50):
            rng = random.Random(5000 + a.seed * 10 + PB + nB)
            S = fruits(NAMES[12:12 + a.KB], rng)
            S.update(plums(PLUMS[2:2 + PB], rng))
            qwords = NAMES[12:12 + a.KB] + PLUMS[2:2 + PB]
            rec = dict(exp="I8", seed=a.seed, KB=a.KB, PB=PB, nB=nB, n_episodes=len(eps), episode_kinds=dict(kinds), mem_loss=mem.loss,
                       ports=mem.ports_summary())
            Q = None
            for mode in ("rule", "light"):
                MEM = mem if mode == "light" else None
                SPLIT_LOG = []
                mem.n_pass = 0
                B, found = build_world(S, qwords, a.seed + 200, nB)
                Q = I1.questions()
                per = defaultdict(lambda: [0, 0])
                labs, agree = [], []
                for q in Q:
                    key = q["word"]
                    if not hasattr(B, "_eff"):
                        B._eff = {}
                    if key not in B._eff:
                        P = all_pairs(B, key)
                        rule_e = I3.effects(P, "revise")
                        if mode == "light":
                            e = light_effects(P, mem, labs)
                            agree += [(e[v] > 0) == (rule_e[v] > 0) for v in I1.CAUSE_VIEWS]
                        else:
                            e = rule_e
                        B._eff[key] = e
                    ans = I1.answer(B, q, B._eff[key])
                    per[q["kind"]][0] += ans == q["ans"]
                    per[q["kind"]][1] += 1
                rec[mode] = dict({k: v[0] / v[1] for k, v in per.items()}, senses_found=found, n_senses=len(S))
                if mode == "light":
                    rec[mode].update(cause_agree=float(np.mean(agree)), cause_labels=dict(Counter(labs)),
                                     split_agree=float(np.mean([r["rule"] == r["light"] for r in SPLIT_LOG])) if SPLIT_LOG else None,
                                     n_light_pass=mem.n_pass)
            rec["total_s"] = time.perf_counter() - t0
            rec["mem_s"] = t_mem
            print(json.dumps({k: v for k, v in rec.items() if k not in ("ports", "episode_kinds")}, ensure_ascii=False), flush=True)
            os.makedirs(os.path.dirname(a.out), exist_ok=True)
            with open(a.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
