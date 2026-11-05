"""介入の研究 実験I9: 経験の結論を、規則ではなく「答えが合ったか」で付ける（成功・失敗から経験を育てる）

I8・I8b の経験の結論（本物・例外あり・見かけ）は、デジタルの規則（I6 の見直し）が付けていた。持ち主の「失敗や反映で経験が増える」に近づけるため、
世界 A では、原因の判断を色・熟し具合・形の 3 つについて全部の組み合わせ（3 × 3 × 3 = 27 通り）で試し、その語の問いに答えて採点を受け
（世界 A の問いの答えは分かっている = 先生の採点）、いちばん多く合った組み合わせの結論を経験にする。規則は使わない。
  - 場面の記号は I8 と同じ（観点・支持・反例・共に変わる・相手の反例・割合）。支持が 0 の観点は「不明」。
  - 同点の組み合わせが複数あるときは、「見かけ」を多く含むもの（原因を少なく見積もる）を選ぶ。
  - 分けるかの経験は I8 と同じく規則の答え（ここでは原因の判断だけを採点で付ける）。
  - --aggregate 1: 同じ場面で、語によって「本物」（熟すと味が変わる果物）と「見かけ」（ウメ）に分かれたら、その場面の経験を「例外あり」にまとめる（2026-11-05 追加）。
  - 器は I8b の labeled（口ごとに 1 種類ずつ）。世界 B（I8 と同じ）で、判断を器に 1 回通す。
比べ: I8b（結論を規則で付けた器）、規則そのもの。
結果: results/i09.jsonl
実行: CUDA_VISIBLE_DEVICES= python intervention/i09_feedback.py --seed 0
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
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i01_small_world as I1  # noqa: E402
import i03_reverse_split as I3  # noqa: E402
import i08_light_judge as I8  # noqa: E402
from i05_more_words import NAMES, FIXED  # noqa: E402
from i06_revise import PLUMS  # noqa: E402
from i07_experience import fruits, plums, all_pairs  # noqa: E402

CONCL = ["本物", "例外あり", "見かけ"]
TIE = "appearance"


def effect_of(lab, ns, nr):
    return 1.0 if lab == "本物" else (ns / max(ns + nr, 1) if lab == "例外あり" else 0.0)


def collect_feedback(seed):
    eps, agree = [], []
    for i, (PA, nA) in enumerate([(0, 50), (2, 50), (6, 50), (0, 10), (2, 10), (6, 10)]):
        rng = random.Random(4000 + seed * 10 + i)
        S = dict(FIXED)
        S.update(fruits(NAMES[:6], rng))
        S.update(plums(PLUMS[:PA], rng))
        I8.SPLIT_LOG = []
        B, _ = I8.build_world(S, ["リンゴ"] + NAMES[:6] + PLUMS[:PA], 100 + seed * 10 + i, nA)
        for r in I8.SPLIT_LOG:
            eps.append(r["scene"] + ["結論:" + ("分ける" if r["rule"] else "分けない")])
        Q = I1.questions()
        for key in sorted(set(B.word)):
            qs = [q for q in Q if q["word"] == key and q["kind"] != "見たまま"]
            if not qs:
                continue
            P = all_pairs(B, key)
            scenes = {v: I8.cause_scene(P, v) for v in I1.CAUSE_VIEWS}
            scores = {}
            for combo in itertools.product(CONCL, repeat=len(I1.CAUSE_VIEWS)):
                e = {v: (0.0 if scenes[v][1] == 0 else effect_of(lab, scenes[v][1], scenes[v][2])) for v, lab in zip(I1.CAUSE_VIEWS, combo)}
                scores[combo] = sum(I1.answer(B, q, e) == q["ans"] for q in qs)
            top = max(scores.values())
            if TIE == "appearance":                                 # 1 回目: 採点が同じなら「見かけ」の多いほう
                best = (None, max((c for c in scores if scores[c] == top), key=lambda c: sum(l == "見かけ" for l in c)))
            else:                                                   # unknown: 採点で決まらない観点は「不明」（経験にしない判断を残す）
                combo = []
                for i in range(len(I1.CAUSE_VIEWS)):
                    labs = {c[i] for c in scores if scores[c] == top}
                    combo.append(next(iter(labs)) if len(labs) == 1 else ("不明" if {"本物", "見かけ"} <= labs else sorted(labs)[0]))
                best = (None, tuple(combo))
            rev = I3.effects(P, "revise")
            for v, lab in zip(I1.CAUSE_VIEWS, best[1]):
                scene, ns, nr = scenes[v]
                lab = "不明" if ns == 0 else lab
                eps.append(scene + ["結論:" + lab])
                agree.append(lab == I8.cause_label(rev, v, ns, nr))
    return eps, float(np.mean(agree))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--KB", type=int, default=12)
    ap.add_argument("--out", default="../Project_Kotoyui_private/intervention/results/i09.jsonl")
    ap.add_argument("--cache", default="../Project_Kotoyui_private/intervention/results/i09_episodes_s{seed}.json")
    ap.add_argument("--tie", default="appearance", choices=["appearance", "unknown"],
                    help="採点が同じ組み合わせが複数あるとき: appearance = 見かけの多いほう、unknown = 決まらない観点は不明（世界 B では見たままを信じる）")
    ap.add_argument("--aggregate", type=int, default=0, help="同じ場面で語により結論が分かれたら「例外あり」にまとめる")
    a = ap.parse_args()
    torch.set_num_threads(2)
    global TIE
    TIE = a.tie
    I8.UNKNOWN_TRUST = a.tie == "unknown"
    if a.tie == "unknown":
        a.cache = a.cache.replace(".json", "_unknown.json")
    I3.MIN_REF, I3.LOOSE = 5.0, True
    t0 = time.perf_counter()
    cache = a.cache.format(seed=a.seed)
    if os.path.exists(cache):
        d = json.load(open(cache, encoding="utf-8"))
        eps, agree_rule = d["eps"], d["agree_rule"]
    else:
        eps, agree_rule = collect_feedback(a.seed)
        json.dump(dict(eps=eps, agree_rule=agree_rule), open(cache, "w", encoding="utf-8"), ensure_ascii=False)
    if a.aggregate:                                                 # 同じ場面で、語によって「本物」と「見かけ」が分かれたら、その場面は「例外あり」
        by = defaultdict(set)
        for e in eps:
            by[tuple(sorted(t for t in e if not t.startswith("結論:")))].add(next(t for t in e if t.startswith("結論:")))
        new = []
        for e in eps:
            sc = tuple(sorted(t for t in e if not t.startswith("結論:")))
            if {"結論:本物", "結論:見かけ"} <= by[sc]:
                e = list(sc) + ["結論:例外あり"]
            new.append(e)
        eps = new
    mem = I8.Memory(eps, a.seed, how="labeled")
    labels = Counter(next(t for t in e if t.startswith("結論:")) for e in eps)
    for PB in (0, 2):
        for nB in (10, 50):
            rng = random.Random(5000 + a.seed * 10 + PB + nB)               # I8 と同じ世界 B
            S = fruits(NAMES[12:12 + a.KB], rng)
            S.update(plums(PLUMS[2:2 + PB], rng))
            qwords = NAMES[12:12 + a.KB] + PLUMS[2:2 + PB]
            I8.MEM = mem
            I8.SPLIT_LOG = []
            B, found = I8.build_world(S, qwords, a.seed + 200, nB)
            Q = I1.questions()
            per = defaultdict(lambda: [0, 0])
            effs, labs = {}, []
            for q in Q:
                if q["word"] not in effs:
                    effs[q["word"]] = I8.light_effects(all_pairs(B, q["word"]), mem, labs)
                per[q["kind"]][0] += I1.answer(B, q, effs[q["word"]]) == q["ans"]
                per[q["kind"]][1] += 1
            I8.MEM = None
            rec = dict(exp="I9", tie=a.tie, aggregate=a.aggregate, seed=a.seed, KB=a.KB, PB=PB, nB=nB, n_episodes=len(eps), labels=dict(labels), agree_with_rule_in_A=agree_rule,
                       light=dict({k: v[0] / v[1] for k, v in per.items()}, senses_found=found, n_senses=len(S), cause_labels=dict(Counter(labs))),
                       total_s=time.perf_counter() - t0)
            print(json.dumps(rec, ensure_ascii=False), flush=True)
            with open(a.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
