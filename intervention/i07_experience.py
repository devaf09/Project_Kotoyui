"""介入の研究 実験I7: 経験の器 — 前提が崩れた場面を覚え、似た推論で思い出して、否定や「たぶん」を返す

持ち主の案（2026-11-04）: 「これまでの前提が崩れるとき、同じように関係する属性で介入が必要かもしれない、という経験を思い出す光の器。
  似たような状況で、緑だから未熟とは限らない、熟しているから酸っぱくないとは限らない、を重ねて通すと否定や思い出しをする」。
方法:
  1. 世界 A（果物 K_A 個 + 反例のリンゴ + ウメ型 P_A 個）で I3 の最終形を学び、原因の比べ（否定・見直し）で起きたことを「経験」にする。
     経験 1 つ = 記号の組（文と同じ扱い）:
       見かけの関係: {原因候補:色, 結果:味, 共に変わる:状態, 結論:見かけ, 例:リンゴ}（色は否定された。否定の手がかりの語を例にする）
       本物の関係:   {原因候補:状態, 結果:味, 共に変わる:色, 結論:本物}
       例外のある関係: {原因候補:状態, 結果:味, 結論:例外あり, 例:ウメ}（熟し具合を否定する組があったが、見直しで取り消した）
  2. 経験の器 = 口 4 つの光の器（I2 と同じ）。経験の組を文と同じように学ぶ。
     思い出し: いまの推論の形（原因候補・結果・共に変わる観点）を入れ、起こりやすさの最も高い口から出る「結論:…」「例:…」の明るさを読む（重ねて通す）。
  3. 世界 B（世界 A に出てこない果物 K_B 個。反例のリンゴもウメも無い）を、少ない文で学ぶ。
     世界 B の中だけでは、色と熟し具合はいつも一緒に変わるので、どちらが味の原因か決められない（区別できない）。
     経験あり: 区別できない観点 v ごとに器へ {原因候補:v, 結果:味, 共に変わる:u} を通し、「見かけ」が「本物」より明るければ v を否定する。
       「例外あり」が明るければ、その観点を使う答えに「たぶん」を付ける。思い出した例（例:リンゴ など）も記録する。
     経験なし: I3 の最終形のまま（区別できないので、色も熟し具合も原因として残る）。
     比べ: 経験を器ではなく表で引いた場合（デジタルの辞書）。
  物差し: 世界 B の塗る・熟す・見たままの正解率。「たぶん」を付けた答えのうち、正しかった割合と、間違えた答えのうち「たぶん」が付いていた割合。
結果: results/i07.jsonl
実行: CUDA_VISIBLE_DEVICES= python intervention/i07_experience.py --seed 0
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

VAL0 = 8


def fruits(names, rng):
    S = {}
    for name in names:
        shape = rng.choice(["丸い", "長い"])
        ripe_c = rng.choice(["赤", "黄", "橙", "紫", "白"])
        ripe_t = ["甘い", "酸っぱい"] if rng.random() < 0.25 else ["甘い"]
        un_t = [rng.choice(["渋い", "酸っぱい"])]
        if un_t == ["酸っぱい"] and "酸っぱい" in ripe_t:
            un_t = ["渋い"]
        S[f"{name}/完熟"] = (name, {"種類": ["果物"], "色": [ripe_c], "形": [shape], "味": ripe_t, "状態": ["完熟"], "される": ["食べる", "切る"]})
        S[f"{name}/未熟"] = (name, {"種類": ["果物"], "色": ["緑"], "形": [shape], "味": un_t, "状態": ["未熟"], "される": ["切る"]})
    return S


def plums(names, rng):
    S = {}
    for name in names:
        shape = rng.choice(["丸い", "長い"])
        S[f"{name}/完熟"] = (name, {"種類": ["果物"], "色": ["黄"], "形": [shape], "味": ["酸っぱい"], "状態": ["完熟"], "される": ["切る"]})
        S[f"{name}/未熟"] = (name, {"種類": ["果物"], "色": ["緑"], "形": [shape], "味": ["酸っぱい"], "状態": ["未熟"], "される": ["切る"]})
    return S


def learn_world(S, qwords, seed, n_train):
    use_senses(S, qwords)
    a = argparse.Namespace(seed=seed, n_train=n_train, min_support=0.0, min_ref=5.0, loose=1, steps=800, max_depth=3)
    train, test = I1.make_world(seed, n_train=n_train)
    leaves = []
    for w in I1.WORDS:
        ss = [(s, toks) for s, toks in train if I1.SENSES[s][0] == w]
        sents = [sorted(I1.SLOT[x] for x in I1.attrs_of(toks, w)) for _, toks in ss]
        leaves += I3.build(w, sents, [s for s, _ in ss], a)
    return I3.LeafBoards(leaves)


def all_pairs(B, key=None):
    return I3.board_pairs(B, key) + I3.port_pairs(B, key) + I3.inner_pairs(B, key, explain=True)


def episodes_from(B):
    """世界 A の原因の比べから経験を作る。"""
    P = all_pairs(B)
    deny, rev = I3.effects(P, "deny"), I3.effects(P, "revise")
    eps = []
    V = I1.CAUSE_VIEWS
    for v in V:
        sup = [(w, d) for w, d, dt, _, _ in P if d[v] >= 0.5 and dt >= 0.5]
        ref = [w for w, d, dt, _, _ in P if d[v] >= 0.5 and dt < 0.2]
        if not sup:
            continue
        co = Counter(u for _, d in sup for u in V if u != v and d[u] >= 0.5)
        partners = [u for u, c in co.items() if c >= 0.5 * len(sup)]
        for u in partners or [None]:
            base = [f"原因候補:{v}", "結果:味"] + ([f"共に変わる:{u}"] if u else [])
            if rev[v] == 0 and ref:
                for w in sorted(set(ref)):
                    eps.append(base + ["結論:見かけ", f"例:{w}"])
            elif deny[v] == 0 and rev[v] > 0:
                for w in sorted(set(ref)):
                    eps.append(base + ["結論:例外あり", f"例:{w}"])
                eps.append(base + ["結論:本物"])
            else:
                eps.append(base + ["結論:本物"])
    return eps, dict(deny=deny, revise=rev)


class Memory:
    """経験の器: 経験の記号の組を、口 M 個の光の器に学ぶ（I2 の mix と同じ）。"""

    def __init__(self, eps, seed, M=4, steps=800):
        self.vocab = sorted({t for e in eps for t in e} | {f"原因候補:{v}" for v in I1.CAUSE_VIEWS} | {f"共に変わる:{v}" for v in I1.CAUSE_VIEWS}
                            | {"結果:味", "結論:見かけ", "結論:本物", "結論:例外あり"})
        self.idx = {t: i for i, t in enumerate(self.vocab)}
        torch.manual_seed(seed)
        self.dev = PortDevice(M, n=64, seed=seed)
        A = len(self.vocab)
        X = torch.zeros(len(eps), A)
        for i, e in enumerate(eps):
            for t in e:
                X[i, self.idx[t]] = 1
        opt = torch.optim.Adam(self.dev.parameters(), lr=0.02)
        for _ in range(steps):
            W = self.bright()
            loss = -(torch.logsumexp(X @ torch.log(W + 1e-9).T, 1) - math.log(M)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            self.W = self.bright().numpy()

    def bright(self):
        return self.dev.forward_field(self.dev.ports).abs()[:, VAL0:VAL0 + len(self.vocab)] ** 2

    def recall(self, query):
        """いまの推論の形を入れ、最も起こりやすい口の明るさで、結論と例を返す（重ねて通す）。"""
        ids = [self.idx[t] for t in query if t in self.idx]
        Wn = self.W / self.W.sum(1, keepdims=True)
        ll = np.log(Wn[:, ids] + 1e-6).sum(1)
        post = np.exp(ll - ll.max())
        post /= post.sum()
        out = post @ Wn                                             # 口の取り分で重ねた明るさ
        concl = {c: float(out[self.idx[f"結論:{c}"]]) for c in ("見かけ", "本物", "例外あり")}
        ex = sorted(((float(out[i]), t[2:]) for t, i in self.idx.items() if t.startswith("例:")), reverse=True)[:2]
        return concl, ex


def table_recall(eps, query):
    """比べ: 経験を表で引く（問いの記号を全部含む経験の結論を数える）。"""
    c = Counter()
    ex = Counter()
    for e in eps:
        if all(t in e for t in query):
            for t in e:
                if t.startswith("結論:"):
                    c[t[3:]] += 1
                if t.startswith("例:"):
                    ex[t[2:]] += 1
    tot = sum(c.values()) or 1
    return {k: c[k] / tot for k in ("見かけ", "本物", "例外あり")}, [(v, k) for k, v in ex.most_common(2)]


def identifiable(P, v):
    """観点 v の支持の組のうち、ほかの原因候補が変わっていない組があるか（世界の中で区別できるか）。"""
    clean = [w for _, d, dt, _, w in P if d[v] >= 0.5 and dt >= 0.5 and all(d[u] < 0.5 for u in I1.CAUSE_VIEWS if u != v)]
    return len(clean) >= 2 and sum(clean) >= I3.MIN_REF            # 1 つだけの組（口どうしの偶然の割れなど）では区別できたとしない


def answer_world(B, mem, eps, use):
    """世界 B の問いに答える。use: none / light / table。戻り値: 種類ごとの正解率、たぶんの記録、思い出し。"""
    Q = I1.questions()
    per = defaultdict(lambda: [0, 0])
    flags, wrongs = [], []
    recalls = {}
    effs = {}
    for q in Q:
        key = q["word"]
        if key not in effs:
            P = all_pairs(B, key)
            e = I3.effects(P, "revise")
            unsure = set()
            if use != "none":
                for v in I1.CAUSE_VIEWS:
                    if e[v] == 0 or identifiable(P, v):
                        continue
                    co = Counter(u for _, d, dt, _, _ in P if d[v] >= 0.5 and dt >= 0.5 for u in I1.CAUSE_VIEWS if u != v and d[u] >= 0.5)
                    u = co.most_common(1)[0][0] if co else None
                    query = [f"原因候補:{v}", "結果:味"] + ([f"共に変わる:{u}"] if u else [])
                    concl, ex = mem.recall(query) if use == "light" else table_recall(eps, query)
                    recalls[f"{v}←{u}"] = dict(concl={k: round(x, 3) for k, x in concl.items()}, ex=[(round(a, 3), b) for a, b in ex])
                    if concl["見かけ"] > concl["本物"]:
                        e[v] = 0.0
                    if e[v] > 0 and concl["例外あり"] >= 0.5 * max(concl["本物"], concl["見かけ"]):
                        unsure.add(v)                               # 否定しなかった観点で、例外の経験が明るい → 「たぶん」
            effs[key] = (e, unsure)
        e, unsure = effs[key]
        ans = I1.answer(B, q, e)
        ok = ans == q["ans"]
        per[q["kind"]][0] += ok
        per[q["kind"]][1] += 1
        changed = q.get("change", (None,))[0]
        flags.append((q["kind"], bool(ok), changed in unsure))
        if not ok:
            wrongs.append(" ".join(q["toks"]))
    res = {k: v[0] / v[1] for k, v in per.items()}
    flagged = [ok for _, ok, f in flags if f]
    wrong = [f for _, ok, f in flags if not ok]
    res["たぶんの数"] = len(flagged)
    res["たぶんの正解率"] = float(np.mean(flagged)) if flagged else None
    res["間違いのうちたぶん"] = float(np.mean(wrong)) if wrong else None
    res["間違い"] = wrongs
    return res, recalls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--KA", type=int, default=6)
    ap.add_argument("--PA", type=int, default=2)
    ap.add_argument("--KB", type=int, default=6)
    ap.add_argument("--nA", type=int, default=50)
    ap.add_argument("--nB", type=int, default=10)
    ap.add_argument("--out", default="../Project_Kotoyui_private/intervention/results/i07.jsonl")
    a = ap.parse_args()
    torch.set_num_threads(2)
    I3.MIN_REF, I3.LOOSE = 5.0, True
    t0 = time.perf_counter()
    rng = random.Random(3000 + a.seed)
    SA = dict(FIXED)
    SA.update(fruits(NAMES[:a.KA], rng))
    SA.update(plums(PLUMS[:a.PA], rng))
    BA = learn_world(SA, ["リンゴ"] + NAMES[:a.KA] + PLUMS[:a.PA], a.seed, a.nA)
    eps, effA = episodes_from(BA)
    mem = Memory(eps, a.seed)
    SB = fruits(NAMES[12:12 + a.KB], rng)
    BB = learn_world(SB, NAMES[12:12 + a.KB], a.seed + 50, a.nB)
    rec = dict(exp="I7", seed=a.seed, KA=a.KA, PA=a.PA, KB=a.KB, nA=a.nA, nB=a.nB, episodes=[" ".join(e) for e in eps], effects_A=effA,
               senses_found_B=len({m.most_common(1)[0][0] for m in BB.members}), n_senses_B=len(SB))
    for use in ("none", "light", "table"):
        res, recalls = answer_world(BB, mem, eps, use)
        rec[use] = res
        rec[f"recall_{use}"] = recalls
    rec["total_s"] = time.perf_counter() - t0
    print(json.dumps({k: v for k, v in rec.items() if k != "episodes"}, ensure_ascii=False), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
