"""介入の研究 実験I3: 語板の分け方を「口 2 つの板に逆から入れる判断」に置き換え、否定に反例の強さを入れる

I1 の分け方は手の規則（文ごとの違いのしきい値、50 文ごとの見直し・分け直し）だった。I2 で、口 2 つの板を文の組で学ぶと、
境目を教えなくても 2 つの光の経路にまとまり、「競合する観点」を数えると分けるかどうかを判断できると分かった。ここでは:
  1. 語ごとに、その語の文を全部読んで口 2 つの板を学ぶ（I2 の mix）。
  2. 逆から属性を入れて口ごとの明るさを読み、競合する観点・片方だけの観点を数える。文の数による確からしさも入れる（--min_support）:
     口ごとに、その観点を含む文（口の取り分で重みづけ）が min_support 未満なら、その観点は数えない（偶然の偏りを除く）。
  3. 「分ける」なら、文を口の取り分で 2 つに分け、それぞれで 1. からやり直す（割れなくなるまで）。分けなかった板の 2 つの口は「同じ物の幅」として残す。
  4. 介入の問い: I1 と同じ問い・同じ答え方。原因を比べる組 = 同じ語の板どうし、分けなかった板の口どうし、
     1 枚の中（I1 と同じ: 色が 2 つあるのに味は 1 つ。重み = 2 つ目の値を含む文の数）。
     否定に反例の強さを入れる（weighted）: 組ごとの重み = 両側の文の数の小さいほう。重みが --min_ref 以上の反例だけで否定する
     （混ざった 1〜2 文が作る見かけの反例を無視する。支持と多数決はしない: 支持の組は交絡していて、色の支持にはならないため）。
  6. 原因の効き目の出し方（2026-11-04 追加、I6）: lsq = 最小二乗（交絡を説明し尽くしで外す）、revise = 否定の見直し
     （否定した観点を、支持の割合の高い順に見て、その観点でしか説明できない支持の組が反例の組より多ければ否定を取り消す）。
  5. 見かけの反例の見分け（「全部（読み直し）」）: 1 枚の中の反例を作る 2 つ目の値が、同じ語のほかの板の主な値（観点の明るさの半分以上）なら、
     その板からの混ざりとして、その反例は使わない（2026-11-04 追加。文ごとの起こりやすさで読み直す形は、混ざった文を学んだ板のほうが
     起こりやすくなってしまい効かなかった）。
物差しは I1 と同じ（分かれ方・振り分け・違和感・介入の問い・見たままの問い）。振り分けと違和感は、板の口ごとの明るさで文の起こりやすさを測って行う。
結果: results/i03.jsonl
実行: CUDA_VISIBLE_DEVICES= python intervention/i03_reverse_split.py --seed 0
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
from i01_small_world import SENSES, SLOTS, SLOT, VIEWS, WORDS, attrs_of, make_world, determined, corrupt, auroc, questions  # noqa: E402
from i02_reverse import PortDevice, VAL0  # noqa: E402


def fit_board(sents, seed, steps, M=2):
    """文（属性の番号の組）から口 M 個の板を学ぶ。戻り値: 明るさ (M, A)、文ごとの口の取り分 (文, M)。"""
    torch.manual_seed(seed)
    dev = PortDevice(M, seed=seed)
    X = torch.zeros(len(sents), len(SLOTS))
    for i, s in enumerate(sents):
        for j in s:
            X[i, j] += 1
    opt = torch.optim.Adam(dev.parameters(), lr=0.02)
    for _ in range(steps):
        W = dev.bright()
        ll = X @ torch.log(W + 1e-9).T
        loss = -(torch.logsumexp(ll, 1) - math.log(M)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        W = dev.bright()
        ll = X @ torch.log(W + 1e-9).T
        post = torch.softmax(ll, 1)
    return W.numpy(), post.numpy()


def judge(W, post, sents, min_support):
    """競合する観点・片方だけの観点（口 2 つ）。文の数の確からしさつき。"""
    comp, excl = [], []
    for v, xs in VIEWS.items():
        ids = [SLOT[(v, x)] for x in xs]
        has = np.array([any(j in ids for j in s) for s in sents], dtype=float)
        sup = post.T @ has                                          # 口ごとの、その観点を含む文の数（取り分で重みづけ）
        s0, s1 = W[0, ids].sum(), W[1, ids].sum()
        h0 = s0 >= 0.05 * W[0].sum() and sup[0] >= min_support
        h1 = s1 >= 0.05 * W[1].sum() and sup[1] >= min_support
        if h0 and h1:
            if 0.5 * np.abs(W[0, ids] / s0 - W[1, ids] / s1).sum() >= 0.5:
                comp.append(v)
        elif h0 != h1 and min(s0 / W[0].sum(), s1 / W[1].sum()) < 0.01:
            excl.append(v)
    return comp, excl, len(comp) >= 2 or (len(comp) >= 1 and len(excl) >= 1)


class Leaf:
    def __init__(self, word, sents, senses, W, post):
        self.word, self.sents, self.senses, self.W, self.post = word, sents, senses, W, post


def build(word, sents, senses, a, depth=0, log=None):
    W, post = fit_board(sents, a.seed * 100 + depth, a.steps)
    comp, excl, sp = judge(W, post, sents, a.min_support)
    if log is not None:
        log.append(dict(word=word, depth=depth, n=len(sents), compete=comp, exclusive=excl, split=bool(sp),
                        members=[dict(Counter(s for s, p in zip(senses, post[:, m]) if p > 0.5)) for m in range(2)]))
    if sp and depth < a.max_depth:
        g = post.argmax(1)
        parts = [[i for i in range(len(sents)) if g[i] == m] for m in range(2)]
        if all(len(p) >= 2 for p in parts):
            out = []
            for p in parts:
                out += build(word, [sents[i] for i in p], [senses[i] for i in p], a, depth + 1, log)
            return out
    return [Leaf(word, sents, senses, W, post)]


class LeafBoards(I1.Boards):
    """I1 の Boards の読み方（route・diff・known・view_dist）を、葉の板の明るさ（口の和）で使う。"""

    def __init__(self, leaves):
        self.mode, self.dark, self.theta = "digital", 0.2, 0.5
        self.leaves = leaves
        self.word = [l.word for l in leaves]
        self.counts = [l.W.sum(0) * 1000 + 1e-9 for l in leaves]     # light() はこれを正規化して読む（= 光の明るさ）
        self.members = [Counter(l.senses) for l in leaves]
        self._cache = None


def loglik(leaf, attrs):
    """板の口ごとの明るさでの、文の属性の組の起こりやすさ（属性あたり、口の最大）。"""
    ids = [SLOT[x] for x in attrs]
    if not ids:
        return 0.0
    W = leaf.W / leaf.W.sum(1, keepdims=True)
    return float(max(np.log(W[m, ids] + 1e-6).sum() for m in range(W.shape[0])) / len(ids))


def port_pairs(B, exclude_word=None):
    """分けなかった板の口どうしを、原因を比べる組にする（同じ物の幅: 色違いのリンゴなど）。重み = 小さいほうの口の文の数。"""
    out = []
    for leaf in B.leaves:
        if leaf.word == exclude_word:
            continue
        W = leaf.W
        nd = {}
        ok = True
        for v in I1.CAUSE_VIEWS + ["味"]:
            ids = [SLOT[(v, x)] for x in VIEWS[v]]
            s0, s1 = W[0, ids].sum(), W[1, ids].sum()
            if s0 < 0.05 * W[0].sum() or s1 < 0.05 * W[1].sum():
                if v == "味":
                    ok = False
                nd[v] = 0.0
                continue
            nd[v] = 0.5 * np.abs(W[0, ids] / s0 - W[1, ids] / s1).sum()
        if not ok:
            continue
        w = float(min(leaf.post.sum(0)))
        out.append((leaf.word, {v: nd[v] for v in I1.CAUSE_VIEWS}, nd["味"], "口どうし", w))
    return out


def inner_pairs(B, exclude_word=None, explain=False):
    """1 枚の中（I1 と同じ）: 原因の観点に 2 つの値があり、味は 1 つ。重み = 2 つ目に多い値を含む文の数（反例の強さ）。"""
    out = []
    L = B.light()
    for r, leaf in enumerate(B.leaves):
        if leaf.word == exclude_word or not B.known(r, "味", L):
            continue
        tn = I1.view_norm(B, r, "味", L)
        if np.sort(tn)[-2] >= B.dark * tn.max():
            continue
        for v in I1.CAUSE_VIEWS:
            if not B.known(r, v, L):
                continue
            cnt = Counter(j for snt in leaf.sents for j in snt if SLOTS[j][0] == v)
            if len(cnt) < 2:
                continue
            others = [u for u in I1.CAUSE_VIEWS if u != v and B.known(r, u, L)
                      and len(Counter(j for snt in leaf.sents for j in snt if SLOTS[j][0] == u)) >= 2]
            if others:
                continue
            w = float(sorted(cnt.values())[-2])
            if explain:                                             # 2 つ目の値が、同じ語のほかの板の「主な値」なら、その板からの混ざりとして使わない
                j2 = sorted(cnt, key=cnt.get)[-2]
                ids = [SLOT[(v, x)] for x in VIEWS[v]]
                mixed = False
                for r2, o in enumerate(B.leaves):
                    if o is leaf or o.word != leaf.word or not B.known(r2, v, L):
                        continue
                    b2 = L[r2, ids] / L[r2, ids].sum()
                    if b2[ids.index(j2)] >= 0.5:
                        mixed = True
                if mixed:
                    continue
            out.append((leaf.word, {u: (1.0 if u == v else 0.0) for u in I1.CAUSE_VIEWS}, 0.0, "1 枚の中", w))
    return out


def board_pairs(B, exclude_word=None):
    """同じ語の板どうし（I1 の語板どうしと同じ）。重み = 小さいほうの板の文の数。"""
    L = B.light()
    out = []
    need = I1.CAUSE_VIEWS if not LOOSE else ["色"]
    idx = [r for r, w in enumerate(B.word) if w != exclude_word and B.known(r, "味", L) and all(B.known(r, v, L) for v in need)]
    for i, r1 in enumerate(idx):
        for r2 in idx[i + 1:]:
            if B.word[r1] != B.word[r2]:
                continue
            d = {v: 0.5 * np.abs(I1.view_norm(B, r1, v, L) - I1.view_norm(B, r2, v, L)).sum()
                 if B.known(r1, v, L) and B.known(r2, v, L) else 0.0 for v in I1.CAUSE_VIEWS}   # 光の届かない観点は差 0 とみなす（--loose）
            dt = 0.5 * np.abs(I1.view_norm(B, r1, "味", L) - I1.view_norm(B, r2, "味", L)).sum()
            out.append((B.word[r1], d, dt, "板どうし", float(min(len(B.leaves[r1].sents), len(B.leaves[r2].sents)))))
    return out


MIN_REF = 5.0
LOOSE = False


def effects(P, how):
    if how == "corr":
        return {v: 1.0 for v in I1.CAUSE_VIEWS}
    if not P:
        return {v: 0.0 for v in I1.CAUSE_VIEWS}
    if how == "lsq":                                                # 最小二乗（Sin3 実験63 と同じ）: 味の差 ≈ Σ 効き目 × 観点の差。説明し尽くしで交絡を外す
        X = np.array([[d[v] for v in I1.CAUSE_VIEWS] for _, d, _, _, _ in P])
        y = np.array([dt for _, _, dt, _, _ in P])
        co = np.linalg.lstsq(X, y, rcond=None)[0]
        return {v: float(np.clip(co[i], 0, 1)) for i, v in enumerate(I1.CAUSE_VIEWS)}
    if how == "revise":                                             # 否定の見直し: まず否定し、否定した観点を「支持 / (支持 + 反例)」の高い順に見直す。
        e = effects(P, "deny")                                      # 観点 v でしか説明できない支持の組（いま認めている観点では説明できない、味の変わった組）が
        cand = []                                                   # 反例の組より多ければ、否定を取り消す（効き目 = 支持 / (支持 + 反例)）。順に見るのは、
        for v in I1.CAUSE_VIEWS:                                    # 色と熟し具合のように一緒に変わる観点で、両方を取り消さないため（説明し尽くし）。
            if e[v] > 0:
                continue
            sup = sum(1 for _, d, dt, _, _ in P if d[v] >= 0.5 and dt >= 0.5)
            ref = sum(1 for _, d, dt, _, _ in P if d[v] >= 0.5 and dt < 0.2)
            if sup:
                cand.append((sup / (sup + ref), v, ref))
        for ratio, v, ref in sorted(cand, reverse=True):
            only = sum(1 for _, d, dt, _, _ in P if d[v] >= 0.5 and dt >= 0.5
                       and not any(e[u] > 0 and d[u] >= 0.5 for u in I1.CAUSE_VIEWS if u != v))
            if only > ref:
                e[v] = ratio
        return e
    e = {}
    for v in I1.CAUSE_VIEWS:
        ref = [w for _, d, dt, _, w in P if d[v] >= 0.5 and dt < 0.2]
        sup = [w for _, d, dt, _, w in P if d[v] >= 0.5 and dt >= 0.5]
        if how == "deny":                                           # 1 つでも反例があれば否定（I1 と同じ）
            e[v] = 1.0 if (sup and not ref) else 0.0
        else:                                                       # weighted: 反例そのものが確か（両側に文が MIN_REF 以上）なときだけ否定
            e[v] = 1.0 if (sup and not [w for w in ref if w >= MIN_REF]) else 0.0
    return e


def run(a):
    t0 = time.perf_counter()
    train, test = make_world(a.seed, n_train=a.n_train)
    log, leaves = [], []
    for w in WORDS:
        ss = [(s, toks) for s, toks in train if SENSES[s][0] == w]
        sents = [sorted(SLOT[x] for x in attrs_of(toks, w)) for _, toks in ss]
        leaves += build(w, sents, [s for s, _ in ss], a, log=log)
    B = LeafBoards(leaves)
    rec = dict(exp="I3", seed=a.seed, n_train_per_sense=a.n_train, min_support=a.min_support, min_ref=a.min_ref, loose=a.loose, steps=a.steps, n_boards=len(leaves),
               boards=[dict(word=l.word, n=len(l.sents), members=dict(Counter(l.senses))) for l in leaves], split_log=log)
    pure = sum(m.most_common(1)[0][1] for m in B.members)
    rec["purity"] = pure / sum(sum(m.values()) for m in B.members)
    major = [m.most_common(1)[0][0] for m in B.members]
    rec["senses_found"] = len(set(major))
    # 振り分け: 同じ語の板のうち、起こりやすさの最大
    ok = n = 0
    for s, toks in test:
        w = SENSES[s][0]
        at = attrs_of(toks, w)
        if determined(w, at) != s:
            continue
        cand = [(loglik(l, at), r) for r, l in enumerate(leaves) if l.word == w]
        r = max(cand)[1]
        n += 1
        ok += major[r] == s
    rec["route_acc"] = ok / max(n, 1)
    # 違和感: 起こりやすさの低さ
    rng = random.Random(a.seed + 99)
    pos, neg = [], []
    for s, toks in test:
        w = SENSES[s][0]
        bad = corrupt(s, toks, rng)
        if bad is None:
            continue
        neg.append(-max(loglik(l, attrs_of(toks, w)) for l in leaves if l.word == w))
        pos.append(-max(loglik(l, attrs_of(bad, w)) for l in leaves if l.word == w))
    rec["anomaly_auroc"] = auroc(pos, neg)
    # 問い
    Q = questions()
    res, items = {}, {}
    for how in ("deny", "weighted", "corr", "lsq", "revise"):
        for src in ("板どうし", "板どうし+口どうし", "板どうし+1 枚の中", "全部", "全部（読み直し）"):
            if how in ("lsq", "revise") and src != "全部（読み直し）":
                continue
            if how in ("weighted", "corr") and src == "全部（読み直し）":
                continue
            per = defaultdict(lambda: [0, 0])
            effs = {}
            for q in Q:
                key = q["word"]
                if key not in effs:
                    P = board_pairs(B, key) + (port_pairs(B, key) if src in ("板どうし+口どうし", "全部") else [])                         + (inner_pairs(B, key) if src in ("板どうし+1 枚の中", "全部") else [])                         + (port_pairs(B, key) + inner_pairs(B, key, explain=True) if src == "全部（読み直し）" else [])
                    effs[key] = effects(P, how)
                ans = I1.answer(B, q, effs[key])
                per[q["kind"]][0] += ans == q["ans"]
                per[q["kind"]][1] += 1
                if (how, src) in (("deny", "全部（読み直し）"), ("corr", "全部"), ("lsq", "全部（読み直し）"), ("revise", "全部（読み直し）")):
                    items.setdefault(how, []).append(bool(ans == q["ans"]))
            res[f"{how}|{src}"] = {k: v[0] / v[1] for k, v in per.items()}
    rec["qa"] = res
    rec["qa_items"] = dict(kind=[q["kind"] for q in Q], **items)
    rec["pairs"] = [(w, {k: round(float(x), 2) for k, x in d.items()}, round(float(dt), 2), s, round(w_, 1))
                    for w, d, dt, s, w_ in board_pairs(B) + port_pairs(B) + inner_pairs(B)]
    rec["total_s"] = time.perf_counter() - t0
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_train", type=int, default=50)
    ap.add_argument("--min_support", type=float, default=3.0, help="口ごとに、観点を含む文がこれ未満なら数えない")
    ap.add_argument("--min_ref", type=float, default=5.0, help="反例として信じる、両側の文の数の下限")
    ap.add_argument("--loose", type=int, default=0, help="板どうしの組に、色と味だけ分かっている板も使う（ほかの観点の差は 0）")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--max_depth", type=int, default=3)
    ap.add_argument("--out", default="intervention/results/i03.jsonl")
    a = ap.parse_args()
    torch.set_num_threads(2)
    global MIN_REF, LOOSE
    MIN_REF, LOOSE = a.min_ref, bool(a.loose)
    rec = run(a)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("boards", "split_log", "pairs")}, ensure_ascii=False))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
