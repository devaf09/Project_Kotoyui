"""介入の研究 実験I33a: 述部グレイン（文脈から機能語を決める）を、まず当てられるかで測る

持ち主（2026-11-09）: 「述部について学ぶグレインを専門に作ってもいいかもしれない」
  「力業で文脈から機能語を判断するグレインを作って、人物の属性（丁寧など）に合わせて学習するのもいい」
I32 の板は (前の述語, 次の述語) だけでつなぎを選ぶので、逆接・理由・同時がほぼ出なかった。
述部グレインは、場面の記号の組（I18 の経験の器 ComboExp「単独+組」と同じ点の式）から機能語を当てる:
  点(結論) = log n(結論) + Σ_記号 log((n(記号, 結論) + 0.1) / (n(結論) + 0.2))
  ComboExp は経験を 1 つずつ持つが、ここでは同じ式を数え上げで持つ（2 万か所あるので）。
場面の記号（どれも作る時に分かるもの）:
  前の述語・次の述語、前の節・次の節の形の印（た・否定・相・丁寧）、極性が替わるか（否定の有り無しが違う）、
  台詞か地の文か、前の行の最後の形の印、話し手の属性（いまは作品のこれまでの台詞の丁寧の割合 = 人物の板の代わり）、
  前の節・次の節に問いの語があるか。
当て物:
  1. つなぎ（行の中の 述語 → 述語）。I32 の (p, q) の板と比べ、型ごとの拾えた割合（逆接・理由・同時）も見る。
  2. 最後の形の印の組（た・丁寧・否定・相）。I32 の「前後の行の割合」と比べる。
作品で 5 つに分け、ほかの作品で数えたものでその作品を当てる（自分の作品は使わない）。
Web 小説は _wn。結果は非公開の置き場だけ。本文はコードに書かない。
実行: python intervention/i33_predicate_grain.py --cache <解析済み JSON> --stats <jsonl>
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i32_function_layers as L  # noqa: E402

FOLDS = 5
ASK = ("何", "どこ", "誰", "どう", "なぜ", "いつ", "どれ", "どちら", "どうして")


def scene_feats(scene):
    """記号 1 つと、記号の組。組は語（p・q・語）どうし以外（語の組は (p, q) だけ）。"""
    sc = sorted(set(scene))
    out = list(sc)
    for a, b in itertools.combinations(sc, 2):
        if a.startswith(("pc:", "qc:")) or b.startswith(("pc:", "qc:")):
            continue
        lex_a, lex_b = a.startswith(("p:", "q:", "語:")), b.startswith(("p:", "q:", "語:"))
        if lex_a and lex_b and not ({a[:2], b[:2]} == {"p:", "q:"}):
            continue
        out.append(a + "&" + b)
    return out


class CountGrain:
    """ComboExp（単独+組）の点の式を数え上げで持つ器。
    smooth="0.1": ComboExp と同じ (n + 0.1) / (n(結論) + 0.2)。
    smooth="全体": 見えていない記号は全体の割合に寄せる (n + P(記号)) / (n(結論) + 1)（少ない結論に引っ張られない）。"""

    def __init__(self, smooth="全体"):
        self.smooth = smooth
        self.n = Counter()
        self.f = defaultdict(Counter)
        self.N = 0

    def feats(self, scene):
        return scene_feats(scene)

    def add(self, scene, lab):
        self.n[lab] += 1
        self.N += 1
        for t in self.feats(scene):
            self.f[t][lab] += 1

    def scores(self, scene):
        fs = [t for t in self.feats(scene) if t in self.f]
        out = {}
        for lab, n in self.n.items():
            s = math.log(n)
            for t in fs:
                if self.smooth == "0.1":
                    s += math.log((self.f[t][lab] + 0.1) / (n + 0.2))
                else:
                    pt = sum(self.f[t].values()) / self.N
                    s += math.log((self.f[t][lab] + pt) / (n + 1)) - math.log(pt)
            out[lab] = s
        return out


class LRGrain:
    """同じ記号で、多値のロジスティック回帰（I19 LRDecoder の多値版）。numpy だけ。"""

    def __init__(self, steps=200, lr=0.5, l2=1e-4):
        self.steps, self.lr, self.l2 = steps, lr, l2

    def fit(self, scenes, labs):
        import numpy as np
        self.np = np
        F = [scene_feats(sc) for sc in scenes]
        cnt = Counter(t for f in F for t in f)
        self.idx = {t: i for i, t in enumerate(sorted(t for t, c in cnt.items() if c >= 2))}
        self.labs = sorted(set(labs))
        li = {x: i for i, x in enumerate(self.labs)}
        rows, cols = [], []
        for r, f in enumerate(F):
            for t in f:
                if t in self.idx:
                    rows.append(r)
                    cols.append(self.idx[t])
        rows, cols = np.array(rows), np.array(cols)
        y = np.array([li[x] for x in labs])
        n, V, K = len(F), len(self.idx), len(self.labs)
        W, b = np.zeros((V, K)), np.log(np.bincount(y, minlength=K) + 1.0)
        b -= b.mean()
        for _ in range(self.steps):
            Z = np.zeros((n, K))
            np.add.at(Z, rows, W[cols])
            Z += b
            Z -= Z.max(1, keepdims=True)
            P = np.exp(Z)
            P /= P.sum(1, keepdims=True)
            P[np.arange(n), y] -= 1
            G = np.zeros((V, K))
            np.add.at(G, cols, P[rows])
            W -= self.lr * (G / n + self.l2 * W) * 10
            b -= self.lr * P.mean(0)
        self.W, self.b = W, b

    def scores(self, scene):
        z = self.b.copy()
        for t in scene_feats(scene):
            if t in self.idx:
                z += self.W[self.idx[t]]
        return dict(zip(self.labs, z))


CONTENT = False


def subtree(T, i):
    kids = L.kids_of(T)
    out, st = set(), [i]
    while st:
        x = st.pop()
        out.add(x)
        st += [k["i"] for k in kids[x]]
    return out


def marks_str(m, pre):
    return [f"{pre}{k}" for k in L.MARKS if m and m[k]]


def collect(works):
    """つなぎの場面と、最後の形の場面。"""
    A, B = [], []
    for wi, w in enumerate(works):
        pol = Counter()          # 台詞の丁寧の割合（人物の板の代わり）
        prev_m = None
        for ln in w["lines"]:
            T = ln["toks"]
            kids = L.kids_of(T)
            byi = {t["i"]: t for t in T}
            q_line = L.isq(ln["text"])
            asks = {t["head"] for t in T if t["lemma"] in ASK}
            subj = {t["head"]: t["lemma"] for t in T if t["dep"] == "nsubj"}
            ctx = (["台詞"] if q_line else ["地"]) + marks_str(prev_m, "前行:")
            if q_line and pol["n"]:
                ctx.append("話し手:丁寧" if pol["丁寧"] / pol["n"] > 0.5 else "話し手:普通")
            root_m = None
            for t in T:
                if t["pos"] not in L.PRED:
                    continue
                if t["dep"] == "advcl" and byi.get(t["head"], {}).get("pos") in L.PRED:
                    h = byi[t["head"]]
                    conn, mp, _ = L.layers(kids, t)
                    _, mq, _ = L.layers(kids, h)
                    sc = ["p:" + t["lemma"], "q:" + h["lemma"]] + marks_str(mp, "p") + marks_str(mq, "q") + ctx
                    if mp["否定"] != mq["否定"]:
                        sc.append("極性が替わる")
                    if t["i"] in asks:
                        sc.append("p問い")
                    if h["i"] in asks:
                        sc.append("q問い")
                    if t["i"] in subj and h["i"] in subj and subj[t["i"]] != subj[h["i"]]:
                        sc.append("主語が替わる")
                    if CONTENT:
                        # 力業の記号: 前の節・次の節の中身の語（節の部分木の内容語）。語どうしの組は作らない
                        sub_p = subtree(T, t["i"])
                        for x in sub_p:
                            if T[x]["pos"] in ("NOUN", "VERB", "ADJ", "ADV") and x != t["i"]:
                                sc.append("pc:" + T[x]["lemma"])
                        for x in range(t["i"] + 1, h["i"]):
                            if x not in sub_p and T[x]["pos"] in ("NOUN", "VERB", "ADJ", "ADV"):
                                sc.append("qc:" + T[x]["lemma"])
                    A.append((wi, sc, conn, t["lemma"], h["lemma"]))
                elif t["dep"] == "ROOT" and root_m is None:
                    conn, m, _ = L.layers(kids, t)
                    root_m = m
                    if conn == "連用":
                        B.append((wi, ["語:" + t["lemma"]] + ctx, "+".join(k for k in L.MARKS if m[k]) or "無"))
            if q_line and root_m is not None:
                pol["n"] += 1
                pol["丁寧"] += root_m["丁寧"]
            if root_m is not None:
                prev_m = root_m
    return A, B


def run(A, B, works):
    res = {}
    # 1. つなぎ
    fold = lambda wi: wi % FOLDS  # noqa: E731
    acc = Counter()
    by_type = defaultdict(Counter)
    for k in range(FOLDS):
        g = CountGrain("0.1")
        g2 = CountGrain("全体")
        tr_sc, tr_y = [], []
        pq = defaultdict(Counter)
        pc = defaultdict(Counter)
        gc = Counter()
        for wi, sc, c, p, q in A:
            if fold(wi) != k:
                g.add(sc, c)
                g2.add(sc, c)
                tr_sc.append(sc)
                tr_y.append(c)
                pq[(p, q)][c] += 1
                pc[p][c] += 1
                gc[c] += 1
        labels = list(gc)
        tot = sum(gc.values())
        g3 = LRGrain()
        g3.fit(tr_sc, tr_y)
        for wi, sc, c, p, q in A:
            if fold(wi) != k:
                continue
            # I32 の (p, q) → p → 全体
            d = {x: (gc[x] + 0.5) / (tot + 0.5 * len(labels)) for x in labels}
            for cnt in (pc.get(p), pq.get((p, q))):
                if cnt:
                    n = sum(cnt.values())
                    lam = n / (n + len(cnt))
                    d = {x: lam * cnt.get(x, 0) / n + (1 - lam) * d[x] for x in labels}
            b32 = max(d, key=d.get)
            t = L.ctype(c)
            acc["n"] += 1
            by_type[t]["n"] += 1
            for nm, best in (("I32 の板", b32), ("グレイン 数え上げ 0.1", None), ("グレイン 数え上げ 全体", None), ("グレイン 回帰", None)):
                if best is None:
                    gg = {"グレイン 数え上げ 0.1": g, "グレイン 数え上げ 全体": g2, "グレイン 回帰": g3}[nm]
                    s_ = gg.scores(sc)
                    best = max(s_, key=s_.get)
                acc[nm] += best == c
                acc[nm + "・型"] += L.ctype(best) == t
                by_type[t][nm] += L.ctype(best) == t
    n = acc.pop("n")
    res["つなぎ"] = dict(n=n, **{k: round(v / n, 4) for k, v in acc.items()},
                       型ごとに拾えた割合={t: {m: (v if m == "n" else round(v / c["n"], 3)) for m, v in c.items()}
                                    for t, c in sorted(by_type.items(), key=lambda x: -x[1]["n"])})
    # 2. 最後の形の印の組
    acc = Counter()
    for k in range(FOLDS):
        trn = [(sc, y) for wi, sc, y in B if fold(wi) != k]
        g = CountGrain("全体")
        for sc, y in trn:
            g.add(sc, y)
        g3 = LRGrain()
        g3.fit([x for x, _ in trn], [y for _, y in trn])
        nolex = lambda sc: [x for x in sc if not x.startswith(("前行:", "話し手:"))]  # noqa: E731
        g4 = LRGrain()
        g4.fit([nolex(x) for x, _ in trn], [y for _, y in trn])
        for wi, sc, y in B:
            if fold(wi) != k:
                continue
            acc["n"] += 1
            for nm, gg, x in (("グレイン 数え上げ 全体", g, sc), ("グレイン 回帰", g3, sc), ("回帰・語と台詞だけ（前の行・話し手なし）", g4, nolex(sc))):
                s_ = gg.scores(x)
                acc[nm] += max(s_, key=s_.get) == y
            acc["いつも 無"] += y == "無"
    n = acc.pop("n")
    res["最後の形の印の組"] = dict(n=n, **{k: round(v / n, 4) for k, v in acc.items()})
    return res


def train_folds(works):
    """展開で使う: 作品の組ごとに、ほかの組の作品で学んだ回帰＋中身の語のグレイン。"""
    global CONTENT
    CONTENT = True
    A, _ = collect(works)
    out = {}
    for k in range(FOLDS):
        g = LRGrain()
        tr = [(sc, c) for wi, sc, c, p, q in A if wi % FOLDS != k]
        g.fit([x for x, _ in tr], [y for _, y in tr])
        out[k] = g
        print("グレイン", k, flush=True)
    return out


def make_grain_dist(g, lines, li, marks):
    """作る行の場面（作る時に分かる記号だけ）から、つなぎの確からしさ。"""
    prev_m = next((marks[lj][1] for lj in range(li - 1, -1, -1) if marks[lj][1] is not None), None)
    q_line = L.isq(lines[li]["text"])
    ctx = (["台詞"] if q_line else ["地"]) + marks_str(prev_m, "前行:")

    def dist(ch, clauses, j):
        c, d = clauses[j], clauses[j + 1]
        sc = ["p:" + c["述語"], "q:" + d["述語"]] + ctx
        for k in ("目的語", "補語", "修飾", "目的語の修飾"):
            if c.get(k):
                sc.append("pc:" + c[k])
            if d.get(k):
                sc.append("qc:" + d[k])
        s = g.scores(sc)
        m = max(s.values())
        z = sum(math.exp(v - m) for v in s.values())
        return {x: math.exp(v - m) / z for x, v in s.items()}
    return dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--content", action="store_true", help="つなぎに、節の中身の語を記号として足す（力業の版）")
    a = ap.parse_args()
    global CONTENT
    CONTENT = a.content
    works = json.load(open(a.cache, encoding="utf-8"))
    A, B = collect(works)
    res = run(A, B, works)
    rec = dict(exp="I33a" + ("・中身の語" if a.content else ""), **res)
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    with open(a.stats, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
