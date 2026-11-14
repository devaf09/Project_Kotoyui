"""I48c: 助詞を波で浮かび上がらせる（持ち主 2026-11-14 の案: 語板に、前後の流れから合う助詞を属性板のように浮かび上がらせる）。

名詞と、それが係る語（述語など）のあいだの助詞（は・が・を・に・へ・で・と・も・の…）を、波の板から読む。

- 助詞ごとに 1 本の波 M[p]（D 点）を持つ。学びでは、出会った（名詞, 係る語, 助詞）ごとに、鍵の波を M[助詞] に重ねるだけ。
- 鍵の波（どれも周波数ごとの掛け算で作る。語の波は I48 と同じ、流れから学んだ波）:
    対   : 名詞の波 × 係る語の波
    係る語: 係る語の波 × 役の印
    名詞 : 名詞の波 × 役の印
    流れ : 前の行までの波 × 名詞の波の共役（名詞がもう話に出ていれば、周波数によらない一定の成分が立つ = 既に出た話題の印）
    行内 : 同じ行の、名詞より前の語の波 × 役の印
- 読む時は、同じ鍵を作って M[p] と共鳴させる（Re<M[p], 鍵>/D）。鍵ごとの明るさを「そろえて」（助詞の間で平均 0・大きさ 1）足し、いちばん明るい助詞を返す。
- 比べる相手: いちばん多い助詞、数えるだけの表（(名詞, 係る語) → 係る語と名詞の表 → 全体、の順に下がる）。

物差し: 外した対話で、隠した助詞を当てる割合（1 番目・3 番目まで）。

実行: python intervention/i48c_particle_wave.py --cache <parsed.json> --D 512,2048 --out i48c.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import zlib
from collections import Counter, defaultdict

import numpy as np

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from i48_wave_board import Waves, content  # noqa: E402

log = logging.getLogger("i48c")
NOUNS = ("NOUN", "PROPN", "PRON")
KEYS = ("対", "係る語", "名詞", "流れ", "行内", "ほかの助詞", "後ろ")


def samples(work):
    """対話 1 本 → (名詞, 係る語, 助詞, 前の行までの中身の語, 同じ行の前の語, 係りの名前) の並び。"""
    out = []
    prev = []
    for ln in work["lines"]:
        T = ln["toks"]
        for t in T:
            if t["pos"] == "ADP" and t["dep"] == "case":
                n = T[t["head"]]
                h = T[n["head"]]
                if n["pos"] in NOUNS and h is not n and t["i"] == n["i"] + 1:
                    before = [x["lemma"] for x in T[: n["i"]] if x["pos"] in ("NOUN", "PROPN", "VERB", "ADJ", "PRON")]
                    sib = ["助:" + x["lemma"] for x in T if x["pos"] == "ADP" and x["dep"] == "case" and x is not t
                           and T[x["head"]]["head"] == h["i"] and T[x["head"]] is not n]      # 同じ語に係る、ほかの名詞の助詞
                    aft = ["後:" + x["lemma"] for x in T if x["head"] == h["i"] and x["i"] > h["i"] and x["pos"] in ("AUX", "SCONJ", "PART")]
                    aft.append("品:" + h["pos"])
                    out.append(dict(n=n["lemma"], h=h["lemma"], p=t["lemma"], ctx=sum(prev[-3:], []), before=before, sib=sib, aft=aft))
        prev.append(content(T))
    return out


class ParticleBoard:
    def __init__(self, waves, parts, D, seed=0, keys=KEYS):
        rng = np.random.default_rng(seed + 7)
        self.wv, self.parts, self.D, self.keys = waves, parts, D, keys
        self.pix = {p: i for i, p in enumerate(parts)}
        self.role = {k: np.exp(1j * rng.uniform(0, 2 * np.pi, D)) for k in KEYS}
        self.marks = {}                                     # 助詞・後ろの語・品詞の印（乱数の位相。出会った時に作る）
        self._rng = rng
        self.M = {k: np.zeros((len(parts), D), np.complex64) for k in keys}     # 助詞ごとの波（鍵の種類ごとに 1 枚。足せば 1 本にもなる）
        self.n = np.zeros(len(parts))

    def w(self, x):
        j = self.wv.ix.get(x)
        return None if j is None else self.wv.W[j]

    def bag(self, ws):
        ids = [self.wv.ix[x] for x in ws if x in self.wv.ix]
        if not ids:
            return None
        v = self.wv.W[ids].sum(0)
        return v / (np.linalg.norm(v) / math.sqrt(self.D) + 1e-9)

    def make_keys(self, s):
        n, h = self.w(s["n"]), self.w(s["h"])
        out = {}
        if n is not None and h is not None:
            out["対"] = n * h
        if h is not None:
            out["係る語"] = h * self.role["係る語"]
        if n is not None:
            out["名詞"] = n * self.role["名詞"]
            c = self.bag(s["ctx"])
            if c is not None:
                out["流れ"] = c * np.conj(n)
        b = self.bag(s["before"])
        if b is not None:
            out["行内"] = b * self.role["行内"]
        for k, f in (("ほかの助詞", "sib"), ("後ろ", "aft")):
            if s.get(f):
                v = sum(self.mark(x) for x in s[f])
                out[k] = v / (np.linalg.norm(v) / math.sqrt(self.D) + 1e-9) * self.role[k]
        return {k: v for k, v in out.items() if k in self.keys}

    def mark(self, x):
        if x not in self.marks:
            self.marks[x] = np.exp(1j * np.random.default_rng(zlib.crc32(x.encode())).uniform(0, 2 * np.pi, self.D))
        return self.marks[x]

    def fit(self, S):
        for s in S:
            j = self.pix.get(s["p"])
            if j is None:
                continue
            self.n[j] += 1
            for k, v in self.make_keys(s).items():
                self.M[k][j] += v
        return self

    def scores(self, s, form="even"):
        tot = np.zeros(len(self.parts))
        for k, v in self.make_keys(s).items():
            r = (self.M[k] @ np.conj(v)).real / self.D          # 助詞ごとの共鳴
            if form == "raw":
                tot += r
            else:
                r = r / np.maximum(self.n, 1.0)                 # 助詞の多さで割る（その助詞の鍵として、どれだけ典型か）
                sd = r.std()
                if sd > 0:
                    tot += (r - r.mean()) / sd                  # そろえる: 鍵ごとに平均 0・大きさ 1
        if form == "even":
            tot = tot + np.log(np.maximum(self.n, 1.0) / self.n.sum())   # もとの多さ（何も手がかりが無い時の向き）
        return tot


class ParticleScoop:
    """すくう形（I48 の波の板と同じ）: 出会った場面ごとに、鍵の波を足した 1 本の波を場面の波として取っておく。
    読む時は、同じ作りの問いの波と共鳴させ、強い top 場面の助詞を、共鳴の強さの 2 乗で重ねる。"""

    def __init__(self, pb, top=30):
        self.pb, self.top = pb, top

    def wave(self, s):
        ks = self.pb.make_keys(s)
        if not ks:
            return None
        v = sum(ks.values())
        return (v / (np.linalg.norm(v) + 1e-9)).astype(np.complex64)

    def fit(self, S):
        K, P = [], []
        for s in S:
            j = self.pb.pix.get(s["p"])
            v = self.wave(s) if j is not None else None
            if v is not None:
                K.append(v)
                P.append(j)
        self.K, self.P = np.stack(K), np.array(P)
        self.prior = np.bincount(self.P, minlength=len(self.pb.parts)).astype(float)
        return self

    def scores(self, s):
        q = self.wave(s)
        if q is None:
            return self.prior
        r = np.abs(self.K @ np.conj(q))
        o = np.argpartition(-r, self.top)[: self.top]
        out = np.bincount(self.P[o], weights=r[o] ** 2, minlength=len(self.pb.parts))
        return out + 1e-6 * self.prior / self.prior.sum()


class CountTable:
    def __init__(self, parts, feats=False):
        self.feats, self.f_ = feats, defaultdict(Counter)   # feats: ほかの助詞・後ろの語も、1 つずつ掛ける（波の鍵と同じ手がかり）
        self.parts, self.pix = parts, {p: i for i, p in enumerate(parts)}
        self.nh, self.h_, self.n_ = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
        self.all = Counter()

    def fit(self, S):
        for s in S:
            if s["p"] in self.pix:
                self.nh[(s["n"], s["h"])][s["p"]] += 1
                self.h_[s["h"]][s["p"]] += 1
                self.n_[s["n"]][s["p"]] += 1
                self.all[s["p"]] += 1
                for x in s["sib"] + s["aft"]:
                    self.f_[x][s["p"]] += 1
        return self

    def scores(self, s):
        tot = sum(self.all.values())
        pr = np.array([self.all[p] / tot for p in self.parts])

        def dist(c):
            z = sum(c.values())
            return np.array([(c[p] + pr[i]) / (z + 1) for i, p in enumerate(self.parts)])
        c = self.nh.get((s["n"], s["h"]))
        if c:
            out = dist(c)
        else:
            h, n = self.h_.get(s["h"]), self.n_.get(s["n"])
            out = pr.copy()
            if h:
                out = out * dist(h) / pr
            if n:
                out = out * dist(n) / pr
        if self.feats:
            for x in s["sib"] + s["aft"]:
                if x in self.f_:
                    out = out * dist(self.f_[x]) / pr
        return out


def measure(score_fn, te, parts):
    pix = {p: i for i, p in enumerate(parts)}
    h1 = h3 = 0
    per = defaultdict(lambda: [0, 0])
    conf = Counter()
    for s in te:
        sc = score_fn(s)
        o = np.argsort(-sc)
        t = pix[s["p"]]
        h1 += o[0] == t
        h3 += t in o[:3]
        per[s["p"]][0] += o[0] == t
        per[s["p"]][1] += 1
        if o[0] != t:
            conf[(s["p"], parts[o[0]])] += 1
    n = max(len(te), 1)
    return dict(top1=round(h1 / n, 4), top3=round(h3 / n, 4), per={p: round(a / b, 3) for p, (a, b) in per.items()},
                confuse=[(a, b, c) for (a, b), c in conf.most_common(6)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--test_works", type=int, default=160)
    ap.add_argument("--D", default="512,2048")
    ap.add_argument("--n_parts", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    works = json.load(open(a.cache, encoding="utf-8"))
    random.Random(a.seed).shuffle(works)
    te_w, tr_w = works[: a.test_works], works[a.test_works:]
    tr = [s for w in tr_w for s in samples(w)]
    parts = [p for p, _ in Counter(s["p"] for s in tr).most_common(a.n_parts)]
    te = [s for w in te_w for s in samples(w) if s["p"] in parts]
    seen = {(s["n"], s["h"]) for s in tr}
    te_new = [s for s in te if (s["n"], s["h"]) not in seen]
    log.info("学び %d、試し %d（うち学びに無い対 %d）、助詞 %s", len(tr), len(te), len(te_new), parts)
    res = dict(n_train=len(tr), n_test=len(te), n_test_new=len(te_new), parts=parts, rows=[])

    def row(name, fn, **kw):
        r = dict(name=name, **kw, all=measure(fn, te, parts), new=measure(fn, te_new, parts))
        res["rows"].append(r)
        log.info("%-34s 全部 %.3f（3 番まで %.3f）　無い対 %.3f（%.3f）", name, r["all"]["top1"], r["all"]["top3"], r["new"]["top1"], r["new"]["top3"])
        return r

    pr = np.array([sum(s["p"] == p for s in tr) for p in parts], float)
    row("いちばん多い助詞", lambda s: pr)
    ct = CountTable(parts).fit(tr)
    row("数えるだけの表", ct.scores)
    row("数える表 ＋ほかの助詞・後ろ", CountTable(parts, True).fit(tr).scores)

    vocab = sorted({x for w in tr_w for ln in w["lines"] for t in ln["toks"] for x in [t["lemma"]] if t["pos"] in ("NOUN", "PROPN", "VERB", "ADJ", "PRON")})
    seq_all = [[[t["lemma"] for t in ln["toks"] if t["pos"] in ("NOUN", "PROPN", "VERB", "ADJ", "PRON")] for ln in w["lines"]] for w in tr_w]
    for D in [int(x) for x in a.D.split(",")]:
        for learned in (False, True):
            wv = Waves(vocab, D, a.seed)
            if learned:
                wv.learn([ln for w in seq_all for ln in w + [[]]])
            tag = f"波 D{D} {'学んだ波' if learned else '印のまま'}"
            pb = ParticleBoard(wv, parts, D, a.seed).fit(tr)
            row(tag + " そろえる", pb.scores, D=D)
            ps = ParticleScoop(pb).fit(tr)
            row(tag + " すくう", ps.scores, D=D)
            if learned:
                for drop in KEYS:
                    ks = tuple(k for k in KEYS if k != drop)
                    pb3 = ParticleBoard(wv, parts, D, a.seed, keys=ks)
                    row(tag + f" すくう −{drop}", ParticleScoop(pb3).fit(tr).scores, D=D)
                row(tag + " 重ねるだけ", lambda s: pb.scores(s, "raw"), D=D)
                both = lambda s: np.log(ct.scores(s) + 1e-9) + pb.scores(s) - np.log(pr / pr.sum())   # noqa: E731
                row(tag + " ＋数える表", both, D=D)
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
