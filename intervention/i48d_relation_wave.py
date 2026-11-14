"""I48d: 語板 v3 の「関係」の最初の試し（持ち主 2026-11-14 の案）。

語板に、属性のほかに「関係」を持たせる。関係 = その語の前に来るつなぎと、後ろに来るつなぎ。
基本の形は「前のつなぎ・今の語・後ろのつなぎ」（が・歩・いている）。行の中のほかの語（今、昨日…）で明るさが変わる。

- つなぎ = 中身の語のあいだの機能語の並び（助詞・助動詞・接続助詞など）を、まとめて 1 つの候補にした物（「て+いる」「ない+た」）。
  何も無い時は「∅」という 1 つの候補。一覧は手で決めず、資料に出た物（学びで min_count 回以上）を全部持つ。
- 持ち方は 2 つを比べる:
    帯   : 語ごとに 1 本の波（後ろの帯・前の帯）。その語の後ろ（前）に出たつなぎの印を重ねた物。平均の関係。
    置き場: 場面（語 1 つの出どころ）ごとに鍵の波を取っておき、問いと共鳴の強い場面のつなぎを 2 乗で重ねる（I48 のすくい方）。
            鍵 = 語、行のほかの語、品詞、（逆回しでは）もう決まっている反対側のつなぎ。
- 試し（外した対話で当てる）:
    後ろ    : 語と行のほかの語から、後ろのつなぎを当てる（順回し）
    前      : 語と行のほかの語から、前のつなぎを当てる
    前（逆回し）: 後ろのつなぎがもう決まっている時に、前のつなぎを当てる（「歩いている」が決まっていて「が」を見つける）
- 比べる相手: いちばん多いつなぎ、数えるだけの表（語 → 品詞 → 全体、の順に下がる）、それに行の語を 1 つずつ掛けた表。

実行: python intervention/i48d_relation_wave.py --cache <parsed.json> --out i48d.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import zlib
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from i48_wave_board import Waves  # noqa: E402

log = logging.getLogger("i48d")
CONTENT = ("NOUN", "PROPN", "PRON", "VERB", "ADJ")
CUE = CONTENT + ("ADV", "NUM")
FUNC = ("ADP", "AUX", "SCONJ", "PART")
NONE = "∅"


def chain(toks):
    return "+".join(t["lemma"] for t in toks) if toks else NONE


def samples(work):
    """対話 1 本 → 中身の語ごとに dict(w 語, pos 品詞, pre 前のつなぎ, post 後ろのつなぎ, cues 行のほかの語, ctx 前の 3 行の語, nxt 後ろの中身の語)。"""
    out, prev = [], []
    for ln in work["lines"]:
        T = ln["toks"]
        idx = [i for i, t in enumerate(T) if t["pos"] in CONTENT]
        cues = [t["lemma"] for t in T if t["pos"] in CUE]
        for k, i in enumerate(idx):
            a = idx[k - 1] + 1 if k else 0
            pre = [t for t in T[a:i]]
            j = i + 1
            while j < len(T) and T[j]["pos"] in FUNC:
                j += 1
            if any(t["pos"] not in FUNC for t in pre):      # 前のあいだに機能語でない物（記号など）がある: 前のつなぎは ∅ にする
                pre = []
            w = T[i]["lemma"]
            out.append(dict(w=w, pos=T[i]["pos"], pre=chain(pre), post=chain(T[i + 1:j]), cues=[c for c in cues if c != w],
                            ctx=sum(prev[-3:], []), nxt=T[idx[k + 1]]["lemma"] if k + 1 < len(idx) else NONE, npost=NONE))
            if k:
                out[-2]["npost"] = out[-1]["post"]            # 前の語から見た「次の語の後ろのつなぎ」（歩 の いている）
        prev.append([t["lemma"] for t in T if t["pos"] in CONTENT])
    return out


def mark(x, D):
    return np.exp(1j * np.random.default_rng(zlib.crc32(x.encode())).uniform(0, 2 * np.pi, D)).astype(np.complex64)


class Relation:
    """関係の板。side = "post"（後ろ）か "pre"（前）。given = もう決まっている反対側のつなぎを鍵に入れるか（逆回し）。"""

    def __init__(self, wv, cands, D, side, given=False, keys=("語", "行", "品詞", "反対側", "対"), top=30):
        self.wv, self.cands, self.D, self.side, self.given, self.keys, self.top = wv, cands, D, side, given, keys, top
        self.cix = {c: i for i, c in enumerate(cands)}
        self.role = {k: mark("役:" + k, D) for k in ("語", "行", "品詞", "反対側", "対")}
        self.w_n = None
        self.cmark = np.stack([mark("つなぎ:" + c, D) for c in cands])

    def w(self, x):
        j = self.wv.ix.get(x)
        return None if j is None else self.wv.W[j].astype(np.complex64)

    def bag(self, ws):
        ids = [self.wv.ix[x] for x in ws if x in self.wv.ix]
        if not ids:
            return None
        v = self.wv.W[ids].sum(0)
        return (v / (np.linalg.norm(v) / math.sqrt(self.D) + 1e-9)).astype(np.complex64)

    def key(self, s):
        parts = []
        w = self.w(s["w"])
        if w is not None and "語" in self.keys:
            parts.append(w * self.role["語"])
        b = self.bag(s["cues"])
        if b is not None and "行" in self.keys:
            parts.append(b * self.role["行"])
            if w is not None and "対" in self.keys:
                parts.append(w * b)                         # 語 × 行のほかの語（「歩 × 今」）
        if "品詞" in self.keys:
            parts.append(mark("品:" + s["pos"], self.D) * self.role["品詞"])
        if self.given == "nxt" and "反対側" in self.keys:   # 次の語とその後ろがもう決まっている（「歩いている」が決まっていて、猫 の後ろを読む）
            n = self.w(s["nxt"])
            if n is not None:
                parts.append(n * self.role["反対側"])
                if w is not None:
                    parts.append(w * n * self.role["反対側"])
            parts.append(mark("つなぎ:" + s["npost"], self.D) * self.role["反対側"] * self.role["品詞"])
        elif self.given and "反対側" in self.keys:
            other = s["post"] if self.side == "pre" else s["pre"]
            parts.append(mark("つなぎ:" + other, self.D) * self.role["反対側"])
        if not parts:
            return None
        v = sum(parts)
        return (v / (np.linalg.norm(v) + 1e-9)).astype(np.complex64)

    def fit(self, S):
        K, P = [], []
        V = len(self.wv.vocab)
        self.band = np.zeros((V, self.D), np.complex64)     # 語ごとの帯: 出たつなぎの印の重ね合わせ
        self.band_n = np.zeros(V)
        for s in S:
            j = self.cix.get(s[self.side])
            if j is None:
                continue
            i = self.wv.ix.get(s["w"])
            if i is not None:
                self.band[i] += self.cmark[j]
                self.band_n[i] += 1
            k = self.key(s)
            if k is not None:
                K.append(k)
                P.append(j)
        self.K, self.P = np.stack(K), np.array(P)
        self.prior = np.bincount(self.P, minlength=len(self.cands)).astype(float)
        return self

    def band_scores(self, s):
        """帯から読む: 語の帯とつなぎの印の共鳴。雑音の床（σ√(2 ln C)、σ = |帯|/(D√2)）より下は 0。"""
        i = self.wv.ix.get(s["w"])
        if i is None or self.band_n[i] == 0:
            return self.prior
        b = self.band[i]
        r = (self.cmark @ np.conj(b)).real / self.D
        floor = np.linalg.norm(b) / (self.D * math.sqrt(2)) * math.sqrt(2 * math.log(len(self.cands)))
        r = np.where(r > floor, r, 0.0)
        return r + 1e-6 * self.prior / self.prior.sum()

    def scoop_scores(self, S):
        """置き場から読む（まとめて）: 問いの波と共鳴の強い top 場面のつなぎを、共鳴の 2 乗で重ねる。"""
        out = np.zeros((len(S), len(self.cands)))
        Q, rows = [], []
        for n, s in enumerate(S):
            q = self.key(s)
            if q is None:
                out[n] = self.prior
            else:
                Q.append(q)
                rows.append(n)
        if Q:
            R = np.abs(self.K @ np.conj(np.stack(Q)).T)     # 場面 × 問い
            o = np.argpartition(-R, self.top, axis=0)[: self.top]
            for c, n in enumerate(rows):
                out[n] = np.bincount(self.P[o[:, c]], weights=R[o[:, c], c] ** 2, minlength=len(self.cands))
        return out + 1e-6 * self.prior / self.prior.sum()


class CountTable:
    def __init__(self, cands, side, cues=False, given=False):
        self.cands, self.cix, self.side, self.cues, self.given = cands, {c: i for i, c in enumerate(cands)}, side, cues, given
        self.w_, self.p_, self.f_ = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
        self.all = Counter()

    def feats(self, s):
        f = ["行:" + c for c in set(s["cues"])] if self.cues else []
        if self.given == "nxt":
            f += ["次:" + s["nxt"], "次後:" + s["npost"]]
        elif self.given:
            f.append("反:" + (s["post"] if self.side == "pre" else s["pre"]))
        return f

    def fit(self, S):
        for s in S:
            c = s[self.side]
            if c in self.cix:
                self.w_[s["w"]][c] += 1
                self.p_[s["pos"]][c] += 1
                self.all[c] += 1
                for x in self.feats(s):
                    self.f_[x][c] += 1
        tot = sum(self.all.values())
        self.pr = np.array([self.all[c] / tot for c in self.cands])
        return self

    def dist(self, c, back):
        z = sum(c.values())
        v = np.zeros(len(self.cands))
        for k, n in c.items():
            v[self.cix[k]] = n
        return (v + back) / (z + 1)

    def scores(self, s):
        pp = self.dist(self.p_.get(s["pos"], Counter()), self.pr)
        out = self.dist(self.w_.get(s["w"], Counter()), pp)
        for x in self.feats(s):
            if x in self.f_:
                out = out * self.dist(self.f_[x], self.pr) / self.pr
        return out


def measure(SC, te, cands, side):
    cix = {c: i for i, c in enumerate(cands)}
    t = np.array([cix[s[side]] for s in te])
    o = np.argsort(-SC, 1)[:, :3]
    h1 = float((o[:, 0] == t).mean())
    h3 = float((o == t[:, None]).any(1).mean())
    nz = t != cix.get(NONE, -1)                             # つなぎが有る場面だけ（∅ を当てるのは易しいので、別に見る）
    return dict(top1=round(h1, 4), top3=round(h3, 4), top1_some=round(float((o[nz, 0] == t[nz]).mean()), 4) if nz.any() else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--test_works", type=int, default=160)
    ap.add_argument("--D", type=int, default=512)
    ap.add_argument("--min_count", type=int, default=5)
    ap.add_argument("--max_scenes", type=int, default=150000)
    ap.add_argument("--test_n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    works = json.load(open(a.cache, encoding="utf-8"))
    works = works["works"] if isinstance(works, dict) else works
    rnd = random.Random(a.seed)
    rnd.shuffle(works)
    te_w, tr_w = works[: a.test_works], works[a.test_works:]
    tr_all = [s for w in tr_w for s in samples(w)]
    tr = tr_all[:: max(1, math.ceil(len(tr_all) / a.max_scenes))]
    te_all = [s for w in te_w for s in samples(w)]
    vocab = sorted({t["lemma"] for w in tr_w for ln in w["lines"] for t in ln["toks"] if t["pos"] in CUE})
    seq = [[t["lemma"] for t in ln["toks"] if t["pos"] in CUE] for w in tr_w for ln in w["lines"] + [dict(toks=[])]]
    wv = Waves(vocab, a.D, a.seed)
    wv.learn(seq)
    res = dict(n_train=len(tr_all), n_scenes=len(tr), D=a.D, tasks={})
    for name, side, given in (("後ろ", "post", False), ("後ろ（逆回し: 次の語とその後ろが決まっている）", "post", "nxt"), ("前", "pre", False), ("前（後ろが決まっている）", "pre", True)):
        cands = [c for c, n in Counter(s[side] for s in tr_all).most_common() if n >= a.min_count]
        cs = set(cands)
        te = [s for s in te_all if s[side] in cs]
        cover = len(te) / max(len(te_all), 1)
        te = random.Random(a.seed + 1).sample(te, min(a.test_n, len(te)))
        log.info("== %s: 候補 %d、試し %d（候補に入る割合 %.3f）、∅ の割合 %.3f", name, len(cands), len(te), cover,
                 sum(s[side] == NONE for s in te) / len(te))
        rows = {}

        def row(label, SC):
            rows[label] = measure(SC, te, cands, side)
            log.info("%-30s 1 番目 %.3f（3 番まで %.3f）　つなぎ有りだけ %.3f", label, rows[label]["top1"], rows[label]["top3"], rows[label]["top1_some"] or 0)

        pr = np.array([sum(1 for s in tr if s[side] == c) for c in cands], float)
        row("いちばん多いつなぎ", np.tile(pr, (len(te), 1)))
        ct = CountTable(cands, side).fit(tr_all)
        row("数える表（語）", np.stack([ct.scores(s) for s in te]))
        ct2 = CountTable(cands, side, cues=True, given=given).fit(tr_all)
        row("数える表 ＋行の語" + ("・反対側" if given else ""), np.stack([ct2.scores(s) for s in te]))
        rel = Relation(wv, cands, a.D, side, given).fit(tr)
        row("波: 語ごとの帯（1 本）", np.stack([rel.band_scores(s) for s in te]))
        B = 256
        row("波: 置き場ですくう", np.concatenate([rel.scoop_scores(te[i:i + B]) for i in range(0, len(te), B)]))
        for drop in ("行", "対") + (("反対側",) if given else ()):
            r2 = Relation(wv, cands, a.D, side, given, keys=tuple(k for k in rel.keys if k != drop)).fit(tr)
            row(f"波: すくう −{drop}", np.concatenate([r2.scoop_scores(te[i:i + B]) for i in range(0, len(te), B)]))
        res["tasks"][name] = dict(n_cands=len(cands), n_test=len(te), cover=round(cover, 4), rows=rows, top_cands=cands[:30])
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
