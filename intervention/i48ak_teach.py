"""I48ak: 介入で選んで覚える（持ち主の方針: ある所まで出力を上げたら、後半は介入で学ぶ）。I48aj の続き。

- 候補: 今の点（I48ag の先読みなし、節・シフト＋元）の上位 K と、三つの共振（I48aj の R1＋R2＋R3、prior なし）の上位 K の和集合。
  選ぶ物: 候補の中で 今×三つ がいちばん大きい物（頭から順に、R3 は置いた物で）。
- 介入: 練習の文（学びに使っていない文）の間ごとに、先生役（試しの場では元の文）が候補の中の元の詰め物を指す。
    指された詰め物: その間の特徴を、全部の板（今の点のグレイン・R1・R2 の 3 つ・R3）に足す（明るい側）。
    選んでいた物が違った時: 選んでいた物の今の点の板の暗い側に DARK_W で足す（I48ai と同じ）。
    元が候補に無い時: 何もしない版（候補の中だけ）と、候補の外でも教える版（候補の外も）の 2 通り。
  足す重さ: 1 回分（数え 1）、または「その板の平均の 1 例分」（板の総量 ÷ 回数 を、この例の量で割った重さ）。
- 比べ: 最初の学び（5,000・2 万・5 万文）ごとに、介入 5,000 文の前後と、同じ 5,000 文をただ学びに足した場合（普通の数え）。
- 教えた数: 自分で作った「〜と〜に行った」の文を 1・3・10 文だけ介入で教えた時の、持ち主の例の「と」の順位。
- 別の板（sep、I48al）: 教えた分を本体に混ぜず介入だけの板に持ち、今の点に sep_ratio の点を足す（教えた帯だけが上がる）。
- 練習の間、板の対数の表は REFRESH 文ごとに作り直す（速さのため。教えた文の効きが少し遅れる）。
手で置く物: K（5）、DARK_W（1/1,000）、REFRESH（100 文）、自作の教える文（TEACH）。ほかは I48aj と同じ。
python intervention/i48ak_teach.py --conllu ... --content ... --func ... --out out.json [--inits 5000 20000 50000] [--practice 5000] [--test 1000] [--r1 型]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import zlib
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i48ab_long_grain as G  # noqa: E402
import i48ad_lookahead as AD  # noqa: E402
import i48ae_marks as AE  # noqa: E402
import i48aj_resonance as AJ  # noqa: E402

MODE = "シフト＋元"
K = 5
DARK_W = 1e-3
REFRESH = 100
TEACH = [("友達", "散歩"), ("先生", "学校"), ("母", "買い物"), ("兄", "公園"), ("彼女", "映画"),
         ("父", "海"), ("姉", "図書館"), ("同僚", "食事"), ("子供", "動物園"), ("祖母", "病院")]   # 「X と Y に 行った」（自作）
EXAMPLES = AJ.EXAMPLES


def keys_of(s):
    return [(l, u) for l, _, u in G.split(s)[0]]


def made_sentence(a, b):
    """自作の文「a と b に 行った。」の要点・詰め物（GiNZA を使わない）。"""
    return [(a, "NOUN"), (b, "NOUN"), ("行く", "VERB")], ["と", "に", "た。"]


def made_items(S, keys, gaps, mk=("過去",)):
    A = [S.attrs(l, u) for l, u in keys]
    n = len(keys)
    return [dict(A=A, i=i, y=gaps[i] if gaps else None, mk=mk, last=i + 1 == n, l=keys[i][0], r=keys[i + 1][0] if i + 1 < n else None)
            for i in range(n)]


def sep_ratio(Wi, ni, logp):
    """介入の板の比: p_i(帯|詰め物) = (介入の数え + 1 回分 × 本体の p(帯|詰め物)) / (介入の回数 + 1) を、本体の p(帯|詰め物) で割った対数。
    介入の板に何も無い帯・教えていない詰め物・「その他」は 0（影響なし。数えの無い帯を −log(回数+1) で下げない）。"""
    pm = np.exp(logp).astype(np.float32)
    pm[-1] = 1.0
    L = np.log((Wi + pm) / (ni[:, None] + 1).astype(np.float32)) - np.log(pm)
    L[Wi == 0] = 0
    L[ni == 0] = 0
    L[-1] = 0
    return L


class Model:
    def __init__(self, S, fill, r1):
        self.S, self.fill, self.r1 = S, fill, r1
        self.feat = lambda x: AE.feats(S, x, MODE)

    # ---- 学び（普通の数え） ----
    def fit(self, tr, hold):
        S = self.S
        TRs = [(s, AE.items(S, s, "節")) for s in tr]
        TR = [x for _, xs in TRs for x in xs]
        cnt = Counter(x["y"] for x in TR)
        self.pair_n = Counter((x["l"], x["r"]) for x in TR if not x["last"])
        self.pg = G.PairGrain(S.D).fit(((*self.feat(x), x["y"]) for x in TR), cnt)
        nc = self.nc = len(self.pg.cls)
        self.ci = {y: i for i, y in enumerate(self.pg.cls) if y is not None}
        self.pr = np.zeros(nc)
        for y, n in cnt.items():
            self.pr[self.cid(y)] += n
        self.Wd = np.zeros_like(self.pg.W)
        self.sep = False                                    # True: 介入の分は本体に混ぜず、介入だけの板（Wi）に持つ
        self.Wi = np.zeros_like(self.pg.W, dtype=np.float32)
        self.ni = np.zeros(nc)
        self.Li = None
        self.log, self.cur_text = [], ""                    # 介入の板の記録（教えた間ごと。番号で 1 つずつ取り消せる）
        self.who = AJ.Whole(S, nc, self.r1)
        self.rR, self.rL, self.rP = AJ.Ratio(nc, S.D), AJ.Ratio(nc, S.D), AJ.Ratio(nc, S.D)
        self.gram = AJ.Gram(nc, self.fill)
        TRK = [keys_of(s) for s, _ in TRs]
        self.who.index(TRK)
        for j, (s, xs) in enumerate(TRs):
            keys = TRK[j]
            if not keys:
                continue
            fic = self.who.fiction(keys, j)
            for x in xs:
                self.add_boards(keys, x, self.cid(x["y"]), 1.0, fic, G.split(s)[1][:x["i"]], base=False)
        self.gram.build()
        HO = [x for s in hold for x in AE.items(S, s, "節")]
        self.best_w = {}
        for b in range(len(AD.EV_BINS)):
            hx = [x for x in HO if AD.ev_bin(self.pair_n[(x["l"], x["r"])] if not x["last"] else 10) == b][:1500]
            F = [self.feat(x) for x in hx]
            self.best_w[b] = max((sum(self.pg.cls[int(np.argmax(w * self.pg.prior + self.pg.logp[:, f] @ v))] == x["y"] for (f, v), x in zip(F, hx)), w)
                                 for w in (0.0, 0.25, 0.5, 1.0))[1] if hx else 1.0
        self.beta = self.choose_beta(hold[:400])
        self.refresh()
        return self

    def clone(self):
        """変わる配列だけを写した物（語板・符号・引き先は分け合う）。"""
        m = copy.copy(self)
        m.pg = copy.copy(self.pg)
        m.pg.W, m.pg.logp, m.pg.prior = self.pg.W.copy(), self.pg.logp.copy(), self.pg.prior.copy()
        m.pr, m.Wd = self.pr.copy(), self.Wd.copy()
        m.Wi, m.ni, m.Li = self.Wi.copy(), self.ni.copy(), None if self.Li is None else self.Li.copy()
        m.log = [dict(e) for e in self.log]
        m.who = copy.copy(self.who)
        m.who.B, m.who.n = self.who.B.copy(), self.who.n.copy()
        for nm in ("rR", "rL", "rP"):
            R = copy.copy(getattr(self, nm))
            R.W = R.W.copy()
            setattr(m, nm, R)
        m.gram = copy.copy(self.gram)
        m.gram.R = copy.copy(self.gram.R)
        m.gram.R.W = self.gram.R.W.copy()
        return m

    def undo(self, numbers):
        """介入の板から、記録の番号の間を取り消す（同じ特徴を引く）。文 1 つ分なら その文の番号を全部。"""
        for k in numbers:
            e = self.log[k]
            if e["取り消し"]:
                continue
            keys = [tuple(z) for z in e["要点"]]
            x = made_items(self.S, keys, None, tuple(e["印"]))[e["間"]]
            f, v = self.feat(x)
            c = self.cid(e["詰め物"])
            np.add.at(self.Wi[c], f, -v / max(1e-9, float(v.sum())))
            self.ni[c] -= 1
            e["取り消し"] = True
        np.maximum(self.Wi, 0, out=self.Wi)
        self.refresh()

    def save_log(self, path):
        with open(path, "w", encoding="utf-8") as fo:
            for e in self.log:
                fo.write(json.dumps(e, ensure_ascii=False) + chr(10))

    def cid(self, y):
        return self.ci.get(y, self.nc - 1)

    def add_boards(self, keys, x, c, w, fic, placed, teach=False, base=True):
        """1 つの間の特徴を、詰め物 c の全部の板に重さ w で足す（学びでも介入でも同じ形）。base=False は今の点の板を除く（学びでは PairGrain.fit が数える）。"""
        S, i, last = self.S, x["i"], x["last"]
        if teach and self.sep:                              # 別の板: 今の点の特徴（和 1）を介入の板にだけ足し、記録を残す（本体は書き換えない）
            f, v = self.feat(x)
            np.add.at(self.Wi[c], f, v / max(1e-9, float(v.sum())))
            self.ni[c] += 1
            self.log.append(dict(番号=len(self.log), 時刻=time.strftime("%F %T"), 文=self.cur_text, 要点=[list(k) for k in keys], 間=x["i"],
                                 印=list(x["mk"]), 最後=x["last"], 詰め物=self.pg.cls[c], 重さ=1, 取り消し=False))
            return
        if base:
            f, v = self.feat(x)
            np.add.at(self.pg.W[c], f, v * self.scale(self.pg.W, c, float(v.sum()), w, teach))
            self.pr[c] += self.scale(None, c, 1.0, w, teach)
        Xi = self.who.gap(keys, i, fic)
        k1 = self.scale_who(c, Xi, w, teach)
        self.who.B[c] += k1 * Xi
        self.who.n[c] += k1
        for R, key in ((self.rR, keys[i]), (self.rL, None if last else keys[i + 1]), (self.rP, None if last else AJ.next_pred(keys, i))):
            if key is None:
                continue
            fa, va = S.attrs(*key)
            np.add.at(R.W[c], fa, va / max(1e-9, float(va.sum())) * self.scale(R.W, c, 1.0, w, teach))
        names = self.gram.state({u for _, u in keys}, placed, last)
        if self.gram.rows is not None and not hasattr(self.gram, "R"):
            self.gram.add(c, names)
        else:
            fg = self.gram.f(names, grow=False)
            if len(fg):
                np.add.at(self.gram.R.W[c], fg, np.ones(len(fg)) / len(fg) * self.scale(self.gram.R.W, c, 1.0, w, teach))

    def scale(self, W, c, mass, w, teach):
        """足す重さ。w が数なら その数（回分）。w が "平均" なら その板の平均の 1 例分 ÷ この例の量。"""
        if w != "平均":
            return float(w)
        if W is None:                                       # prior の数えは 1 回分
            return 1.0
        n = max(1.0, self.pr[c])
        return float(W[c].sum()) / n / max(1e-9, mass)

    def scale_who(self, c, Xi, w, teach):
        if w != "平均":
            return float(w)
        n = max(1.0, self.who.n[c])
        avg = float(np.abs(self.who.B[c]).sum()) / n
        return avg / max(1e-9, float(np.abs(Xi).sum()))

    def choose_beta(self, hold):
        out = {}
        HOs = [(s, AE.items(self.S, s, "節")) for s in hold]
        self.who.build()
        for nm, obj in (("R1", self.who), ("R2右向き", self.rR), ("R2左向き", self.rL), ("R2述語", self.rP), ("R3", self.gram)):
            best = None
            for b in AJ.BETAS:
                obj.build(b)
                hit = 0
                for s, xs in HOs:
                    keys = keys_of(s)
                    if not keys:
                        continue
                    gaps = G.split(s)[1]
                    fic = self.who.fiction(keys) if nm == "R1" else ()
                    for i, g in enumerate(gaps):
                        last = i + 1 == len(gaps)
                        if nm == "R1":
                            sc = self.who.score(self.who.gap(keys, i, fic))
                        elif nm == "R2右向き":
                            sc = self.rR.score(*self.S.attrs(*keys[i]))
                        elif nm == "R2左向き":
                            if last:
                                continue
                            sc = self.rL.score(*self.S.attrs(*keys[i + 1]))
                        elif nm == "R2述語":
                            p = None if last else AJ.next_pred(keys, i)
                            if not p:
                                continue
                            sc = self.rP.score(*self.S.attrs(*p))
                        else:
                            sc = self.gram.score(self.gram.state({u for _, u in keys}, gaps[:i], last))
                        hit += AJ.ranks(sc)[self.cid(g)] < 3
                if best is None or hit > best[0]:
                    best = (hit, b)
            out[nm] = best[1]
            obj.build(best[1])
        return out

    def refresh(self):
        pg = self.pg
        tot = pg.W.sum(1, keepdims=True)
        pg.logp = np.log((pg.W + G.ALPHA) / (tot + G.ALPHA * pg.D))
        if not G.OTHER:
            pg.logp[-1] = -1e9
        pg.prior = np.log((self.pr + 1) / (self.pr.sum() + len(self.pr)))
        self.Wn = pg.W / np.maximum(1e-9, np.linalg.norm(pg.W, axis=1, keepdims=True))
        self.who.build(self.beta["R1"])
        self.rR.build(self.beta["R2右向き"]), self.rL.build(self.beta["R2左向き"]), self.rP.build(self.beta["R2述語"])
        self.gram.R.build(self.beta["R3"])
        self.Li = sep_ratio(self.Wi, self.ni, pg.logp) if self.ni.any() else None

    # ---- 点 ----
    def gap_scores(self, keys, x, fic):
        S, pg, i, last = self.S, self.pg, x["i"], x["last"]
        f, v = self.feat(x)
        pw = self.best_w[AD.ev_bin(self.pair_n[(x["l"], x["r"])] if not last else 10)]
        s = pw * pg.prior + pg.logp[:, f] @ v - np.log1p(self.Wd[:, f] / (pg.W[:, f] + G.ALPHA)) @ v
        s = s - s.max()
        p = np.exp(s)
        p /= p.sum()
        if not G.OTHER:
            p[-1] = 0
        parts = [self.rR.score(*S.attrs(*keys[i]))]
        if not last:
            parts.append(self.rL.score(*S.attrs(*keys[i + 1])))
            pr_ = AJ.next_pred(keys, i)
            if pr_:
                parts.append(self.rP.score(*S.attrs(*pr_)))
        now = np.log(np.maximum(p, 1e-30)).astype(np.float32)
        if self.Li is not None:
            now = now + self.Li[:, f] @ (v / max(1e-9, float(v.sum())))
        return dict(今=now, R1=self.who.score(self.who.gap(keys, i, fic)), R2=np.mean(parts, axis=0))

    def walk(self, keys, xs, gold=None):
        """頭から順に置く。間ごとに (候補, 選んだ物, 点の辞書) を返す。gold があれば R3 の状態は元の詰め物で置いた物（練習の時）。"""
        fic = self.who.fiction(keys)
        ups = {u for _, u in keys}
        placed, out = [], []
        for x in xs:
            g = self.gap_scores(keys, x, fic)
            r3 = self.gram.score(self.gram.state(ups, placed, x["last"]))
            three = g["R1"] + g["R2"] + r3
            both = g["今"] + three
            if not G.OTHER:
                three[-1] = both[-1] = -1e9
            cand = (set(np.argsort(-g["今"])[:K]) | set(np.argsort(-three)[:K])) - (set() if G.OTHER else {self.nc - 1})
            pick = max(cand, key=lambda c: both[c])
            out.append(dict(候補=cand, 選び=int(pick), 今=g["今"], 三つ=three, **{"今×三つ": both}))
            placed.append(gold[x["i"]] if gold is not None else self.pg.cls[pick])
        return out, fic

    # ---- 介入 ----
    def teach(self, keys, xs, gaps, outside, w, only=None, text=""):
        """only: その詰め物の間だけを教える（ほかの間は見るだけ）。text: 記録に残す文。"""
        self.cur_text = text or "".join(k[0] for k in keys)
        res, fic = self.walk(keys, xs, gold=gaps)
        n = Counter()
        for x, r in zip(xs, res):
            y = self.cid(x["y"])
            if y == self.nc - 1 or (only is not None and x["y"] != only):
                continue
            if y in r["候補"] or outside:
                self.add_boards(keys, x, y, w, fic, gaps[:x["i"]], teach=True)
                n["教えた"] += 1
                n["候補の外"] += y not in r["候補"]
            if r["選び"] != y:
                f, v = self.feat(x)
                np.add.at(self.Wd[r["選び"]], f, v * DARK_W)
                n["暗い側"] += 1
        return n

    # ---- 測る ----
    def evaluate(self, te, surf):
        c = Counter()
        for s in te:
            xs = AE.items(self.S, s, "節")
            if not xs:
                continue
            cont, gaps, head = G.split(s)
            keys = keys_of(s)
            res, _ = self.walk(keys, xs)
            seq = [self.pg.cls[r["選び"]] for r in res]
            c["文"] += 1
            c["間n"] += len(gaps) - 1
            for i, (g, y, r) in enumerate(zip(seq, gaps, res)):
                c[":間" if i + 1 < len(gaps) else ":文末"] += g == y
                c[":の"] += g == "の" and i + 1 < len(gaps)
                c["候補n"] += 1
                c[":候補"] += self.cid(y) in r["候補"]
                rk = AJ.ranks(r["今×三つ"])[self.cid(y)]
                c[":3"] += rk < 3
                c[":5"] += rk < 5
            sent = "".join(surf.form(l, u, g) + g for (l, f_, u), g in zip(cont, seq))
            c[":文"] += G.norm(head + sent) == G.norm(s["text"])
        d = lambda k, n: round(c[k] / max(1, c[n]), 3)
        return dict(間=d(":間", "間n"), 文末=d(":文末", "文"), 文=d(":文", "文"), の=d(":の", "間n"), 候補に元=d(":候補", "候補n"),
                    上位3=d(":3", "候補n"), 上位5=d(":5", "候補n"), 例=self.examples())

    def examples(self):
        out = {}
        for keys, gi, want in EXAMPLES:
            xs = made_items(self.S, list(keys), None)
            res, _ = self.walk(list(keys), xs)
            r = res[gi]
            wc = self.cid(want)
            out[want] = dict(順位=int(AJ.ranks(r["今×三つ"])[wc]) + 1, 候補に入る=wc in r["候補"], 選び=self.pg.cls[r["選び"]])
        return out


def main():
    ap = argparse.ArgumentParser()
    for k in ("conllu", "content", "func", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--inits", type=int, nargs="*", default=[5000, 20000, 50000])
    ap.add_argument("--practice", type=int, default=5000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--D", type=int, default=8192)
    ap.add_argument("--r1", default="基準", choices=AJ.R1_VARS)
    a = ap.parse_args()
    t0 = time.time()
    sents = [s for s in G.read_conllu(a.conllu) if s["toks"]]
    te = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 == 0][: a.test]
    tr_all = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 != 0]
    hold = tr_all[-2000:]
    tr_all = tr_all[:-2000]
    cb = json.load(open(a.content, encoding="utf-8"))
    func = G.Func(json.load(open(a.func, encoding="utf-8")))
    S = G.AttrSpace(cb, func, a.D, bands=("B", "C", "D", "E", "G"))
    fill = AD.Fill(S)
    surf = G.Surface()
    top = max(a.inits)
    pr = tr_all[top: top + a.practice]                     # 練習の文: どの最初の学びにも入っていない文
    for s in tr_all[:top] + pr:
        fill.see(s)
        cont, gaps, _ = G.split(s)
        for (l, f_, u), g in zip(cont, gaps):
            surf.add(l, f_, u, g)
    res = dict(R1の型=a.r1, 練習の文=len(pr), 試しの文=len(te), K=K, DARK_W=DARK_W, REFRESH=REFRESH, 学び={})
    print(f"読み込み {time.time() - t0:.0f} 秒", flush=True)

    def save():
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    for n0 in a.inits:
        t1 = time.time()
        base = Model(S, fill, a.r1).fit(tr_all[:n0], hold)
        o = dict(β=base.beta, prior=base.best_w, 前=base.evaluate(te, surf))
        res["学び"][str(n0)] = o
        print(n0, "前", round(time.time() - t1), "秒", json.dumps(o["前"], ensure_ascii=False), flush=True)
        save()
        # 同じ練習の文を、ただ学びに足す（普通の数え）
        t2 = time.time()
        plain = Model(S, fill, a.r1).fit(tr_all[:n0] + pr, hold)
        o["ただ足す"] = plain.evaluate(te, surf)
        print(n0, "ただ足す", round(time.time() - t2), "秒", json.dumps(o["ただ足す"], ensure_ascii=False), flush=True)
        save()
        del plain
        for outside in (False, True):
            for w in (1.0, "平均"):
                t2 = time.time()
                m = base.clone()
                n = Counter()
                for j, s in enumerate(pr):
                    xs = AE.items(S, s, "節")
                    if xs:
                        n.update(m.teach(keys_of(s), xs, G.split(s)[1], outside, w))
                    if (j + 1) % REFRESH == 0:
                        m.refresh()
                m.refresh()
                nm = f"介入・{'候補の外も' if outside else '候補の中だけ'}・{'1 回分' if w == 1.0 else '平均の 1 例分'}"
                o[nm] = dict(数=dict(n), 後=m.evaluate(te, surf))
                print(n0, nm, round(time.time() - t2), "秒", json.dumps(o[nm], ensure_ascii=False), flush=True)
                save()
                del m
        if n0 == top:                                       # 自作の文を 1・3・10 文だけ教える
            o["自作を教える"] = {}
            for outside in (False, True):
                for k in (1, 3, 10):
                    m = base.clone()
                    n = Counter()
                    for a_, b_ in TEACH[:k]:
                        keys, gaps = made_sentence(a_, b_)
                        n.update(m.teach(keys, made_items(S, keys, gaps), gaps, outside, 1.0))
                        m.refresh()
                    nm = f"{'候補の外も' if outside else '候補の中だけ'}・{k} 文"
                    o["自作を教える"][nm] = dict(数=dict(n), 例=m.examples())
                    print(n0, "自作", nm, json.dumps(o["自作を教える"][nm], ensure_ascii=False), flush=True)
                    del m
            save()
    res["秒"] = round(time.time() - t0)
    save()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
