"""I48bf: 周波数で位置を持つ語板（tools/i48_freq_board.py の覚え方）を、本番と同じ 5 万文の学び・1,000 文の試しで測る（案、未確定）。

行 = 頭の詰め物・内容語・詰め物・内容語・詰め物 …（詰め物は本体の詰め物のクラス 1 つを 1 つの単位にする。機能語も入る）。
単位ごとに、属性の帯（品詞・機能語・B・C・D・E・G・名前）ごとに 100 個の周波数の値を持つ（帯の中は属性ごとの乱数の位相、
名前の帯は単位の名前だけの乱数の位相を重み ε で混ぜる。詰め物は名前の符号だけ: 機能語の属性の値を入れると詰め物どうしが似すぎて
（2,000〜5,000 文の小さい試しで、詰め物の値どうしの近さの平均 .42、いちばん近い相手で .93）読めなかったため）。周波数のずれ ν_j = j × W / 100（検波した後の物。1,000 倍の搬送は
整数の位置では同じになるので持たない。tools/i48_freq_board2.py で本物の波と一致を確かめた）。
覚え方: 行の中の全部の組（p, q）で、単位 p の記憶の箱 R_p に「単位 q の値 × e^(−i 2π ν (q − p))」を足す（マイナス位置あり）。
読み方（そろい具合 FM で読む。I48bc と同じ形）: 箱 R_b を距離 d だけ回して候補 c の値と重ね、帯を 4 つずつに分けた 32 本の小帯ごとの重なり z を出す。
  強さ n̂_c = max(0, Re Σ z)、そろい具合 FM = |Σ z| / Σ |z|、FM の床 = 使っていない値 32 本の FM の 平均 ＋ 標準偏差 × √(2 ln V)。
  詰め物の点（差し込み）: その箱・距離での全部の候補の n̂ の和を N̂ として、点 = log((n̂ + p_c) / (N̂ + 1) / p_c)、FM が床を越えた候補だけ。
  箱の数え N が床 2 ln V に届かない語、N̂ が床に届かない距離は点を足さない（本体に頼る）。
比べる形: 隣だけ（間の左の語の d = +1、右の語の d = −1。I48bc・I48bd の表と同じ範囲） ／ 行の全部の内容語（それぞれの距離で）。
  W = 1 と 0.25。W = 1・全部では「1 語を仮に決めてから残りを読み直す」形も 1 つ試す
  （行の全部の間をまず読み、1 位と 2 位の差のいちばん大きい間の詰め物を仮に決め、その詰め物の箱からの点も足して、ほかの間を読み直す）。
測る物:
  1 間の穴埋め（本体と同じ AP.measure: 間・文末・文。本体 .5125 / .5836 / .0771、表 .5711 と並べる）
  2 行の位置ごとに、ほかの単位（すぐ隣だけ ／ 全部）から読んだ 1 位（FM で床を越えた中で n̂ のいちばん大きい物）: 内容語の位置で品詞が合う率・語が合う率、
    詰め物の位置で詰め物が合う率。当てずっぽう（いつも一番多い物）と並べる
  3 行全体の整合: 試しの行と、語順を入れ替えた行（隣り合う内容語 2 つ）／内容語 1 つを品詞の違う語に替えた行を見分ける AUC。
    位置の点 = 本当の単位の n̂ を同じ種類の候補の中で平均 0・ばらつき 1 にそろえた値、と、本当の単位の FM − 床。行の平均と最小の両方
  5 記憶の箱の数と混み具合（箱ごとの「相手 × 距離」の種類の数 ÷ 値の数 800）、大きさ、秒
手で置く物: 帯ごとの周波数の数 100、小帯 4、名前の重み ε 0.1（内容語）・1.0（詰め物）、2・3 の読みの範囲（隣だけ・全部）、床の値の数 32、W の並び（1・0.25）、乱数の種、
  入れ替え・替える語の選び方（品詞の違う語、数えが床以上の語から乱数で）。
python intervention/i48bf_freq_position.py --conllu ... --content ... --func ... --out out.json [--train 50000] [--test 1000]
"""
from __future__ import annotations

import argparse
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
import i48ap_units as AP  # noqa: E402
import i48at_recheck as RC  # noqa: E402
import i48ax_sayable as AX  # noqa: E402
from i48core import learn as LN  # noqa: E402
from i48core import model as MD  # noqa: E402

BANDS = ("品詞", "機能語", "B", "C", "D", "E", "G", "名前")
NC = 100
NSUB = 4
EPS = 0.1
EPS_F = 1.0                         # 詰め物は名前の符号だけ（属性の値を入れると詰め物どうしが似すぎる。小さい試しで近さの平均 .42、1 位が当てずっぽうより下）
P_PROBE = 32
WS = (1.0, 0.25)
TOPK = 4
F = len(BANDS) * NC


def shape(key):
    r = np.random.default_rng(zlib.crc32(key.encode()))
    return np.exp(2j * np.pi * r.random(NC)).astype(np.complex64) / np.sqrt(NC)


def content_names(S, lemma, upos):
    out = [("品詞", "品詞:" + upos, 1.0)]
    w = S.cb.get(lemma)
    if w is not None:
        for b in ("B", "C", "D", "E", "G"):
            for a, v in sorted(w["attrs"].get(b, {}).items(), key=lambda z: -z[1])[:TOPK]:
                out.append((b, b + "|" + a, float(v)))
    return out


def func_names(S, t):
    out = [("品詞", "品詞:" + t["upos"], 1.0)]
    w = S.func.board(dict(lemma=t["lemma"], form=t["form"], xpos=t["xpos"]))
    if w is not None:
        for b, A in w["attrs"].items():
            for a, v in A.items():
                out.append(("機能語", "機能語|" + b + ":" + a, float(v)))
    else:
        out.append(("機能語", "記号:" + t["form"], 1.0))
    return out


def value(names, name, eps=None):
    eps = EPS if eps is None else eps
    v = np.zeros(F, dtype=np.complex64)
    for bi, b in enumerate(BANDS[:-1]):
        x = np.zeros(NC, dtype=np.complex64)
        for bb, nm, wt in names:
            if bb == b:
                x += wt * shape(b + "|" + nm)
        n = np.linalg.norm(x)
        if n > 0:
            v[bi * NC:(bi + 1) * NC] = x / n
    n = np.linalg.norm(v)
    v = (1 - eps) * (v / n if n > 0 else v)
    v[(len(BANDS) - 1) * NC:] = eps * shape("名前|" + name)
    return v / np.linalg.norm(v)


def auc(pos, neg):
    """pos（元の行）の値が neg（崩した行）より大きい確からしさ。"""
    a = np.array(list(neg) + list(pos))
    r = a.argsort().argsort() + 1.0
    rp = r[len(neg):].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


class FreqPos:
    """差し込み: 周波数で位置を持つ記憶の箱で、詰め物の点を足す。ctx は "隣" か "全部"、fix は「1 語を仮に決めてから読み直す」。"""

    def __init__(self, m, S, W, seed=0):
        self.S, self.W = S, W
        self.nu = np.tile(np.arange(NC) * W / NC, len(BANDS)).astype(np.float64)
        self.ctx, self.fix = "隣", False
        self.nc = m.nc
        r = np.random.default_rng(seed + 7919)
        pr = np.exp(2j * np.pi * r.random((P_PROBE, F))).astype(np.complex64)
        self.probe = pr / np.linalg.norm(pr, axis=1, keepdims=True)
        self.cache, self.fcache = {}, {}

    def clone(self):
        return self

    def refresh(self, m):
        pass

    def E(self, d):
        return np.exp(-2j * np.pi * self.nu * d).astype(np.complex64)

    def line(self, m, s, known=True):
        cont, gaps, head = G.split(s)
        u = [self.K + m.cid(head)]
        for (l, _, up), g in zip(cont, gaps):
            u.append(self.kid.get((l, up), -1))
            u.append(self.K + m.cid(g))
        return u

    def fit(self, m, tr):
        t0 = time.time()
        keys = Counter()
        cls_tok = {}
        for s in tr:
            cont, gaps, head = G.split(s)
            for l, _, up in cont:
                keys[(l, up)] += 1
            T = s["toks"]
            ci = [i for i, t in enumerate(T) if G.is_content(t)]
            if not ci:
                continue
            bounds = [(0, ci[0])] + [(ci[k] + 1, ci[k + 1] if k + 1 < len(ci) else len(T)) for k in range(len(ci))]
            for g, (a, b) in zip([head] + gaps, bounds):
                c = m.cid(g)
                if c not in cls_tok:
                    cls_tok[c] = T[a:b]
        self.keys = list(keys)
        self.kid = {k: i for i, k in enumerate(self.keys)}
        self.K = K = len(self.keys)
        U = K + m.nc
        V = np.zeros((U, F), dtype=np.complex64)
        for i, (l, up) in enumerate(self.keys):
            V[i] = value(content_names(self.S, l, up), l + "|" + up)
        for c in range(m.nc):
            nm = [x for t in cls_tok.get(c, []) for x in func_names(self.S, t)]
            y = m.pg.cls[c]
            V[K + c] = value(nm, "詰め物|" + ("その他" if y is None else (y or "空")), EPS_F)
        self.V = V
        self.U = U
        lens = [2 * len(G.split(s)[0]) + 1 for s in tr]
        self.Lmax = max(lens)
        Et = np.stack([self.E(d) for d in range(-self.Lmax, self.Lmax + 1)])
        R = np.zeros((U, F), dtype=np.complex64)
        N = np.zeros(U, dtype=np.int64)
        codes = []
        span = 2 * self.Lmax + 1
        for s in tr:
            u = np.array(self.line(m, s))
            L = len(u)
            if L < 2:
                continue
            P = np.arange(L)
            Dm = P[None, :] - P[:, None]                    # q − p
            X = V[u]
            C = (Et[Dm + self.Lmax] * X[None, :, :]).sum(1) - X   # 自分自身（d = 0）を除く
            np.add.at(R, u, C)
            np.add.at(N, u, 1)
            w_ = getattr(self, "win", None)                  # I48bg: 覚える組の距離を限る時は、混み具合もその範囲で数える
            mask = (Dm != 0) if w_ is None else ((Dm != 0) & (np.abs(Dm) <= w_))
            codes.append(((u[:, None].astype(np.int64) * U + u[None, :]) * span + (Dm + self.Lmax))[mask])
        self.R, self.N = R, N
        cls_cnt = Counter()
        for s in tr:
            for g in G.split(s)[1]:
                cls_cnt[m.cid(g)] += 1
        tot = sum(cls_cnt.values())
        self.p = np.array([(cls_cnt[c] + 1) / (tot + m.nc) for c in range(m.nc)], dtype=np.float64)
        kc = np.array([keys[k] for k in self.keys], dtype=np.float64)
        self.pk = kc / kc.sum()
        self.floor_n = 2 * np.log(m.nc)
        self.ok_box = N >= self.floor_n
        cd = np.unique(np.concatenate(codes))
        box = cd // (U * span)
        per = np.bincount(box, minlength=U)
        used = per[N > 0]
        load = used / F
        self.stats = dict(学び秒=round(time.time() - t0), 箱の数=int((N > 0).sum()), 内容語の箱=int((N[:K] > 0).sum()), 詰め物の箱=int((N[K:] > 0).sum()),
                          数えが床以上の箱=int(self.ok_box.sum()), 床=round(float(self.floor_n), 2), 行の長さの最大=self.Lmax,
                          箱ごとの相手x距離の種類=dict(平均=round(float(used.mean()), 1), 中央=float(np.median(used)),
                                              上位1割=float(np.quantile(used, 0.9)), 最大=int(used.max())),
                          混み具合_種類割る値の数=dict(平均=round(float(load.mean()), 3), 中央=round(float(np.median(load)), 3),
                                              一を越える箱の割合=round(float((load > 1).mean()), 4), 最大=round(float(load.max()), 1)),
                          記憶の箱_MB=round(R.nbytes / 1e6, 1), 値_MB=round(V.nbytes / 1e6, 1))
        return self

    def read(self, b, d, cand):
        """箱 b を距離 d で読み、候補 cand（単位の番号の並び）の (n̂, FM, 床) を返す。"""
        Q = np.conj(self.R[b]) * self.E(d)
        Z = (self.V[cand] * Q[None, :]).reshape(len(cand), len(BANDS) * NSUB, -1).sum(2)
        tot = Z.sum(1)
        nhat = np.maximum(0.0, np.real(tot))
        fm = np.abs(tot) / np.maximum(1e-9, np.abs(Z).sum(1))
        Zp = (self.probe * Q[None, :]).reshape(P_PROBE, len(BANDS) * NSUB, -1).sum(2)
        fp = np.abs(Zp.sum(1)) / np.maximum(1e-9, np.abs(Zp).sum(1))
        return nhat, fm, float(fp.mean() + fp.std() * np.sqrt(2 * np.log(len(cand))))

    def pts(self, b, d):
        key = (b, d)
        if key not in self.cache:
            out = None
            if b >= 0 and self.ok_box[b]:
                nhat, fm, floor = self.read(b, d, np.arange(self.K, self.K + self.nc))
                Nh = float(nhat.sum())
                if Nh >= self.floor_n:
                    s = np.log((nhat + self.p) / (Nh + 1) / self.p)
                    out = np.where(fm > floor, s, 0.0).astype(np.float32)
            self.cache[key] = out
        return self.cache[key]

    def gap_pts(self, keys, i):
        n = len(keys)
        pg = 2 * i + 2
        ks = [i, i + 1] if self.ctx == "隣" else list(range(n))
        out = np.zeros(self.nc, dtype=np.float32)
        for k in ks:
            if k >= n:
                continue
            v = self.pts(self.kid.get(tuple(keys[k]), -1), pg - (2 * k + 1))
            if v is not None:
                out += v
        return out

    def score(self, m, keys, x, f, v):
        i = x["i"]
        if not self.fix:
            out = self.gap_pts(keys, i)
        else:
            kt = tuple(map(tuple, keys))
            if kt not in self.fcache:
                n = len(keys)
                base = [self.gap_pts(keys, j) for j in range(n)]
                marg = [float(np.diff(np.sort(b)[-2:])[0]) for b in base]
                j = int(np.argmax(marg))
                c = int(np.argmax(base[j]))
                fin = []
                for k in range(n):
                    if k == j:
                        fin.append(base[k])
                        continue
                    add = self.pts(self.K + c, (2 * k + 2) - (2 * j + 2))
                    fin.append(base[k] + (add if add is not None else 0))
                self.fcache[kt] = fin
            out = self.fcache[kt][i].copy()
        if not G.other_ok(m):
            out[-1] = 0
        return out

    # ---- 2・3 行の読み ----
    def line_read(self, u, p, cand, ctx="全部"):
        Q = np.zeros(F, dtype=np.complex64)
        for q, b in enumerate(u):
            if q != p and b >= 0 and self.ok_box[b] and (ctx == "全部" or abs(q - p) == 1):
                Q += np.conj(self.R[b]) * self.E(p - q)
        tot = self.V[cand] @ Q                                    # Re Σ V_c Q（全部の小帯の和）
        return np.real(tot), Q

    def fm_of(self, Q, c):
        Z = (self.V[c] * Q).reshape(len(BANDS) * NSUB, -1).sum(1)
        return float(np.abs(Z.sum()) / max(1e-9, np.abs(Z).sum()))

    def floor_of(self, Q, ncand):
        Zp = (self.probe * Q[None, :]).reshape(P_PROBE, len(BANDS) * NSUB, -1).sum(2)
        fp = np.abs(Zp.sum(1)) / np.maximum(1e-9, np.abs(Zp).sum(1))
        return float(fp.mean() + fp.std() * np.sqrt(2 * np.log(ncand)))


def line_tests(fp, m, te, ctx, seed=5):
    t0 = time.time()
    ck = np.array([i for i in range(fp.K) if fp.ok_box[i]])
    cg = np.arange(fp.K, fp.K + fp.nc)
    up = np.array([fp.keys[i][1] for i in range(fp.K)])
    rng = np.random.default_rng(seed)
    c2 = dict(内容語_位置=0, 品詞が合う=0, 語が合う=0, 詰め物_位置=0, 詰め物が合う=0)
    top_pos = Counter(up[ck]).most_common(1)[0][0]
    cls_freq = int(np.argmax(fp.p))
    guess = dict(品詞=0, 詰め物=0)
    lines = {"元": [], "語順の入れ替え": [], "品詞の違う語に替える": []}

    def pos_scores(u):
        z, f = [], []
        for p in range(1, len(u)):
            if u[p] < 0:
                continue
            cand = ck if p % 2 == 1 else cg
            am, Q = fp.line_read(u, p, cand, ctx)
            j = np.searchsorted(cand, u[p]) if p % 2 == 1 else u[p] - fp.K
            if p % 2 == 1 and (j >= len(cand) or cand[j] != u[p]):
                continue
            z.append((am[j] - am.mean()) / max(1e-9, am.std()))
            f.append(fp.fm_of(Q, u[p]) - fp.floor_of(Q, len(cand)))
        return z, f

    for s in te:
        u = fp.line(m, s)
        if len(u) < 3:
            continue
        # 2 位置ごとの 1 位
        for p in range(1, len(u)):
            if u[p] < 0:
                continue
            cand = ck if p % 2 == 1 else cg
            am, Q = fp.line_read(u, p, cand, ctx)
            fl = fp.floor_of(Q, len(cand))
            best = None
            for j in np.argsort(-am)[:200]:
                if fp.fm_of(Q, cand[j]) > fl:
                    best = int(cand[j])
                    break
            if best is None:
                best = int(cand[int(np.argmax(am))])
            if p % 2 == 1:
                c2["内容語_位置"] += 1
                c2["品詞が合う"] += fp.keys[best][1] == fp.keys[u[p]][1]
                c2["語が合う"] += best == u[p]
                guess["品詞"] += top_pos == fp.keys[u[p]][1]
            else:
                c2["詰め物_位置"] += 1
                c2["詰め物が合う"] += best == u[p]
                guess["詰め物"] += (fp.K + cls_freq) == u[p]
        # 3 崩した行
        kpos = [p for p in range(1, len(u), 2) if u[p] >= 0 and fp.ok_box[u[p]]]
        if len(kpos) < 2:
            continue
        lines["元"].append(pos_scores(u))
        a = [k for k in range(len(kpos) - 1) if kpos[k + 1] == kpos[k] + 2]
        if a:
            k = a[int(rng.integers(len(a)))]
            u2 = list(u)
            u2[kpos[k]], u2[kpos[k + 1]] = u2[kpos[k + 1]], u2[kpos[k]]
            lines["語順の入れ替え"].append(pos_scores(u2))
        p = kpos[int(rng.integers(len(kpos)))]
        other = ck[up[ck] != fp.keys[u[p]][1]]
        u3 = list(u)
        u3[p] = int(other[int(rng.integers(len(other)))])
        lines["品詞の違う語に替える"].append(pos_scores(u3))
    out = {"2_位置ごとの1位": dict(内容語の位置=c2["内容語_位置"], 品詞が合う率=round(c2["品詞が合う"] / max(1, c2["内容語_位置"]), 4),
                                 語が合う率=round(c2["語が合う"] / max(1, c2["内容語_位置"]), 4),
                                 品詞_当てずっぽう=round(guess["品詞"] / max(1, c2["内容語_位置"]), 4),
                                 詰め物の位置=c2["詰め物_位置"], 詰め物が合う率=round(c2["詰め物が合う"] / max(1, c2["詰め物_位置"]), 4),
                                 詰め物_当てずっぽう=round(guess["詰め物"] / max(1, c2["詰め物_位置"]), 4))}
    agg = {}
    for nm, L in lines.items():
        agg[nm] = {"強さ_平均": [float(np.mean(z)) for z, _ in L], "強さ_最小": [float(np.min(z)) for z, _ in L],
                   "FM_平均": [float(np.mean(f)) for _, f in L], "FM_最小": [float(np.min(f)) for _, f in L]}
    r3 = {}
    for nm in ("語順の入れ替え", "品詞の違う語に替える"):
        r3[nm] = dict(行=len(lines[nm]), **{k: round(auc(agg["元"][k], agg[nm][k]), 4) for k in agg["元"]})
    out["3_行全体の整合_AUC"] = dict(元の行=len(lines["元"]), **r3)
    out["2・3_秒"] = round(time.time() - t0)
    return out


def main():
    ap = argparse.ArgumentParser()
    for k in ("conllu", "content", "func", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--train", type=int, default=50000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--D", type=int, default=8192)
    ap.add_argument("--r1", default="架空・前")
    a = ap.parse_args()
    t0 = time.time()
    sents = [s for s in G.read_conllu(a.conllu) if s["toks"]]
    te = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 == 0][: a.test]
    tr_all = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 != 0]
    hold, tr = tr_all[-2000:], tr_all[:-2000][: a.train]
    ref = tr_all[:-2000][a.train:]
    cb = json.load(open(a.content, encoding="utf-8"))
    func = G.Func(json.load(open(a.func, encoding="utf-8")))
    S = G.AttrSpace(cb, func, a.D, bands=("B", "C", "D", "E", "G"))
    fill, surf = AD.Fill(S), G.Surface()
    for s in tr:
        fill.see(s)
        cont, gaps, _ = G.split(s)
        for (l, f_, u), g in zip(cont, gaps):
            surf.add(l, f_, u, g)
    base = MD.Model(S, fill, a.r1).fit(tr, hold)
    print(f"学び {time.time() - t0:.0f} 秒", flush=True)
    for s in te:
        AE.items(S, s, "節")
    import i48core.measure as ME
    for _, _, keys, _ in ME.WANT:
        MD.made_items(S, keys, None)
    fill_ref = AD.Fill(S)
    for s in ref:
        fill_ref.see(s)
    say, nat = AX.Sayable(S, ref), RC.Natural(fill_ref, ref)
    res = dict(学びの文=len(tr), 試しの文=len(te), 周波数の数=NC, 帯=list(BANDS), 小帯=NSUB, 名前の重み=EPS, 床の値=P_PROBE,
               比べ_I48bd=dict(本体=dict(間=0.5125, 文末=0.5836, 文=0.0771), 表=dict(間=0.5711)), 型={})

    def save():
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    def report(nm, m):
        t1 = time.time()
        r = AP.measure(m, te, surf)
        r["言えるか"] = AX.score(m, te, say, nat)
        r["筋の無い共振"] = LN.monitor(m, te)
        r["秒"] = round(time.time() - t1)
        res["型"][nm] = r
        print(nm, r["秒"], "秒", json.dumps({k: r[k] for k in ("間", "文末", "文", "と_人", "と_人以外")}, ensure_ascii=False), flush=True)
        save()
    report("本体", base)
    for W in WS:
        fp = FreqPos(base, S, W).fit(base, tr)
        res.setdefault("箱", {})[f"W {W}"] = fp.stats
        print(f"W {W}", json.dumps(fp.stats, ensure_ascii=False), flush=True)
        save()
        for ctx, fix in [("隣", False), ("全部", False)] + ([("全部", True)] if W == WS[0] else []):
            m = base.clone()
            fp.ctx, fp.fix = ctx, fix
            m.extras.append(fp)
            m.refresh()
            report(f"本体 ＋ 周波数の箱（W {W}・{ctx}" + ("・1 語を仮に決めて読み直す" if fix else "") + "）", m)
        for ctx in ("隣", "全部"):
            nm = f"W {W}・読む範囲 {ctx}"
            res.setdefault("行の読み", {})[nm] = line_tests(fp, base, te, ctx)
            print(nm, "行の読み", json.dumps(res["行の読み"][nm], ensure_ascii=False), flush=True)
            save()
        del fp
    res["秒"] = round(time.time() - t0)
    save()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
