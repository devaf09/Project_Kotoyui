"""I47d: 否定を、意味とは別の周波数の帯として持つ試し。

I47c で、否定（ない・ぬ）を流れから学ばせても板としては現れなかった（否定の付いた語は元の語とほぼ同じ所に
居て、動く向きもそろわない）。そこで、否定を学ばせず、最初から別の帯として持たせる。

  意味の帯   : 語の波 m（食べる と 食べない で同じ。流れから学ぶ）
  向きの帯   : 述語だけ、m に「肯定の鍵 A」か「否定の鍵 N」を掛けた波（位相をずらす = 周波数をずらす）
  行の波     : 語の波の重ね合わせ [意味の帯 ; 向きの帯]

  読み出し   : 向きの帯に鍵の逆を掛けて m と共鳴させると、その語が肯定で居るか・否定で居るかが別々に出る。
               鍵が違うので、食べる と 食べない を重ねても打ち消し合わない。

比べる相手:
  逆位相     : 否定は m の符号を返す（位相 π）。食べる と 食べない を重ねると消える。
  別の語     : I47c の形（食べる＋否 を別の語として流れから学ぶ）。意味が元の語と分かれる。

測ること:
  read  : 行を k 本重ねた波から「どの語が、どちらの向きで居るか」を読めるか（AUC。しきい値は置かない）。
          同じ語が肯定と否定の両方で重なっている時に、両方を読めるか。
  scoop : 否定のある行を問いにして共鳴した場面をすくう。同じ述語が居る場面の割合（意味が保たれるか）と、
          その述語が同じ向き（否定）で居る割合（向きが保たれるか）。次の行の語の当たりも見る。
  --fig : 意味の板（1 枚を共有）の上に、向きの帯から読み出した 否定の強さ を色で重ねた図。

使い方:
  python intervention/i47d_negation_band.py --cache wn50x5_parsed.json --out i47d_wn.json --fig i47d_wn.png
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter

import numpy as np

from i47_gpu import Scenes
from i47b_layers_negation import NEG, Waves, content, som

PRED = ("VERB", "ADJ")


def split(toks):
    """行の語を (語, 向き) にする。向き: 0 = 帯なし（名詞など）、1 = 肯定の述語、2 = 否定の述語。"""
    pos = {t["lemma"]: t["pos"] for t in toks}
    out = []
    for w in content(toks, neg=True):
        if w.endswith(NEG):
            out.append((w[: -len(NEG)], 2))
        else:
            out.append((w, 1 if pos.get(w) in PRED else 0))
    return out


def auc(pos, neg):
    """居る組の読み出しが、居ない組より強い割合。"""
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    r = allv.argsort().argsort() + 1.0
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


class Band:
    def __init__(self, wv, seed=0, own=True):
        rng = np.random.default_rng(seed + 7)
        self.wv = wv
        self.B = wv.base if own else wv.W                   # 向きの帯に載せる波: 語の自分の印（own）か、流れで寄せた波か
        self.key = {1: np.exp(1j * rng.uniform(0, 2 * np.pi, wv.D)), 2: np.exp(1j * rng.uniform(0, 2 * np.pi, wv.D))}

    def line(self, ws, idf, mode="band", beta=1.0):
        """行の波。mode: band = 鍵を掛けた帯、pi = 逆位相、none = 意味の帯だけ。"""
        m = np.zeros(self.wv.D, complex)
        b = np.zeros(self.wv.D, complex)
        n = 0
        for w, p in ws:
            j = self.wv.ix.get(w)
            if j is None:
                continue
            a = idf.get(w, 1.0)
            m += a * self.wv.W[j]
            n += 1
            if p and mode == "band":
                b += a * self.B[j] * self.key[p]
            elif p and mode == "pi":
                b += a * self.B[j] * (1 if p == 1 else -1)
        if not n:
            return None
        v = np.concatenate([m, beta * b]) if mode != "none" else np.concatenate([m, 0 * b])
        return v / (np.linalg.norm(v) + 1e-9)

    def read(self, v, j, p, mode="band"):
        """行の波 v の向きの帯から、語 j が向き p で居る強さを読む。"""
        b = v[self.wv.D:]
        if mode == "band":                                  # 鍵の逆を掛けると、居る語は位相がそろって実部に出る
            return max(0.0, float(np.real(np.vdot(self.B[j], b * self.key[p].conj()))))
        s = float(np.real(np.vdot(self.B[j], b)))
        return max(0.0, s if p == 1 else -s)


def part_read(lines, wv, bd, idf, a, res):
    rng = random.Random(a.seed)
    pool = [ws for ws in lines if any(p == 2 for _, p in ws) and sum(w in wv.ix for w, _ in ws) >= 2]
    preds = sorted({w for ws in lines for w, p in ws if p and w in wv.ix})
    out = {}
    for k in (1, 4, 16, 64):
        sc = {m: ([], []) for m in ("band", "pi")}
        both = {m: [] for m in ("band", "pi")}
        for _ in range(a.trials):
            pick = rng.sample(pool, k)
            here = {(w, p) for ws in pick for w, p in ws if p and w in wv.ix}
            for m in sc:
                v = sum(x for x in (bd.line(ws, idf, m) for ws in pick) if x is not None)
                for w, p in here:
                    sc[m][0].append(bd.read(v, wv.ix[w], p, m))
                    if (w, 3 - p) in here:
                        both[m].append(bd.read(v, wv.ix[w], p, m))
                for w, p in here:                           # 居ない組: 同じ語の逆の向き と、関係ない述語
                    if (w, 3 - p) not in here:
                        sc[m][1].append(bd.read(v, wv.ix[w], 3 - p, m))
                    o = preds[rng.randrange(len(preds))]
                    if (o, p) not in here:
                        sc[m][1].append(bd.read(v, wv.ix[o], p, m))
        out[f"{k} 行を重ねる"] = {
            ("鍵の帯（I47d）" if m == "band" else "逆位相"): dict(
                読み分け=round(auc(np.array(sc[m][0]), np.array(sc[m][1])), 3),
                肯定と否定が両方居る語の読み分け=round(auc(np.array(both[m]), np.array(sc[m][1])), 3) if both[m] else None,
                両方居る組=len(both[m]))
            for m in sc}
    res["読み出し"] = out


def part_scoop(works, wv, bd, idf, a, res):
    te = set(range(a.test_works))
    train, test = [], []
    for wi, wk in enumerate(works):
        L = wk["lines"]
        sp = [split(ln["toks"]) for ln in L]
        dl = [ln["text"].startswith("「") for ln in L]
        for i in range(1, len(L)):
            if dl[i - 1] and dl[i] and sp[i - 1] and sp[i]:
                (test if wi in te else train).append((sp[i - 1], sp[i]))
    freq = Counter(w for _, b in train for w, _ in b)
    prior = [w for w, _ in freq.most_common(200)]
    z = sum(freq.values())
    negrate = Counter()
    seen = Counter()
    for q, _ in train:
        for w, p in q:
            if p:
                seen[w] += 1
                negrate[w] += p == 2
    out = {}
    Z = np.zeros(2 * wv.D, complex)
    for name, mode, beta in (("意味の帯だけ（I47a）", "none", 0.0), ("鍵の帯 重さ 0.5", "band", 0.5), ("鍵の帯 重さ 1（I47d）", "band", 1.0),
                             ("鍵の帯 重さ 2", "band", 2.0), ("逆位相 重さ 1", "pi", 1.0)):
        K = Scenes(np.stack([x if (x := bd.line(q, idf, mode, beta)) is not None else Z for q, _ in train]))
        hit = n = 0
        keep = pol = base = nq = 0.0
        for q, b in test:
            v = bd.line(q, idf, mode, beta)
            if v is None:
                continue
            top, rt = K.top(v, a.top)
            c = Counter()
            for j, rj in zip(top, rt):
                for w, _ in set(train[j][1]):
                    c[w] += rj ** 2
            s = sum(c.values()) or 1.0
            sc = Counter({w: (1 - a.lam) * x / s for w, x in c.items()})
            for w in prior:
                sc[w] += a.lam * freq[w] / z
            B = {w for w, _ in b}
            hit += bool(B & {w for w, _ in sc.most_common(10)})
            n += 1
            for w, p in q:                                  # 否定の述語ごとに、すくった場面でどう居るか
                if p != 2 or seen[w] < a.min_n:
                    continue
                has = [dict(train[j][0]).get(w) for j in top]
                has = [h for h in has if h]
                keep += len(has) / len(top)
                if has:
                    pol += sum(h == 2 for h in has) / len(has)
                    base += negrate[w] / seen[w]
                    nq += 1
        nn = max(1.0, nq)
        out[name] = dict(次の行の当たり=round(hit / max(1, n), 4), 同じ述語の居る場面=round(keep / nn, 3),
                         その述語が否定で居る割合=round(pol / nn, 3), ふだんの否定の割合=round(base / nn, 3), 問い=int(nq))
    res["すくい上げ"] = dict(train=len(train), test=len(test), 結果=out)


def draw(lines, wv, bd, idf, df, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in font_manager.findSystemFonts():
        if any(k in f for k in ("NotoSansCJK", "NotoSansJP", "IPAGothic", "ipag", "TakaoGothic")):
            font_manager.fontManager.addfont(f)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
            break
    words = [w for w, _ in df.most_common(a.map_words) if w in wv.ix]
    C = wv.C[[wv.ix[w] for w in words]]
    C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
    _, cell = som(np.concatenate([C.real, C.imag], 1), side=a.side, seed=a.seed)
    truth = Counter()
    tot = Counter()
    for ws in lines:
        for w, p in ws:
            if p:
                tot[w] += 1
                truth[w] += p == 2
    pw = [w for w in words if tot[w]]
    Wp = wv.W[[wv.ix[w] for w in pw]].conj()
    Bp = bd.B[[wv.ix[w] for w in pw]].conj()

    def readall(v):                                         # 述語ぜんぶの、肯定・否定の読み出し
        b = v[wv.D:]
        return [np.maximum(0.0, np.real(Bp @ (b * bd.key[p].conj()))) for p in (1, 2)]

    def evenv(x, times=1):                                  # I47b と同じそろえ方: 強い所の平均を 1 にし、1 に近い所だけ残す
        y = x.copy()
        for _ in range(times):
            nz = y[y > 0]
            if not nz.size:
                break
            s = nz[nz >= nz.mean()].mean()
            y = np.where(y / (s + 1e-9) > 0.5, y / (s + 1e-9), 0.0)
        return y
    vs = [x for x in (bd.line(ws, idf, "band") for ws in lines) if x is not None]
    reads = {"全部を 1 度に重ねて読む": readall(sum(vs))}
    for k in (4, 16, 64):
        acc = [np.zeros(len(pw)), np.zeros(len(pw))]
        acc2 = [np.zeros(len(pw)), np.zeros(len(pw))]
        acc4 = [np.zeros(len(pw)), np.zeros(len(pw))]
        for i in range(0, len(vs), k):
            v = sum(vs[i:i + k])
            r = readall(v)
            pres = np.maximum(0.0, np.real(Bp @ v[: wv.D]))  # 意味の帯で、語の自分の印が共鳴する強さ
            for t, ac in ((1, acc2), (4, acc4)):
                here = evenv(pres, t) > 0
                for q in (0, 1):
                    ac[q] += np.where(here, r[q], 0.0)
            for q in (0, 1):
                acc[q] += r[q]
        reads[f"{k} 行ずつ重ね、意味の帯を 4 回そろえて残った語だけ向きを読んで足す"] = acc4
        reads[f"{k} 行ずつ重ねて読み、足す"] = acc
        reads[f"{k} 行ずつ重ね、意味の帯を 1 回そろえて残った語だけ向きを読んで足す"] = acc2
    tr_ = np.array([truth[w] / tot[w] for w in pw])
    big = np.array([tot[w] >= a.min_n for w in pw])
    cors = {}
    for name, (ra, rn) in reads.items():
        sh = rn / (ra + rn + 1e-9)
        ok = big & ((ra + rn) > 0)
        cors[name] = dict(語ごとの相関=round(float(np.corrcoef(sh[ok], tr_[ok])[0, 1]), 3), 読めた語=int(ok.sum()), 語=int(big.sum()))
    use = max(cors, key=lambda n: cors[n]["語ごとの相関"])
    ra, rn = reads[use]
    side = a.side
    grids = {k: np.zeros(side * side) for k in ("n", "aff", "neg", "true", "tn")}
    cellof = dict(zip(words, cell))
    for i, w in enumerate(pw):
        c = cellof[w]
        grids["n"][c] += 1
        grids["aff"][c] += ra[i]
        grids["neg"][c] += rn[i]
        grids["true"][c] += truth[w]
        grids["tn"][c] += tot[w]
    m = grids["n"] > 0
    share = np.full(side * side, np.nan)
    share[m] = grids["neg"][m] / (grids["neg"][m] + grids["aff"][m] + 1e-9)
    true = np.full(side * side, np.nan)
    true[m] = grids["true"][m] / grids["tn"][m]
    dens = np.full(side * side, np.nan)
    dens[m] = np.log1p(grids["tn"][m])
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.6))
    for x, g, t, cm in ((ax[0], dens, "意味の板（述語の多さ）", "Greys"),
                        (ax[1], share, "否定の帯から読み出した、否定の割合", "RdBu_r"),
                        (ax[2], true, "資料で数えた、否定の割合（答え）", "RdBu_r")):
        kw = {} if cm == "Greys" else dict(vmin=0, vmax=0.6)
        im = x.imshow(g.reshape(side, side), cmap=cm, **kw)
        x.set_title(t, fontsize=11)
        x.set_xticks([])
        x.set_yticks([])
        fig.colorbar(im, ax=x, fraction=0.046)
    best = {}
    for w, c in zip(words, cell):                           # 否定の多い升に、その升でいちばん多い語を書く
        if tot[w] >= a.min_n and (c not in best or tot[w] > tot[best[c]]):
            best[c] = w
    lab = sorted((c for c in best if not np.isnan(true[c])), key=lambda c: -true[c])[:14] + \
        sorted((c for c in best if not np.isnan(true[c])), key=lambda c: true[c])[:6]
    for c in lab:
        for x in ax[1:]:
            x.text(c % side, c // side, best[c], fontsize=7, ha="center", va="center", color="black")
    fig.suptitle("I47d: 意味の板は 1 枚のまま、否定は別の帯の色として重なる（赤 = 否定で使われることが多い所）", fontsize=12)
    fig.tight_layout()
    fig.savefig(a.fig, dpi=130)
    ok = m & ~np.isnan(share)
    return dict(読み方ごと=cors, 図に使った読み方=use, 升ごとの相関=round(float(np.corrcoef(share[ok], true[ok])[0, 1]), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--test_works", type=int, default=10)
    ap.add_argument("--D", type=int, default=512)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--mix", type=float, default=0.6)
    ap.add_argument("--min_n", type=int, default=5)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--side", type=int, default=28)
    ap.add_argument("--map_words", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--band_wave", default="own", choices=("own", "learned"), help="向きの帯に載せる波")
    ap.add_argument("--fig")
    ap.add_argument("--out")
    a = ap.parse_args()
    works = json.load(open(a.cache, encoding="utf-8"))
    works = works["works"] if isinstance(works, dict) else works
    random.Random(a.seed).shuffle(works)
    tr = works[a.test_works:]
    lines = [split(ln["toks"]) for wk in tr for ln in wk["lines"]]
    seq = [[w for w, _ in ws] for ws in lines]
    df = Counter(w for ws in seq for w in set(ws))
    idf = {w: math.log((len(seq) + 1) / (c + 0.5)) for w, c in df.items()}
    wv = Waves(sorted(w for w, c in df.items() if c >= 2), D=a.D, seed=a.seed)
    wv.learn(seq, mix=a.mix)
    bd = Band(wv, a.seed, own=a.band_wave == "own")
    res = dict(向きの帯の波=a.band_wave, vocab=len(wv.vocab), 行=len(lines), 否定のある行=sum(any(p == 2 for _, p in ws) for ws in lines))
    part_read(lines, wv, bd, idf, a, res)
    part_scoop(works, wv, bd, idf, a, res)
    if a.fig:
        res["板の色と答え"] = draw(lines, wv, bd, idf, df, a)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
