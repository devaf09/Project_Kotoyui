"""介入の研究 I47b・I47c: 4 層の時系列ですくい上げるたびに強い所を平均 1 にそろえる／否定・未・非・不・無 の板を色づけて見る

持ち主（2026-11-12）:
  「重ねすぎると全体がぼやける件は、4 層くらいの時系列の層にわけて、すくい上げるたびに強い部分を平均 1 にするようにすれば、
    ぼやけた部分は消えていくんじゃないか」
  「否定形を主体とする接頭辞で波打つ板がどう変化するのかも気になる。否定形や、未、非、などの周波数の板を、つけていない意味の分と
    どう違うか色づけて表示して欲しい」

I47b（--part layers）:
  層 = 今の行（1 行）・少し前まで（3 行）・場面（10 行）・話の流れ（30 行）。層ごとに、その窓の語の波の重ね合わせで過去の場面と共鳴させ、
  強く共鳴した場面の「次の台詞」の語を重ねて板にする。すくい上げるたびに、強い所（平均より上）の平均が 1 になるようにそろえ、
  残す方が近い所（|1 − 明るさ| < |0 − 明るさ|）だけ残す（ぼやけた所は消える）。4 枚を重ねて、もう一度そろえる。
  物差し: 外した作品の台詞で、次の台詞の中身の語が上位 10 語に入る割合。そろえない時・1 層だけの時と比べる。
I47c（--part neg）:
  否定の付いた述語（行か＋ない）を「語＋否」の別の語として流れから学び、付いていない語の板と比べる。
  接頭辞（未・非・不・無）は、付いた語（不自然）と付いていない語（自然）の組で比べる。
  板 = 似た語が隣り合うように並べた 2D の板（語の波で作る自己組織の地図）。語の波をその板に当てた明るさを色で描く。
  図: 付いていない語の板／付いた語の板／差（赤 = 付くと明るくなる所、青 = 暗くなる所）。全部の組の差の平均 =「否定の板」。
  数: 否定で板がどれだけ動くか（付いた語と元の語の重なり）、動く向きが語をまたいでそろっているか（そろっていれば、否定は 1 枚の板として足せる）。

  python intervention/i47b_layers_negation.py --cache <wn50x5_parsed.json> --part layers
  python intervention/i47b_layers_negation.py --cache <wn50x5_parsed.json> --part neg --fig out.png
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict

import numpy as np

CONTENT = ("NOUN", "PROPN", "VERB", "ADJ")
STOP = {"する", "いる", "ある", "なる", "言う", "思う", "こと", "もの", "よう", "それ", "これ", "の", "ん", "方", "人", "今", "何", "自分", "私", "俺", "僕", "君", "あなた"}
NEG = "＋否"
PREFIX = ("未", "非", "不", "無")


def content(toks, neg=False):
    """中身の語。neg=True の時、ない・ぬ の付いた述語は「語＋否」にする。"""
    negh = {t["head"] for t in toks if t["pos"] == "AUX" and t["lemma"] in ("ない", "ぬ")} if neg else set()
    out = []
    for t in toks:
        if t["pos"] in CONTENT and t["lemma"] not in STOP:
            out.append(t["lemma"] + NEG if t["i"] in negh and t["pos"] in ("VERB", "ADJ") else t["lemma"])
        elif neg and t["i"] in negh and t["pos"] in ("VERB", "ADJ"):   # する・いる・ある なども、否定の時は残す
            out.append(t["lemma"] + NEG)
    return out


class Waves:
    """語の波: 乱数の位相（自分の印）と、流れから学んだ波（一緒に出た語の印の重ね合わせから、どの語にも共通の向きを引いたもの）。"""

    def __init__(self, vocab, D=512, seed=0):
        rng = np.random.default_rng(seed)
        self.vocab = vocab
        self.ix = {w: i for i, w in enumerate(vocab)}
        self.D = D
        self.base = np.exp(1j * rng.uniform(0, 2 * np.pi, (len(vocab), D)))
        self.C = np.zeros_like(self.base)                   # 流れの波
        self.W = self.base.copy()

    def learn(self, seq, mix=0.6, rounds=2):
        for _ in range(rounds):
            src = self.W
            acc = np.zeros_like(self.W)
            n = np.zeros(len(self.ix))
            for k, ws in enumerate(seq):
                ctx = set(ws) | set(seq[k - 1] if k else []) | set(seq[k + 1] if k + 1 < len(seq) else [])
                ids = [self.ix[w] for w in ctx if w in self.ix]
                if len(ids) < 2:
                    continue
                s = src[ids].sum(0)
                for w in set(ws):
                    j = self.ix.get(w)
                    if j is not None:
                        acc[j] += s - src[j]
                        n[j] += len(ids) - 1
            m = n > 0
            C = np.zeros_like(self.W)
            C[m] = acc[m] / n[m, None]
            C[m] -= C[m].mean(0)                            # 共通の向き（搬送波）を引く
            C[m] /= (np.linalg.norm(C[m], axis=1, keepdims=True) / math.sqrt(self.D) + 1e-9)
            self.C = C
            W = (1 - mix) * self.base + mix * C
            self.W = W / np.maximum(np.abs(W), 1e-9)

    def bag(self, ws, idf):
        ids = [(self.ix[w], idf.get(w, 1.0)) for w in ws if w in self.ix]
        if not ids:
            return None
        v = sum(self.W[j] * a for j, a in ids)
        return v / (np.linalg.norm(v) + 1e-9)


def even(c):
    """すくい上げた板をそろえる: 強い所（平均より上）の平均を 1 にし、残す方が近い所（> 0.5）だけ残す。"""
    if not c:
        return c
    vals = np.array(list(c.values()))
    strong = vals[vals >= vals.mean()]
    s = strong.mean() if len(strong) else 1.0
    return Counter({w: v / s for w, v in c.items() if abs(1.0 - v / s) < abs(0.0 - v / s)})


def part_layers(works, a):
    random.Random(a.seed).shuffle(works)
    tr_w, te_w = works[a.test_works:], works[:a.test_works]
    wins = (1, 3, 10, 30)

    def scenes(ws_):
        out = []
        for wk in ws_:
            L = wk["lines"]
            cw = [content(ln["toks"]) for ln in L]
            dl = [ln["text"].startswith("「") for ln in L]
            for i in range(1, len(L)):
                if dl[i - 1] and dl[i] and cw[i - 1] and cw[i]:
                    out.append(([sum(cw[max(0, i - w):i], []) for w in wins], cw[i]))
        return out
    train, test = scenes(tr_w), scenes(te_w)
    seq = [content(ln["toks"]) for wk in tr_w for ln in wk["lines"]]
    df = Counter(w for ws in seq for w in set(ws))
    idf = {w: math.log((len(seq) + 1) / (c + 0.5)) for w, c in df.items()}
    wv = Waves(sorted(df), D=a.D, seed=a.seed)
    wv.learn(seq, mix=a.mix)
    freq = Counter(w for _, b in train for w in b)
    prior = [w for w, _ in freq.most_common(200)]
    common = set(prior[:50])
    z = sum(freq.values())
    Z = np.zeros(a.D, complex)
    K = [np.stack([wv.bag(c[l], idf) if wv.bag(c[l], idf) is not None else Z for c, _ in train]) for l in range(len(wins))]

    def scoop(l, q):
        r = np.abs(K[l].conj() @ q)
        c = Counter()
        for j in np.argsort(-r)[: a.top]:
            for w in set(train[j][1]):
                c[w] += r[j] ** 2
        return c

    def rank(c):
        s = sum(c.values()) or 1.0
        sc = Counter({w: (1 - a.lam) * v / s for w, v in c.items()})
        for w in prior:
            sc[w] += a.lam * freq[w] / z
        return [w for w, _ in sc.most_common(10)]

    modes = {"ふだんの多さ": lambda bs: Counter(),
             "1 層（今の行だけ）": lambda bs: bs[0],
             "4 層を重ねる（そろえない）": lambda bs: sum((Counter({w: v / (sum(b.values()) or 1) for w, v in b.items()}) for b in bs), Counter()),
             "4 層、すくうたびにそろえる（I47b）": lambda bs: even(sum((even(b) for b in bs), Counter())),
             "今の行 + 場面（1・10 行）をそろえる": lambda bs: even(even(bs[0]) + even(bs[2]))}
    for l, w in enumerate(wins):
        modes[f"層 {w} 行だけ"] = (lambda l_: (lambda bs: bs[l_]))(l)
    tot = {m: [0.0, 0, 0.0, 0, 0] for m in modes}
    for ctx, b in test:
        qs = [wv.bag(ctx[l], idf) for l in range(len(wins))]
        bs = [scoop(l, q) if q is not None else Counter() for l, q in enumerate(qs)]
        B, Bi = set(b), set(b) - common
        for m, f in modes.items():
            c = f(bs)
            top = set(rank(c))
            t = tot[m]
            t[0] += len(B & top) / len(B)
            t[1] += bool(B & top)
            if Bi:
                t[2] += len(Bi & top) / len(Bi)
                t[3] += 1
            t[4] += len(c)
    n = len(test)
    res = dict(train=len(train), test=n, vocab=len(wv.vocab), 層の窓=list(wins))
    for m, t in tot.items():
        res[m] = dict(再現率=round(t[0] / n, 4), 当たり=round(t[1] / n, 4), 多い語を除いた再現率=round(t[2] / max(1, t[3]), 4), 板に残る語の数=round(t[4] / n, 1))
    return res


def som(X, side=28, epochs=12, seed=0):
    """似た語が隣り合う 2D の板（自己組織の地図）。X: 語 × 実数の特徴（波の実部・虚部）。"""
    rng = np.random.default_rng(seed)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    M = X[rng.choice(len(X), side * side, replace=len(X) < side * side)].copy()
    gy, gx = np.divmod(np.arange(side * side), side)
    d2 = (gy[:, None] - gy[None]) ** 2 + (gx[:, None] - gx[None]) ** 2
    for e in range(epochs):
        sig = max(0.8, side / 3 * (1 - e / epochs))
        bmu = np.argmax(X @ M.T, 1)
        H = np.exp(-d2 / (2 * sig * sig))                   # 板の上で近い升ほど、一緒に寄る
        num = H[:, bmu] @ X
        den = H[:, bmu].sum(1, keepdims=True)
        M = num / np.maximum(den, 1e-9)
        M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    return M, np.argmax(X @ M.T, 1)


def part_neg(works, a):
    seq = [content(ln["toks"], neg=True) for wk in works for ln in wk["lines"]]
    df = Counter(w for ws in seq for w in set(ws))
    vocab = sorted(w for w, c in df.items() if c >= 2)
    wv = Waves(vocab, D=a.D, seed=a.seed)
    wv.learn(seq, mix=a.mix)
    C = wv.C / (np.linalg.norm(wv.C, axis=1, keepdims=True) + 1e-9)
    R = np.concatenate([C.real, C.imag], 1)                 # 板に当てる時は、実数の特徴にする
    cos = lambda x, y: float(R[wv.ix[x]] @ R[wv.ix[y]])
    groups = {"否定（ない・ぬ）": [(w[: -len(NEG)], w) for w in vocab if w.endswith(NEG) and w[: -len(NEG)] in wv.ix
                              and df[w] >= a.min_n and df[w[: -len(NEG)]] >= a.min_n]}
    for p in PREFIX:
        groups["接頭辞「" + p + "」"] = [(w[1:], w) for w in vocab if w.startswith(p) and len(w) >= 2 and w[1:] in wv.ix
                                    and not w.endswith(NEG) and df[w] >= 2 and df[w[1:]] >= 2]
    rng = np.random.default_rng(a.seed)
    res = dict(vocab=len(vocab), 行=len(seq))
    dirs = {}
    for g, ps in groups.items():
        if len(ps) < 3:
            res[g] = dict(組=len(ps), 例=[y for _, y in ps])
            continue
        Dv = np.stack([R[wv.ix[y]] - R[wv.ix[x]] for x, y in ps])
        Dn = Dv / (np.linalg.norm(Dv, axis=1, keepdims=True) + 1e-9)
        G = Dn @ Dn.T
        same = (G.sum() - len(ps)) / (len(ps) * (len(ps) - 1))   # 動く向きが、語をまたいでそろっている度合い
        oth = [w for w in vocab if df[w] >= a.min_n and not w.endswith(NEG)]
        rnd = np.stack([R[wv.ix[oth[rng.integers(len(oth))]]] - R[wv.ix[x]] for x, _ in ps])   # 物差し: 元の語から、関係ない語への向き
        rn = rnd / (np.linalg.norm(rnd, axis=1, keepdims=True) + 1e-9)
        Gr = rn @ rn.T
        base = (Gr.sum() - len(ps)) / (len(ps) * (len(ps) - 1))
        # 板として足せるか: ほかの組の動きの平均を元の語に足すと、付いた語に近づくか（1 組ずつ外して確かめる）
        gain, before, after = 0, [], []
        for i, (x, y) in enumerate(ps):
            op = np.delete(Dv, i, 0).mean(0)
            v = R[wv.ix[x]] + op
            v /= np.linalg.norm(v) + 1e-9
            b0, a0 = cos(x, y), float(v @ R[wv.ix[y]])
            before.append(b0)
            after.append(a0)
            gain += a0 > b0
        res[g] = dict(組=len(ps), 元の語との重なり=round(float(np.mean(before)), 3), 動く向きのそろい=round(float(same), 3),
                      関係ない語への向きのそろい=round(float(base), 3), 板を足した後の重なり=round(float(np.mean(after)), 3),
                      板を足すと近づいた組=f"{gain}/{len(ps)}", 例=[y for _, y in sorted(ps, key=lambda p: -df[p[1]])[:10]])
        dirs[g] = (Dv.mean(0), ps)
    if a.fig:
        draw(wv, R, df, groups, dirs, a)
    return res


def draw(wv, R, df, groups, dirs, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    for f in ("/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf", "C:/Windows/Fonts/meiryo.ttc", "C:/Windows/Fonts/YuGothM.ttc"):
        try:
            fm.fontManager.addfont(f)
            plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name()
            break
        except Exception:
            pass
    side = a.side
    words = [w for w, c in df.most_common() if w in wv.ix and not w.endswith(NEG)][: a.map_words]
    M, cell = som(R[[wv.ix[w] for w in words]], side=side, seed=a.seed)
    label = {}
    for w, c in zip(words, cell):                           # 升の名前 = その升でいちばん多い語
        label.setdefault(int(c), w)
    board = lambda v: (M @ (v / (np.linalg.norm(v) + 1e-9))).reshape(side, side)
    neg = groups["否定（ない・ぬ）"]
    ex = [p for p in sorted(neg, key=lambda p: -df[p[1]]) if p[0] not in STOP][:3]
    for g in ("接頭辞「不」", "接頭辞「無」", "接頭辞「未」", "接頭辞「非」"):
        ps = sorted(groups.get(g, []), key=lambda p: -min(df[p[0]], df[p[1]]))
        ex += ps[:1]
    means = [(g, d) for g, (d, _) in dirs.items()]
    mrows = (len(means) + 2) // 3
    rows = len(ex) + mrows
    fig, ax = plt.subplots(rows, 3, figsize=(13.5, 4.3 * rows))
    fig.suptitle("否定・未・非・不・無 で、波打つ板はどう変わるか（I47c）\n板 = 似た語が隣り合うように並べた 2D の板。色 = その語の波を当てた時の明るさ", fontsize=15, y=0.995)

    def mark(axx, B, k=6, sign=1):
        idx = np.argsort(-(B * sign).ravel())[: k * 3]
        used = []
        for j in idx:
            y, x = divmod(int(j), side)
            if j in label and all(abs(y - u) + abs(x - v) > 3 for u, v in used):
                used.append((y, x))
                axx.text(x, y, label[j], fontsize=9, ha="center", va="center", color="black",
                         bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
            if len(used) >= k:
                break

    for r, (x, y) in enumerate(ex):
        b0, b1 = board(R[wv.ix[x]]), board(R[wv.ix[y]])
        vm = max(abs(b0).max(), abs(b1).max())
        for c, (B, ttl) in enumerate(((b0, f"付いていない「{x}」"), (b1, f"付いた「{y.replace(NEG, '＋ない')}」"))):
            im = ax[r, c].imshow(B, cmap="viridis", vmin=-vm, vmax=vm)
            ax[r, c].set_title(ttl, fontsize=13)
            mark(ax[r, c], B)
        d = b1 - b0
        dm = abs(d).max()
        ax[r, 2].imshow(d, cmap="RdBu_r", vmin=-dm, vmax=dm)
        ax[r, 2].set_title("差（赤 = 付くと明るくなる、青 = 暗くなる）", fontsize=13)
        mark(ax[r, 2], d, k=4)
        mark(ax[r, 2], d, k=4, sign=-1)
    for c in range(3 * mrows):
        axx = ax[len(ex) + c // 3, c % 3]
        if c < len(means):
            g, dvec = means[c]
            B = board(dvec)
            dm = abs(B).max()
            axx.imshow(B, cmap="RdBu_r", vmin=-dm, vmax=dm)
            axx.set_title(f"「{g}」の板 = 全部の組（{len(dirs[g][1])}）の差の平均", fontsize=12)
            mark(axx, B, k=5)
            mark(axx, B, k=5, sign=-1)
        else:
            axx.axis("off")
    for axx in ax.ravel():
        axx.set_xticks([])
        axx.set_yticks([])
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(a.fig, dpi=110)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--part", default="layers", choices=("layers", "neg"))
    ap.add_argument("--test_works", type=int, default=10)
    ap.add_argument("--D", type=int, default=512)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--mix", type=float, default=0.6)
    ap.add_argument("--min_n", type=int, default=5)
    ap.add_argument("--side", type=int, default=28)
    ap.add_argument("--map_words", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fig")
    ap.add_argument("--out")
    a = ap.parse_args()
    works = json.load(open(a.cache, encoding="utf-8"))
    res = part_layers(works, a) if a.part == "layers" else part_neg(works, a)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
