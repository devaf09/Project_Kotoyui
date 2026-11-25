"""I48az: 間どうしを結ぶ（持ち主の案: 語ごとに「前の間の詰め物」と「後ろの間の詰め物」を組で持ち、並べて共振がいちばん良い組を選ぶ。
映画の前が「と」なら後ろは「を」のように、前後の組で持てば、文が長いほど決まりやすくなるはず）。本体は書き換えない。

(a) 3 つ組の板: 学びの文の語 k ごとに、(前の間の詰め物 a, 語 k, 後ろの間の詰め物 b) を数える。頭の語の a は頭の詰め物（「頭」の印つき）。
    行は学びに出た (a, b) の組、列は語の属性の周波数（波の長さ DK に畳む）。−1・−2・−3 の語は、属性の周波数を 1・2・3 × SHIFT
    ずらして同じ波に重ねる（語の順番の周波数ずらし）。点 = Σ x_d log(p(d|a,b) / p(d))（R2 の比の板と同じ形、ならし β は取りおきで選ぶ）。
    学びに無い (a, b) の組は 0（数えが無いので鳴らない）。
(b) 選び: 各間の候補は、本体の点 s = 今 ＋ R1 ＋ R2（＋ R3）で、雑音の床（中央値 ＋ σ √(2 ln V)、σ = 1.4826 × MAD）より上の物と、
    いちばん上の 1 つ。文の全部の間をまとめて、Σ s ＋ Σ 3 つ組の点 がいちばん大きい並びを列をたどる形（ビタビ）で選ぶ（厳密）。
(c) 決まらない間: その間をその詰め物にした時の最大（前後から）の 1 位と 2 位の差が、その間の床 σ √(2 ln V) の中なら「決まらない」。
    I48at の違和感の順（試しの文で出す）の上位 1 割・2 割と重なるかを見る。
(d) 比べ: 本体の walk（R3 あり）・R3 を抜いた walk・3 つ組の列（R3 なし）・3 つ組の列 ＋ R3（R3 は本体の walk で置いた並びの状態）。
    測る物: 間・文末・文、助詞ごとの当たりと害、と の人・人以外、文の長さごとの当たり、言える（物差しは学んでいない文）、
    と の例（9 つ ＋ 友達・映画・見る）の選び、筋の無い共振、想定外の反応。
(f) 強くなりすぎない合わせ（持ち主の問い: AM と FM）: 3 つ組の点を、今の対数の比（回数が効く AM 型）のほかに、回数によらない FM 型でも出す。
    FM 型: 行の長さを 1 にそろえた板と、語の波の向きの近さ cos。床は回数から決める: 行 r の床 = √(2 ln V) / √n_r（V = その間の
    後ろの候補の数、n_r = 行の回数。回数が少ないほど床が高い）。cos が床を越えた時だけ log(cos / 床) を足す（越えなければ 0）。
手で置く物: DK（2,048）、−3 まで（持ち主の案）。ほかは本体と同じ。
python intervention/past/i48az_link_gaps.py --conllu ... --content ... --func ... --out out.json [--train 50000] [--test 1000]
"""
from __future__ import annotations
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする

import argparse
import json
import os
import sys
import time
import zlib
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import i48ab_long_grain as G  # noqa: E402
import i48ad_lookahead as AD  # noqa: E402
import i48ae_marks as AE  # noqa: E402
import i48an_attr_board as AN  # noqa: E402
import i48at_recheck as RC  # noqa: E402
import i48ax_sayable as AX  # noqa: E402
from i48core import boards as BD  # noqa: E402
from i48core import measure as ME  # noqa: E402
from i48core import model as MD  # noqa: E402

DK = 2048
BACK = 3
LENS = ((2, 2), (3, 3), (4, 4), (5, 6), (7, 99))


def floor_of(s, live):
    v = s[live]
    med = float(np.median(v))
    sig = 1.4826 * float(np.median(np.abs(v - med)))
    return med, sig * np.sqrt(2 * np.log(max(2, int(live.sum()))))


class Triple:
    """(前の間の詰め物, 語, 後ろの間の詰め物) の板。"""

    def __init__(self, m, dk=DK, back=BACK):
        self.m, self.S, self.dk, self.back = m, m.S, dk, back
        self.HEAD = m.nc                                     # 頭の印（クラスの外の番号）
        self.mode = "AM"

    def word_x(self, keys, k):
        """語 k の波（−1〜−back の語は属性の周波数を j × SHIFT ずらして重ねる）。和 1。"""
        fs, vs = [], []
        for j in range(self.back + 1):
            if k - j < 0:
                break
            f, v = self.S.attrs(*keys[k - j])
            fs.append((f + j * self.S.shift) % self.dk)
            vs.append(np.asarray(v, dtype=np.float32))
        f, v = np.concatenate(fs), np.concatenate(vs)
        return f, v / max(1e-9, float(v.sum()))

    def triples(self, s):
        keys = MD.keys_of(s)
        if not keys:
            return
        _, gaps, head = G.split(s)
        for k in range(len(keys)):
            a = self.HEAD if k == 0 else self.m.cid(gaps[k - 1])
            yield a, k, self.m.cid(gaps[k]), keys

    def fit(self, tr, hold):
        pairs = {}
        for s in tr:
            for a, k, b, keys in self.triples(s):
                pairs.setdefault((a, b), len(pairs))
        self.row = pairs
        self.W = np.zeros((len(pairs), self.dk), dtype=np.float32)
        self.n = np.zeros(len(pairs))
        for s in tr:
            for a, k, b, keys in self.triples(s):
                r = pairs[(a, b)]
                f, v = self.word_x(keys, k)
                np.add.at(self.W[r], f, v)
                self.n[r] += 1
        self.by_a = defaultdict(list)
        for (a, b), r in pairs.items():
            self.by_a[a].append((b, r))
        best = None
        HO = [t for s in hold[:400] for t in self.triples(s)]
        for beta in BD.BETAS:
            self.build(beta)
            hit = 0
            for a, k, b, keys in HO:
                bs = self.by_a.get(a)
                if not bs:
                    continue
                f, v = self.word_x(keys, k)
                sc = [(float(self.L[r, f] @ v), bb) for bb, r in bs]
                hit += max(sc)[1] == b
            if best is None or hit > best[0]:
                best = (hit, beta)
        self.beta = best[1]
        self.build(self.beta)
        return self

    def build(self, beta):
        allw = self.W.sum(0).astype(np.float64)
        pd = (allw + G.ALPHA) / (allw.sum() + G.ALPHA * self.dk)
        self.L = (np.log((self.W + beta * pd) / (self.n[:, None] + beta)) - np.log(pd)).astype(np.float32)
        self.Wn = self.W / np.maximum(1e-9, np.linalg.norm(self.W, axis=1, keepdims=True))   # FM 型: 行の長さを 1 に

    def phi(self, A, B, keys, k):
        """候補 A（前の間）× B（後ろの間）の 3 つ組の点の表。学びに無い組は 0。"""
        f, v = self.word_x(keys, k)
        out = np.zeros((len(A), len(B)), dtype=np.float32)
        if self.mode == "FM":
            xd = np.zeros(self.dk, dtype=np.float32)
            np.add.at(xd, f, v)
            xd /= max(1e-9, float(np.linalg.norm(xd)))
            lnV = np.sqrt(2 * np.log(max(2, len(B))))
        for i, a in enumerate(A):
            for j, b in enumerate(B):
                r = self.row.get((a, b))
                if r is None:
                    continue
                if self.mode == "FM":
                    cos = float(self.Wn[r] @ xd)
                    fl = lnV / np.sqrt(self.n[r])
                    out[i, j] = np.log(cos / fl) if cos > fl else 0.0
                else:
                    out[i, j] = self.L[r, f] @ v
        return out


def unary(m, keys, xs, r3_seq=None):
    """各間の本体の点 s = 今 ＋ R1 ＋ R2（＋ R3、置いた並び r3_seq の状態で）。"""
    fic = m.who.fiction(keys)
    ups = {u for _, u in keys}
    out, logic = [], []
    for x in xs:
        g = m.gap_scores(keys, x, fic)
        s = g["今"] + g["R1"] + g["R2"]
        r3 = None
        if r3_seq is not None:
            r3 = m.gram.score(m.gram.state(ups, r3_seq[:x["i"]], x["last"]))
            s = s + r3
        if not G.other_ok(m):
            s = s.copy()
            s[-1] = -1e9
        out.append(s)
        logic.append(g["R2"] + (r3 if r3 is not None else 0))
    return out, logic


def chain(m, tri, keys, xs, us):
    """列をたどって、Σ s ＋ Σ 3 つ組の点 がいちばん大きい並びと、間ごとの「決まらない」。"""
    live = np.ones(m.nc, dtype=bool)
    if not G.other_ok(m):
        live[-1] = False
    C, T = [], []
    for s in us:
        med, t = floor_of(s, live)
        c = set(np.nonzero(live & (s > med + t))[0].tolist()) | {int(np.argmax(s))}
        C.append(sorted(c))
        T.append(t)
    n = len(xs)
    P = [tri.phi([tri.HEAD], C[0], keys, 0)[0]]          # 語 0: (頭, 間 0)
    for k in range(1, n):
        P.append(tri.phi(C[k - 1], C[k], keys, k))         # 語 k: (間 k−1, 間 k)
    U = [us[k][C[k]] for k in range(n)]
    fw = [U[0] + P[0]]
    back = []
    for k in range(1, n):
        tot = fw[-1][:, None] + P[k]
        back.append(np.argmax(tot, axis=0))
        fw.append(tot.max(0) + U[k])
    bw = [np.zeros(len(C[-1]), dtype=np.float32)]
    for k in range(n - 1, 0, -1):
        bw.insert(0, (P[k] + (U[k] + bw[0])[None, :]).max(1))
    j = int(np.argmax(fw[-1]))
    seq = [j]
    for k in range(n - 2, -1, -1):
        j = int(back[k][j])
        seq.insert(0, j)
    seq = [C[k][j] for k, j in enumerate(seq)]
    undecided = []
    for k in range(n):
        mm = np.sort(fw[k] + bw[k])[::-1]
        undecided.append(len(mm) > 1 and mm[0] - mm[1] < T[k])
    return seq, undecided, C


def evaluate(m, te, seqs, surf, say, nat):
    c = Counter()
    by_len = defaultdict(lambda: [0, 0])
    to = dict(人=[0, 0], 他=[0, 0])
    sayc = Counter()
    for s, seq in zip(te, seqs):
        if seq is None:
            continue
        cont, gaps, head = G.split(s)
        keys = MD.keys_of(s)
        xs = AE.items(m.S, s, "節")
        out = [m.pg.cls[k] for k in seq]
        c["文"] += 1
        L = len(gaps) - 1
        for i, (g, y) in enumerate(zip(out, gaps)):
            if i + 1 == len(gaps):
                c["文末当たり"] += g == y
                continue
            c["間"] += 1
            c["間当たり"] += g == y
            for lo, hi in LENS:
                if lo <= L <= hi:
                    by_len[f"{lo}〜{hi}" if hi != lo else str(lo)][0] += 1
                    by_len[f"{lo}〜{hi}" if hi != lo else str(lo)][1] += g == y
            if g != y and g is not None:
                k = say.judge(xs[i], g)
                sayc["外れ"] += 1
                sayc[k] += 1
            for p in ME.PARTICLES:
                if y == p:
                    c[p + "n"] += 1
                    c[p + "当たり"] += g == p
                elif g == p:
                    c[p + "害"] += 1
            if y == "と":
                h = "人" if AN.is_human(m.S, keys[i]) else "他"
                to[h][0] += 1
                to[h][1] += g == "と"
        sent = "".join(surf.form(l, u, g or "") + ("□" if g is None else g) for (l, f_, u), g in zip(cont, out))   # 空の詰め物（""）はそのまま、その他だけ □
        c["文当たり"] += G.norm(head + sent) == G.norm(s["text"])
    d = lambda a, b: round(c[a] / max(1, c[b]), 4)
    return dict(間=d("間当たり", "間"), 文末=d("文末当たり", "文"), 文=d("文当たり", "文"),
                助詞={p: dict(数=c[p + "n"], 当たり=d(p + "当たり", p + "n"), 害=c[p + "害"]) for p in ME.PARTICLES},
                と_人=dict(数=to["人"][0], 当たり=round(to["人"][1] / max(1, to["人"][0]), 3)),
                と_人以外=dict(数=to["他"][0], 当たり=round(to["他"][1] / max(1, to["他"][0]), 3)),
                長さごとの間の当たり={k: dict(数=v[0], 当たり=round(v[1] / max(1, v[0]), 4)) for k, v in sorted(by_len.items())},
                外れのうち言える_言えそう=round((sayc["言える"] + sayc["言えそう"]) / max(1, sayc["外れ"]), 4))


EXTRA = [("友達・映画・見る", [("友達", "NOUN"), ("映画", "NOUN"), ("見る", "VERB")])]


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
    fill = AD.Fill(S)
    surf = G.Surface()
    for s in tr:
        fill.see(s)
        cont, gaps, _ = G.split(s)
        for (l, f_, u), g in zip(cont, gaps):
            surf.add(l, f_, u, g)
    m = MD.Model(S, fill, a.r1).fit(tr, hold)                   # 本体を先に学ぶ（属性の周波数の並びを前の試しと同じにする）
    res = dict(学びの文=len(tr), 試しの文=len(te), DK=DK, 後ろ=BACK, 型={})

    def save():
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"学び {time.time() - t0:.0f} 秒", flush=True)
    base = ME.measure(m, te, surf)
    res["型"]["本体の walk（R3 あり）"] = base
    print("本体", json.dumps({k: base[k] for k in ("間", "文末", "文")}, ensure_ascii=False), flush=True)
    save()
    for s in te:
        AE.items(S, s, "節")
    fill_ref = AD.Fill(S)
    for s in ref:
        fill_ref.see(s)
    say, nat = AX.Sayable(S, ref), RC.Natural(fill_ref, ref)
    t1 = time.time()
    tri = Triple(m).fit(tr, hold)
    res["3 つ組"] = dict(組の数=len(tri.row), β=tri.beta, 学び秒=round(time.time() - t1))
    print("3 つ組", json.dumps(res["3 つ組"], ensure_ascii=False), flush=True)
    walk_seqs, noR3, ch, chR3, fm, fmR3 = [], [], [], [], [], []
    und_all, und_r3 = [], []
    m_no = m.clone()
    zero = np.zeros(m.nc, dtype=np.float32)
    m_no.gram.score = lambda names: zero
    unexp = Counter()
    unexp_ex, n_unexp, n_unexp_hit, n_changed = [], 0, 0, 0
    nolog = dict(選んだ数=0, 筋の無い共振=0, 当たり=0, 外れ=0)
    for s in te:
        xs = AE.items(S, s, "節")
        if not xs:
            for L in (walk_seqs, noR3, ch, chR3, fm, fmR3):
                L.append(None)
            continue
        keys = MD.keys_of(s)
        gaps = G.split(s)[1]
        rs, _ = m.walk(keys, xs)
        w = [r["選び"] for r in rs]
        walk_seqs.append(w)
        noR3.append([r["選び"] for r in m_no.walk(keys, xs)[0]])
        us, logic = unary(m, keys, xs)
        seq, und, _ = chain(m, tri, keys, xs, us)
        ch.append(seq)
        und_all += und
        us3, logic3 = unary(m, keys, xs, r3_seq=[m.pg.cls[c] for c in w])
        seq3, und3, _ = chain(m, tri, keys, xs, us3)
        chR3.append(seq3)
        und_r3 += und3
        tri.mode = "FM"
        fm.append(chain(m, tri, keys, xs, us)[0])
        fmR3.append(chain(m, tri, keys, xs, us3)[0])
        tri.mode = "AM"
        for k, (c, x) in enumerate(zip(seq3, xs)):
            nolog["選んだ数"] += 1
            lg, wv = logic3[k].copy(), m.gap_scores(keys, x, m.who.fiction(keys))["R1"].copy()
            lg[-1] = wv[-1] = -1e9
            weak = lg[c] <= np.median(lg[:-1])
            if weak and int(np.argmax(wv)) == c:
                nolog["筋の無い共振"] += 1
                ok = m.pg.cls[c] == gaps[x["i"]]
                nolog["当たり"] += ok
                nolog["外れ"] += not ok
            if c != w[k]:
                n_changed += 1
                if weak:                                          # 3 つ組の点が選びを替え、論理の筋が中央値以下
                    n_unexp += 1
                    n_unexp_hit += m.pg.cls[c] == gaps[x["i"]]
                    unexp[(x["l"], m.pg.cls[c])] += 1
                    if len(unexp_ex) < 20:
                        unexp_ex.append(dict(要点="・".join(kk[0] for kk in keys), 間=x["i"], 元=gaps[x["i"]], 本体=m.pg.cls[w[k]], 列=m.pg.cls[c]))
    for nm, seqs in (("本体の walk（R3 あり）", walk_seqs), ("R3 を抜いた walk", noR3), ("3 つ組の列（R3 なし）", ch), ("3 つ組の列 ＋ R3", chR3),
                     ("FM 型の列（R3 なし）", fm), ("FM 型の列 ＋ R3", fmR3)):
        r = evaluate(m, te, seqs, surf, say, nat)
        res["型"].setdefault(nm, {}).update(r if nm != "本体の walk（R3 あり）" else {k: r[k] for k in ("長さごとの間の当たり", "外れのうち言える_言えそう", "と_人", "と_人以外")})
        print(nm, json.dumps({k: r[k] for k in ("間", "文末", "文", "と_人", "外れのうち言える_言えそう")}, ensure_ascii=False), flush=True)
        save()
    res["決まらない間"] = dict(R3なし=int(sum(und_all)), R3あり=int(sum(und_r3)), 間の数=len(und_all))
    rows = RC.recheck(m, te, nat)                                 # 試しの文での違和感の材料（同じ順の間）
    order = RC.oddness(rows)
    und = np.array(und_r3[: len(rows)])
    for q, nm in ((.1, "上位1割"), (.2, "上位2割")):
        top = np.zeros(len(rows), dtype=bool)
        top[order[: int(len(rows) * q)]] = True
        res["決まらない間"][f"決まらない間のうち違和感の{nm}に入る割合"] = round(float((und & top).sum() / max(1, und.sum())), 4)
    res["決まらない間"]["違和感の順と並びが合うか"] = len(rows) == len(und_r3)
    res["筋の無い共振"] = nolog
    res["想定外の反応"] = dict(列で替わった選び=n_changed, 想定外=n_unexp, 当たり=n_unexp_hit, 外れ=n_unexp - n_unexp_hit,
                         くり返す組=[dict(左の語=a_, 詰め物=b_, 回数=v) for (a_, b_), v in unexp.most_common(10) if v >= 2], 例=unexp_ex)
    ex = {}
    for kind, name, keys, gi in ME.WANT:
        xs = MD.made_items(S, keys, None)
        us, _ = unary(m, keys, xs)
        seq, und, _ = chain(m, tri, keys, xs, us)
        ex[name] = dict(列の選び=m.pg.cls[seq[gi]], 決まらない=bool(und[gi]))
    for name, keys in EXTRA:
        xs = MD.made_items(S, keys, None)
        us, _ = unary(m, keys, xs)
        seq, und, _ = chain(m, tri, keys, xs, us)
        rs, _ = m.walk(keys, xs)
        ex[name] = dict(列の選び=[m.pg.cls[c] for c in seq], 本体の選び=[m.pg.cls[r["選び"]] for r in rs], 決まらない=[bool(u) for u in und])
    res["例"] = ex
    res["秒"] = round(time.time() - t0)
    print("決まらない", json.dumps(res["決まらない間"], ensure_ascii=False), "想定外", n_unexp, flush=True)
    save()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
