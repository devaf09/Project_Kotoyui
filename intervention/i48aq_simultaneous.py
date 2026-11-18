"""I48aq: 文の全部の間を同時に照らす（持ち主の方針: 同時に照らしたい。高速に照らす方法が無いと、つらくなる）。

今の形（左から順）: 間ごとに 今の点・R1・R2（置いた物に依らない）と R3（左に置いた詰め物の文法グレイン）を足し、左から 1 つずつ決める。
同時の形:
  (0) 置いた物に依らない点（今の点・R1・R2）を、文の束 × 間 × 候補 の 1 枚の行列で一度に出す（帯 × 間 の重さの行列との掛け算）。
  (1) 1 回目: 前後の詰め物は「置いた物に依らない点の分布」（その点の softmax、温度なし）で柔らかく置いた状態として、全部の間を同時に照らす。
  (2) 各間の候補の分布を、前後の間の手がかりに戻して、もう一度全部を同時に照らす。
  (3) どの間の選びも変わらなくなるまで繰り返す（回数は手で決めない。安全のための上限 LIMIT だけ置き、何回で落ち着いたかを数える）。
  前後の手がかりは、両側の文法グレイン R3両: 文の中の品詞、間が文末か途中か、左の間の詰め物の機能語（「左置:」）と右の間の詰め物の機能語（「右置:」）。
  学びは元の詰め物で置いた状態（両側とも分かっている）、試しは分布で重みをつけた柔らかい状態。
  選ぶ物は今の形と同じく、今の点の上位 K と 三つの上位 K の和集合の中で、今＋R1＋R2＋R3両 がいちばん大きい物。
比べ: 左から順（I48ak の walk）、先読み（R3 を幅 BEAM で、I48aj の三つ・先読み と同じ形）、同時。当たり（間・文末・文、助詞ごとの当たりと害）と、
  1 文あたりの時間（間の数ごと）。同時は文の束ごとに一度に出す（束の大きさ BATCH は CPU の行列の大きさのためだけ。答えは変わらない）。
手で置く物: LIMIT（20、安全のため）、BATCH（64 文）、BEAM（10、比べの先読みだけ）。
python intervention/i48aq_simultaneous.py --conllu ... --content ... --func ... --out out.json [--train 50000] [--test 1000]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i48ab_long_grain as G  # noqa: E402
import i48ad_lookahead as AD  # noqa: E402
import i48ae_marks as AE  # noqa: E402
import i48aj_resonance as AJ  # noqa: E402
import i48ak_teach as AK  # noqa: E402
import i48ao_many_particles as AO  # noqa: E402

LIMIT = 20
BATCH = 64
BEAM = 10
K = AK.K


class Gram2:
    """両側の文法グレイン（左置・右置）。"""

    def __init__(self, n_cls, fill):
        self.fill, self.idx, self.n_cls = fill, {}, n_cls
        self.rows = []
        self.fw = {}

    def func_words(self, g):
        if g not in self.fw:
            self.fw[g] = sorted({t["lemma"] for t in self.fill.toks.get(g, []) if t["upos"] in AJ.FUNC_UPOS})
        return self.fw[g]

    def names_gold(self, ups, gaps, i):
        out = {"品詞:" + u: 1.0 for u in ups}
        out["間:文末" if i + 1 == len(gaps) else "間:途中"] = 1.0
        for j, g in enumerate(gaps):
            if j == i:
                continue
            for w in self.func_words(g):
                k = ("左置:" if j < i else "右置:") + w
                out[k] = out.get(k, 0) + 1.0
        return out

    def vec(self, names, grow=False):
        if grow:
            for n in names:
                self.idx.setdefault(n, len(self.idx))
        f = np.array([self.idx[n] for n in names if n in self.idx], dtype=np.int64)
        v = np.array([w for n, w in names.items() if n in self.idx], dtype=np.float64)
        return f, v

    def add(self, c, names):
        self.rows.append((c, *self.vec(names, grow=True)))

    def build(self, beta):
        self.R = AJ.Ratio(self.n_cls, max(1, len(self.idx)))
        for c, f, v in self.rows:
            if len(f):
                self.R.add(c, f, v)
        self.R.build(beta)
        return self


class Batch:
    """文の束 × 間 × 候補 を行列で出す（置いた物に依らない点）。"""

    def __init__(self, m):
        self.m = m

    def fixed(self, sents_items):
        """sents_items: [(keys, xs)]。返り値: 間ごとの [今, R1, R2] を (候補 × 間の数の和) の行列で。"""
        m, S, D = self.m, self.m.S, self.m.S.D
        pg = m.pg
        gaps = [(keys, x) for keys, xs in sents_items for x in xs]
        n = len(gaps)
        Vb = np.zeros((D, n), dtype=np.float32)
        Xr1 = np.zeros((D, n), dtype=np.complex64)
        R2 = {nm: np.zeros((D, n), dtype=np.float32) for nm in ("R", "L", "P")}
        has = np.zeros((3, n), dtype=np.float32)
        pw = np.zeros(n, dtype=np.float32)
        fic = ()
        for g, (keys, x) in enumerate(gaps):
            if x["i"] == 0:
                fic = m.who.fiction(keys)
            f, v = m.feat(x)
            np.add.at(Vb[:, g], f, v)
            Xr1[:, g] = m.who.gap(keys, x["i"], fic)
            i, last = x["i"], x["last"]
            pw[g] = m.best_w[AD.ev_bin(m.pair_n[(x["l"], x["r"])] if not last else 10)]
            for k, (nm, key) in enumerate((("R", keys[i]), ("L", None if last else keys[i + 1]), ("P", None if last else AJ.next_pred(keys, i)))):
                if key is None:
                    continue
                fa, va = S.attrs(*key)
                np.add.at(R2[nm][:, g], fa, va / max(1e-9, float(va.sum())))
                has[k, g] = 1
        s = pw[None, :] * pg.prior[:, None].astype(np.float32) + pg.logp.astype(np.float32) @ Vb
        s = s - s.max(0, keepdims=True)
        p = np.exp(s)
        p /= p.sum(0, keepdims=True)
        p[-1] = 0
        now = np.log(np.maximum(p, 1e-30)).astype(np.float32)
        who = m.who
        m_all = np.real(np.conj(who.Ball)[None, :] @ Xr1)[0] / who.N
        mm = (np.real(np.conj(who.B) @ Xr1) + who.beta * m_all[None, :]) / (who.n[:, None] + who.beta)
        r1 = np.where(m_all[None, :] > 0, np.log(np.maximum(AJ.R1_FLOOR, mm / np.where(m_all > 0, m_all, 1)[None, :])), 0).astype(np.float32)
        r1[-1] = -1e9
        parts = [m.rR.L @ R2["R"], m.rL.L @ R2["L"], m.rP.L @ R2["P"]]
        r2 = sum(pt * h[None, :] for pt, h in zip(parts, has)) / np.maximum(1, has.sum(0))[None, :]
        return now, r1, r2.astype(np.float32), gaps


def run_simul(m, g2, sents_items):
    """同時に照らす。返り値: 文ごとの選び（候補の番号）と、落ち着くまでの回数。"""
    now, r1, r2, gaps = Batch(m).fixed(sents_items)
    fixed = r1 + r2
    nc = m.nc
    # 文ごとの間の範囲
    spans, k = [], 0
    for keys, xs in sents_items:
        spans.append((k, k + len(xs), keys, xs))
        k += len(xs)
    base = now + fixed
    base[-1] = -1e9
    P = np.exp(base - base.max(0, keepdims=True))
    P /= P.sum(0, keepdims=True)                                 # 1 回目の柔らかい置き方: 置いた物に依らない点の分布
    fw = [g2.func_words(c) if c is not None else [] for c in m.pg.cls]
    word_ids = {}
    for c, ws in enumerate(fw):
        for w in ws:
            word_ids.setdefault(w, []).append(c)
    picks = np.full(k, -1)
    rounds = np.zeros(len(spans), dtype=int)
    done = np.zeros(len(spans), dtype=bool)
    ups_all = [{u for _, u in keys} for _, _, keys, _ in spans]
    for it in range(1, LIMIT + 1):
        cols = []
        for sid, (a, b, keys, xs) in enumerate(spans):
            n = b - a
            for i in range(n):
                names = {"品詞:" + u: 1.0 for u in ups_all[sid]}
                names["間:文末" if i + 1 == n else "間:途中"] = 1.0
                for w, cs in word_ids.items():
                    lft = float(P[cs, a:a + i].sum()) if i > 0 else 0.0
                    rgt = float(P[cs, a + i + 1:b].sum()) if i + 1 < n else 0.0
                    if lft > 0:
                        names["左置:" + w] = lft
                    if rgt > 0:
                        names["右置:" + w] = rgt
                cols.append(names)
        S3 = np.zeros((len(g2.idx), k), dtype=np.float32)
        for g, names in enumerate(cols):
            f, v = g2.vec(names)
            if len(f):
                S3[f, g] = v / v.sum()
        r3 = g2.R.L @ S3
        three = fixed + r3
        both = now + three
        three[-1] = both[-1] = -1e9
        new = np.empty(k, dtype=int)
        for g in range(k):
            cand = (set(np.argpartition(-now[:, g], K)[:K]) | set(np.argpartition(-three[:, g], K)[:K])) - {nc - 1}
            new[g] = max(cand, key=lambda c: both[c, g])
        for sid, (a, b, _, _) in enumerate(spans):
            if not done[sid] and it > 1 and np.array_equal(new[a:b], picks[a:b]):
                done[sid] = True
                rounds[sid] = it - 1
        picks = new
        if done.all():
            break
        Pn = np.exp(both - both.max(0, keepdims=True))
        P = Pn / Pn.sum(0, keepdims=True)
    rounds[~done] = LIMIT
    return [picks[a:b] for a, b, _, _ in spans], rounds


def run_beam(m, keys, xs):
    """左から順に、R3（左に置いた物）を幅 BEAM で先読み（I48aj の三つ・先読み と同じ形、選ぶ点は 今＋三つ）。"""
    fic = m.who.fiction(keys)
    ups = {u for _, u in keys}
    gs = [m.gap_scores(keys, x, fic) for x in xs]
    bm = [((), 0.0)]
    for i, x in enumerate(xs):
        nb = []
        for seq, sc in bm:
            r3 = m.gram.score(m.gram.state(ups, [m.pg.cls[c] for c in seq], x["last"]))
            s = gs[i]["今"] + gs[i]["R1"] + gs[i]["R2"] + r3
            s[-1] = -1e9
            for c in np.argsort(-s)[:BEAM]:
                nb.append((seq + (int(c),), sc + float(s[c])))
        bm = sorted(nb, key=lambda z: -z[1])[:BEAM]
    return list(bm[0][0])


def score_seqs(m, te_items, seqs, surf):
    c = defaultdict(int)
    for (s, keys, xs), seq in zip(te_items, seqs):
        cont, gaps, head = G.split(s)
        out = [m.pg.cls[k] for k in seq]
        c["文"] += 1
        for i, (g, y) in enumerate(zip(out, gaps)):
            if i + 1 == len(gaps):
                c["文末当たり"] += g == y
                continue
            c["間"] += 1
            c["間当たり"] += g == y
            for p in AO.PARTICLES:
                if y == p:
                    c[p + "n"] += 1
                    c[p + "当たり"] += g == p
                elif g == p:
                    c[p + "害"] += 1
        sent = "".join(surf.form(l, u, g) + g for (l, f_, u), g in zip(cont, out))
        c["文当たり"] += G.norm(head + sent) == G.norm(s["text"])
    d = lambda a, b: round(c[a] / max(1, c[b]), 4)
    return dict(間=d("間当たり", "間"), 文末=d("文末当たり", "文"), 文=d("文当たり", "文"),
                助詞={p: dict(数=c[p + "n"], 当たり=d(p + "当たり", p + "n"), 害=c[p + "害"]) for p in AO.PARTICLES})


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
    m = AK.Model(S, fill, a.r1).fit(tr, hold)
    g2 = Gram2(m.nc, fill)
    for s in tr:
        keys = AK.keys_of(s)
        gaps = G.split(s)[1]
        ups = {u for _, u in keys}
        for i, g in enumerate(gaps):
            g2.add(m.cid(g), g2.names_gold(ups, gaps, i))
    best = None                                                  # β は R3 と同じく取りおきの文で選ぶ（両側とも元の詰め物で置いた状態）
    for b in AJ.BETAS:
        g2.build(b)
        hit = 0
        for s in hold[:400]:
            keys = AK.keys_of(s)
            gaps = G.split(s)[1]
            ups = {u for _, u in keys}
            for i, g in enumerate(gaps):
                f, v = g2.vec(g2.names_gold(ups, gaps, i))
                hit += AJ.ranks(g2.R.score(f, v))[m.cid(g)] < 3 if len(f) else 0
        if best is None or hit > best[0]:
            best = (hit, b)
    g2.build(best[1])
    print(f"学び {time.time() - t0:.0f} 秒、R3両の帯 {len(g2.idx)}、β {best[1]}", flush=True)
    te_items = [(s, AK.keys_of(s), AE.items(S, s, "節")) for s in te]
    te_items = [t for t in te_items if t[2]]
    res = dict(学びの文=len(tr), 試しの文=len(te_items), LIMIT=LIMIT, BATCH=BATCH, BEAM=BEAM, R3両のβ=best[1], 形={}, 時間={})
    times = defaultdict(lambda: defaultdict(list))
    # 左から順
    seqs = []
    for s, keys, xs in te_items:
        t1 = time.perf_counter()
        rs, _ = m.walk(keys, xs)
        times["左から順"][len(xs)].append(time.perf_counter() - t1)
        seqs.append([r["選び"] for r in rs])
    res["形"]["左から順"] = score_seqs(m, te_items, seqs, surf)
    print("左から順", json.dumps(res["形"]["左から順"], ensure_ascii=False), flush=True)
    # 先読み
    seqs = []
    for s, keys, xs in te_items:
        t1 = time.perf_counter()
        seqs.append(run_beam(m, keys, xs))
        times["先読み"][len(xs)].append(time.perf_counter() - t1)
    res["形"]["先読み"] = score_seqs(m, te_items, seqs, surf)
    print("先読み", json.dumps(res["形"]["先読み"], ensure_ascii=False), flush=True)
    # 同時（間の数ごとの束で時間を測る）
    by_n = defaultdict(list)
    for t in te_items:
        by_n[len(t[2])].append(t)
    seqs_map, rounds_all = {}, []
    for n, items in sorted(by_n.items()):
        for b in range(0, len(items), BATCH):
            chunk = items[b:b + BATCH]
            t1 = time.perf_counter()
            picks, rounds = run_simul(m, g2, [(keys, xs) for _, keys, xs in chunk])
            dt = (time.perf_counter() - t1) / len(chunk)
            times["同時"][n].extend([dt] * len(chunk))
            for (s, _, _), pk in zip(chunk, picks):
                seqs_map[s["id"]] = list(pk)
            rounds_all.extend(int(r) for r in rounds)
    seqs = [seqs_map[s["id"]] for s, _, _ in te_items]
    res["形"]["同時"] = score_seqs(m, te_items, seqs, surf)
    res["落ち着くまでの回数"] = {str(k): rounds_all.count(k) for k in sorted(set(rounds_all))}
    print("同時", json.dumps(res["形"]["同時"], ensure_ascii=False), res["落ち着くまでの回数"], flush=True)
    for nm, d in times.items():
        res["時間"][nm] = {str(n): dict(文=len(v), 一文の秒=round(float(np.mean(v)), 4)) for n, v in sorted(d.items())}
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res["時間"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
