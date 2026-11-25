"""I48bh: 新しい語板 1 版目（I48bc の使われ方の波、DK 2,048、FM→AM、少ない語は I48be の形で借りる）の符号を替えて測る（案、未確定）。

符号（種類「後ろ」「前」ごと、詰め物のクラスごとに長さ DK の複素の値。長さ √DK にそろえる）:
  (a) 今の白い符号: クラスごとの乱数の位相（I48bc のまま）。属性と関係が無いので、これ自体が「名前だけの符号」。
  (b) 太さ 6: (a) を幅 6 の箱で巡回にならした物（tools/past/i48_roundtrip_thick.py と同じならし方）。
  (c) 属性の符号 ＋ 名前 0.1: 詰め物の機能語の属性（品詞と機能語の板の属性。I48bf と同じ名前の付け方）ごとの乱数の位相を重みで足した物を
      0.9、(a) の符号を 0.1 で混ぜる（どちらも長さ 1 にしてから）。機能語の無いクラス（空・その他）は (a) のまま。
  (d) (c) を太さ 6 にならした物。
  あわせて、属性の符号だけ（名前 0）を (c0) として並べる。
比べる形（どれも本体に差し込み、同じ 5 万文・1,000 文）: 本体、数えの表 ＋ 借りる（I48be と同じ）、波 (a)〜(d)・(c0) ＋ 借りる。
  使われ方の波には位置のずれが無いので、太さがずれへの強さとして効く所は無い（効かなければ「効かない」と記録する）。
測る物: AP.measure（間・文末・文・助詞ごとの当たりと害・と の人）、筋の無い共振、本体から替わった選びと想定外の反応、
  符号どうしの近さ（クラスの組の |コサイン| の平均と、いちばん近い相手の平均）。
もう 1 つ: 行の合わなさ（本体・表 ＋ 借りる・波 (a) ＋ 借りる）。試しの文の内容語 1 つを、品詞の違う語（学びの数えが床以上の語から乱数）に
  替えた行と元の行で、間ごとに「いちばん点の高い詰め物の点 − 本当の詰め物の点」（今 ＋ R1 ＋ R2。R3 の文法の板は置いた順に依るので入れない）を出し、
  行の中の最大（いちばん合わない所）で見分ける AUC。
手で置く物: DK 2,048、帯 16、太さ 6、名前の重み 0.1、属性の名前（I48bf と同じ）、替える語の選び方と乱数の種、ほかは I48bc・I48be と同じ。
python intervention/past/i48bh_codes.py --conllu ... --content ... --func ... --out out.json [--train 50000] [--test 1000]
"""
from __future__ import annotations
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする

import argparse
import json
import os
import sys
import time
import zlib
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import i48ab_long_grain as G  # noqa: E402
import i48ad_lookahead as AD  # noqa: E402
import i48ae_marks as AE  # noqa: E402
import i48ap_units as AP  # noqa: E402
import i48at_recheck as RC  # noqa: E402
import i48ax_sayable as AX  # noqa: E402
import i48bc_usage_board as BC  # noqa: E402
import i48bd_table_vs_wave as BD  # noqa: E402
import i48be_borrow as BE  # noqa: E402
import i48bf_freq_position as BF  # noqa: E402
from i48core import learn as LN  # noqa: E402
from i48core import model as MD  # noqa: E402

DK = 2048
WIDTH = 6
EPS = 0.1


def thick(C, w):
    k = np.zeros(C.shape[1], dtype=np.complex64)
    k[:w] = 1.0
    X = np.fft.ifft(np.fft.fft(C, axis=1) * np.fft.fft(k)[None, :], axis=1).astype(np.complex64)
    return X / np.linalg.norm(X, axis=1, keepdims=True) * np.sqrt(C.shape[1])


def attr_codes(C, names_of, kind, dk):
    """クラスごとの属性の符号（属性の無いクラスは None）。"""
    out = []
    for nm in names_of:
        if not nm:
            out.append(None)
            continue
        x = np.zeros(dk, dtype=np.complex64)
        for _, a, wt in nm:
            r = np.random.default_rng(zlib.crc32((kind + "|" + a).encode()))
            x += wt * np.exp(2j * np.pi * r.random(dk)).astype(np.complex64)
        out.append(x / np.linalg.norm(x))
    return out


def mixed(C, A, eps):
    X = C.copy()
    for i, a in enumerate(A):
        if a is not None:
            v = (1 - eps) * a + eps * C[i] / np.linalg.norm(C[i])
            X[i] = v / np.linalg.norm(v) * np.sqrt(C.shape[1])
    return X


def code_stats(C):
    Cn = C / np.linalg.norm(C, axis=1, keepdims=True)
    M = np.abs(Cn @ np.conj(Cn).T)
    np.fill_diagonal(M, 0)
    iu = np.triu_indices(len(C), 1)
    return dict(組の近さ_平均=round(float(M[iu].mean()), 4), いちばん近い相手_平均=round(float(M.max(1).mean()), 4))


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
    res = dict(学びの文=len(tr), 試しの文=len(te), DK=DK, 帯=BC.B, 太さ=WIDTH, 名前の重み=EPS,
               比べ_I48be=dict(表_借りる=dict(間=0.5917, 文末=0.6136, 文=0.1351)), 符号={}, 型={})

    def save():
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    q0 = None

    def report(nm, m):
        t1 = time.time()
        r = AP.measure(m, te, surf)
        r["言えるか"] = AX.score(m, te, say, nat)
        r["筋の無い共振"] = LN.monitor(m, te)
        q = BD.picks(m, te)
        if q0 is not None:
            ch = un = hit = 0
            te_x = [s for s in te if AE.items(S, s, "節")]
            for s, a0, a1 in zip(te_x, q0, q):
                keys, gaps = MD.keys_of(s), G.split(s)[1]
                xs = AE.items(S, s, "節")
                fic = m.who.fiction(keys)
                ups = {u for _, u in keys}
                for x, c0, c1 in zip(xs, a0, a1):
                    if c0 == c1:
                        continue
                    ch += 1
                    g = m.gap_scores(keys, x, fic)
                    logic = g["R2"] + m.gram.score(m.gram.state(ups, [m.pg.cls[c] for c in a1[:x["i"]]], x["last"]))
                    if logic[c1] <= np.median(logic[:-1]):
                        un += 1
                        hit += m.pg.cls[c1] == gaps[x["i"]]
            r.update(本体から替わる選び=ch, 想定外の反応=dict(想定外=un, 当たり=hit, 外れ=un - hit))
        r["秒"] = round(time.time() - t1)
        res["型"][nm] = r
        print(nm, r["秒"], "秒", json.dumps({k: r[k] for k in ("間", "文末", "文", "と_人", "と_人以外")}, ensure_ascii=False), flush=True)
        save()
        return q

    def with_(x):
        m = base.clone()
        m.extras.append(x)
        m.refresh()
        return m

    q0 = report("本体", base)
    # 数えの表 ＋ 借りる（I48be と同じ）
    ub = BC.UsageBoard(base, dk=DK).fit(base, tr)
    rows, nf = BE.usage_vectors(ub)
    nbrs = BE.neighbours(ub, rows, nf)
    tb = BE.BorrowTable(base).fit(base, tr, ub, nbrs)
    m_tab = with_(tb)
    report("数えの表 ＋ 借りる", m_tab)
    # 詰め物のクラスごとの機能語（属性の符号の材料。I48bf と同じ取り方）
    cls_tok = {}
    for s in tr:
        T = s["toks"]
        ci = [i for i, t in enumerate(T) if G.is_content(t)]
        if not ci:
            continue
        cont, gaps, head = G.split(s)
        bounds = [(0, ci[0])] + [(ci[k] + 1, ci[k + 1] if k + 1 < len(ci) else len(T)) for k in range(len(ci))]
        for g, (x0, x1) in zip([head] + gaps, bounds):
            c = base.cid(g)
            if c not in cls_tok:
                cls_tok[c] = T[x0:x1]
    names_of = [[x for t in cls_tok.get(c, []) for x in BF.func_names(S, t)] if base.pg.cls[c] else [] for c in range(base.nc)]
    variants = {}
    for k in ub.KINDS:
        C = ub.code[k]
        A = attr_codes(C, names_of, k, DK)
        variants.setdefault("(a) 今の白い符号", {})[k] = C
        variants.setdefault("(b) 太さ 6", {})[k] = thick(C, WIDTH)
        variants.setdefault("(c0) 属性の符号だけ", {})[k] = mixed(C, A, 0.0)
        variants.setdefault("(c) 属性の符号 ＋ 名前 0.1", {})[k] = mixed(C, A, EPS)
        variants.setdefault("(d) 太さ 6 ＋ 属性 ＋ 名前 0.1", {})[k] = thick(mixed(C, A, EPS), WIDTH)
    m_wave_a = None
    for nm, codes in variants.items():
        t1 = time.time()
        u = BC.UsageBoard(base, dk=DK)
        u.code = {k: v.astype(np.complex64) for k, v in codes.items()}
        u.fit(base, tr)
        w = BE.borrow_waves(u, nbrs)
        w.mode = "FM→AM"
        res["符号"][nm] = {k: code_stats(v) for k, v in codes.items()}
        res["符号"][nm]["学び秒"] = round(time.time() - t1)
        m = with_(w)
        report(nm + " ＋ 借りる", m)
        if nm.startswith("(a)"):
            m_wave_a = m
        else:
            del m, w, u
    # 行の合わなさ: 内容語 1 つを品詞の違う語に替えた行の見分け
    t1 = time.time()
    rng = np.random.default_rng(5)
    vocab = [key for key, n in ub.N["後ろ"].items()]
    up_of = np.array([k[1] for k in vocab])
    lines = []
    for s in te:
        xs = AE.items(S, s, "節")
        keys = MD.keys_of(s)
        if len(keys) < 2 or not xs:
            continue
        p = int(rng.integers(len(keys)))
        other = [vocab[i] for i in np.nonzero(up_of != keys[p][1])[0]]
        nk = other[int(rng.integers(len(other)))]
        keys2 = list(keys)
        keys2[p] = nk
        A2 = [S.attrs(l, u_) for l, u_ in keys2]
        xs2 = [dict(x, A=A2, l=keys2[x["i"]][0], r=keys2[x["i"] + 1][0] if x["i"] + 1 < len(keys2) else None) for x in xs]
        lines.append((keys, xs, keys2, xs2, [base.cid(g) for g in G.split(s)[1]]))

    def worst(m, keys, xs, ys):
        fic = m.who.fiction(keys)
        vals = []
        for x, y in zip(xs, ys):
            g = m.gap_scores(keys, x, fic)
            sc = g["今"] + g["R1"] + g["R2"]
            vals.append(float(sc.max() - sc[y]))
        return max(vals), float(np.mean(vals))
    out = {}
    for nm, m in (("本体", base), ("数えの表 ＋ 借りる", m_tab), ("(a) 今の白い符号 ＋ 借りる", m_wave_a)):
        o_max, c_max, o_mean, c_mean = [], [], [], []
        for keys, xs, keys2, xs2, ys in lines:
            a1, b1 = worst(m, keys, xs, ys)
            a2, b2 = worst(m, keys2, xs2, ys)
            o_max.append(a1), c_max.append(a2), o_mean.append(b1), c_mean.append(b2)
        out[nm] = dict(行=len(lines), いちばん合わない所で見分けるAUC=round(BF.auc([-v for v in o_max], [-v for v in c_max]), 4),
                       行の平均で見分けるAUC=round(BF.auc([-v for v in o_mean], [-v for v in c_mean]), 4))
        print("行の合わなさ", nm, out[nm], flush=True)
    res["行の合わなさ_品詞の違う語に替える"] = out
    res["行の合わなさ_秒"] = round(time.time() - t1)
    res["秒"] = round(time.time() - t0)
    save()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
