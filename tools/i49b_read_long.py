"""I49b: 物差し 6 の小さな試し。長い文章（青空文庫）を器（I49a の Vessel）に 1 文ずつ読ませ、読み進むうちの変化を数える（数えだけ）。

持ち主と決めた物差し 6: 長い文章を読ませ、読み進むうちの変化を見る。場面の切り替えと、その時の変化に注意する。
器の作りは変えない（Vessel(2048, bands=20, longest=1024) をそのまま使う）。

作品: 元の本文（Shift_JIS）を、区切りの印つきで読み直す。本文は 2 本目の「-----」の行の後から「底本：」の前まで。
  ルビ（《…》と ｜）を除き、注記 ［＃…］ は印として読んでから除く。1 文ごとに、直前に
  見出し（［＃「…」は大／中／小見出し］のある行）・空き行・改ページ等（［＃改ページ］［＃改丁］［＃改段］［＃改見開き］）があったかの印を持たせる。
  見出しの行そのものは読ませない（印だけ）。文は 。！？ で切る。作品は、見出しの注記がある作品のうち、文が 1,000 以上の物を種 0 で 5 つ選ぶ。
入力（1 文 = 1 回の hear）:
  A  属性の袋: 内容語（NOUN・PROPN・VERB・ADJ）の属性（語板の帯 B・C・D・E・G の上位）を足して長さ 1 にした物（float32、D 2048）
  B  流れから学んだ意味の波: 波の板の語の波 W の、内容語（I48 の STOP を除く）の idf の重みの和を長さ 1 にした物（complex64、D 2048）
  ベクトルが 0 の文（内容語が無い・板に無い）は tick にする（その数も数える）。
数える物（波 20 本それぞれと平均）:
  a 区切りの直後の文の外れが、そうでない文より大きいか（AUC。見出し・空き行・改ページを別々に）
  b 区切りの後、外れが元の水準（区切りの前 20 文の中央値）以下に戻るまでの文の数（波ごとの中央値、200 文で打ち切り）
  c 同じ作品の文の順をばらばらにした時（種 0）と比べた外れの平均の差（ばらばら − 順）
  d 5 作品を順に続けて読ませた時、作品の替わり目 4 か所の跳ね（替わり目の外れ − 前の 10 文の外れの平均）と、
    同じ作品の中の見出しの跳ねの比べ
  e 見張り: 張り付きの回数（波ごと）、死んだ刻み
  f 感度の行き先: 波ごとの中央値と、上限・下限に張り付いた割合（最後の刻みと、全部の刻み）
手で置く物: 1 作品の文の上限 3,000、元の水準の窓 20 文、戻りの打ち切り 200 文、跳ねの前の窓 10 文、区切りの形（上の 3 つ）、種 0。
python tools/i49b_read_long.py --src <元の本文の置き場> --conllu <青空文庫の conllu> --content <content_boards.json> --func <func_boards.json> \
    --wave <波の板（.npz と .json の名前）> --out <stats.json> [--npz <記録の置き場>]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "intervention"))
import i48ab_long_grain as G  # noqa: E402
import i48_wave_board as W48  # noqa: E402
from i49a_inner_vessel import Vessel  # noqa: E402

MAX_SENT = 3000
BASE_WIN, MAX_BACK, JUMP_WIN = 20, 200, 10
HEAD = re.compile(r"［＃.*?(大|中|小)?見出し］")
PAGE = re.compile(r"［＃改(ページ|丁|段|見開き)］")
NOTE = re.compile(r"［＃.*?］")
RUBY = re.compile(r"《.*?》")
CONTENT_POS = ("NOUN", "PROPN", "VERB", "ADJ")


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return None
    a = np.array(list(neg) + list(pos), dtype=float)
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    r = (np.bincount(inv, weights=a.argsort().argsort() + 1.0) / cnt)[inv]
    return round(float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))), 4)


def body_of(path):
    raw = open(path, "rb").read().decode("cp932", errors="replace").replace("\r\n", "\n")
    L = raw.split("\n")
    dash = [i for i, ln in enumerate(L) if ln.startswith("-----")]
    if len(dash) < 2:
        return None
    end = next((i for i, ln in enumerate(L) if ln.startswith("底本：")), len(L))
    return L[dash[1] + 1: end]


def sentences_with_marks(lines):
    """行の並び → [(文, dict(見出し, 空き行, 改ページ))]。印は、その文の直前に区切りがあった時だけ立つ。"""
    out, pend = [], dict(見出し=False, 空き行=False, 改ページ=False)
    for ln in lines:
        if PAGE.search(ln):
            pend["改ページ"] = True
        if HEAD.search(ln):
            pend["見出し"] = True
            continue                                    # 見出しの行そのものは読ませない
        text = RUBY.sub("", NOTE.sub("", ln)).replace("｜", "").strip().strip("　")
        if not text:
            if not PAGE.search(ln):
                pend["空き行"] = True
            continue
        for s in re.findall(r"[^。！？]+[。！？」』）]*", text):
            s = s.strip("　 ")
            if not s:
                continue
            out.append((s, dict(pend)))
            pend = dict(見出し=False, 空き行=False, 改ページ=False)
    return out


def pick_works(src_dir, conllu, n=5, seed=0):
    ids = [ln.split("= ")[1].strip().split("-")[1] for ln in open(conllu, encoding="utf-8") if ln.startswith("# newdoc id = ")]
    files = {}
    for f in os.listdir(src_dir):
        files.setdefault(f.split("_")[0], []).append(f)
    ids = [i for i in ids if i in files]
    random.Random(seed).shuffle(ids)
    got = []
    for wid in ids:
        B = body_of(os.path.join(src_dir, sorted(files[wid])[0]))
        if not B or not any(HEAD.search(x) for x in B):
            continue
        S = sentences_with_marks(B)
        if len(S) >= 1000:
            got.append(S[:MAX_SENT])
        if len(got) == n:
            break
    return got


class Vecs:
    def __init__(self, a):
        cb = json.load(open(a.content, encoding="utf-8"))
        func = G.Func(json.load(open(a.func, encoding="utf-8")))
        self.S = G.AttrSpace(cb, func, 2048)
        z = np.load(a.wave + ".npz")
        self.W = z["W"]
        m = json.load(open(a.wave + ".json", encoding="utf-8"))
        self.ix = {w: i for i, w in enumerate(m["vocab"]) if w is not None}
        self.idf = m["idf"]

    def A(self, doc):
        y = np.zeros(2048, dtype=np.float32)
        for t in doc:
            if t.pos_ in CONTENT_POS:
                f, v = self.S.attrs(t.lemma_, t.pos_, t.tag_, t.text)
                np.add.at(y, f, v)
        n = float(np.linalg.norm(y))
        return y / n if n > 0 else None

    def B(self, doc):
        ws = [t.lemma_ for t in doc if t.pos_ in W48.CONTENT and t.lemma_ not in W48.STOP and t.lemma_ in self.ix]
        if not ws:
            return None
        v = (self.W[[self.ix[w] for w in ws]] * np.asarray([self.idf.get(w, 1.0) for w in ws], dtype=np.float32)[:, None]).sum(0)
        n = float(np.linalg.norm(v))
        return (v / n).astype(np.complex64) if n > 0 else None


def run(seq):
    """seq: ベクトル（None は tick）の並び → 器、文ごとの外れ（tick の文は nan）。"""
    v = Vessel(2048, bands=20, longest=1024)
    sur = np.full((len(seq), v.N), np.nan, dtype=np.float32)
    for i, x in enumerate(seq):
        if x is None:
            v.tick()
        else:
            sur[i] = v.hear(x)
    return v, sur


def measure_work(v, sur, marks):
    N = sur.shape[1]
    heard = ~np.isnan(sur[:, 0])
    out = {}
    for kind in ("見出し", "空き行", "改ページ"):
        m = np.array([mk[kind] for mk in marks]) & heard
        other = heard & ~np.array([any(mk.values()) for mk in marks])
        out[f"a_{kind}"] = dict(数=int(m.sum()), 波ごとのAUC=[auc(sur[m, k], sur[other, k]) for k in range(N)],
                                平均のAUC=auc(np.nanmean(sur[m], 1), np.nanmean(sur[other], 1)) if m.any() else None)
    # b 戻るまでの文の数
    brk = [i for i, mk in enumerate(marks) if any(mk.values()) and i >= BASE_WIN and heard[i]]
    back = [[] for _ in range(N)]
    for i in brk:
        for k in range(N):
            base = np.nanmedian(sur[i - BASE_WIN: i, k])
            if np.isnan(base) or not sur[i, k] > base:
                continue
            j = next((d for d in range(1, MAX_BACK + 1) if i + d < len(sur) and not np.isnan(sur[i + d, k]) and sur[i + d, k] <= base), MAX_BACK)
            back[k].append(j)
    out["b_戻るまでの文の数の中央値"] = [float(np.median(b)) if b else None for b in back]
    out["b_跳ねた区切りの数"] = [len(b) for b in back]
    g = np.array(v.log["gain"])
    out["e_張り付き"] = v.watch["張り付き"].tolist()
    out["e_死んだ刻み"] = int(v.watch["死んだ"])
    out["f_感度の中央値"] = np.round(np.median(g, 0), 3).tolist()
    out["f_上限に張り付いた割合"] = np.round((g >= v.gmax * 0.999).mean(0), 4).tolist()
    out["f_下限に張り付いた割合"] = np.round((g <= 1.0 / v.gmax * 1.001).mean(0), 4).tolist()
    out["外れの平均"] = np.round(np.nanmean(sur, 0), 4).tolist()
    return out


KS = (1, 2, 5, 10, 20, 50, 100, 200)


def cos_bands(A, B):
    """波ごとの向きの近さ |cos|（複素）。どちらかが 0 の波は nan。"""
    na, nb = np.linalg.norm(A, axis=1), np.linalg.norm(B, axis=1)
    d = np.abs(np.sum(np.conj(A) * B, axis=1))
    out = np.full(A.shape[0], np.nan, dtype=np.float64)
    ok = (na > 0) & (nb > 0)
    out[ok] = d[ok] / (na[ok] * nb[ok])
    return out


def direction(seq, points):
    """5 作品を続けた並び seq を器に読ませ、points（名前 → 文の番号の並び）の各点 i について、i を聞く直前の状態と、
    i − 1 + k 文目を聞いた後の状態の |cos| を波ごとに出す（器のコードは変えず、v.S の写しを要る間だけ持つ）。
    あわせて、文ごとに、いちばん速い波といちばん遅い波の向きのずれ（1 − |cos|）を出す。"""
    v = Vessel(2048, bands=20, longest=1024)
    want = {}
    for nm, idx in points.items():
        for i in idx:
            want.setdefault(i, []).append(nm)
    pending = []                                       # (始めの文, 写し, 名前の並び)
    res = {nm: {k: [] for k in KS} for nm in points}
    gap = np.full(len(seq), np.nan)
    for i, x in enumerate(seq):
        if i in want:
            pending.append((i, v.S.copy(), want[i]))
        if x is None:
            v.tick()
        else:
            v.hear(x)
        gap[i] = 1.0 - cos_bands(v.S[:1], v.S[-1:])[0]
        keep = []
        for (b, S0, nms) in pending:
            k = i - b + 1
            if k in KS:
                c = cos_bands(S0, v.S)
                for nm in nms:
                    res[nm][k].append(c)
            if k < KS[-1]:
                keep.append((b, S0, nms))
        pending = keep
    out = {}
    for nm in points:
        out[nm] = dict(数=len(points[nm]), **{f"k{k}": np.round(np.nanmean(np.array(res[nm][k]), 0), 4).tolist() if res[nm][k] else None for k in KS})
    return out, gap, v


def main_direction(a, works, seqs):
    """b の測り直し: 区切りの前後の状態の向き（B の入力、5 作品を続けて読む）。"""
    cat, kinds = [], dict(見出し=[], 空き行=[], 作品の替わり目=[])
    for wi, (w, s) in enumerate(zip(works, seqs)):
        base = len(cat)
        if wi > 0:
            kinds["作品の替わり目"].append(base)
        for i, (_, m) in enumerate(w):
            if i == 0:
                continue
            if m["見出し"]:
                kinds["見出し"].append(base + i)
            elif m["空き行"]:
                kinds["空き行"].append(base + i)
        cat += s
    brk = set(i for v_ in kinds.values() for i in v_)
    near = set(j for i in brk for j in range(i, i + 3))
    cand = [i for i in range(1, len(cat) - KS[-1]) if i not in near]
    n_all = sum(len(v_) for v_ in kinds.values())
    ctrl = sorted(random.Random(0).sample(cand, n_all))
    points = dict(kinds, 区切りでない所=ctrl)
    out, gap, v = direction(cat, points)
    heard = np.array([x is not None for x in cat])
    other = [i for i in range(1, len(cat)) if i not in brk and heard[i]]
    gap_auc = {nm: auc(gap[[i for i in idx if heard[i]]], gap[other]) for nm, idx in kinds.items()}
    gap_mean = {nm: round(float(np.nanmean(gap[idx])), 4) for nm, idx in points.items()}
    return dict(向きの近さ=out, 速い波と遅い波のずれ_AUC=gap_auc, 速い波と遅い波のずれ_平均=gap_mean,
                ずれの平均_全部=round(float(np.nanmean(gap)), 4), 張り付き=v.watch["張り付き"].tolist(), 死んだ刻み=int(v.watch["死んだ"]))


PAUSES = (0, 10, 100, 1000)
POST = 5


def points_of(works, seqs):
    """5 作品を続けた並びと、区切りの種類ごとの文の番号、同じ数の区切りでない所（種 0、区切りの直後 2 文を除く）。"""
    cat, kinds = [], dict(見出し=[], 空き行=[], 作品の替わり目=[])
    for wi, (w, s) in enumerate(zip(works, seqs)):
        base = len(cat)
        if wi > 0:
            kinds["作品の替わり目"].append(base)
        for i, (_, m) in enumerate(w):
            if i == 0:
                continue
            if m["見出し"]:
                kinds["見出し"].append(base + i)
            elif m["空き行"]:
                kinds["空き行"].append(base + i)
        cat += s
    brk = set(i for v_ in kinds.values() for i in v_)
    near = set(j for i in brk for j in range(i, i + 3))
    cand = [i for i in range(1, len(cat) - KS[-1]) if i not in near]
    ctrl = sorted(random.Random(0).sample(cand, len(brk)))
    return cat, dict(kinds, 区切りでない所=ctrl)


def pause_run(cat, points, L):
    """points の各文の前に L 刻みの間（tick）を差し込んで読ませる。器のコードは変えず、張り付きの数の増え方を刻みごとに見る。"""
    v = Vessel(2048, bands=20, longest=1024)
    kind_of = {i: nm for nm, idx in points.items() for i in idx}
    N = v.N
    pin_post, pin_other, pin_pause = np.zeros(N, np.int64), np.zeros(N, np.int64), np.zeros(N, np.int64)
    pin_post_kind = {nm: np.zeros(N, np.int64) for nm in points}
    gain_end = {nm: [] for nm in points}
    sur_first = {nm: [] for nm in points}
    sur_all = []
    post_left, post_kind = 0, None

    def step(fn, *x):
        before = v.watch["張り付き"].copy()
        out = fn(*x)
        return out, v.watch["張り付き"] - before
    for i, x in enumerate(cat):
        nm = kind_of.get(i)
        if nm is not None and L > 0:
            for _ in range(L):
                _, d = step(v.tick)
                pin_pause += d
            gain_end[nm].append(v.gain.copy())
        if nm is not None:
            post_left, post_kind = POST, nm
        if x is None:
            _, d = step(v.tick)
            sur = None
        else:
            sur, d = step(v.hear, x)
            sur_all.append(sur)
        if post_left > 0:
            pin_post += d
            pin_post_kind[post_kind] += d
            if post_left == POST and sur is not None:
                sur_first[post_kind].append(sur)
            post_left -= 1
        else:
            pin_other += d
    n_post = sum(len(v_) for v_ in points.values()) * POST
    n_other = len(cat) - n_post
    res = dict(間の長さ=L, 刻み=int(v.watch["刻み"]), 死んだ刻み=int(v.watch["死んだ"]),
               張り付き_間の直後5文=pin_post.tolist(), 張り付き_それ以外の文=pin_other.tolist(), 張り付き_間の中=pin_pause.tolist(),
               張り付き_直後5文あたり=round(float(pin_post.sum()) / max(1, n_post), 4), 張り付き_それ以外1文あたり=round(float(pin_other.sum()) / max(1, n_other), 4),
               種類ごと={})
    sa = np.array(sur_all)
    res["外れの平均_全部"] = round(float(sa.mean()), 4)
    for nm in points:
        g = np.array(gain_end[nm]) if gain_end[nm] else None
        sf = np.array(sur_first[nm]) if sur_first[nm] else None
        res["種類ごと"][nm] = dict(
            数=len(points[nm]), 張り付き_直後5文=pin_post_kind[nm].tolist(),
            張り付き_直後5文の1か所あたり=round(float(pin_post_kind[nm].sum()) / max(1, len(points[nm])), 3),
            間の終わりの感度_中央値=np.round(np.median(g, 0), 3).tolist() if g is not None else None,
            間の終わりの感度_上限の割合=np.round((g >= v.gmax * 0.999).mean(0), 3).tolist() if g is not None else None,
            直後の文の外れ_波ごと=np.round(sf.mean(0), 4).tolist() if sf is not None else None,
            直後の文の外れ_平均=round(float(sf.mean()), 4) if sf is not None else None)
    return res


def main():
    ap = argparse.ArgumentParser()
    for k in ("src", "conllu", "content", "func", "wave", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--npz", default=None)
    ap.add_argument("--direction", action="store_true", help="b の測り直しだけ（B の入力、区切りの前後の状態の向き）")
    ap.add_argument("--pause", action="store_true", help="区切りと区切りでない所に黙る刻みを差し込み、張り付きを数える（B の入力）")
    a = ap.parse_args()
    if a.pause:
        t0 = time.time()
        works = pick_works(a.src, a.conllu)
        import spacy
        nlp = spacy.load("ja_ginza")
        vec = Vecs(a)
        seqs = [[vec.B(d) for d in nlp.pipe([s for s, _ in w], batch_size=256)] for w in works]
        cat, points = points_of(works, seqs)
        res = dict(間=[], 直後の文の数=POST)
        for L in PAUSES:
            t1 = time.time()
            r = pause_run(cat, points, L)
            r["秒"] = round(time.time() - t1)
            res["間"].append(r)
            print("間", L, r["秒"], "秒", flush=True)
            json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        res["秒"] = round(time.time() - t0)
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("済み", res["秒"], "秒", flush=True)
        return
    if a.direction:
        t0 = time.time()
        works = pick_works(a.src, a.conllu)
        import spacy
        nlp = spacy.load("ja_ginza")
        vec = Vecs(a)
        seqs = [[vec.B(d) for d in nlp.pipe([s for s, _ in w], batch_size=256)] for w in works]
        res = main_direction(a, works, seqs)
        res["秒"] = round(time.time() - t0)
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("済み", res["秒"], "秒", flush=True)
        return
    t0 = time.time()
    works = pick_works(a.src, a.conllu)
    print("作品", [len(w) for w in works], f"{time.time() - t0:.0f} 秒", flush=True)
    import spacy
    nlp = spacy.load("ja_ginza")
    vec = Vecs(a)
    docs = []
    for w in works:
        t1 = time.time()
        docs.append(list(nlp.pipe([s for s, _ in w], batch_size=256)))
        print("読み", len(w), f"{time.time() - t1:.0f} 秒", flush=True)
    res = dict(作品=[dict(文=len(w), 見出し=sum(m["見出し"] for _, m in w), 空き行=sum(m["空き行"] for _, m in w),
                         改ページ=sum(m["改ページ"] for _, m in w)) for w in works], 入力={})
    half = None
    for name in ("A", "B"):
        f = getattr(vec, name)
        seqs = [[f(d) for d in ds] for ds in docs]
        r = dict(tickにした文=[sum(x is None for x in s) for s in seqs], 作品ごと=[], ばらばらとの差=[])
        t1 = time.time()
        for wi, (w, s) in enumerate(zip(works, seqs)):
            marks = [m for _, m in w]
            v, sur = run(s)
            half = v.half.tolist()
            mw = measure_work(v, sur, marks)
            if wi == 0:
                mw["1刻みのミリ秒"] = round((time.time() - t1) / len(s) * 1000, 3)
            r["作品ごと"].append(mw)
            if a.npz:
                os.makedirs(a.npz, exist_ok=True)
                v.dump(os.path.join(a.npz, f"i49b_{name}_w{wi + 1}.npz"))
            perm = list(range(len(s)))
            random.Random(0).shuffle(perm)
            _, sur2 = run([s[i] for i in perm])
            r["ばらばらとの差"].append(np.round(np.nanmean(sur2, 0) - np.nanmean(sur, 0), 4).tolist())
            print(name, "作品", wi + 1, f"{time.time() - t1:.0f} 秒", flush=True)
        # d 5 作品を続けて
        cat, starts, heads = [], [], []
        for w, s in zip(works, seqs):
            starts.append(len(cat))
            heads += [len(cat) + i for i, (_, m) in enumerate(w) if m["見出し"] and i > 0]
            cat += s
        v, sur = run(cat)

        def jump(i):
            if np.isnan(sur[i, 0]):
                i = next((j for j in range(i, min(len(sur), i + 5)) if not np.isnan(sur[j, 0])), i)
            prev = sur[max(0, i - JUMP_WIN): i]
            return sur[i] - np.nanmean(prev, 0) if not np.isnan(sur[i, 0]) and len(prev) else None
        jw = [x for x in (jump(i) for i in starts[1:]) if x is not None]
        jh = [x for x in (jump(i) for i in heads if i >= JUMP_WIN) if x is not None]
        r["d_続けて読む"] = dict(替わり目の跳ね_波ごと=np.round(np.mean(jw, 0), 4).tolist() if jw else None,
                             見出しの跳ね_波ごとの平均=np.round(np.mean(jh, 0), 4).tolist() if jh else None,
                             替わり目の跳ね_平均=round(float(np.mean(jw)), 4) if jw else None,
                             見出しの跳ね_平均=round(float(np.mean(jh)), 4) if jh else None, 見出しの数=len(jh),
                             張り付き=v.watch["張り付き"].tolist(), 死んだ刻み=int(v.watch["死んだ"]))
        r["秒"] = round(time.time() - t1)
        res["入力"][name] = r
        json.dump(dict(res, 半減期=half), open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    res["半減期"] = half
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("済み", res["秒"], "秒", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
