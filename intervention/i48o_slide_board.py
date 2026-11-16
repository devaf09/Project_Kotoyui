"""I48o: 状況のスライド（I48j）を、保存・読み込みできる板にする。bot の行づくりの述語の語尾を、状況で選ぶための板。

板 = I48j の語の帯（語ごとに 1 本、つなぎの印を状況の幅でずらして重ねた物）と品詞の帯 ＋ 状況の読み手。
状況の読み手（手で決めない。資料の行の並びから数える）:
  行の状況 = 行の機能語の原形の袋（です・ます・た・てる …）の、過去・続き・丁寧 の有り無し（I48i の素の袋）。語の無い行は「読めない」。
  返事の行の最後の述語の状況を、軸ごとに数える表で当てる。
    過去・続き: 相手の今の行の状況 × 自分の前の行の状況
    丁寧      : 自分の前の行の状況だけ（相手に合わせるのではなく、自分のふだんの話し方を保つ）
  表で多い方（1 か 0）を、その軸の状況とする。手がかりが無い（行が無い・語が無い）か、表にその組が無い時は「読めない」。
  手がかりで決め方が変わらない軸（どの組でも同じ方に決まる）は、手がかりが何も教えていないので、いつも「読めない」。
読む時: 読めない軸は、その軸の 0 と 1 の両方の幅で読んで足す（= その軸を見ない帯の読み、11.3 の全体の帯）。雑音の床も足した数に合わせる。

測る物（外した対話の、行の最後の述語の語尾）:
  状況なしの帯・数える表、正解の状況を渡した帯（上限）、読んだ状況を渡した帯。読み手の軸ごとの当たりと、読めた割合。
  何も出ない割合（帯にも品詞の帯にも、床を越える候補が無い）。混ぜた資料では、資料ごとにも出す。

実行: python intervention/i48o_slide_board.py --cache <rpc.json> [--extra_cache <混ぜる資料>] --D 2048 --out i48o.json [--board <名前>]
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i48e_relation_split as E  # noqa: E402
import i48j_situation_slide as J  # noqa: E402

log = logging.getLogger("i48o")
AX = [a for a, _ in J.AXES]
KEYS = {"過去": ("相手", "自分"), "続き": ("相手", "自分"), "丁寧": ("自分",)}
FINAL_PART = "PART"                                         # 語尾の ね・よ・か など（軸の外。bot では今の決め方のまま）


def line_post(toks):
    """行の最後の中身の語の後ろのつなぎ（E.chain の形）。中身の語が無ければ None。"""
    T = toks
    idx = [i for i, t in enumerate(T) if t["pos"] in E.CONTENT]
    if not idx:
        return None
    i = idx[-1]
    j = i + 1
    while j < len(T) and T[j]["pos"] in E.FUNC:
        j += 1
    return E.chain(T[i + 1:j])


def line_sit(toks):
    """行の状況 = 行の機能語の原形の袋の、過去・続き・丁寧 の有り無し（I48i の素の袋）。機能語も中身の語も無い行は None。"""
    if not any(t["pos"] in E.CONTENT or t["pos"] in E.FUNC for t in toks):
        return None
    return J.situation("+".join(t["lemma"] for t in toks if t["pos"] in E.FUNC) or E.NONE)


def tail_of(toks_after):
    """述語の後ろの機能語の並びのうち、最後に続く終助詞（PART）の原形の並び。"""
    k = len(toks_after)
    while k > 0 and toks_after[k - 1]["pos"] == FINAL_PART:
        k -= 1
    return "+".join(t["lemma"] for t in toks_after[k:])


class SitReader:
    """返事の行の最後の述語の状況を、相手の今の行と自分の前の行の状況から、軸ごとに数える表で当てる。"""

    def __init__(self):
        self.c = {a: defaultdict(Counter) for a in AX}

    @staticmethod
    def feats(ax, partner, own):
        i = AX.index(ax)
        f = dict(相手=None if partner is None else partner[i], 自分=None if own is None else own[i])
        return tuple(f[k] for k in KEYS[ax])

    def fit(self, rows):
        for r in rows:
            for i, a in enumerate(AX):
                self.c[a][self.feats(a, r["partner"], r["own"])][r["sit"][i]] += 1
        return self

    def decide(self, a, k):
        c = self.c[a].get(k)
        if all(x is None for x in k) or not c or c[0] == c[1]:
            return None
        return int(c[1] > c[0])

    def informative(self, a):
        """手がかりで決め方が変わる軸か（手がかりのある組の決め方が 1 つだけなら、手がかりは何も教えていない = 読めない軸）。"""
        return len({self.decide(a, k) for k in self.c[a] if any(x is not None for x in k)} - {None}) > 1

    def read(self, partner, own):
        return tuple(self.decide(a, self.feats(a, partner, own)) if self.informative(a) else None for a in AX)

    def to_json(self):
        return {a: [[list(k), dict(v)] for k, v in self.c[a].items()] for a in AX}

    @classmethod
    def from_json(cls, d):
        r = cls()
        for a in AX:
            for k, v in d[a]:
                r.c[a][tuple(k)] = Counter({int(x): n for x, n in v.items()})
        return r


def reader_rows(work):
    """対話 1 本 → 返事の行ごとの (相手の今の行の状況、自分の前の行の状況、返事の行の最後の述語の場面)。"""
    S = [s for s in E.samples(work) if s["host"] == "述"]
    last = {}
    for s in S:
        last[s["li"]] = s
    L = work["lines"]
    out = []
    for li, s in sorted(last.items()):
        if li < 1:
            continue
        out.append(dict(s=s, sit=J.situation(s["post"]), partner=line_sit(L[li - 1]["toks"]),
                        own=line_sit(L[li - 2]["toks"]) if li >= 2 else None))
    return out


class SlideBoard:
    def __init__(self, band, reader):
        self.band, self.reader = band, reader
        self.cands = band.cands

    @classmethod
    def build(cls, works, D=2048, stride=1, min_count=5):
        tr = [s for w in works for s in E.samples(w) if s["host"] == "述"]
        cnt = Counter(s["post"] for s in tr)
        cands = [c for c, n in cnt.most_common() if n >= min_count]
        band = J.SlideBand(cands, D, stride).fit(tr)
        rows = [r for w in works for r in reader_rows(w) if r["s"]["post"] in band.cix]
        return cls(band, SitReader().fit(rows))

    def read(self, b, sit):
        """帯 b を状況 sit で読む。sit の None の軸は 0 と 1 の両方の幅で読んで足す。"""
        bd = self.band
        combos = list(itertools.product(*[(0, 1) if x is None else (x,) for x in sit]))
        r = np.zeros(len(self.cands))
        for c in combos:                                       # 印を k ずらして読む = 帯を −k ずらして読む（大きい印の表をずらさない）
            r += (bd.cmark @ np.conj(np.roll(b, -bd.k(c)))).real / bd.D
        floor = np.linalg.norm(b) / (bd.D * math.sqrt(2)) * math.sqrt(2 * math.log(len(self.cands))) * math.sqrt(len(combos))
        return np.where(r > floor, r, 0.0)

    def scores(self, w, pos, sit):
        b = self.band.band.get(w)
        r = self.read(b, sit) if b is not None else None
        src = "語"
        if r is None or r.max() <= 0:
            pb = self.band.pband.get(pos)
            r, src = (self.read(pb, sit), "品詞") if pb is not None else (np.zeros(len(self.cands)), "無し")
            if r.max() <= 0:
                src = "無し"
        return r, src

    def batch_scores(self, ws, poses, sits, dev=None, B=1024):
        """scores を行の束でまとめて（測る時のため）。読むのは線形なので、行ごとに ずらした帯の和 を 1 本作ってから、候補の印とまとめて掛ける。"""
        bd = self.band
        C = len(self.cands)
        lf = math.sqrt(2 * math.log(C)) / (bd.D * math.sqrt(2))
        cm = None
        if dev is not None:
            import torch
            cm = torch.from_numpy(bd.cmark).to(dev)

        def run(store, keys):
            out = np.zeros((len(keys), C), np.float32)
            ok = np.zeros(len(keys), bool)
            for s0 in range(0, len(keys), B):
                V, fl, ix = [], [], []
                for n, (key, sit) in enumerate(keys[s0:s0 + B]):
                    b = store.get(key)
                    if b is None:
                        continue
                    combos = list(itertools.product(*[(0, 1) if x is None else (x,) for x in sit]))
                    V.append(sum(np.roll(b, -bd.k(c)) for c in combos))
                    fl.append(np.linalg.norm(b) * lf * math.sqrt(len(combos)))
                    ix.append(s0 + n)
                if not V:
                    continue
                V = np.stack(V).astype(np.complex64)
                if cm is not None:
                    import torch
                    r = (torch.conj(torch.from_numpy(V).to(dev)) @ cm.T).real.cpu().numpy() / bd.D
                else:
                    r = (np.conj(V) @ bd.cmark.T).real / bd.D
                fl = np.array(fl, np.float32)[:, None]
                r = np.where(r > fl, r, 0.0)
                out[ix], ok[ix] = r, True
            return out, ok

        r, ok = run(bd.band, list(zip(ws, sits)))
        src = ["語" if o and m > 0 else "" for o, m in zip(ok, r.max(1))]
        need = [i for i in range(len(ws)) if src[i] != "語"]
        if need:
            rp, okp = run(bd.pband, [(poses[i], sits[i]) for i in need])
            for j, i in enumerate(need):
                r[i] = rp[j]
                src[i] = "品詞" if okp[j] and rp[j].max() > 0 else "無し"
        return r, src

    def save(self, path):
        bw, bp = sorted(self.band.band), sorted(self.band.pband)
        meta = dict(kind="i48o_slide", cands=self.cands, D=self.band.D, stride=self.band.stride, words=bw, poses=bp, reader=self.reader.to_json())
        json.dump(meta, open(path + ".json", "w", encoding="utf-8"), ensure_ascii=False)
        np.savez(path + ".npz", B=np.stack([self.band.band[w] for w in bw]).astype(np.complex64),
                 P=np.stack([self.band.pband[p] for p in bp]).astype(np.complex64))

    @classmethod
    def load(cls, path):
        meta = json.load(open(path + ".json", encoding="utf-8"))
        z = np.load(path + ".npz")
        band = J.SlideBand(meta["cands"], meta["D"], meta["stride"])
        B, P = z["B"], z["P"]                                  # npz は引くたびに読み直すので、1 回だけ読む
        band.band = {w: B[i] for i, w in enumerate(meta["words"])}
        band.pband = {p: P[i] for i, p in enumerate(meta["poses"])}
        return cls(band, SitReader.from_json(meta["reader"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--extra_cache", help="混ぜる 2 つ目の資料（文体の違う会話）")
    ap.add_argument("--extra_name", default="混ぜた資料")
    ap.add_argument("--test_works", type=int, default=500)
    ap.add_argument("--D", type=int, default=2048)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--min_count", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="測る時の掛け算の置き場（cuda:0 など）")
    ap.add_argument("--board", help="学びの対話で作った板を保存する名前（.json と .npz）")
    ap.add_argument("--out")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    t0 = time.time()
    works = json.load(open(a.cache, encoding="utf-8"))
    works = works["works"] if isinstance(works, dict) else works
    for w in works:
        w["_src"] = "RPC"
    random.Random(a.seed).shuffle(works)
    te_w, tr_w = works[: a.test_works], works[a.test_works:]
    if a.extra_cache:
        ex = json.load(open(a.extra_cache, encoding="utf-8"))
        ex = ex["works"] if isinstance(ex, dict) else ex
        for w in ex:
            w["_src"] = a.extra_name
        random.Random(a.seed).shuffle(ex)
        k = len(ex) // 5
        te_w, tr_w = te_w + ex[:k], tr_w + ex[k:]
    sb = SlideBoard.build(tr_w, a.D, a.stride, a.min_count)
    cands, cix = sb.cands, sb.band.cix
    log.info("学び %d 対話、候補 %d、帯の語 %d（%.0f 秒）", len(tr_w), len(cands), len(sb.band.band), time.time() - t0)
    if a.board:
        sb.save(a.board)
        log.info("板を保存: %s", a.board)
    tr_s = [s for w in tr_w for s in E.samples(w) if s["host"] == "述" and s["post"] in cix]
    ct0 = J.CountSit(cands, False).fit(tr_s)
    plain = J.SlideBand(cands, a.D, slide=False).fit(tr_s)
    pb = SlideBoard(plain, sb.reader)
    dev = None
    if a.device:
        import torch
        dev = torch.device(a.device)
    res = dict(cache=os.path.basename(a.cache), extra=os.path.basename(a.extra_cache) if a.extra_cache else None, D=a.D, seed=a.seed,
               n_cands=len(cands), sources={})
    for src in sorted({w["_src"] for w in te_w}):
        rows = [r for w in te_w if w["_src"] == src for r in reader_rows(w) if r["s"]["post"] in cix]
        t = np.array([cix[r["s"]["post"]] for r in rows])
        read = [sb.reader.read(r["partner"], r["own"]) for r in rows]
        o = dict(n=len(rows), 状況の割合={ax: round(float(np.mean([r["sit"][i] for r in rows])), 3) for i, ax in enumerate(AX)})
        o["読み手"] = {ax: dict(読めた=round(float(np.mean([x[i] is not None for x in read])), 3),
                               当たり=round(float(np.mean([x[i] == r["sit"][i] for x, r in zip(read, rows) if x[i] is not None])), 3)
                               if any(x[i] is not None for x in read) else None,
                               いちばん多い方=round(max(o["状況の割合"][ax], 1 - o["状況の割合"][ax]), 3)) for i, ax in enumerate(AX)}
        rows_o = {}

        def row(label, SC, extra=None):
            od = np.argsort(-SC, 1)[:, :3]
            rr = dict(top1=round(float((od[:, 0] == t).mean()), 4), top3=round(float((od == t[:, None]).any(1).mean()), 4),
                      何も出ない=round(float((SC.max(1) <= 0).mean()), 4), **(extra or {}))
            if label != "数える表 状況なし":
                got = [J.situation(cands[int(j)]) if m > 0 else None for j, m in zip(SC.argmax(1), SC.max(1))]
                rr["丁寧が合う"] = round(float(np.mean([g[2] == r["sit"][2] for g, r in zip(got, rows) if g])), 4)
                rr["過去が合う"] = round(float(np.mean([g[0] == r["sit"][0] for g, r in zip(got, rows) if g])), 4)
            rows_o[label] = rr
            log.info("[%s] %-28s 1 番目 %.3f（3 番まで %.3f）何も出ない %.3f %s", src, label, rr["top1"], rr["top3"], rr["何も出ない"],
                     {k: v for k, v in rr.items() if k not in ("top1", "top3", "何も出ない")})

        row("数える表 状況なし", np.stack([ct0.scores(r["s"], None) for r in rows]))
        ws, ps = [r["s"]["w"] for r in rows], [r["s"]["pos"] for r in rows]
        R, _ = pb.batch_scores(ws, ps, [(0, 0, 0)] * len(rows), dev)
        row("帯 状況なし", R)
        for label, sits in (("帯 正解の状況（上限）", [r["sit"] for r in rows]), ("帯 読んだ状況", read),
                            ("板 全部読めない（全体の帯）", [(None, None, None)] * len(rows))):
            R, src_ = sb.batch_scores(ws, ps, sits, dev)
            row(label, R, dict(出どころ=dict(Counter(src_))))
        o["rows"] = rows_o
        res["sources"][src] = o
        log.info("[%s] 行 %d、状況の割合 %s、読み手 %s", src, len(rows), o["状況の割合"], o["読み手"])
    res["seconds"] = round(time.time() - t0)
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log.info("済み %.0f 秒", time.time() - t0)


if __name__ == "__main__":
    main()
