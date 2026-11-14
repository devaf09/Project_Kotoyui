"""I48i: 状況の波（場の空気）を関係の板の鍵に入れる試し（持ち主 2026-11-14 の案）。

案: 過去・進行などの時の波や、尊敬・謙譲・丁寧の波を渡すと、文全体の語尾がそろってそちらへ寄る。
語板は「その状況への反応」を持つ。状況の波そのものは、語の外の場に 1 本置く。

ここでの状況の波 = つなぎの素（語尾・助詞の並びを作っている 1 つ 1 つの機能語: です・ます・た・てる・れる・ね・よ…）の印の重ね合わせ。
軸（時・丁寧さ・敬意）は手で決めない。どこから読むかを 3 つ試す:
    自分の前: 同じ話し手の前の行（2 行前と 4 行前。対話は 1 行ずつ交代と見なす）
    相手の前: 1 つ前の行（相手）
    行の中  : 同じ行のほかの語の後ろのつなぎ（当てたい語の自分のつなぎは入れない。答えが漏れるので）
鍵に足す物: 状況の波 × 役の印、語の波 × 状況の波。

測る物: 名詞の後ろ（助詞）と述語の後ろ（語尾）の 1 番目の当たり。係り方は入れない。
比べる相手: 同じ手がかりの数える表（素を 1 つずつ掛ける）。
「素の一致」: 当てた語尾と正解の語尾で、素の集まりがどれだけ重なるか（丁寧さだけ合っている、などを拾う物差し）。

実行: python intervention/i48i_situation_wave.py --cache <parsed.json> --D 512 --out i48i.json
"""
from __future__ import annotations

import argparse
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
from i48_wave_board import Waves  # noqa: E402

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

log = logging.getLogger("i48i")
SRC = ("自分の前", "相手の前", "行の中")


def elems(chain):
    return [] if chain == E.NONE else chain.split("+")


def samples(work):
    """E.samples に、状況の素を足す: s["sit"] = {自分の前: Counter, 相手の前: Counter, 行の中: Counter}。"""
    S = E.samples(work)
    by_line = defaultdict(Counter)
    for s in S:
        by_line[s["li"]].update(elems(s["post"]))
    for s in S:
        li = s["li"]
        s["sit"] = {"自分の前": by_line.get(li - 2, Counter()) + by_line.get(li - 4, Counter()), "相手の前": by_line.get(li - 1, Counter()),
                    "行の中": by_line[li] - Counter(elems(s["post"]))}
    return S


class RelationS(E.Relation):
    """鍵 = 語・品詞・印・係る語 ＋ 状況の波（srcs で選んだ出どころごとに 1 本）。"""

    def __init__(self, wv, cands, D, srcs, idf, **kw):
        super().__init__(wv, cands, D, ("語", "印", "係"), **kw)
        self.srcs, self.idf = tuple(srcs), idf
        self.srole = {x: (E.mark("役:状況:" + x, D), E.mark("役:状況対:" + x, D)) for x in SRC}
        self.emark = {}

    def sit_wave(self, c):
        v = None
        for e, n in c.items():
            if e not in self.emark:
                self.emark[e] = E.mark("素:" + e, self.D)
            x = self.idf.get(e, 1.0) * math.sqrt(n) * self.emark[e]
            v = x if v is None else v + x
        if v is None:
            return None
        return (v / (np.linalg.norm(v) / math.sqrt(self.D) + 1e-9)).astype(np.complex64)

    def key(self, s):
        R, parts = self.role, []
        w = self.w(s["w"])
        if w is not None:
            parts.append(w * R["語"])
        parts.append(E.mark("品:" + s["pos"], self.D) * R["品詞"])
        fv = sum(self.fmark[f"{f}:{s['flags'][f]}"] for f in E.FLAGS)
        fv = fv / (np.linalg.norm(fv) / math.sqrt(self.D) + 1e-9)
        parts.append(fv * R["印"])
        if w is not None:
            parts.append(w * fv * R["対"])
        h = self.w(s["head"])
        if h is not None:
            parts.append(h * R["係"])
            if w is not None:
                parts.append(w * h * R["係対"])
        for x in self.srcs:
            sv = self.sit_wave(s["sit"][x])
            if sv is not None:
                parts.append(sv * self.srole[x][0])
                if w is not None:
                    parts.append(w * sv * self.srole[x][1])
        v = sum(parts)
        return (v / (np.linalg.norm(v) + 1e-9)).astype(np.complex64)


class CountS(E.CountTable):
    def __init__(self, cands, srcs):
        super().__init__(cands, ("印", "係"))
        self.srcs = tuple(srcs)

    def feats(self, s):
        return super().feats(s) + [f"{x}:{e}" for x in self.srcs for e in s["sit"][x]]


def elem_overlap(SC, te, cands):
    """当てたつなぎと正解のつなぎの、素の重なり（Jaccard）の平均。∅ どうしは 1。"""
    o = SC.argmax(1)
    tot = 0.0
    for i, s in zip(o, te):
        a, b = set(elems(cands[i])), set(elems(s["post"]))
        tot += 1.0 if not a and not b else len(a & b) / max(len(a | b), 1)
    return round(tot / len(te), 4)


AXES = {"丁寧": ("です", "ます", "ござる"), "過去": ("た",), "続き": ("てる", "いる", "とる")}   # 物差しのためだけの軸（鍵には使わない）


def axis_match(SC, te, cands):
    """当てたつなぎと正解のつなぎで、軸の素が「有る・無い」が合った割合と、正解で有る割合。"""
    o = SC.argmax(1)
    out = {}
    for ax, es in AXES.items():
        p = np.array([any(e in es for e in elems(cands[i])) for i in o])
        t = np.array([any(e in es for e in elems(s["post"])) for s in te])
        out[ax] = dict(合う=round(float((p == t).mean()), 4), 正解で有る=round(float(t.mean()), 4))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--extra_cache", help="混ぜる 2 つ目の資料（文体の違う会話。文体が 1 色の資料だけでは、状況の波に見分ける物が無い）")
    ap.add_argument("--test_works", type=int, default=500)
    ap.add_argument("--train_works", type=int, default=0)
    ap.add_argument("--D", type=int, default=512)
    ap.add_argument("--min_count", type=int, default=5)
    ap.add_argument("--max_scenes", type=int, default=200000)
    ap.add_argument("--test_n", type=int, default=4000)
    ap.add_argument("--hosts", default="述,名")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    t0 = time.time()
    works = json.load(open(a.cache, encoding="utf-8"))
    works = works["works"] if isinstance(works, dict) else works
    random.Random(a.seed).shuffle(works)
    te_w, tr_w = works[: a.test_works], works[a.test_works:]
    if a.train_works:
        tr_w = tr_w[: a.train_works]
    if a.extra_cache:                                       # 2 つ目の資料は 1/5 を試しに、残りを学びに足す
        ex = json.load(open(a.extra_cache, encoding="utf-8"))
        ex = ex["works"] if isinstance(ex, dict) else ex
        random.Random(a.seed).shuffle(ex)
        k = len(ex) // 5
        te_w, tr_w = te_w + ex[:k], tr_w + ex[k:]
        log.info("混ぜた: 2 つ目の資料 学び %d・試し %d", len(ex) - k, k)
    tr_all = [s for w in tr_w for s in samples(w)]
    te_all = [s for w in te_w for s in samples(w)]
    vocab = sorted({t["lemma"] for w in tr_w for ln in w["lines"] for t in ln["toks"] if t["pos"] in E.CUE})
    seq = [[t["lemma"] for t in ln["toks"] if t["pos"] in E.CUE] for w in tr_w for ln in w["lines"] + [dict(toks=[])]]
    wv = Waves(vocab, a.D, a.seed)
    dev = torch.device(a.device) if (a.device and torch is not None) else None
    wv.learn(seq, dev=dev)
    wv.W = wv.W.astype(np.complex64)
    wv.base = None
    df = Counter(e for s in tr_all for e in set(elems(s["post"])))
    idf = {e: math.log(len(tr_all) / n) for e, n in df.items()}
    res = dict(cache=os.path.basename(a.cache), D=a.D, n_train_works=len(tr_w), n_elems=len(df), tasks={})
    sets = [(), ("自分の前",), ("相手の前",), ("行の中",), ("自分の前", "相手の前"), ("自分の前", "相手の前", "行の中")]
    for host in a.hosts.split(","):
        tr_h = [s for s in tr_all if s["host"] == host]
        cnt = Counter(s["post"] for s in tr_h)
        cands = [c for c, n in cnt.most_common() if n >= a.min_count]
        cs = set(cands)
        tr_c = [s for s in tr_h if s["post"] in cs]
        te = [s for s in te_all if s["host"] == host and s["post"] in cs]
        te = random.Random(a.seed + 1).sample(te, min(a.test_n, len(te)))
        tr_k = tr_c[:: max(1, math.ceil(len(tr_c) / a.max_scenes))]
        name = "名詞の後ろ（助詞）" if host == "名" else "述語の後ろ（語尾）"
        log.info("== %s: 候補 %d、置き場 %d、試し %d、状況が有る割合 %s", name, len(cands), len(tr_k), len(te),
                 {x: round(float(np.mean([bool(s["sit"][x]) for s in te])), 3) for x in SRC})
        rows = {}

        def row(label, SC):
            r = rows[label] = E.measure(SC, te, cands, host)
            r["素の一致"] = elem_overlap(SC, te, cands)
            r["軸"] = axis_match(SC, te, cands)
            log.info("%-36s 1 番目 %.3f（3 番まで %.3f）素の一致 %.3f　丁寧 %.3f 過去 %.3f 続き %.3f", label, r["top1"], r["top3"], r["素の一致"],
                     r["軸"]["丁寧"]["合う"], r["軸"]["過去"]["合う"], r["軸"]["続き"]["合う"])

        B = 256
        for srcs in sets:
            ct = CountS(cands, srcs).fit(tr_c)
            row("数える表 " + ("・".join(srcs) if srcs else "状況なし"), np.stack([ct.scores(s) for s in te]))
        for srcs in sets:
            rel = RelationS(wv, cands, a.D, srcs, idf, device=dev).fit(tr_k)
            row("波 " + ("・".join(srcs) if srcs else "状況なし"), np.concatenate([rel.scoop_scores(te[i:i + B]) for i in range(0, len(te), B)]))
            del rel
            if dev is not None:
                torch.cuda.empty_cache()
        res["tasks"][name] = dict(n_cands=len(cands), n_scenes=len(tr_k), n_test=len(te), rows=rows)
    res["seconds"] = round(time.time() - t0)
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log.info("済み %.0f 秒", time.time() - t0)


if __name__ == "__main__":
    main()
