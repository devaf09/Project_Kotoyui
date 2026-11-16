"""I48h: 関係の板の鍵に、語板の属性の帯（直交の並び）を入れる試し。

持ち主の問い（2026-11-14）: 属性を割合のまま持つのに比べて、直交の周波数で持つ利点はあるか。
利点が出るとしたら、波のまま計算に混ぜる所。ここでは関係の板（I48e・I48f）の鍵に、その語の属性の帯の波を足す。
ねらい: 語そのものを知らなくても、属性が似た語の場面と共鳴して、助詞・語尾を借りられるか。

属性の帯の波（語ごと）:
- 帯ごとに、属性 j を周波数 j に置く（直交。I48g で、割合のままと同じ数に戻ると確かめた形）。
  = 割合の並びを長さ D に伸ばして FFT した物に、帯ごとの共通の乱数の位相を掛ける。
- 帯ごとに大きさをそろえて足す。2 つの語の波の共鳴 = 帯ごとの割合の並びの似かた（cos）の和。
鍵に足す物: 属性の波 × 役の印、属性の波 × 係る語の波（「クラス × 述語」）。

測る物（名詞の後ろ = 助詞、述語の後ろ = 語尾。係り方は入れない）:
- 語・印・係る語（I48f の板の鍵）と、それに属性を足した物。
- 語の鍵を外した形（知らない語のつもり）で、属性があると戻るか。
- 試しを、学びでの語の回数で分ける（0 回 = 知らない語、1〜9 回 = 少ない語、10 回以上）。
- 比べる相手: 数える表（語 → クラス → 品詞 と下がる。クラス = D の帯のいちばん大きい属性）。

実行: python intervention/i48h_relation_attr_key.py --cache <parsed.json> --boards <content_boards.json> --D 2048 --out i48h.json
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
import zlib
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i48e_relation_split as E  # noqa: E402
from i48_wave_board import Waves  # noqa: E402

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

log = logging.getLogger("i48h")


class AttrWaves:
    """語板（割合の表）→ 語ごとの属性の波（直交の並び）。"""

    def __init__(self, boards, D, bands, seed=0):
        self.D, self.bands = D, tuple(bands)
        self.board = {}
        for b in boards:                                    # 同じ名前が品詞ちがいで有る時は、多い方
            if b["attrs"] and (b["name"] not in self.board or b["freq"] > self.board[b["name"]]["freq"]):
                self.board[b["name"]] = b
        self.aix = {}
        for band in self.bands:
            names = sorted({x for b in self.board.values() for x in b["attrs"].get(band, {})})
            assert len(names) <= D, f"帯 {band} の属性 {len(names)} が D {D} より多い"
            self.aix[band] = {x: i for i, x in enumerate(names)}
        self.common = {band: np.exp(1j * np.random.default_rng(zlib.crc32(f"属性の帯:{band}:{seed}".encode())).uniform(0, 2 * np.pi, D))
                       for band in self.bands}
        self.memo = {}

    def wave(self, w):
        if w not in self.memo:
            b, v = self.board.get(w), None
            if b is not None:
                parts = []
                for band in self.bands:
                    d = b["attrs"].get(band)
                    if not d:
                        continue
                    r = np.zeros(self.D)
                    for x, p in d.items():
                        r[self.aix[band][x]] = p
                    r /= np.linalg.norm(r) + 1e-12
                    parts.append(np.fft.fft(r) * self.common[band])          # 大きさ √D（語の波 1 本と同じ強さ）
                if parts:
                    v = (sum(parts) / math.sqrt(len(parts))).astype(np.complex64)
            self.memo[w] = v
        return self.memo[w]

    def cls(self, w):
        b = self.board.get(w)
        dd = b["attrs"].get("D") if b else None
        return max(dd, key=dd.get) if dd else None


class RelationA(E.Relation):
    """I48e の Relation に、鍵「属」（属性の波）を足す。keys に "語" が無ければ語の波を入れない（知らない語のつもり）。"""

    def __init__(self, wv, cands, D, keys, aw, **kw):
        super().__init__(wv, cands, D, keys, **kw)
        self.aw = aw
        self.arole = {k: E.mark("役:" + k, D) for k in ("属", "属係")}

    def w(self, x):
        return super().w(x)

    def key(self, s):
        k, R, parts = self.keys, self.role, []
        w = self.w(s["w"]) if "語" in k else None
        if w is not None:
            parts.append(w * R["語"])
        parts.append(E.mark("品:" + s["pos"], self.D) * R["品詞"])
        if "印" in k:
            fv = sum(self.fmark[f"{f}:{s['flags'][f]}"] for f in E.FLAGS)
            fv = fv / (np.linalg.norm(fv) / math.sqrt(self.D) + 1e-9)
            parts.append(fv * R["印"])
            if w is not None:
                parts.append(w * fv * R["対"])
        h = self.w(s["head"]) if "係" in k else None
        if h is not None:
            parts.append(h * R["係"])
            if w is not None:
                parts.append(w * h * R["係対"])
        if "属" in k:
            av = self.aw.wave(s["w"])
            if av is not None:
                parts.append(av * self.arole["属"])
                if h is not None:
                    parts.append(av * h * self.arole["属係"])
        v = sum(parts)
        return (v / (np.linalg.norm(v) + 1e-9)).astype(np.complex64)


class CountA:
    """数える表: 語 → クラス → 品詞 → 全体 と下がる。係る語・印は I48e と同じく 1 つずつ掛ける。use_word=False は語を見ない。"""

    def __init__(self, cands, aw, use_word=True, use_cls=True):
        self.cands, self.cix, self.aw, self.use_word, self.use_cls = cands, {c: i for i, c in enumerate(cands)}, aw, use_word, use_cls
        self.w_, self.c_, self.p_, self.f_ = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
        self.all = Counter()

    def feats(self, s):
        f = [f"印:{x}:{s['flags'][x]}" for x in E.FLAGS] + ["係:" + s["head"]]
        c = self.aw.cls(s["w"]) if self.use_cls else None
        if c:
            f.append(f"類係:{c}:{s['head']}")
        return f

    def fit(self, S):
        for s in S:
            c = s["post"]
            self.w_[s["w"]][c] += 1
            k = self.aw.cls(s["w"])
            if k:
                self.c_[k][c] += 1
            self.p_[s["pos"]][c] += 1
            self.all[c] += 1
            for x in self.feats(s):
                self.f_[x][c] += 1
        tot = sum(self.all.values())
        self.pr = np.array([self.all[c] / tot for c in self.cands])
        return self

    def dist(self, c, back):
        z = sum(c.values())
        v = np.zeros(len(self.cands))
        for k, n in c.items():
            v[self.cix[k]] = n
        return (v + back) / (z + 1)

    def scores(self, s):
        out = self.dist(self.p_.get(s["pos"], Counter()), self.pr)
        k = self.aw.cls(s["w"]) if self.use_cls else None
        if k:
            out = self.dist(self.c_.get(k, Counter()), out)
        if self.use_word:
            out = self.dist(self.w_.get(s["w"], Counter()), out)
        for x in self.feats(s):
            if x in self.f_:
                out = out * self.dist(self.f_[x], self.pr) / self.pr
        return out


def split_measure(SC, te, cands, host, n_train):
    """全部と、学びでの語の回数ごと（0、1〜9、10 以上）の 1 番目の当たり。"""
    cix = {c: i for i, c in enumerate(cands)}
    t = np.array([cix[s["post"]] for s in te])
    ok = SC.argmax(1) == t
    n = np.array([n_train.get(s["w"], 0) for s in te])
    r = E.measure(SC, te, cands, host)
    for name, m in (("知らない語", n == 0), ("少ない語", (n > 0) & (n < 10)), ("多い語", n >= 10)):
        r[name] = round(float(ok[m].mean()), 4) if m.any() else None
        r[name + "の数"] = int(m.sum())
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--boards", required=True, help="語板 v2 の content_boards.json（割合の表）")
    ap.add_argument("--bands", default="D;A,C,D;A,B,C,D,E,G", help="試す帯の組（; で区切る）")
    ap.add_argument("--test_works", type=int, default=500)
    ap.add_argument("--train_works", type=int, default=0)
    ap.add_argument("--D", type=int, default=2048)
    ap.add_argument("--min_count", type=int, default=5)
    ap.add_argument("--max_scenes", type=int, default=100000)
    ap.add_argument("--test_n", type=int, default=4000)
    ap.add_argument("--hosts", default="名,述")
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
    boards = json.load(open(a.boards, encoding="utf-8"))
    tr_all = [s for w in tr_w for s in E.samples(w)]
    te_all = [s for w in te_w for s in E.samples(w)]
    vocab = sorted({t["lemma"] for w in tr_w for ln in w["lines"] for t in ln["toks"] if t["pos"] in E.CUE})
    seq = [[t["lemma"] for t in ln["toks"] if t["pos"] in E.CUE] for w in tr_w for ln in w["lines"] + [dict(toks=[])]]
    wv = Waves(vocab, a.D, a.seed)
    dev = torch.device(a.device) if (a.device and torch is not None) else None
    wv.learn(seq, dev=dev)
    wv.W = wv.W.astype(np.complex64)
    wv.base = None
    band_sets = [tuple(x.split(",")) for x in a.bands.split(";")]
    aws = {bs: AttrWaves(boards, a.D, bs, a.seed) for bs in band_sets}
    aw_cls = AttrWaves(boards, a.D, ("D",), a.seed)
    res = dict(cache=os.path.basename(a.cache), boards=a.boards.replace("\\", "/").split("/")[-2], D=a.D, n_train_works=len(tr_w), tasks={})
    for host in a.hosts.split(","):
        tr_h = [s for s in tr_all if s["host"] == host]
        cnt = Counter(s["post"] for s in tr_h)
        cands = [c for c, n in cnt.most_common() if n >= a.min_count]
        cs = set(cands)
        tr_c = [s for s in tr_h if s["post"] in cs]
        n_train = Counter(s["w"] for s in tr_c)
        te = [s for s in te_all if s["host"] == host and s["post"] in cs]
        te = random.Random(a.seed + 1).sample(te, min(a.test_n, len(te)))
        tr_k = tr_c[:: max(1, math.ceil(len(tr_c) / a.max_scenes))]
        name = "名詞の後ろ（助詞）" if host == "名" else "述語の後ろ（語尾）"
        has = np.mean([aw_cls.board.get(s["w"]) is not None for s in te])
        log.info("== %s: 候補 %d、置き場 %d、試し %d、語板に有る語の割合 %.3f", name, len(cands), len(tr_k), len(te), has)
        rows = {}

        def row(label, SC):
            r = rows[label] = split_measure(SC, te, cands, host, n_train)
            log.info("%-40s 1 番目 %.3f（3 番まで %.3f）　知らない語 %s（%d）　少ない語 %s（%d）　多い語 %s", label, r["top1"], r["top3"],
                     r["知らない語"], r["知らない語の数"], r["少ない語"], r["少ない語の数"], r["多い語"])

        for label, kw in (("数える表 語・印・係る語", dict(use_cls=False)), ("数える表 語・クラス・印・係る語", dict()),
                          ("数える表 印・係る語（語なし）", dict(use_word=False, use_cls=False)), ("数える表 クラス・印・係る語（語なし）", dict(use_word=False))):
            ct = CountA(cands, aw_cls, **kw).fit(tr_c)
            row(label, np.stack([ct.scores(s) for s in te]))
        B = 256
        variants = [("波 語・印・係る語（I48f の鍵）", ("語", "印", "係"), band_sets[0]), ("波 印・係る語（語なし）", ("印", "係"), band_sets[0])]
        for bs in band_sets:
            variants += [(f"波 語・印・係る語 ＋属性 {''.join(bs)}", ("語", "印", "係", "属"), bs), (f"波 印・係る語 ＋属性 {''.join(bs)}（語なし）", ("印", "係", "属"), bs)]
        for label, keys, bs in variants:
            rel = RelationA(wv, cands, a.D, keys, aws[bs], device=dev).fit(tr_k)
            row(label, np.concatenate([rel.scoop_scores(te[i:i + B]) for i in range(0, len(te), B)]))
            del rel
            if dev is not None:
                torch.cuda.empty_cache()
        res["tasks"][name] = dict(n_cands=len(cands), n_scenes=len(tr_k), n_test=len(te), in_board=round(float(has), 4), rows=rows)
    res["seconds"] = round(time.time() - t0)
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log.info("済み %.0f 秒", time.time() - t0)


if __name__ == "__main__":
    main()
