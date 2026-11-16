"""I48j: 状況を「周波数のスライド」で渡して、語尾を切り替える試し（持ち主 2026-11-14 の案）。

案: 学ぶ時に、語板に現在進行形や過去形を状況ごとに学ばせておく。会話で状況や空気が変わったら、
周波数をスライドするだけで切り替わる（歩く・公園 → 過去: 公園を 歩いていた、現在にスライド: 公園を 歩いている）。

作り:
- 状況 = 3 つの軸の有り無し（過去・続き・丁寧）。語尾の素から自動で読む（た / いる・てる・とる / です・ます・ござる）。
  スライドの幅 k = 過去 1 ＋ 続き 2 ＋ 丁寧 4（0〜7）に stride を掛けた数。波を周波数の方向に k 個ずらす（np.roll）。
  乱数の位相の波は、ずらすと元の波とほぼ重ならない。
- 語の帯（語ごとに 1 本）= Σ（その語の後ろに出たつなぎの印を、その状況の幅だけずらした物）。
- 読む時: 渡したい状況の幅だけずらしたつなぎの印と、帯を共鳴させる。その状況で学んだつなぎだけが明るい。
  帯に無い時（その語のその状況が資料に無い）は、品詞の帯に下がる。雑音の床より下は読まない。

測る物（述語の後ろ = 語尾。外した対話で）:
1. 状況を渡した時の当たり（渡さない帯、数える表と比べる）。状況は正解の語尾から読んだ物（「こう言いたい」を渡す場面のつもり）。
2. 切り替え: 軸を 1 つ裏返した状況を渡して読む。出たつなぎが、裏返した軸は切り替わり、ほかの軸は保たれている割合。

実行: python intervention/i48j_situation_slide.py --cache <parsed.json> --D 2048 --out i48j.json
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

log = logging.getLogger("i48j")
AXES = (("過去", ("た",)), ("続き", ("いる", "てる", "とる")), ("丁寧", ("です", "ます", "ござる")))


def elems(chain):
    return [] if chain == E.NONE else chain.split("+")


def situation(chain):
    """つなぎ → 状況（軸ごとの 0/1 の組）。"""
    es = set(elems(chain))
    return tuple(int(any(e in es for e in ax)) for _, ax in AXES)


def shift_of(sit, stride):
    return sum(b << i for i, b in enumerate(sit)) * stride


class SlideBand:
    """語ごとの帯。slide=False は状況を見ない帯（I48e の帯と同じ）。"""

    def __init__(self, cands, D, stride=1, slide=True):
        self.cands, self.D, self.stride, self.slide = cands, D, stride, slide
        self.cix = {c: i for i, c in enumerate(cands)}
        self.cmark = np.stack([E.mark("つなぎ:" + c, D) for c in cands])
        self.csit = [situation(c) for c in cands]
        self.band, self.pband = {}, {}

    def k(self, sit):
        return shift_of(sit, self.stride) if self.slide else 0

    def fit(self, S):
        for s in S:
            j = self.cix.get(s["post"])
            if j is None:
                continue
            v = np.roll(self.cmark[j], self.k(self.csit[j]))
            for store, key in ((self.band, s["w"]), (self.pband, s["pos"])):
                if key not in store:
                    store[key] = np.zeros(self.D, np.complex64)
                store[key] += v
        return self

    def read(self, b, sit):
        """帯 b を、状況 sit の幅で読む。雑音の床より下は 0。"""
        m = np.roll(self.cmark, self.k(sit), axis=1)
        r = (m @ np.conj(b)).real / self.D
        floor = np.linalg.norm(b) / (self.D * math.sqrt(2)) * math.sqrt(2 * math.log(len(self.cands)))
        return np.where(r > floor, r, 0.0)

    def scores(self, s, sit):
        b = self.band.get(s["w"])
        r = self.read(b, sit) if b is not None else None
        src = "語"
        if r is None or r.max() <= 0:
            pb = self.pband.get(s["pos"])
            r, src = (self.read(pb, sit), "品詞") if pb is not None else (np.zeros(len(self.cands)), "無し")
        return r, src


class CountSit:
    """数える表: (語, 状況) → つなぎ。無ければ (品詞, 状況) に下がる。use_sit=False は状況を見ない。"""

    def __init__(self, cands, use_sit=True):
        self.cands, self.cix, self.use_sit = cands, {c: i for i, c in enumerate(cands)}, use_sit
        self.w_, self.p_ = defaultdict(Counter), defaultdict(Counter)

    def fit(self, S):
        for s in S:
            if s["post"] in self.cix:
                sit = situation(s["post"]) if self.use_sit else None
                self.w_[(s["w"], sit)][s["post"]] += 1
                self.p_[(s["pos"], sit)][s["post"]] += 1
        return self

    def scores(self, s, sit):
        sit = sit if self.use_sit else None
        v = np.zeros(len(self.cands))
        for c, n in self.p_.get((s["pos"], sit), {}).items():
            v[self.cix[c]] = n
        v = v / max(v.sum(), 1) * 1e-3
        for c, n in self.w_.get((s["w"], sit), {}).items():
            v[self.cix[c]] += n
        return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--extra_cache", help="混ぜる 2 つ目の資料（文体の違う会話）")
    ap.add_argument("--test_works", type=int, default=500)
    ap.add_argument("--train_works", type=int, default=0)
    ap.add_argument("--D", type=int, default=2048)
    ap.add_argument("--stride", type=int, default=1, help="スライドの 1 目盛りの幅（周波数の数）")
    ap.add_argument("--min_count", type=int, default=5)
    ap.add_argument("--test_n", type=int, default=4000)
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
    if a.extra_cache:
        ex = json.load(open(a.extra_cache, encoding="utf-8"))
        ex = ex["works"] if isinstance(ex, dict) else ex
        random.Random(a.seed).shuffle(ex)
        k = len(ex) // 5
        te_w, tr_w = te_w + ex[:k], tr_w + ex[k:]
    tr = [s for w in tr_w for s in E.samples(w) if s["host"] == "述"]
    te_all = [s for w in te_w for s in E.samples(w) if s["host"] == "述"]
    cnt = Counter(s["post"] for s in tr)
    cands = [c for c, n in cnt.most_common() if n >= a.min_count]
    cs = set(cands)
    tr = [s for s in tr if s["post"] in cs]
    te = random.Random(a.seed + 1).sample([s for s in te_all if s["post"] in cs], min(a.test_n, len(te_all)))
    csit = {c: situation(c) for c in cands}
    log.info("述語の後ろ: 候補 %d、学び %d、試し %d。状況の割合（過去・続き・丁寧）%s", len(cands), len(tr), len(te),
             [round(float(np.mean([csit[s["post"]][i] for s in te])), 3) for i in range(len(AXES))])
    t = np.array([cands.index(s["post"]) for s in te])
    res = dict(cache=os.path.basename(a.cache), D=a.D, stride=a.stride, n_cands=len(cands), n_train=len(tr), n_test=len(te), rows={}, switch={})

    def row(label, SC, extra=None):
        o = np.argsort(-SC, 1)[:, :3]
        r = dict(top1=round(float((o[:, 0] == t).mean()), 4), top3=round(float((o == t[:, None]).any(1).mean()), 4), **(extra or {}))
        res["rows"][label] = r
        log.info("%-34s 1 番目 %.3f（3 番まで %.3f）%s", label, r["top1"], r["top3"], extra or "")

    for label, use in (("数える表 状況なし", False), ("数える表 状況を渡す", True)):
        ct = CountSit(cands, use).fit(tr)
        row(label, np.stack([ct.scores(s, csit[s["post"]]) for s in te]))
    plain = SlideBand(cands, a.D, slide=False).fit(tr)
    row("帯 状況なし（I48e の帯）", np.stack([plain.scores(s, None)[0] for s in te]))
    sb = SlideBand(cands, a.D, a.stride).fit(tr)
    out = [sb.scores(s, csit[s["post"]]) for s in te]
    row("帯 状況をスライドで渡す", np.stack([r for r, _ in out]), dict(Counter(src for _, src in out)))

    # 切り替え: 軸を 1 つ裏返した状況を渡す
    for i, (ax, _) in enumerate(AXES):
        n = flipped = kept = attested = empty = 0
        ex = []
        for s in te:
            sit = csit[s["post"]]
            new = tuple(b ^ (j == i) for j, b in enumerate(sit))
            r, src = sb.scores(s, new)
            n += 1
            if r.max() <= 0:
                empty += 1
                continue
            c = cands[int(r.argmax())]
            got = csit[c]
            flipped += got[i] == new[i]
            kept += all(got[j] == new[j] for j in range(len(AXES)) if j != i)
            attested += src == "語"
            if len(ex) < 12 and src == "語" and got == new:
                ex.append(f"{s['w']}: {s['post']} → {c}")
        res["switch"][ax] = dict(n=n, 軸が切り替わった=round(flipped / n, 4), ほかの軸が保たれた=round(kept / n, 4), その語の帯から出た=round(attested / n, 4),
                                 何も出ない=round(empty / n, 4), 例=ex)
        log.info("切り替え %s: 切り替わった %.3f、ほかの軸が保たれた %.3f、その語の帯から %.3f、何も出ない %.3f", ax, flipped / n, kept / n, attested / n, empty / n)
        log.info("    例: %s", " ／ ".join(ex[:8]))
    res["seconds"] = round(time.time() - t0)
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
