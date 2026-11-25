"""I49zzh: 外付けを使って外れに変わった問い（害）を、具体的に抜き出す（持ち主の頼み「具体的な値を２０個ほど出してくれる？」）。

場面は I49zzd と同じ（話を並べ替えて半分に分け、本体に前半だけ聞かせ、後半で問う。介入 100%、外付けのまま）。
害 = 本体の答えは当たっていたのに、外付けの答えを使って外れた問い。
出す物（問いごと）: 隠した席、見せた 2 語、正解、本体の答え・段・点の上位 3 つ、外付けの答え・段・点の上位 3 つ、
  選ぶ時に見た数え（その段の組での、本体の当たり・外付けの当たり）、その話の中で何番目の 3 つ組か、
  外付けがその時までに聞いた 3 つ組の数、正解の語を外付けがその席で聞いた事が有るか。
抜き出し方: 害の問いを出た順に並べ、黄金比の並びで n 個を取る（頭から取らない。乱数は使わない）。
注意: 出力には材料の語が入る。出力はリポジトリに置かず、外へも出さない（持ち主の文から作った物の決まり）。
引数: --text 材料 --wave 語の波 --content 内容語の板 --out 記録.json [--k 1] [--n 20]
手で置いた物: 場面の作り方、抜き出す数。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, HERE)

import i49zp_text_talks_run as ZP  # noqa: E402

SEATS = ("だれが", "何を", "どうした")


class Peek:
    """部品を包んで、直前の ask の結果を取っておく（部品の動きは変えない）。"""

    def __init__(self, inner):
        self.inner, self.out = inner, None

    def ask(self, si, tri, cands):
        self.out = self.inner.ask(si, tri, cands)
        return self.out

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _name(w):
    return w[0] if isinstance(w, tuple) else w


def _top(o, n=3):
    pts = (o or {}).get("点") or {}
    return [[_name(c), round(float(v), 4)] for c, v in sorted(pts.items(), key=lambda kv: -float(kv[1]))[:n]]


def pick(n_all, n):
    """0 … n_all-1 から、黄金比の並びで n 個（出た順に並べ直す）。"""
    if n_all <= n:
        return list(range(n_all))
    return sorted(ZP.order(n_all, 1)[:n])


def collect(body, ext, first, second):
    """→ (害の問いの並び, 当たりに変わった数, 問いの数)。"""
    from i49zzd_taught_body import TaughtBody
    for tris, _ in first:
        for tri in tris:
            body.learn(tri)
        body.tick()
    pb, pe = Peek(body), Peek(ext)
    t = TaughtBody(pb, pe)
    harm, up, n, heard, seen = [], 0, 0, 0, [set(), set(), set()]
    for tris, words in second:
        for ti, tri in enumerate(tris):
            for si in range(3):
                cands = sorted({w for w in words if w not in {tri[j] for j in range(3) if j != si}} | {tri[si]}, key=str)
                if len(cands) < 2:
                    continue
                pe.out = None
                a = t.ask(si, tuple(None if j == si else tri[j] for j in range(3)), cands)["答え"]
                n += 1
                _, _, ba, ea = t.last
                cell, used = t.log[-1]
                up += used and ba != tri[si] and a == tri[si]
                if used and ba == tri[si] and a != tri[si]:
                    sn = t.seen[-1]
                    harm.append(dict(隠した席=SEATS[si], 見せた語=[_name(tri[j]) for j in range(3) if j != si], 正解=_name(tri[si]),
                                     本体=dict(答え=_name(ba), 段=cell[0], 上位=_top(pb.out)), 外付け=dict(答え=_name(ea), 段=cell[1], 上位=_top(pe.out)),
                                     選ぶ時に見た数え=dict(本体の当たり=sn[0], 外付けの当たり=sn[1]), 話の中で何番目の三つ組=ti + 1,
                                     外付けが聞いた三つ組=heard, 正解の語を外付けがその席で聞いた事が有る=tri[si] in seen[si]))
                t.teach(si, tri, a)
            t.settle(tri)
            heard += 1
            for j in range(3):
                seen[j].add(tri[j])
        t.rest()
    return harm, up, n


def main():
    import i49_foundation as FD
    import i49c_read_words as Rw
    import i49m_attr_base as AB
    ap = argparse.ArgumentParser()
    for k in ("text", "wave", "content", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    talks = []
    for f in a.text.split(","):
        talks += ZP.talks_of([l.strip() for l in open(f, encoding="utf-8").read().splitlines()])[0]
    wv = Rw.WordVecs.load(a.wave)

    def wave_of(w):
        v = wv.of(*w)
        return wv.of(w[0], "PROPN") if v is None and w[1] == "NOUN" else v

    full = AB.AttrBase(json.load(open(a.content, encoding="utf-8")))
    plain = [(t[0], t[1]) for t in talks]
    seq = [plain[i] for i in ZP.order(len(plain), a.k)]
    harm, up, n = collect(FD.body(wave_of, full.of), FD.body(wave_of, full.of), seq[:len(seq) // 2], seq[len(seq) // 2:])
    cells = {}
    for h in harm:
        key = "%s｜%s" % (h["本体"]["段"], h["外付け"]["段"])
        cells[key] = cells.get(key, 0) + 1
    res = dict(問い=n, 当たりに変わった=up, 外れに変わった_害=len(harm), 害の段の組=dict(sorted(cells.items(), key=lambda kv: -kv[1])),
               害のうち_正解の語を外付けがその席で聞いた事が有る=sum(h["正解の語を外付けがその席で聞いた事が有る"] for h in harm),
               抜き出し=[harm[i] for i in pick(len(harm), a.n)])
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "抜き出し"}, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
