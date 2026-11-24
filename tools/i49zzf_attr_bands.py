"""I49zzf: 語板の属性の帯ごとに、語の近さへの効き方を数える（語板を作り直すかどうかを考えるための数え。未確定の試し）。

問い: 語の近さ（属性の波の cos）は、どの帯の属性で決まっているか。その近さは「同じ使われ方をする語」を近くに置けているか。
  席ごと（だれが・何を・どうした）に、材料に出た語（1 語 1 回、属性を持つ語だけ）の全部の組について数える。
  「同じ使われ方」: だれが・何を の席では、同じ「どうした」の語と 1 度でも一緒に出た 2 語。どうした の席では、同じ「何を」の語と一緒に出た 2 語。
  帯ごとに出す物:
    属性の数、その帯の属性を持つ語の数
    近さの取り分: 全部の帯で測った cos のうち、その帯の属性から来る分の平均（全部の帯で足すと cos の平均になる）
    その帯だけの見分け: その帯の属性だけで測った cos で、「同じ使われ方の組」が「そうでない組」より上に来る割合（.5 = 見分けられない）
    その帯を抜いた見分け: その帯を抜いた残りで測った時の同じ割合。全部の帯の見分けより上がるなら、その帯は見分けを下げている
  見分けの割合は、順位の和で数える（同じ値は半分ずつ）。乱数は使わない。
手で置いた物: 「同じ使われ方」の決め方、cos で測る事、1 語 1 回で数える事。
材料の語や例は出さない。出すのは帯の名前と集計の数だけ。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "intervention"))

SEATS = ("だれが", "何を", "どうした")


def rank_share(score, same):
    """同じ使われ方の組が、そうでない組より上に来る割合（順位の和。同じ値は平均の順位）。"""
    n1 = int(same.sum())
    n0 = int(len(same) - n1)
    if not n1 or not n0:
        return None
    order = np.argsort(score, kind="stable")
    s = score[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    r = np.empty(len(s), dtype=np.float64)
    r[order] = ranks
    return round(float((r[same].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)), 4)


def _cos(m):
    n = np.linalg.norm(m, axis=1)
    u = m / np.where(n > 0, n, 1)[:, None]
    return u @ u.T


def band_report(base, talks):
    """base: AttrBase。talks: [(3 つ組の並び, ...), ...]。→ 席ごとの数え。"""
    bands = sorted({b for b, _ in base.ix})
    cols = {b: np.array([i for (bb, _), i in base.ix.items() if bb == b], dtype=np.int64) for b in bands}
    out = {}
    for s in range(3):
        other = 2 if s < 2 else 1
        ctx = {}
        for t in talks:
            for tri in t[0]:
                ctx.setdefault(tri[s], set()).add(tri[other])
        words = [w for w in sorted(ctx, key=str) if base.of(w) is not None]
        if len(words) < 3:
            continue
        m = np.stack([np.real(base.of(w)).astype(np.float32) for w in words])
        iu = np.triu_indices(len(words), 1)
        same = np.array([bool(ctx[words[i]] & ctx[words[j]]) for i, j in zip(*iu)])
        full = _cos(m)[iu]
        n = np.linalg.norm(m, axis=1)
        u = m / np.where(n > 0, n, 1)[:, None]
        d = dict(語=len(words), 組=int(len(same)), 同じ使われ方の組=int(same.sum()), cosの平均=round(float(full.mean()), 4),
                 全部の帯の見分け=rank_share(full, same), 帯={})
        for b in bands:
            c = cols[b]
            has = int((np.abs(m[:, c]).sum(axis=1) > 0).sum())
            if not has:
                continue
            rest = np.setdiff1d(np.arange(m.shape[1]), c)
            d["帯"][b] = dict(属性=int(len(c)), 持つ語=has,
                             近さの取り分=round(float((u[:, c] @ u[:, c].T)[iu].mean()), 4),
                             力の取り分=round(float((u[:, c] ** 2).sum(axis=1).mean()), 4),
                             その帯だけの見分け=rank_share(_cos(m[:, c])[iu], same),
                             その帯を抜いた見分け=rank_share(_cos(m[:, rest])[iu], same))
        out[SEATS[s]] = d
    return out


def band_filter(base, spec):
    """品詞ごとに、使う帯を変えた attr_of を返す（向きの長さは変えず、使わない帯の所を 0 にする）。
    spec: "VERB:-D,NOUN:-E" なら、動詞は D を抜き、名詞は E を抜く。"VERB:+C+B" なら、動詞は C と B だけ。書いていない品詞は全部の帯。
    使う帯に属性が 1 つも無い語は None（属性を持たない語と同じ扱い）。"""
    rule = {}
    for part in [x for x in (spec or "").split(",") if x.strip()]:
        pos, r = part.strip().split(":")
        mask = np.ones(base.n_attr, dtype=np.float32) if r[0] == "-" else np.zeros(base.n_attr, dtype=np.float32)
        for b in [x for x in r.replace("-", "+").split("+") if x]:
            ix = [i for (bb, _), i in base.ix.items() if bb == b]
            assert ix, "帯が無い: %s" % b
            mask[ix] = 0.0 if r[0] == "-" else 1.0
        rule[pos] = mask
    cache = {}

    def of(word):
        if word not in cache:
            v = base.of(word)
            m = rule.get(word[1]) if isinstance(word, tuple) else None
            if v is not None and m is not None:
                v = (v * m).astype(np.complex64)
                v = v if np.abs(v).sum() > 0 else None
            cache[word] = v
        return cache[word]

    return of


def main():
    import i49m_attr_base as AB
    import i49zp_text_talks_run as ZP
    ap = argparse.ArgumentParser()
    for k in ("text", "content", "out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    talks = []
    for f in a.text.split(","):
        talks += ZP.talks_of([l.strip() for l in open(f, encoding="utf-8").read().splitlines()])[0]
    res = band_report(AB.AttrBase(json.load(open(a.content, encoding="utf-8"))), talks)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
