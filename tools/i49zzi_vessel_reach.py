"""I49zzi: 器（内面の波）に届かない語を数える（内面波の実験の前の下調べ。見るだけの道具）。

問い: 器は語の属性の波を受け取る。属性を持たない語は器を動かせない。材料のうち、どれだけが器に届かないか。
  席ごと（だれが・何を・どうした）に出す物:
    語の種類の数と、属性を持つ種類の割合
    出た回数（3 つ組の席の延べ）と、属性を持つ語が座っていた割合
    属性を持つ語の、属性の数の平均
  3 つ組ごとに出す物: 3 つの席のうち、属性を持つ語がいくつ有るか（0〜3）の数。0 の組は器を全く動かせない。
  話ごとに出す物: 属性を持つ語が 1 つも無い話の数。
手で置いた物: 無し（数えるだけ）。
材料の語や例は出さない。出すのは集計の数だけ。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "intervention"))

SEATS = ("だれが", "何を", "どうした")


def reach(attr_of, talks):
    """attr_of: 語 → 属性の波（無ければ None）。talks: [(3 つ組の並び, ...), ...]。→ 集計。"""
    kinds = [dict() for _ in SEATS]
    hits = [0, 0, 0]
    slots = [0, 0, 0]
    per_tri = [0, 0, 0, 0]
    empty_talks = 0
    for t in talks:
        any_hit = False
        for tri in t[0]:
            n = 0
            for s in range(3):
                w = tri[s]
                if w not in kinds[s]:
                    v = attr_of(w)
                    kinds[s][w] = 0 if v is None else int((np.abs(v) > 0).sum())
                slots[s] += 1
                if kinds[s][w]:
                    hits[s] += 1
                    n += 1
            per_tri[n] += 1
            any_hit = any_hit or n > 0
        empty_talks += 0 if any_hit else 1
    out = {}
    for s, name in enumerate(SEATS):
        has = [c for c in kinds[s].values() if c]
        out[name] = dict(語の種類=len(kinds[s]), 属性を持つ種類=len(has),
                         種類の割合=round(len(has) / max(1, len(kinds[s])), 4),
                         出た回数=slots[s], 属性を持つ語の回数=hits[s],
                         回数の割合=round(hits[s] / max(1, slots[s]), 4),
                         属性の数の平均=round(float(np.mean(has)), 2) if has else None)
    total = sum(per_tri)
    out["3つ組ごとの届く席の数"] = {str(i): per_tri[i] for i in range(4)}
    out["全く届かない3つ組の割合"] = round(per_tri[0] / max(1, total), 4)
    out["話"] = len(talks)
    out["全く届かない話"] = empty_talks
    return out


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
    base = AB.AttrBase(json.load(open(a.content, encoding="utf-8")))
    res = reach(base.of, talks)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
