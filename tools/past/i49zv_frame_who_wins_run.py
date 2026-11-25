"""I49zv: 席を増やすと、「正解が上位に居るのに、属性の似た語に負ける」問いは減るか（数えだけ）。

I49zu で、この負け方は材料のせいではないと分かった。I49zr・I49zs で、足した席（に・で など）を見せると同じ問いが当たりやすくなった。
ここでは、その当たりやすさが「似た語との選び分け」から来ているかを見る。
見る問い: 足した席が有る枠の、が・を・どうした を隠す問い（I49zr の「足した席を見せると」と同じ問い）。形は 波で解く形（FrameSolve・属性）。
  足した席を 見せない ／ 見せる の 2 通りで、同じ問いごとに
    正解の順位（1 位 ／ 2〜3 位 ／ それより下・点が無い）
    2〜3 位で負けた問いの、1 位の語と正解の近さ・正解とほかの候補の近さ（属性の波どうし）
    見せない時の順位 → 見せた時の順位 の移り方
材料と話の作り方、並べ替えは tools/past/i49zr_frame_run.py と同じ（材料はリポジトリに入れない）。
引数: --text 材料（カンマ区切り） --content 内容語の板 --out 記録.json
手で置く物: 上位を 3 位までとする事、並べ替えの k（0 と 1）。
"""
from __future__ import annotations
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, HERE)

import i49zp_text_talks_run as ZP  # noqa: E402
import i49zr_frame_run as R  # noqa: E402
from i49zq_frame import SEATS  # noqa: E402

RANKS = ("1 位", "2〜3 位", "それより下")


def look(attr_of, talks, show):
    """→ 問いの名札 → (順位の組, 1 位と正解の近さ, 正解とほかの候補の近さ)。足した席が有る枠の、核の席の問いだけ。"""
    import i49zl_who_wins_after_run as ZL
    import i49zq_frame as Q
    cos = ZL.attr_cos(attr_of)
    rel = Q.FrameSolve(Q.FrameVec(attr_of))
    out = {}
    for ti, (frames, words, _) in enumerate(talks):
        for fi, f in enumerate(frames):
            q = show(f)
            if not R.is_core(f):
                for si in R.CORE:
                    w = q[si]
                    shown = {x for j, x in enumerate(q) if j != si and x is not None}
                    if w is None or not shown:
                        continue
                    cands = sorted({x for x in words if x not in shown} | {w}, key=str)
                    if len(cands) < 2:
                        continue
                    pts = rel.ask(si, tuple(None if j == si else x for j, x in enumerate(q)), cands)["点"]
                    order = sorted((c for c in pts if pts[c] > 0), key=lambda c: (-pts[c], cands.index(c)))
                    k = order.index(w) if w in order else 99
                    rank = RANKS[0] if k == 0 else RANKS[1] if k < 3 else RANKS[2]
                    nw = no = None
                    if rank == RANKS[1]:
                        nw = cos(None, order[0], w)
                        rest = [cos(None, c, w) for c in cands if c not in (w, order[0])]
                        rest = [x for x in rest if x is not None]
                        no = sum(rest) / len(rest) if rest else None
                    out[(ti, fi, si)] = (rank, nw, no)
            rel.learn(f)
        rel.tick()
    return out


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def table(recs):
    n = max(1, len(recs))
    d = dict(問い=len(recs))
    for r in RANKS:
        d[r] = round(sum(v[0] == r for v in recs.values()) / n, 4)
    lost = [v for v in recs.values() if v[0] == RANKS[1]]
    d["2〜3 位で負けた問い"] = dict(問い=len(lost), 一位と正解の近さ=mean([v[1] for v in lost]), 正解とほかの候補の近さ=mean([v[2] for v in lost]))
    d["席ごと"] = {SEATS[si]: {r: sum(v[0] == r for k, v in recs.items() if k[2] == si) for r in RANKS} for si in R.CORE}
    return d


def run(attr_of, talks, ks=(0, 1)):
    out = {}
    for k in ks:
        seq = [talks[i] for i in ZP.order(len(talks), k)]
        a, b = look(attr_of, seq, R.strip), look(attr_of, seq, lambda f: f)
        keys = [x for x in a if x in b]
        a, b = {x: a[x] for x in keys}, {x: b[x] for x in keys}
        move = {r: {s: sum(1 for x in keys if a[x][0] == r and b[x][0] == s) for s in RANKS} for r in RANKS}
        out["元の順" if k == 0 else "並べ替え %d" % k] = dict(見せない=table(a), 見せる=table(b), 移り方=move)
    return out


def main():
    import i49m_attr_base as AB
    ap = argparse.ArgumentParser()
    for k in ("text", "content", "out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    t0 = time.time()
    talks = []
    for f in a.text.split(","):
        talks += R.talks_of([l.strip() for l in open(f, encoding="utf-8").read().splitlines()])[0]
    full = AB.AttrBase(json.load(open(a.content, encoding="utf-8")))
    res = dict(話=len(talks), 枠=sum(len(t[0]) for t in talks))
    res.update(run(full.of, talks))
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
