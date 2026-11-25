"""I49x: 解く形で、見せた 2 語の点を「足す」か「掛ける」か（持ち主の脈絡の案「語が増えるたび状況が収束する」から。未確定の試し）。数えだけ。

今の解く形は、見せた 2 語の波を足して 1 本にしてから候補と比べる。これは「どちらかと合う語」も残す。
収束は「どちらとも合う語だけ残す」動きなので、見せた語ごとに別々に点を出し、合わせ方を変えて比べる。
  元のまま          2 語の波を足してから比べる（今までの形）
  別々に出して足す    語ごとの点を足す（元のままとの違いは、割り方だけ。比べ役）
  掛ける            語ごとの点（負は 0）を掛ける
  小さい方          語ごとの点の小さい方
  掛ける・無ければ足す  掛けた点が全部 0 の時だけ、足した点で答える
片方の語しか点を出せない問いは、どの版もその 1 つの点で答える。
採点は、正しい物と、正しいと言える物（I49w と同じ集まり、話と青空）を合わせた物の両方。
流れは tools/i49h_tally_run.py と同じ。引数は tools/i49m_attr_run.py と同じ（--content 付き）。記録はリポジトリの外。
手で置く物: 負の点を 0 にする事。
"""
from __future__ import annotations
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, HERE)

import i49h_tally_run as HR  # noqa: E402
import i49w_also_right_run as WR  # noqa: E402

SEATS = ("だれが", "何を", "どうした")
VERS = ("元のまま", "別々に出して足す", "掛ける", "小さい方", "掛ける・無ければ足す")


def answers(rel, si, q, cands):
    base = rel.ask(si, q, cands)
    out = {"元のまま": base["答え"]}
    per = []
    for sj in range(3):
        if sj == si:
            continue
        one = tuple(q[j] if j == sj else None for j in range(3))
        p = rel.ask(si, one, cands)["点"]
        if p:
            per.append({c: max(0.0, v) for c, v in p.items()})

    def best(s):
        if not s or max(s.values()) <= 0:
            return None
        return max(s, key=lambda c: (s[c], -cands.index(c)))
    if len(per) < 2:
        a = best(per[0]) if per else None
        for v in VERS[1:]:
            out[v] = a
        return out
    keys = [c for c in cands if c in per[0] or c in per[1]]
    add = {c: per[0].get(c, 0.0) + per[1].get(c, 0.0) for c in keys}
    mul = {c: per[0].get(c, 0.0) * per[1].get(c, 0.0) for c in keys}
    out["別々に出して足す"] = best(add)
    out["掛ける"] = best(mul)
    out["小さい方"] = best({c: min(per[0].get(c, 0.0), per[1].get(c, 0.0)) for c in keys})
    out["掛ける・無ければ足す"] = out["掛ける"] if out["掛ける"] is not None else out["別々に出して足す"]
    return out


def stream(rel, talks, ok):
    n = 0
    right = {v: 0 for v in VERS}
    also = {v: 0 for v in VERS}
    blank = {v: 0 for v in VERS}
    seat = {v: [[0, 0] for _ in SEATS] for v in VERS}
    n_seat = [0, 0, 0]
    for tris, words in talks:
        for tri in tris:
            for si in range(3):
                cands = sorted({w for w in words if w not in {tri[j] for j in range(3) if j != si}} | {tri[si]}, key=str)
                if len(cands) < 2:
                    continue
                q = tuple(None if j == si else tri[j] for j in range(3))
                n += 1
                n_seat[si] += 1
                good = ok.get((si, q), set())
                for v, a in answers(rel, si, q, cands).items():
                    r = a == tri[si]
                    o = a is not None and (r or a in good)
                    right[v] += r
                    also[v] += o
                    blank[v] += a is None
                    seat[v][si][0] += r
                    seat[v][si][1] += o
            rel.learn(tri)
        rel.tick()
    d = max(1, n)
    return {v: dict(正しい物=round(right[v] / d, 4), 合わせて=round(also[v] / d, 4), 答えなし=round(blank[v] / d, 4),
                    席ごと={SEATS[i]: dict(正しい物=round(seat[v][i][0] / max(1, n_seat[i]), 4),
                                        合わせて=round(seat[v][i][1] / max(1, n_seat[i]), 4)) for i in range(3)})
            for v in VERS}


def run(wave_of, attr_of, talks, aoz):
    import i49k_seat_vec as KV

    def safe(f):
        return lambda w: None if w is None else f(w)
    ok = WR.ok_sets([t for tris, _ in talks for t in tris] + [tuple(t) for t in aoz])
    out = {}
    for start, pre in (("白紙から", ()), ("青空の 3 つ組を先に", aoz)):
        out[start] = {}
        for form, base in (("解く・属性（席あり）", attr_of), ("解く・語の波（席あり）", wave_of)):
            u = KV.Unbind(KV.SeatVec(safe(base), weight="数", bind="位相"), online=True)
            for tri in pre:
                u.sv.hear(tri)
            out[start][form] = stream(u, talks, ok)
            print(start, form, {v: d["正しい物"] for v, d in out[start][form].items()}, flush=True)
    return out


def main():
    import i49m_attr_base as AB
    i = sys.argv.index("--content")
    content = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    a, vec_of, talks, aoz, t0 = HR.load()
    full = AB.AttrBase(json.load(open(content, encoding="utf-8")))
    res = dict(会話=len(talks), 青空の組=len(aoz))
    res.update(run(vec_of, full.of, talks, aoz))
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
