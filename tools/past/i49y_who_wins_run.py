"""I49y: 正解が上位に入っているのに 1 位になれない問いで、何が 1 位を取っているかを数える。数えだけ。

I49t で、解く形は正解を 3 位までには入れているのに、1 位にできない問いが多いと分かった。
正解が 2〜3 位だった問いを集め、1 位を取った語と正解を比べる。
  1 位が「正しいと言える物」だった割合（I49w の集まり、話と青空）
  1 位の方が、正解より多く聞いた語だった割合（聞いた回数 = 語の波に足した数）。同じ・少ないの割合も
  1 位と正解の、元の向き（属性）の近さの平均。比べ役として、正解とほかの候補の近さの平均
  1 位がこの話でもう出ていた割合、正解がこの話でもう出ていた割合
比べ役として、正解が 1 位だった問いでの「正解の方が 2 位より多く聞いた語だった割合」も出す。
形は 解く・属性（席あり）。流れは tools/i49h_tally_run.py と同じ。引数は tools/i49m_attr_run.py と同じ（--content 付き）。記録はリポジトリの外。
手で置く物: 「上位」を 3 位までとする事。
"""
from __future__ import annotations
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, HERE)

import i49h_tally_run as HR  # noqa: E402
import i49w_also_right_run as WR  # noqa: E402

SEATS = ("だれが", "何を", "どうした")


def _cos(sv, a, b):
    x, y = sv._base(a), sv._base(b)
    if x is None or y is None:
        return None
    return float(np.real(np.vdot(x, y)))


def who(rel, talks, ok):
    sv = rel.sv
    keys = ("問い", "一位が正しいと言える物", "一位の方が多く聞いた", "同じだけ聞いた", "一位の方が少なく聞いた",
            "一位がこの話でもう出た", "正解がこの話でもう出た")
    lost = [{k: 0 for k in keys} for _ in range(4)]            # 席 3 つ + 全体
    near_w, near_o = [[] for _ in range(4)], [[] for _ in range(4)]
    won = dict(問い=0, 正解の方が多く聞いた=0, 同じだけ聞いた=0, 正解の方が少なく聞いた=0, 正解がこの話でもう出た=0)
    n = 0
    for tris, words in talks:
        said = set()
        for tri in tris:
            for si in range(3):
                cands = sorted({w for w in words if w not in {tri[j] for j in range(3) if j != si}} | {tri[si]}, key=str)
                if len(cands) < 2:
                    continue
                q = tuple(None if j == si else tri[j] for j in range(3))
                n += 1
                pts = rel.ask(si, q, cands)["点"]
                order = sorted((c for c in pts if pts[c] > 0), key=lambda c: (-pts[c], cands.index(c)))
                ans = tri[si]
                if ans not in order[:3]:
                    continue
                na = sv.n_add.get(ans, 0)
                if order[0] == ans:
                    if len(order) > 1:
                        nb = sv.n_add.get(order[1], 0)
                        won["問い"] += 1
                        won["正解の方が多く聞いた" if na > nb else "同じだけ聞いた" if na == nb else "正解の方が少なく聞いた"] += 1
                        won["正解がこの話でもう出た"] += ans in said
                    continue
                top = order[0]
                nt = sv.n_add.get(top, 0)
                c1 = _cos(sv, top, ans)
                others = [v for v in (_cos(sv, c, ans) for c in cands if c not in (ans, top)) if v is not None]
                for d in (si, 3):
                    x = lost[d]
                    x["問い"] += 1
                    x["一位が正しいと言える物"] += top in ok.get((si, q), ())
                    x["一位の方が多く聞いた" if nt > na else "同じだけ聞いた" if nt == na else "一位の方が少なく聞いた"] += 1
                    x["一位がこの話でもう出た"] += top in said
                    x["正解がこの話でもう出た"] += ans in said
                    if c1 is not None:
                        near_w[d].append(c1)
                    if others:
                        near_o[d].append(sum(others) / len(others))
            rel.learn(tri)
            said.update(tri)
        rel.tick()

    def rate(x):
        d = max(1, x["問い"])
        return {k: (v if k == "問い" else round(v / d, 4)) for k, v in x.items()}
    out = dict(全部の問い=n, 正解が一位だった問い=rate(won))
    for d, name in enumerate(SEATS + ("全体",)):
        r = rate(lost[d])
        r["一位と正解の近さの平均"] = round(float(np.mean(near_w[d])), 4) if near_w[d] else None
        r["正解とほかの候補の近さの平均"] = round(float(np.mean(near_o[d])), 4) if near_o[d] else None
        out.setdefault("正解が二〜三位だった問い", {})[name] = r
    return out


def run(attr_of, talks, aoz):
    import i49k_seat_vec as KV
    ok = WR.ok_sets([t for tris, _ in talks for t in tris] + [tuple(t) for t in aoz])
    out = {}
    for start, pre in (("白紙から", ()), ("青空の 3 つ組を先に", aoz)):
        u = KV.Unbind(KV.SeatVec(attr_of, weight="数", bind="位相"), online=True)
        for tri in pre:
            u.sv.hear(tri)
        out[start] = who(u, talks, ok)
        print(start, out[start]["正解が二〜三位だった問い"]["全体"], flush=True)
    return out


def main():
    import i49m_attr_base as AB
    i = sys.argv.index("--content")
    content = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    a, vec_of, talks, aoz, t0 = HR.load()
    full = AB.AttrBase(json.load(open(content, encoding="utf-8")))
    res = dict(会話=len(talks), 青空の組=len(aoz))
    res.update(run(full.of, talks, aoz))
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
