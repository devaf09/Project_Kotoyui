"""I49z: 仮の板（前の文の属性だけを緩く持つ板）で、並んだ上位から選び直す（持ち主の案、未確定）。数えだけ。

持ち主の言葉:「語はなく、前の文章の属性だけを緩く持ってて、古い属性は消え、近い属性と重要な属性と何度も出る属性だけど保持する板を前に置く」
板 = 前に聞いた 3 つ組の語の属性の向きを足した物（何度も出る属性ほど強く残る）。3 つ組を聞くたびに薄れる（古い属性は消える）。
「重要な属性」は決め方が手で置く値になるので、ここでは入れていない。
並べる物（どれも 解く・属性（席あり）の答えを、一位の九割以内の候補の中か、上位 3 の中で、板に一番近い語に替える）:
  板無し
  薄れない・話ごとに空（I49u の流れの波と同じ。比べ役）
  半分になるまで 1 / 2 / 4 / 8 組（話をまたいで持つ）
  席ごとの板（こちらで足した形: その席に前に来た語の属性だけを持つ板を、席ごとに 3 枚）で、同じ薄れ方
採点は、正しい物と、正しいと言える物を合わせた物（I49w の集まり、話と青空）。席ごとにも出す。
流れは tools/i49h_tally_run.py と同じ。引数は tools/i49m_attr_run.py と同じ（--content 付き）。記録はリポジトリの外。
手で置く物: 薄れる速さ（1・2・4・8 組を並べて見る）、九割、上位 3。
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
BOARDS = [("薄れない・話ごとに空", dict())] + [("半分まで %d 組" % h, dict(half=h, carry=True)) for h in (1, 2, 4, 8)]
VERS = [("板無し", dict(pick="そのまま"))]
for pick, pname in (("九割以内", "九割以内"), ("上位", "上位 3")):
    for seat, sname in ((False, "1 枚の板"), (True, "席ごとの板")):
        for bname, kw in BOARDS:
            VERS.append(("%s｜%s｜%s" % (pname, sname, bname), dict(pick=pick, top=3, by_seat=seat, **kw)))


def stream(fp, talks, ok):
    n, right, also = 0, 0, 0
    seat = [[0, 0, 0] for _ in SEATS]
    for tris, words in talks:
        for tri in tris:
            for si in range(3):
                cands = sorted({w for w in words if w not in {tri[j] for j in range(3) if j != si}} | {tri[si]}, key=str)
                if len(cands) < 2:
                    continue
                q = tuple(None if j == si else tri[j] for j in range(3))
                a = fp.ask(si, q, cands)["答え"]
                r = a == tri[si]
                o = a is not None and (r or a in ok.get((si, q), ()))
                n += 1
                right += r
                also += o
                seat[si][0] += 1
                seat[si][1] += r
                seat[si][2] += o
            fp.learn(tri)
        fp.tick()
    d = max(1, n)
    return dict(正しい物=round(right / d, 4), 合わせて=round(also / d, 4), 答えが変わった数=fp.picked,
                席ごと={SEATS[i]: dict(正しい物=round(x[1] / max(1, x[0]), 4), 合わせて=round(x[2] / max(1, x[0]), 4))
                      for i, x in enumerate(seat)})


def run(attr_of, talks, aoz):
    import i49k_seat_vec as KV
    from i49u_flow_pick import FlowPick
    ok = WR.ok_sets([t for tris, _ in talks for t in tris] + [tuple(t) for t in aoz])
    out = {}
    for start, pre in (("白紙から", ()), ("青空の 3 つ組を先に", aoz)):
        out[start] = {}
        for name, kw in VERS:
            u = KV.Unbind(KV.SeatVec(attr_of, weight="数", bind="位相"), online=True)
            for tri in pre:
                u.sv.hear(tri)
            out[start][name] = stream(FlowPick(u, attr_of, **kw), talks, ok)
            print(start, name, out[start][name]["正しい物"], out[start][name]["合わせて"], flush=True)
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
