"""I49zd: 仮の板に、属性ではなく「語そのものの波」を持たせる（流れを添わせる形の続き。未確定の試し）。数えだけ。

I49u で一番効いた選び直しは「この話でもう出た語を選ぶ」という決まりだった。I49y では、取り違えの相手は属性が似た語だった。
そこで、板に語の波（語ごとに違う向き）を足して薄れさせ、決まりを書かずに同じ働きが出るかを見る。
並べる物（解く・属性（席あり）の答えを、板に一番近い候補に替える）:
  板無し ／ 出た語・上位 3（決まりで書いた比べ役）
  選び方: 九割以内 ／ 上位 3   板: 1 枚 ／ 席ごと   薄れ: 半分まで 1・2・4 組（話をまたぐ） ／ 薄れない（話ごとに空）
採点は、正しい物と、合わせて（I49w の集まり、話と青空）。板無しから 当たりに変わった数・外れに変わった数 も出す。
流れは tools/i49h_tally_run.py と同じ。引数は tools/i49m_attr_run.py と同じ（--content 付き）。記録はリポジトリの外。
手で置く物: 薄れる速さ、九割、上位 3。
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
import i49za_fold_run as ZR  # noqa: E402

VERS = [("板無し", "属性", dict(pick="そのまま")), ("出た語・上位 3", "属性", dict(pick="出た語", top=3))]
for pick, pname in (("九割以内", "九割以内"), ("上位", "上位 3")):
    for seat, sname in ((False, "1 枚"), (True, "席ごと")):
        for hname, kw in [("半分まで %d 組" % h, dict(half=h, carry=True)) for h in (1, 2, 4)] + [("薄れない", dict())]:
            VERS.append(("語の波の板｜%s｜%s｜%s" % (pname, sname, hname), "語の波", dict(pick=pick, top=3, by_seat=seat, **kw)))


def run(wave_of, attr_of, talks, aoz):
    import i49k_seat_vec as KV
    from i49u_flow_pick import FlowPick
    ok = WR.ok_sets([t for tris, _ in talks for t in tris] + [tuple(t) for t in aoz])
    out = {}
    for start, pre in (("白紙から", ()), ("青空の 3 つ組を先に", aoz)):
        out[start], base = {}, None
        for name, board, kw in VERS:
            u = KV.Unbind(KV.SeatVec(attr_of, weight="数", bind="位相"), online=True)
            for tri in pre:
                u.sv.hear(tri)
            fp = FlowPick(u, wave_of if board == "語の波" else attr_of, **kw)
            recs = ZR.stream(fp, talks, ok)
            out[start][name] = ZR.summary(recs, fp.picked, base)
            base = recs if base is None else base
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
    res.update(run(vec_of, full.of, talks, aoz))
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
