"""I49zn: 属性＋固有の位相の解く形で、「正解が上位に居るのに、属性の似た語に負ける」問いが減ったかを数える。数えだけ。

I49zl と同じ数え（tools/i49y_who_wins_run.py の who）を、3 つの形で並べる:
  属性だけ ／ 属性＋固有の位相（混ぜ 1・1,024 本） ／ 属性＋固有の位相で、読む時の振幅を直す（毎回・席ごと・.01）
近さは、どの形でも、属性の波どうしで測る（固有の位相は物差しに入れない。I49zl と同じ物差し）。
流れ・引数は tools/i49zl_who_wins_after_run.py と同じ（--content 付き）。記録はリポジトリの外。
手で置く物: 「上位」を 3 位までとする事、混ぜ具合、固有の位相の本数、直す速さ、temp .05。
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, HERE)

import i49h_tally_run as HR  # noqa: E402
import i49w_also_right_run as WR  # noqa: E402
import i49y_who_wins_run as Y  # noqa: E402
import i49zl_who_wins_after_run as ZL  # noqa: E402

ID1 = dict(mix=1.0, n_id=1024)
VERS = (("属性だけ", None, None), ("属性＋固有の位相", ID1, None), ("属性＋固有の位相＋振幅", ID1, ZL.AMP))


def run(attr_of, talks, aoz):
    import i49k_seat_vec as KV
    from i49zj_phase_read import PhaseRead
    from i49zm_two_channel import TwoChannel
    ok = WR.ok_sets([t for tris, _ in talks for t in tris] + [tuple(t) for t in aoz])
    old, Y._cos = Y._cos, ZL.attr_cos(attr_of)
    out = {}
    try:
        for start, pre in (("白紙から", ()), ("青空の 3 つ組を先に", aoz)):
            out[start] = {}
            for name, two, read in VERS:
                base_of = attr_of if two is None else TwoChannel(attr_of, **two).of
                sv = KV.SeatVec(base_of, weight="数", bind="位相")
                rel = PhaseRead(sv, **read) if read else KV.Unbind(sv, online=True)
                for tri in pre:
                    sv.hear(tri)
                    if read:
                        rel.hear_seen(tri)
                d = Y.who(rel, talks, ok)
                n = max(1, d["全部の問い"])
                d["正解が二〜三位だった割合"] = round(d["正解が二〜三位だった問い"]["全体"]["問い"] / n, 4)
                d["正解が一位だった割合"] = round(d["正解が一位だった問い"]["問い"] / n, 4)
                out[start][name] = d
                print(start, name, d["正解が二〜三位だった割合"], d["正解が二〜三位だった問い"]["全体"], flush=True)
    finally:
        Y._cos = old
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
