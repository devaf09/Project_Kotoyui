"""I49i: 向きの無い語を座り方の近さで借りる形（intervention/i49i_seat.SeatRelation）を、文の 3 つ組で元の表と比べる。数えだけ。

材料・流れ・数え方は tools/i49h_tally_run.py と同じ（会話を 1 本ずつ、答える前に問う）。
並べる物: Relation（元）／ SeatRelation 無い語 ／ SeatRelation 全部。始め方: 白紙から ／ 青空の 3 つ組を先に learn してから。
足して数える物: 問いのうち、見せた 2 語のどちらかに向きが無い問いの数と、その中だけの 1 位の割合（元と比べる）。座り方の近さを使った回数。
記録はリポジトリの外。手で置く物: i49h_tally_run と同じ（100 本ごとの区切り、half 1,024、揺れの目安の式）。
python tools/past/i49i_seat_run.py （引数は tools/i49h_tally_run.py と同じ）
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


def makers(vec_of):
    import i49e_relation as RL
    import i49i_seat as ST
    return (("Relation", lambda: RL.Relation(vec_of, half=HR.HALF)),
            ("無い語", lambda: ST.SeatRelation(vec_of, half=HR.HALF, where="無い語")),
            ("全部", lambda: ST.SeatRelation(vec_of, half=HR.HALF, where="全部")))


def no_vec_flags(vec_of, talks):
    """stream_rec と同じ並びで、見せた 2 語のどちらかに向きが無い問いかどうか。"""
    out = []
    for tris, words in talks:
        for tri in tris:
            for si in range(3):
                shown = {tri[j] for j in range(3) if j != si}
                if len({w for w in words if w not in shown} | {tri[si]}) < 2:
                    continue
                out.append(any(vec_of(w) is None for w in shown))
    return out


def run(vec_of, talks, pre):
    """HR.run_all に、向きの無い語を含む問いだけの 1 位の割合を足す。"""
    import i49e_relation as RL
    flags = no_vec_flags(vec_of, talks)
    res = {"向きの無い語を含む問い": dict(数=sum(flags), 割合=round(sum(flags) / max(1, len(flags)), 4))}
    for name, make in makers(vec_of):
        rel = make()
        for tri in pre:
            rel.learn(tri)
        t1 = time.time()
        recs = HR.stream_rec(rel, talks)
        assert len(recs) == len(flags)
        r = HR.summary(recs, len(talks))
        r["秒"] = round(time.time() - t1, 1)
        sub = [x for x, f in zip(recs, flags) if f]
        rest = [x for x, f in zip(recs, flags) if not f]
        r["向きの無い語を含む問いの一位"] = round(sum(x[2] for x in sub) / max(1, len(sub)), 4)
        r["ほかの問いの一位"] = round(sum(x[2] for x in rest) / max(1, len(rest)), 4)
        if isinstance(rel, RL.Relation) and hasattr(rel, "used"):
            r["座り方の近さを使った回数"] = rel.used
        if name == "Relation":
            base = recs
        else:
            r["Relation との違い"] = HR.compare(base, recs)
        res[name] = r
    return res


def main():
    a, vec_of, talks, aoz, t0 = HR.load()
    res = dict(会話=len(talks), 青空の組=len(aoz))
    print("用意", round(time.time() - t0), flush=True)
    for name, pre in (("白紙から", []), ("青空を先に", aoz)):
        res[name] = run(vec_of, talks, pre)
        print(name, round(time.time() - t0), flush=True)
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
