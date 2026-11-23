"""I49u: 並んだ上位から、話の流れで選び直す試し（intervention/i49u_flow_pick.FlowPick）。数えだけ。

流れは tools/i49h_tally_run.py と同じ（答える前に問う）。引数は tools/i49m_attr_run.py と同じ（--content 付き）。
始め方: 白紙から ／ 青空の 3 つ組を先に。元の形: 関係の表 ／ 解く・属性（席あり）。流れの波の元の向きは属性の割合。
返す物: 版ごとの一位の割合、席ごと、元の形から 当たりに変わった数・外れに変わった数、選び直しで答えが変わった数。
手で置く物: FlowPick の物（上位の数 3 と 5、九割）。記録はリポジトリの外。
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

SEATS = ("だれが", "何を", "どうした")
PICKS = (("そのまま", dict(pick="そのまま")),
         ("流れ・上位 3", dict(pick="上位", top=3)),
         ("流れ・上位 5", dict(pick="上位", top=5)),
         ("流れ・九割以内", dict(pick="九割以内")),
         ("流れ・順位を足す", dict(pick="順位を足す")),
         ("流れ（自分の分を引く）・上位 3", dict(pick="上位", top=3, minus_self=True)),
         ("流れ（自分の分を引く）・順位を足す", dict(pick="順位を足す", minus_self=True)),
         ("出た語・上位 3", dict(pick="出た語", top=3)))


def one(fp, talks, base=None):
    recs = HR.stream_rec(fp, talks)
    n = max(1, len(recs))
    d = dict(一位=round(sum(r[2] for r in recs) / n, 4), 答えが変わった数=fp.picked,
             席ごと={s: round(sum(r[2] for r in recs if r[1] == i) / max(1, sum(1 for r in recs if r[1] == i)), 4)
                   for i, s in enumerate(SEATS)})
    if base is not None:
        d["当たりに変わった数"] = sum(1 for b, r in zip(base, recs) if not b[2] and r[2])
        d["外れに変わった数"] = sum(1 for b, r in zip(base, recs) if b[2] and not r[2])
    return d, recs


def run(wave_of, attr_of, talks, aoz):
    import i49e_relation as RL
    import i49k_seat_vec as KV
    from i49u_flow_pick import FlowPick

    def table():
        return RL.Relation(wave_of, half=HR.HALF)

    def solve():
        return KV.Unbind(KV.SeatVec(attr_of, weight="数", bind="位相"), online=True)
    out = {}
    for start, pre in (("白紙から", ()), ("青空の 3 つ組を先に", aoz)):
        out[start] = {}
        for form, mk in (("関係の表", table), ("解く・属性（席あり）", solve)):
            res, base = {}, None
            for name, kw in PICKS:
                rel = mk()
                for tri in pre:
                    rel.learn(tri)
                res[name], recs = one(FlowPick(rel, attr_of, **kw), talks, base)
                base = recs if base is None else base
                print(start, form, name, res[name]["一位"], flush=True)
            out[start][form] = res
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
