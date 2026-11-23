"""I49s: 関係の表と、表を持たない解く形の差が、どの問いで出ているかを調べる。数えだけ。

関係の表は、段を上から順に使う（見せた 2 語とまったく同じ組の数が有ればそれだけを使い、無ければ下の段へ）。解く形は、全部を 1 本の波に足して比べる。
同じ流れで両方に答えさせ、表が使った段ごと・席ごとに、問いの数、表の当たり、解く形の当たり、片方だけ当たった数を出す。
引数は tools/i49m_attr_run.py と同じ（--content 付き）。始め方: 白紙から ／ 青空の 3 つ組を先に。
解く形: 解く・属性（席あり）と、解く・属性・席の印。
手で置く物: 無し（比べる 2 つの形の物だけ）。記録はリポジトリの外。
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


def split(base, other):
    """base = 表の記録、other = 解く形の記録（同じ問いの並び）。表の段ごと・席ごとに数える。"""
    assert len(base) == len(other)
    out = {}
    for b, o in zip(base, other):
        assert b[:2] == o[:2]
        for key in (b[3], b[3] + "・" + SEATS[b[1]], "全部", "全部・" + SEATS[b[1]]):
            d = out.setdefault(key, dict(問い=0, 表=0, 解く=0, 表だけ当たり=0, 解くだけ当たり=0))
            d["問い"] += 1
            d["表"] += bool(b[2])
            d["解く"] += bool(o[2])
            d["表だけ当たり"] += bool(b[2] and not o[2])
            d["解くだけ当たり"] += bool(o[2] and not b[2])
    for d in out.values():
        d["表の当たり"] = round(d["表"] / d["問い"], 4)
        d["解くの当たり"] = round(d["解く"] / d["問い"], 4)
    return out


def run(wave_of, attr_of, talks, aoz):
    import i49e_relation as RL
    import i49k_seat_vec as KV
    out = {}
    for start, pre in (("白紙から", ()), ("青空の 3 つ組を先に", aoz)):
        r = RL.Relation(wave_of, half=HR.HALF)
        for tri in pre:
            r.learn(tri)
        base = HR.stream_rec(r, talks)
        res = {}
        for name, mark in (("解く・属性（席あり）", False), ("解く・属性・席の印", True)):
            sv = KV.SeatVec(attr_of, weight="数", bind="位相", mark=mark)
            for tri in pre:
                sv.hear(tri)
            res[name] = split(base, HR.stream_rec(KV.Unbind(sv, online=True), talks))
        out[start] = res
        print(start, flush=True)
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
