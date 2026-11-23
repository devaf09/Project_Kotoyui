"""I49r: 席の印（その席によく座る語、を波に足しておく）を、解く・属性（席あり）に足して比べる。数えだけ。

関係の表と解く形の差は、席ごとに見ると「だれが」で大きい（.573 対 .505）。表は「その席によく座る語」を数えで持っている。
解く形にも、席ごとに決まった 1 本の波（席の印）を、その席で聞くたびに語の波へ足して持たせる。表は持たない。足すだけ。
引数は tools/i49m_attr_run.py と同じ（--content 付き）。始め方: 白紙から ／ 青空の 3 つ組を先に。
並べる物: 関係の表 ／ 解く・属性（席あり）／ 解く・属性・席の印 ／ 解く・語の波（席あり）／ 解く・語の波・席の印。
手で置く物: 席の印の波の作り方（席の番号と周波数の番号から決めた位相）、印を足す重さ 1、探す時に印を足す重さ 1。
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
import i49j_shift_vec_run as JR  # noqa: E402


def run(wave_of, attr_of, talks, aoz):
    import i49e_relation as RL
    import i49k_seat_vec as KV
    out = {}
    for start, pre in (("白紙から", ()), ("青空の 3 つ組を先に", aoz)):
        def table():
            r = RL.Relation(wave_of, half=HR.HALF)
            for tri in pre:
                r.learn(tri)
            return r

        def solve(base, mark):
            def make():
                sv = KV.SeatVec(base, weight="数", bind="位相", mark=mark)
                for tri in pre:
                    sv.hear(tri)
                return KV.Unbind(sv, online=True)
            return make
        mk = (("関係の表", table), ("解く・属性（席あり）", solve(attr_of, False)), ("解く・属性・席の印", solve(attr_of, True)),
              ("解く・語の波（席あり）", solve(wave_of, False)), ("解く・語の波・席の印", solve(wave_of, True)))
        out[start] = JR.run(attr_of, talks, [], mk)
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
