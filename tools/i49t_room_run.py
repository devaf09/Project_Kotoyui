"""I49t: 共振の「揺らぎの余地」を数える（持ち主の言葉「共振だと揺らぎが少ない気がする」から）。数えだけ。

今の答え方は、点がいちばん高い候補を 1 つ選ぶ。ここでは選び方は変えずに、次を数える。
  正解が何位までに入るか（1・2・3・5 位まで）
  1 位と 2 位の点の近さ（(1 位 − 2 位) ÷ 1 位）の分かれ方、1 位の 9 割以上の点を持つ候補の数
  同じ問い（同じ席、同じ見せた 2 語）がまた出た時に、正解が前と違った割合（言葉の側に元から有る揺らぎ）
引数は tools/i49m_attr_run.py と同じ（--content 付き）。始め方: 白紙から ／ 青空の 3 つ組を先に。形: 関係の表 ／ 解く・属性（席あり）。
手で置く物: 「近い」の区切り（1 位の 9 割）、近さを分ける 10 個の区切り。記録はリポジトリの外。
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


def room(rel, talks):
    n = 0
    at = {1: 0, 2: 0, 3: 0, 5: 0}
    gaps = [0] * 10
    near, answered, n_cands = 0, 0, 0
    seen, again, changed = {}, 0, 0
    by_seat = [dict(問い=0, 一位=0, 三位まで=0) for _ in SEATS]
    for tris, words in talks:
        for tri in tris:
            for si in range(3):
                cands = sorted({w for w in words if w not in {tri[j] for j in range(3) if j != si}} | {tri[si]}, key=str)
                if len(cands) < 2:
                    continue
                q = tuple(None if j == si else tri[j] for j in range(3))
                key = (si, q)
                if key in seen:
                    again += 1
                    changed += seen[key] != tri[si]
                seen[key] = tri[si]
                o = rel.ask(si, q, cands)
                n += 1
                n_cands += len(cands)
                by_seat[si]["問い"] += 1
                pts = o["点"]
                if o["答え"] is None or not pts:
                    continue
                answered += 1
                order = sorted(pts, key=lambda c: (-pts[c], cands.index(c)))
                rank = order.index(tri[si]) + 1 if tri[si] in pts else None
                for k in at:
                    at[k] += rank is not None and rank <= k
                by_seat[si]["一位"] += rank == 1
                by_seat[si]["三位まで"] += rank is not None and rank <= 3
                s1 = pts[order[0]]
                s2 = pts[order[1]] if len(order) > 1 else 0.0
                g = (s1 - s2) / s1 if s1 > 0 else 1.0
                gaps[min(9, max(0, int(g * 10)))] += 1
                near += sum(1 for c in order if pts[c] >= 0.9 * s1)
            rel.learn(tri)
        rel.tick()
    return dict(問い=n, 答えた問い=answered, 候補の数の平均=round(n_cands / max(1, n), 1),
                正解が入る割合={"%d 位まで" % k: round(v / n, 4) for k, v in at.items()},
                一位と二位の近さの分かれ方=[round(v / max(1, answered), 4) for v in gaps],
                一位の九割以上の候補の数の平均=round(near / max(1, answered), 2),
                同じ問いがまた出た数=again, 正解が前と違った割合=round(changed / max(1, again), 4),
                席ごと={SEATS[i]: dict(一位=round(d["一位"] / max(1, d["問い"]), 4), 三位まで=round(d["三位まで"] / max(1, d["問い"]), 4))
                      for i, d in enumerate(by_seat)})


def run(wave_of, attr_of, talks, aoz):
    import i49e_relation as RL
    import i49k_seat_vec as KV
    out = {}
    for start, pre in (("白紙から", ()), ("青空の 3 つ組を先に", aoz)):
        r = RL.Relation(wave_of, half=HR.HALF)
        sv = KV.SeatVec(attr_of, weight="数", bind="位相")
        for tri in pre:
            r.learn(tri)
            sv.hear(tri)
        out[start] = {"関係の表": room(r, talks), "解く・属性（席あり）": room(KV.Unbind(sv, online=True), talks)}
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
