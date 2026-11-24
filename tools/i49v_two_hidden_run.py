"""I49v: 2 つの席を同時に隠す問いで、「1 位を出さずに候補を掛け算にする」案を測る（持ち主の案、未確定）。数えだけ。

持ち主の言葉:「１位を出さずに候補を文章全体で掛け算にするとより良い結果が出そうな気がする。でも文字数が増えると計算が増える。」
流れは tools/i49h_tally_run.py と同じ（答える前に問う）。ただし 3 つ組の 1 語だけを見せ、残りの 2 席を当てさせる。
候補は 2 席とも同じ: その話の内容語から見せた語を除き、正解の 2 語を足した物。
選び方:
  別々             席ごとに、見せた 1 語だけから 1 位を選ぶ
  順に・前から      前の席の 1 位を決めてから、それも見せた事にして後ろの席を選ぶ
  順に・後ろから    後ろの席を先に決める
  掛け算・上位 k    席ごとの上位 k 個を残す。組（前の語, 後ろの語）ごとに「相手を見せた時の自分の点」を両側で出して掛ける。一番大きい組
  足し算・上位 k    上と同じ点を、掛けずに足す（掛け算である事が効くのかを見る比べ役）
数える物: 2 席とも当たった割合、席ごとの当たり、見せた席ごと、上位 k に 2 席とも正解が入っている割合（掛け算の上限）、問う回数。
引数は tools/i49m_attr_run.py と同じ（--content 付き）。記録はリポジトリの外。
手で置く物: 残す数 k（3 と 5）、負の点を 0 にする事、同じ語を 2 席に置かない事。
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
KS = (3, 5)


def _top(o, cands, k=None):
    pts = o["点"]
    order = sorted((c for c in pts if pts[c] > 0), key=lambda c: (-pts[c], cands.index(c)))
    return order if k is None else order[:k]


def _fill(q, si, w):
    t = list(q)
    t[si] = w
    return tuple(t)


def two(rel, talks):
    names = ["別々", "順に・前から", "順に・後ろから"] + ["%s・上位 %d" % (m, k) for k in KS for m in ("掛け算", "足し算")]
    both = {n: 0 for n in names}
    one = {n: 0 for n in names}
    by_shown = {n: [0, 0, 0] for n in names}
    n_shown = [0, 0, 0]
    room = {k: 0 for k in KS}
    n, asks = 0, 0
    for tris, words in talks:
        for tri in tris:
            for sk in range(3):
                a, b = [j for j in range(3) if j != sk]
                cands = sorted({w for w in words if w != tri[sk]} | {tri[a], tri[b]}, key=str)
                if len(cands) < 3:
                    continue
                q = tuple(tri[j] if j == sk else None for j in range(3))
                n += 1
                n_shown[sk] += 1
                oa, ob = rel.ask(a, q, cands), rel.ask(b, q, cands)
                asks += 2
                ta, tb = _top(oa, cands), _top(ob, cands)
                ans = {"別々": (ta[0] if ta else None, tb[0] if tb else None)}
                # 順に
                x = ans["別々"][0]
                y = None
                if x is not None:
                    t = _top(rel.ask(b, _fill(q, a, x), [c for c in cands if c != x]), cands)
                    asks += 1
                    y = t[0] if t else None
                ans["順に・前から"] = (x, y if y is not None else next((c for c in tb if c != x), None))
                y = ans["別々"][1]
                x = None
                if y is not None:
                    t = _top(rel.ask(a, _fill(q, b, y), [c for c in cands if c != y]), cands)
                    asks += 1
                    x = t[0] if t else None
                ans["順に・後ろから"] = (x if x is not None else next((c for c in ta if c != y), None), y)
                # 掛け算・足し算
                kmax = max(KS)
                A, B = ta[:kmax], tb[:kmax]
                sa = {cb: rel.ask(a, _fill(q, b, cb), [c for c in A if c != cb])["点"] for cb in B}
                sb = {ca: rel.ask(b, _fill(q, a, ca), [c for c in B if c != ca])["点"] for ca in A}
                asks += len(A) + len(B)
                for k in KS:
                    room[k] += tri[a] in ta[:k] and tri[b] in tb[:k]
                    for m in ("掛け算", "足し算"):
                        best, bv = None, 0.0
                        for i, ca in enumerate(ta[:k]):
                            for j, cb in enumerate(tb[:k]):
                                if ca == cb:
                                    continue
                                u, v = max(0.0, sa[cb].get(ca, 0.0)), max(0.0, sb[ca].get(cb, 0.0))
                                s = u * v if m == "掛け算" else u + v
                                if s > bv:
                                    best, bv = (ca, cb), s
                        ans["%s・上位 %d" % (m, k)] = best if best is not None else ans["別々"]
                for name, (x, y) in ans.items():
                    ok = (x == tri[a]) + (y == tri[b])
                    one[name] += ok
                    both[name] += ok == 2
                    by_shown[name][sk] += ok == 2
            rel.learn(tri)
        rel.tick()
    d = max(1, n)
    return dict(問い=n, 問う回数=asks,
                上位に二席とも正解が入る割合={"上位 %d" % k: round(v / d, 4) for k, v in room.items()},
                版={name: dict(二席とも当たり=round(both[name] / d, 4), 席ごとの当たり=round(one[name] / (2 * d), 4),
                              見せた席ごと={SEATS[i]: round(by_shown[name][i] / max(1, n_shown[i]), 4) for i in range(3)})
                   for name in names})


def run(wave_of, attr_of, talks, aoz):
    import i49e_relation as RL
    import i49k_seat_vec as KV

    def safe(f):
        return lambda w: None if w is None else f(w)
    wave_of, attr_of = safe(wave_of), safe(attr_of)
    out = {}
    for start, pre in (("白紙から", ()), ("青空の 3 つ組を先に", aoz)):
        r = RL.Relation(wave_of, half=HR.HALF)
        u = KV.Unbind(KV.SeatVec(attr_of, weight="数", bind="位相"), online=True)
        for tri in pre:
            r.learn(tri)
            u.sv.hear(tri)
        out[start] = {}
        for form, rel in (("関係の表", r), ("解く・属性（席あり）", u)):
            t0 = time.time()
            out[start][form] = two(rel, talks)
            out[start][form]["秒"] = round(time.time() - t0)
            print(start, form, out[start][form]["版"]["別々"]["二席とも当たり"], out[start][form]["秒"], flush=True)
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
