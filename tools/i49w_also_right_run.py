"""I49w: 「正しい物」と「正しいと言える物」を両方数える（持ち主の問い「正しい物と正しいと言えるもの、両方をカウントする方法」から）。数えだけ。

同じ問い（同じ席、同じ見せた 2 語）で実際に使われた語を全部集めて、正解の集まりにする。答えを 3 つに分ける。
  正しい物            この文の正解そのもの（今までの数え方）
  正しいと言える物      この文の正解ではないが、別の所で同じ 2 語と一緒にその席で使われていた語
  どちらでもない
集まりは 2 通り: 話だけ（自作の話 600 本の全部） ／ 話と青空（青空の 3 つ組も足す）。
採点のための集まりなので、後の話の分も入れる（覚える側には渡さない）。まだ誰も書いていない正しい組は拾えないので、下限の数になる。
流れは tools/i49h_tally_run.py と同じ。引数は tools/i49m_attr_run.py と同じ（--content 付き）。記録はリポジトリの外。
手で置く物: 無し（「同じ問い」を席と見せた 2 語で決める事だけ）。
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


def ok_sets(tris):
    ok = {}
    for tri in tris:
        for si in range(3):
            ok.setdefault((si, tuple(None if j == si else tri[j] for j in range(3))), set()).add(tri[si])
    return ok


def grade(rel, talks, ok_talk, ok_wide):
    n = 0
    hit = {"正しい物": 0, "言える・話だけ": 0, "言える・話と青空": 0}
    seat = [{"問い": 0, "正しい物": 0, "言える・話だけ": 0, "言える・話と青空": 0} for _ in SEATS]
    size = dict(話だけ=0, 話と青空=0)
    many = dict(話だけ=0, 話と青空=0)
    for tris, words in talks:
        for tri in tris:
            for si in range(3):
                cands = sorted({w for w in words if w not in {tri[j] for j in range(3) if j != si}} | {tri[si]}, key=str)
                if len(cands) < 2:
                    continue
                q = tuple(None if j == si else tri[j] for j in range(3))
                ans = rel.ask(si, q, cands)["答え"]
                n += 1
                seat[si]["問い"] += 1
                cs = set(cands)
                for name, ok in (("話だけ", ok_talk), ("話と青空", ok_wide)):
                    s = ok.get((si, q), set()) & cs
                    size[name] += len(s)
                    many[name] += len(s) > 1
                    if ans is not None and ans != tri[si] and ans in s:
                        hit["言える・" + name] += 1
                        seat[si]["言える・" + name] += 1
                if ans == tri[si]:
                    hit["正しい物"] += 1
                    seat[si]["正しい物"] += 1
            rel.learn(tri)
        rel.tick()
    d = max(1, n)
    out = dict(問い=n, 正しい物=round(hit["正しい物"] / d, 4))
    for name in ("話だけ", "話と青空"):
        k = "言える・" + name
        out[name] = dict(正しいと言える物=round(hit[k] / d, 4), 合わせて=round((hit["正しい物"] + hit[k]) / d, 4),
                         候補の中の正解の数の平均=round(size[name] / d, 2), 正解が二つ以上有る問いの割合=round(many[name] / d, 4),
                         席ごと={SEATS[i]: dict(正しい物=round(x["正しい物"] / max(1, x["問い"]), 4),
                                             合わせて=round((x["正しい物"] + x[k]) / max(1, x["問い"]), 4))
                               for i, x in enumerate(seat)})
    return out


def run(wave_of, attr_of, talks, aoz):
    import i49e_relation as RL
    import i49k_seat_vec as KV
    ok_talk = ok_sets(t for tris, _ in talks for t in tris)
    ok_wide = ok_sets([t for tris, _ in talks for t in tris] + [tuple(t) for t in aoz])
    out = {}
    for start, pre in (("白紙から", ()), ("青空の 3 つ組を先に", aoz)):
        r = RL.Relation(wave_of, half=HR.HALF)
        u = KV.Unbind(KV.SeatVec(attr_of, weight="数", bind="位相"), online=True)
        for tri in pre:
            r.learn(tri)
            u.sv.hear(tri)
        out[start] = {"関係の表": grade(r, talks, ok_talk, ok_wide), "解く・属性（席あり）": grade(u, talks, ok_talk, ok_wide)}
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
