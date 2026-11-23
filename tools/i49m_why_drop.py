"""I49m: 属性で解く形が「聞くほど下がる」のは、どの問いで外れるようになるからか。数えだけの下調べ（理由の見立てを立てる前に、中身を見る）。

材料・流れ・引数は tools/i49m_attr_run.py と同じ（--content 付き）。解く・属性（聞きながら育つ形、長さで割る）を流し、問いごとに次を記録して、
100 本ごと・席ごと・始め方ごとにまとめる:
  候補の数、そのうち作った向きを持つ候補の数（競う相手の数）、正解が作った向きを持つか、正解の順位、
  正解がそれまでに聞かれた回数（足した回数）、外れた時に 1 位になった語が聞かれた回数、
  外れた時の 1 位の語が、見せた 2 語のどちらかと同じ席の組で前に座った事があるか（まったく座っていないのに勝ったのか）。
まとめる物:
  区切りごと: 問いの数、候補の数の平均、競う相手の数の平均、正解が向きを持つ割合、1 位の割合、正解が向きを持つ問いだけの 1 位の割合、
              正解の順位 ÷ 競う相手の数 の平均（相手の数が増えた分を除いた、順位の良さ。0 に近いほど良い）、
              外れた時: 1 位の語の方が正解より多く聞かれていた割合、1 位の語が見せた語と座った事が無かった割合
  正解が聞かれた回数ごと（0、1〜3、4〜15、16〜63、64 以上）: 問いの数と 1 位の割合
記録はリポジトリの外。手で置く物: 100 本ごとの区切り、回数の区切り（0・1〜3・4〜15・16〜63・64 以上）。
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

BINS = ((0, 0, "0"), (1, 3, "1〜3"), (4, 15, "4〜15"), (16, 63, "16〜63"), (64, 10 ** 9, "64 以上"))


def trace(base_of, talks, pre):
    import i49k_seat_vec as KV
    sv = KV.SeatVec(base_of, weight="数", bind="位相")
    u = KV.Unbind(sv, online=True)
    for tri in pre:
        u.learn(tri)
    rows = []
    for ti, (tris, words) in enumerate(talks):
        for tri in tris:
            for si in range(3):
                shown = [j for j in range(3) if j != si]
                cands = sorted({w for w in words if w not in {tri[j] for j in shown}} | {tri[si]}, key=str)
                if len(cands) < 2:
                    continue
                o = u.ask(si, tuple(None if j == si else tri[j] for j in range(3)), cands)
                s = o["点"]
                ans = tri[si]
                rank = None if ans not in s else 1 + sum(1 for c in s if s[c] > s[ans])
                top = o["答え"]
                sat = None
                if top is not None and top != ans:
                    ctx = sv.ctx.get(top, {})
                    sat = any((si, j, tri[j]) in ctx for j in shown)
                rows.append(dict(t=ti, si=si, n=len(cands), nv=len(s), has=ans in s, rank=rank, hit=top == ans,
                                 heard=sv.n_add.get(ans, 0), top_heard=None if top is None else sv.n_add.get(top, 0), sat=sat))
            u.learn(tri)
    return rows


def _mean(xs):
    xs = list(xs)
    return round(sum(xs) / len(xs), 4) if xs else None


def block(rows):
    wrong = [r for r in rows if not r["hit"] and r["top_heard"] is not None and r["has"]]
    return dict(問い=len(rows), 候補の数=_mean(r["n"] for r in rows), 競う相手の数=_mean(r["nv"] for r in rows),
                正解が向きを持つ割合=_mean(r["has"] for r in rows), 一位=_mean(r["hit"] for r in rows),
                正解が向きを持つ問いの一位=_mean(r["hit"] for r in rows if r["has"]),
                順位を相手の数で割った平均=_mean((r["rank"] - 1) / max(1, r["nv"] - 1) for r in rows if r["has"] and r["nv"] > 1),
                外れた時_一位の方が多く聞かれていた割合=_mean(r["top_heard"] > r["heard"] for r in wrong),
                外れた時_一位が見せた語と座っていなかった割合=_mean(not r["sat"] for r in wrong if r["sat"] is not None))


def summarize(rows, n_talks, every=100):
    from i49e_relation import SLOTS
    out = {"区切りごと": {}, "席ごとの区切り": {s: {} for s in SLOTS}, "正解が聞かれた回数ごと": {}}
    for end in range(every, n_talks + 1, every):
        cur = [r for r in rows if end - every <= r["t"] < end]
        out["区切りごと"][str(end)] = block(cur)
        for si, s in enumerate(SLOTS):
            out["席ごとの区切り"][s][str(end)] = block([r for r in cur if r["si"] == si])
    for lo, hi, name in BINS:
        x = [r for r in rows if lo <= r["heard"] <= hi]
        out["正解が聞かれた回数ごと"][name] = dict(問い=len(x), 一位=_mean(r["hit"] for r in x), 順位を相手の数で割った平均=_mean(
            (r["rank"] - 1) / max(1, r["nv"] - 1) for r in x if r["has"] and r["nv"] > 1))
    out["全体"] = block(rows)
    return out


def main():
    import i49m_attr_base as AB
    i = sys.argv.index("--content")
    content = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    a, _, talks, aoz, t0 = HR.load()
    full = AB.AttrBase(json.load(open(content, encoding="utf-8")))
    res = dict(会話=len(talks), 青空の組=len(aoz))
    for start, pre in (("白紙から", []), ("青空を先に", aoz)):
        res[start] = summarize(trace(full.of, talks, pre), len(talks))
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
