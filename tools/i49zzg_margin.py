"""I49zzg: 1 位と 2 位の点の差ごとに、どちらが当たるかを数える（見るだけの数え。揺らぎを育てる案の初めの一歩、未確定の試し）。

持ち主の言葉:「むしろその揺らぎを大きくする方法を考えてほしい」。
今の数えの揺れ（同じ数えで 1 問だけ動く）は、1 位と 2 位の点がほぼ同じ問いで、小数の下の桁が順位を決めている物。
「同じ」と見なす幅を広げて、その中から別の手掛かり（器の向きなど）で選ぶ形を考える前に、幅をどこまで広げて良いかを数える。
  場面: 話を並べ替えて半分に分け、本体（i49_foundation.body）に前半を聞かせ、後半は問うてから聞かせる（「本体が後半も聞く」と同じ）。
  問いごとに、本体が答えに使った点の 1 位と 2 位を取り、差の割合 d =（1 位 − 2 位）÷ 1 位 を出す。
  答えた段（2語・片方だけ借りる・述語側・もう片方・借りる・解く）と d の区切りごとに、問い・1 位が当たり・2 位が当たり を数える。
  1 位と 2 位の当たりが五分に近い区切りが、「揺らしても損をしない幅」の目安になる。
手で置いた物: d の区切り（見るための区切りで、選び方には使わない）、場面の作り方（半分に分ける、並べ替えの k）。乱数は使わない。
材料の語や例は出さない。出すのは段の名前と集計の数だけ。
引数: --text 材料（カンマ区切り） --wave 語の波 --content 内容語の板 --out 記録.json [--ks 1,2,3]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, HERE)

import i49zp_text_talks_run as ZP  # noqa: E402

BINS = ((0.0, "0（全く同じ）"), (1e-6, "〜0.000001"), (1e-4, "〜0.0001"), (1e-2, "〜0.01"), (0.05, "〜0.05"), (0.1, "〜0.1"),
        (0.2, "〜0.2"), (0.5, "〜0.5"), (float("inf"), "それより上"))


def bin_of(d):
    return next(name for lim, name in BINS if d <= lim)


def margin(points, cands):
    """点（候補 → 点）から、(1 位, 2 位, 差の割合) を返す。順は、点が大きい順、同じ点なら候補の並びで先の物（本体の選び方と同じ）。"""
    order = sorted(cands, key=lambda c: (-float(points.get(c, 0.0)), cands.index(c)))
    s1, s2 = float(points.get(order[0], 0.0)), float(points.get(order[1], 0.0))
    return order[0], order[1], ((s1 - s2) / abs(s1) if s1 != 0 else 0.0)


def count(body, first, second):
    """→ 段 → 区切り → dict(問い, 1位が当たり, 2位が当たり)。答えられなかった問い・候補が 1 つの問いは「数えない」に入れる。"""
    for tris, _ in first:
        for tri in tris:
            body.learn(tri)
        body.tick()
    out, skip = {}, 0
    for tris, words in second:
        for tri in tris:
            for si in range(3):
                cands = sorted({w for w in words if w not in {tri[j] for j in range(3) if j != si}} | {tri[si]}, key=str)
                if len(cands) < 2:
                    continue
                o = body.ask(si, tuple(None if j == si else tri[j] for j in range(3)), cands)
                if o["答え"] is None or not o.get("点"):
                    skip += 1
                    continue
                a, b, d = margin(o["点"], cands)
                c = out.setdefault(o["段"], {}).setdefault(bin_of(d), dict(問い=0, 一位が当たり=0, 二位が当たり=0))
                c["問い"] += 1
                c["一位が当たり"] += a == tri[si]
                c["二位が当たり"] += b == tri[si]
            body.learn(tri)
        body.tick()
    return out, skip


def run(wave_of, attr_of, talks, ks=(1, 2, 3)):
    import i49_foundation as FD
    plain = [(t[0], t[1]) for t in talks]
    res = {}
    for k in ks:
        seq = [plain[i] for i in ZP.order(len(plain), k)]
        by, skip = count(FD.body(wave_of, attr_of), seq[:len(seq) // 2], seq[len(seq) // 2:])
        total = {}
        for lv in by.values():
            for name, c in lv.items():
                t = total.setdefault(name, dict(問い=0, 一位が当たり=0, 二位が当たり=0))
                for key in c:
                    t[key] += c[key]
        order = [name for _, name in BINS]
        res["元の順" if k == 0 else "並べ替え %d" % k] = dict(
            数えない問い=skip, 全部の段={n: total[n] for n in order if n in total},
            段ごと={lv: {n: by[lv][n] for n in order if n in by[lv]} for lv in sorted(by)})
        print(k, res["元の順" if k == 0 else "並べ替え %d" % k]["全部の段"], flush=True)
    return res


def main():
    import i49c_read_words as Rw
    import i49m_attr_base as AB
    ap = argparse.ArgumentParser()
    for k in ("text", "wave", "content", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--ks", default="1,2,3")
    a = ap.parse_args()
    t0 = time.time()
    talks = []
    for f in a.text.split(","):
        talks += ZP.talks_of([l.strip() for l in open(f, encoding="utf-8").read().splitlines()])[0]
    wv = Rw.WordVecs.load(a.wave)

    def wave_of(w):
        v = wv.of(*w)
        return wv.of(w[0], "PROPN") if v is None and w[1] == "NOUN" else v

    full = AB.AttrBase(json.load(open(a.content, encoding="utf-8")))
    res = run(wave_of, full.of, talks, ks=tuple(int(x) for x in a.ks.split(",")))
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
