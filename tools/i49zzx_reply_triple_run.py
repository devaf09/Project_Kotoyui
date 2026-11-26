"""I49zzx: 前の 3 つ組から、次の 3 つ組を「1 つの返事」として組む（持ち主の輪の ②→③ を小さくつなぐ。未確定の試し）。

I49zzw（74 節）は席ごとに当てた。ここでは 3 つの席を合わせて 1 つの 3 つ組にし、組んだ物を丸ごと数える。
材料と場面は I49zzw と同じ（「A が B を C する」の文が並んだテキスト。話を並べ替えて半分に分け、前半を聞かせ、後半は問うてから聞かせる）。
組み方（返事の決め方。案が幾つか有るので、単純な形を並べて全部数える）:
  全部引き継ぎ:             前の 3 つ組をそのまま返す（オウム返しの比べ役）。
  席ごと（全部、遷移の波）:  3 つの席とも、遷移の波で別々に選ぶ。
  席ごと（人は引き継ぎ）:    だれが は前の語、何を・どうした は遷移の波で別々に選ぶ。
  動きを引いて、物は本体:    だれが は前の語、どうした は遷移の波。何を は、本体に（だれが, ?, どうした）を問うて埋める。
  物を引いて、動きは本体:    だれが は前の語、何を は遷移の波。どうした は、本体に（だれが, 何を, ?）を問うて埋める。
  本体が答えない時は、遷移の波の答えを使う。
数える物: 席ごとの当たり、3 席とも当たり、2 席以上当たり、組んだ 3 つ組が「前に丸ごと聞いた 3 つ組」である割合（まとまり）、前の 3 つ組と同じ割合。
  比べ: 本物の次の 3 つ組が、前に丸ごと聞いた物である割合。
手で置いた物: 次元、属性を位相へ写す大きさ（I49zzw と同じ）、場面の作り方、組み方の並び。乱数は使わない。材料の語や例は出さない。
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
import i49zzw_next_triple_run as W  # noqa: E402

FORMS = ("全部引き継ぎ", "席ごと（全部、遷移の波）", "席ごと（人は引き継ぎ）", "動きを引いて、物は本体", "物を引いて、動きは本体")
KEYS = ("だれが", "何を", "どうした", "3 席とも", "2 席以上", "前に丸ごと聞いた 3 つ組", "前の 3 つ組と同じ")


def fill(body, si, tri, cands, fallback):
    """本体に、席 si を問うて埋める。答えない時は fallback。→ (語, 本体が答えたか)"""
    q = tuple(None if j == si else tri[j] for j in range(3))
    cs = [c for c in cands if c not in {tri[j] for j in range(3) if j != si}]
    o = body.ask(si, q, cs) if len(cs) >= 2 else {"答え": None}
    return (o["答え"], True) if o["答え"] is not None else (fallback, False)


def replies(prev, cands, wave, body=None):
    """→ 組み方 → (3 つ組, 本体が答えたか)。body が無い時は、本体を使う組み方を作らない。"""
    w = tuple(W.pick(cands, wave.scores(si, prev, cands)) for si in range(3))
    out = {FORMS[0]: (tuple(prev), None), FORMS[1]: (w, None), FORMS[2]: ((prev[0], w[1], w[2]), None)}
    if body is not None:
        o, ok = fill(body, 1, (prev[0], None, w[2]), cands, w[1])
        out[FORMS[3]] = ((prev[0], o, w[2]), ok)
        v, ok = fill(body, 2, (prev[0], w[1], None), cands, w[2])
        out[FORMS[4]] = ((prev[0], w[1], v), ok)
    return out


def count(first, second, attr_of, body=None, dims=None, scale=None):
    wave = W.Wave(W.DIMS if dims is None else dims, attr_of, W.SCALE if scale is None else scale)
    heard = set()

    def hear(tris):
        for a, b in zip(tris, tris[1:]):
            wave.learn(a, b)
        for t in tris:
            heard.add(tuple(t))
            if body is not None:
                body.learn(t)
        if body is not None:
            body.tick()

    for tris, _ in first:
        hear(tris)
    names = FORMS if body is not None else FORMS[:3]
    out = {n: dict(問い=0, 本体が答えた=0, **{k: 0 for k in KEYS}) for n in names}
    ref = dict(問い=0, 本物の次が前に丸ごと聞いた物=0, 本物の次が前と同じ=0)
    for tris, words in second:
        cands = sorted(words, key=str)
        for a, b in zip(tris, tris[1:]):
            ref["問い"] += 1
            ref["本物の次が前に丸ごと聞いた物"] += tuple(b) in heard
            ref["本物の次が前と同じ"] += tuple(b) == tuple(a)
            for n, (r, ok) in replies(a, cands, wave, body).items():
                c = out[n]
                hits = [r[si] == b[si] for si in range(3)]
                c["問い"] += 1
                c["本体が答えた"] += bool(ok)
                for si in range(3):
                    c[KEYS[si]] += hits[si]
                c["3 席とも"] += sum(hits) == 3
                c["2 席以上"] += sum(hits) >= 2
                c["前に丸ごと聞いた 3 つ組"] += tuple(r) in heard
                c["前の 3 つ組と同じ"] += tuple(r) == tuple(a)
        hear(tris)
    out["比べ"] = ref
    return out


def run(attr_of, talks, make_body=None, ks=(1, 2, 3), dims=None, scale=None):
    plain = [(t[0], t[1]) for t in talks]
    res, total = {}, {}
    for k in ks:
        seq = [plain[i] for i in ZP.order(len(plain), k)]
        d = count(seq[:len(seq) // 2], seq[len(seq) // 2:], attr_of, make_body() if make_body else None, dims, scale)
        res["元の順" if k == 0 else "並べ替え %d" % k] = d
        for n, c in d.items():
            t = total.setdefault(n, {})
            for key, v in c.items():
                t[key] = t.get(key, 0) + v
        print(k, {n: (c["問い"], c.get("3 席とも"), c.get("前に丸ごと聞いた 3 つ組")) for n, c in d.items() if n != "比べ"}, flush=True)
    res["合計"] = total
    return res


def main():
    import i49_foundation as FD
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
    res = dict(話=len(talks), 手で置いた値=dict(DIMS=W.DIMS, SCALE=W.SCALE),
               **run(full.of, talks, lambda: FD.body(wave_of, full.of), ks=tuple(int(x) for x in a.ks.split(","))))
    res["秒"] = round(time.time() - t0, 1)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res["合計"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
