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
  本体への仮の問い（組むための問い）は、本体の中の「どの段が当たるか」の数えに入れない。後半は、今までの場面と同じく、聞く前にその 3 つ組を本体に問う。
    （run70・run71 はここが違っていた: 仮の問いが、次に聞いた別の 3 つ組で採点されていた。75 節の続き 2。）
  話の形（78 節から。どれも、物は本体で埋め、前と同じ 3 つ組は返さない）:
    動きは出た語を除く: どうした を、その話でもう出た動きを除いて遷移の波で引く。／人は話の中の人から: だれが を、その話でもう出た人の中から遷移の波で選ぶ。／両方。
  （前と同じ物は返さない）: 上の 3 つの組み方で、組んだ 3 つ組が前の 3 つ組と同じになった時だけ、最後に決めた席を、前の語を除いて選び直す
    （席ごと は どうした を、本体で埋める形は 本体で埋めた席を）。本物の次の 3 つ組が前と同じ事は無かった（75 節）ので足した形。
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
NOREP = "（前と同じ物は返さない）"
TALK = ("話の形（動きは出た語を除く）", "話の形（人は話の中の人から）", "話の形（両方）")
KEYS = ("だれが", "何を", "どうした", "3 席とも", "2 席以上", "前に丸ごと聞いた 3 つ組", "前の 3 つ組と同じ")


def fill(body, si, tri, cands, fallback, avoid=None):
    """本体に、席 si を問うて埋める。答えない時は fallback。avoid は候補から除く語。→ (語, 本体が答えたか)"""
    q = tuple(None if j == si else tri[j] for j in range(3))
    cs = [c for c in cands if c != avoid and c not in {tri[j] for j in range(3) if j != si}]
    n = len(getattr(body, "_wait", ()))
    o = body.ask(si, q, cs) if len(cs) >= 2 else {"答え": None}
    if hasattr(body, "_wait"):
        del body._wait[n:]          # 仮の問いは、本体の「どの段が当たるか」の数えに入れない（次に聞く 3 つ組の問いではないので）
    return (o["答え"], True) if o["答え"] is not None else (fallback, False)


def replies(prev, cands, wave, body=None, hist=None, allow=None):
    """→ 組み方 → (3 つ組, 本体が答えたか)。body が無い時は、本体を使う組み方を作らない。
    hist（その話の、ここまでの 3 つ組。prev を含む）を渡すと、話の形（78 節）も作る。
    allow(席, 語) を渡すと、その席に置いて良い語だけから選ぶ（候補を広げた時に、動きの席へ物の語を置かないため）。"""
    prev = tuple(prev)
    sc = [wave.scores(si, prev, cands) for si in range(3)]
    if allow is not None:
        sc = [[x if allow(si, c) else float("-inf") for x, c in zip(sc[si], cands)] for si in range(3)]

    def fill_(body, si, tri, cands, fallback, avoid=None):
        return fill(body, si, tri, cands if allow is None else [c for c in cands if allow(si, c)], fallback, avoid)
    w = tuple(W.pick(cands, sc[si]) for si in range(3))

    def other(si):
        """席 si を、前の語を除いて遷移の波で選び直す。"""
        keep = [i for i, c in enumerate(cands) if c != prev[si] and (allow is None or allow(si, c))]
        return W.pick([cands[i] for i in keep], [sc[si][i] for i in keep]) if keep else prev[si]

    seat = (prev[0], w[1], w[2])
    out = {FORMS[0]: (prev, None), FORMS[1]: (w, None), FORMS[2]: (seat, None),
           FORMS[2] + NOREP: ((prev[0], w[1], other(2)) if seat == prev else seat, None)}
    if body is not None:
        o, ok = fill_(body, 1, (prev[0], None, w[2]), cands, w[1])
        out[FORMS[3]] = out[FORMS[3] + NOREP] = ((prev[0], o, w[2]), ok)
        if out[FORMS[3]][0] == prev:
            o, ok = fill_(body, 1, (prev[0], None, w[2]), cands, other(1), avoid=prev[1])
            out[FORMS[3] + NOREP] = ((prev[0], o, w[2]), ok)
        v, ok = fill_(body, 2, (prev[0], w[1], None), cands, w[2])
        out[FORMS[4]] = out[FORMS[4] + NOREP] = ((prev[0], w[1], v), ok)
        if out[FORMS[4]][0] == prev:
            v, ok = fill_(body, 2, (prev[0], w[1], None), cands, other(2), avoid=prev[2])
            out[FORMS[4] + NOREP] = ((prev[0], w[1], v), ok)
        if hist is not None:
            def among(si, keep):
                ix = [i for i, c in enumerate(cands) if keep(c) and (allow is None or allow(si, c))]
                return W.pick([cands[i] for i in ix], [sc[si][i] for i in ix]) if ix else None

            people, verbs = {t[0] for t in hist}, {t[2] for t in hist}
            who = among(0, lambda c: c in people) or prev[0]
            act = among(2, lambda c: c not in verbs) or w[2]
            for name, s0, v0 in zip(TALK, (prev[0], who, who), (act, w[2], act)):
                o, ok = fill_(body, 1, (s0, None, v0), cands, w[1])
                if (s0, o, v0) == prev:
                    o, ok = fill_(body, 1, (s0, None, v0), cands, other(1), avoid=prev[1])
                out[name] = ((s0, o, v0), ok)
    return out


def count(first, second, attr_of, body=None, dims=None, scale=None):
    wave = W.Wave(W.DIMS if dims is None else dims, attr_of, W.SCALE if scale is None else scale)
    heard = set()

    def hear(tris, words=None):
        for a, b in zip(tris, tris[1:]):
            wave.learn(a, b)
        for t in tris:
            heard.add(tuple(t))
            if body is not None:
                if words is not None:       # 後半は、今までの場面と同じく、聞く前にその 3 つ組を本体に問う（本体の中の数えを育てるため。答えは使わない）
                    for si in range(3):
                        cs = sorted({w for w in words if w not in {t[j] for j in range(3) if j != si}} | {t[si]}, key=str)
                        if len(cs) >= 2:
                            body.ask(si, tuple(None if j == si else t[j] for j in range(3)), cs)
                body.learn(t)
        if body is not None:
            body.tick()

    for tris, _ in first:
        hear(tris)
    names = [n for f in (FORMS if body is not None else FORMS[:3]) for n in ((f,) if f in FORMS[:2] else (f, f + NOREP))]
    names += list(TALK) if body is not None else []
    out = {n: dict(問い=0, 本体が答えた=0, **{k: 0 for k in KEYS}) for n in names}
    ref = dict(問い=0, 本物の次が前に丸ごと聞いた物=0, 本物の次が前と同じ=0)
    for tris, words in second:
        cands = sorted(words, key=str)
        for i in range(1, len(tris)):
            a, b = tris[i - 1], tris[i]
            ref["問い"] += 1
            ref["本物の次が前に丸ごと聞いた物"] += tuple(b) in heard
            ref["本物の次が前と同じ"] += tuple(b) == tuple(a)
            for n, (r, ok) in replies(a, cands, wave, body, [tuple(t) for t in tris[:i]]).items():
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
        hear(tris, words)
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
