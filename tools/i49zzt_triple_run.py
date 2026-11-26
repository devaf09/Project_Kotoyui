"""I49zzt: 言葉の 3 つ組で、掛け合わせた波（数える／外れた時だけ位相を動かす）を、今の本体と比べる道具（`intervention/i49zzt_triple.py`。未確定の試し）。

場面は「本体が後半も聞く」と同じ（tools/i49zzg_margin.py と同じ作り）: 話を並べ替えて半分に分け、前半を聞かせ、後半は問うてから聞かせる。
問いは、3 つ組の 1 つの席を隠して、話の中の語から選ぶ形。どの形も、同じ問いに同じ候補で答える。
比べる物:
  本体（i49_foundation.body）／掛け合わせて数える（TripleCount）／外れた時だけ位相を動かす（TripleNudge。動かす大きさと、前半を聞く回数を変える）。
分けて数える物（聞いていない組に届くかを見るため）:
  「隠した席の他の 2 語の組を、前に聞いた事が有るか」と「正解の語を、その席で前に聞いた事が有るか」。
手で置いた物: FORMS（次元・動かす大きさ・前半を聞く回数）、場面の作り方。乱数は使わない。
材料の語や例は出さない。出すのは集計の数だけ。
--attr を付けると、位相を語板の属性から始める形（数えるだけ／外れた時だけ動かす。写す大きさを変える）も、同じ問いで数える。
引数: --text 材料（カンマ区切り） --wave 語の波 --content 内容語の板 --out 記録.json [--ks 1,2,3] [--attr]
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
import i49zzt_triple as T  # noqa: E402

# 名前 → (次元, 動かす大きさ, 前半を聞く回数)
FORMS = {"動かす（4096 次元、大きさ .1、1 回）": (4096, 0.1, 1), "動かす（4096 次元、大きさ .3、1 回）": (4096, 0.3, 1),
         "動かす（4096 次元、大きさ .3、4 回）": (4096, 0.3, 4), "動かす（1024 次元、大きさ .3、4 回）": (1024, 0.3, 4)}
COUNT_DIMS = 4096
# 位相を語板の属性から始める形（--attr を付けた時）。名前 → (動かす大きさ。None は数えるだけ, 属性を位相へ写す大きさ)
ATTR_FORMS = {"属性から始めて数える（写す大きさ 1）": (None, 1.0), "属性から始めて数える（2）": (None, 2.0), "属性から始めて数える（4）": (None, 4.0),
              "属性から始めて数える（8）": (None, 8.0), "属性から始めて動かす（2、大きさ .03）": (0.03, 2.0), "属性から始めて動かす（2、大きさ .1）": (0.1, 2.0),
              "属性から始めて動かす（4、大きさ .03）": (0.03, 4.0), "属性から始めて動かす（4、大きさ .1）": (0.1, 4.0)}
KINDS = ("全部", "2 語の組を前に聞いた", "2 語の組は初めて", "2 語の組は初めてで、正解の語はその席で聞いた")


def cands_of(tri, si, words):
    return sorted({w for w in words if w not in {tri[j] for j in range(3) if j != si}} | {tri[si]}, key=str)


def count(body, first, second, forms=None, count_dims=None, attr_of=None, attr_forms=None):
    forms = FORMS if forms is None else forms
    cd = COUNT_DIMS if count_dims is None else count_dims
    models = {"掛け合わせて数える": T.TripleCount(cd)}
    passes, counters = {}, ["掛け合わせて数える"]
    for name, (dims, eta, p) in forms.items():
        models[name] = T.TripleNudge(dims, eta)
        passes[name] = p
    if attr_of is not None:
        for name, (eta, scale) in (ATTR_FORMS if attr_forms is None else attr_forms).items():
            if eta is None:
                models[name] = T.TripleCount(cd, attr_of=attr_of, scale=scale)
                counters.append(name)
            else:
                models[name] = T.TripleNudge(cd, eta, attr_of=attr_of, scale=scale)
                passes[name] = 1
    pairs, seat = set(), set()

    def note(tri):
        for si in range(3):
            pairs.add((si,) + tuple(tri[j] for j in range(3) if j != si))
            seat.add((si, tri[si]))

    for tris, words in first:
        for tri in tris:
            body.learn(tri)
            for name in counters:
                models[name].learn(tri)
            note(tri)
        body.tick()
    for name, p in passes.items():
        for _ in range(p):
            for tris, words in first:
                for tri in tris:
                    models[name].learn(tri, sorted(words, key=str))
    names = ["本体"] + list(models)
    out = {k: dict(問い=0, 当てずっぽう=0.0, **{n: 0 for n in names}) for k in KINDS}
    for tris, words in second:
        for tri in tris:
            for si in range(3):
                cands = cands_of(tri, si, words)
                if len(cands) < 2:
                    continue
                q = tuple(None if j == si else tri[j] for j in range(3))
                seen = (si,) + tuple(tri[j] for j in range(3) if j != si) in pairs
                kinds = ["全部", "2 語の組を前に聞いた" if seen else "2 語の組は初めて"]
                if not seen and (si, tri[si]) in seat:
                    kinds.append("2 語の組は初めてで、正解の語はその席で聞いた")
                ans = {"本体": body.ask(si, q, cands)["答え"]}
                for n, m in models.items():
                    ans[n] = m.ask(si, q, cands)["答え"]
                for k in kinds:
                    out[k]["問い"] += 1
                    out[k]["当てずっぽう"] += 1 / len(cands)
                    for n in names:
                        out[k][n] += ans[n] == tri[si]
            body.learn(tri)
            for name in counters:
                models[name].learn(tri)
            for name in passes:
                models[name].learn(tri, sorted(words, key=str))
            note(tri)
        body.tick()
    for c in out.values():
        c["当てずっぽう"] = round(c["当てずっぽう"], 1)
    return out


def run(wave_of, attr_of, talks, ks=(1, 2, 3), forms=None, count_dims=None, attr=False, attr_forms=None):
    import i49_foundation as FD
    plain = [(t[0], t[1]) for t in talks]
    res = {}
    for k in ks:
        seq = [plain[i] for i in ZP.order(len(plain), k)]
        res["元の順" if k == 0 else "並べ替え %d" % k] = count(FD.body(wave_of, attr_of), seq[:len(seq) // 2], seq[len(seq) // 2:], forms, count_dims,
                                                        attr_of if attr else None, attr_forms)
        print(k, res["元の順" if k == 0 else "並べ替え %d" % k]["全部"], flush=True)
    total = {}
    for d in res.values():
        for kind, c in d.items():
            t = total.setdefault(kind, {})
            for key, v in c.items():
                t[key] = round(t.get(key, 0) + v, 1)
    res["合計"] = total
    return res


def main():
    import i49c_read_words as Rw
    import i49m_attr_base as AB
    ap = argparse.ArgumentParser()
    for k in ("text", "wave", "content", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--ks", default="1,2,3")
    ap.add_argument("--attr", action="store_true")   # 位相を語板の属性から始める形も数える
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
    res = dict(話=len(talks), 手で置いた値=dict(FORMS={n: list(v) for n, v in FORMS.items()}, COUNT_DIMS=COUNT_DIMS,
                                              ATTR_FORMS={n: list(v) for n, v in ATTR_FORMS.items()} if a.attr else None),
               **run(wave_of, full.of, talks, ks=tuple(int(x) for x in a.ks.split(",")), attr=a.attr))
    res["秒"] = round(time.time() - t0, 1)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res["合計"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
