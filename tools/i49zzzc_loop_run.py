"""I49zzzc: ここまでの部品を通しでつないで、文を出す（持ち主の案、ロードマップの追記 16 の案 3-1。最小の輪。未確定の試し）。

持ち主の案: 「相手の行 → 75節の組み方（人・動き・本体で物） → 76節の直した I42（意味の印＋状況波の丁寧度で活用生成） → 発話出力 → 最初へ戻す、
  という一連の流れをまずは1つのスクリプトで通し、実際の日本語の応答文を10〜20往復出力して目視確認する。」
流れ:
  1. 前の行の 3 つ組を受け取る（材料は「A が B を C する」のテキスト。切り方は tools/i49zp_text_talks_run.py）。
  2. 返事の 3 つ組を組む（tools/i49zzx_reply_triple_run.py の replies）。だれが は話の中の人から遷移の波で、どうした は遷移の波で、
     何を は本体に問うて埋める。前と同じ 3 つ組は返さない。
  3. 文の形にする（intervention/i49zzy_form.py の Former。流れは κ=3 で混ぜる形）。「だれが が 何を を どうした」＋意味の印。
     丁寧は、前の行からは選べない（73 節）ので、状態として固定する（言い切りと丁寧の 2 つを並べて出す）。
  4. 出した 3 つ組を、次の入り口に戻す（自分で続ける形）。
場面は今までと同じ（話を並べ替えて半分に分け、前半を聞かせる。後半の話で試し、試した後に聞かせる）。
候補は 2 通り: 狭い（その話の語。先の行の語も入っているので、当たりやすい作り）／広い（ここまでに聞いた全部の語と、その話でここまでに出た語）。
出す物:
  相手に返す: 後半の初めの話（--talks 個）で、前の行ごとに、本物の次の行と、組んだ返事（狭い／広い）を並べる。
  自分で続ける: 同じ話の初めの 1 行から、返事を次の入り口に戻して --steps 回続ける（広い候補。組み方 2 通り: 上の形／その話で出た動きは返さない形）。
  数え: 後半の問いから等しい間隔で --sample 個取り、広い候補での席ごとの当たりと、3 席とも当たりを数える。
出た文は材料の語を使うので、`--out` はリポジトリに入れない。
手で置いた物: 組み方の選び方、丁寧を固定する事、κ=3、場面の作り方、広い候補の作り方。乱数は使わない。
引数: --text 材料（カンマ区切り） --wave 語の波 --content 内容語の板 --cache 形を組む部品の材料（解析済み JSON） --out 記録.json
      [--k 1] [--talks 2] [--steps 10] [--sample 300]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "intervention"), os.path.join(HERE, "..", "intervention", "past")]

import i49zp_text_talks_run as ZP  # noqa: E402
import i49zzw_next_triple_run as W  # noqa: E402
import i49zzx_reply_triple_run as X  # noqa: E402

FORM = X.TALK[1]              # だれが は話の中の人から、どうした は遷移の波、何を は本体。前と同じ 3 つ組は返さない
FORM_NEW = X.TALK[2]          # 上に足して、その話で出た動きは返さない
COUNTED = (X.FORMS[3] + X.NOREP, FORM, FORM_NEW)
KAPPA = 3.0
MARKS = (("言い切り", ()), ("丁寧", ("丁寧",)))


def line_of(former, tri, marks=()):
    """3 つ組と意味の印 → 文。former が無い時や組めない時は、動詞をそのまま置く。"""
    v = former.build(tri[2][0], marks, fix=("混ぜる", KAPPA, False)) if former is not None else None
    return "%sが%sを%s。" % (tri[0][0], tri[1][0], v or tri[2][0])


def lines_of(former, tri):
    return {name: line_of(former, tri, m) for name, m in MARKS}


def run(first, second, attr_of, body, former=None, talks=2, steps=10, sample=300, dims=None, scale=None):
    wave = W.Wave(W.DIMS if dims is None else dims, attr_of, W.SCALE if scale is None else scale)
    vocab = set()

    def hear(tris, words=None):
        vocab.update(w for t in tris for w in t)
        for a, b in zip(tris, tris[1:]):
            wave.learn(a, b)
        for t in tris:
            if words is not None:
                for si in range(3):
                    cs = sorted({w for w in words if w not in {t[j] for j in range(3) if j != si}} | {t[si]}, key=str)
                    if len(cs) >= 2:
                        body.ask(si, tuple(None if j == si else t[j] for j in range(3)), cs)
            body.learn(t)
        body.tick()

    for tris, _ in first:
        hear(tris)
    total = sum(len(t) - 1 for t, _ in second)
    every = max(total // max(sample, 1), 1)
    cnt = {n: dict(問い=0, だれが=0, 何を=0, どうした=0, **{"3 席とも": 0}) for n in COUNTED}
    reply, free, q = [], [], 0
    for ti, (tris, words) in enumerate(second):
        tris = [tuple(t) for t in tris]
        narrow = sorted(words, key=str)
        show = ti < talks
        for i in range(1, len(tris)):
            a, b, hist = tris[i - 1], tris[i], tris[:i]
            if not show and q % every:
                q += 1
                continue
            q += 1
            wide = sorted(vocab | {w for t in hist for w in t}, key=str)
            rw = X.replies(a, wide, wave, body, hist)
            for n in COUNTED:
                r, c = rw[n][0], cnt[n]
                hits = [r[si] == b[si] for si in range(3)]
                c["問い"] += 1
                for si, s in enumerate(ZP.SEATS):
                    c[s] += hits[si]
                c["3 席とも"] += all(hits)
            if show:
                rn = X.replies(a, narrow, wave, body, hist)[FORM][0]
                reply.append(dict(話=ti, 前の行=line_of(None, a), 本物の次の行=line_of(None, b), 返事_狭い候補=lines_of(former, rn),
                                  返事_広い候補=lines_of(former, rw[FORM][0]), 当たった席_広い候補=[s for si, s in enumerate(ZP.SEATS) if rw[FORM][0][si] == b[si]]))
        if show:
            wide = sorted(vocab | set(tris[0]), key=str)
            for name in (FORM, FORM_NEW):
                hist, said = [tris[0]], []
                for _ in range(steps):
                    r = X.replies(hist[-1], sorted(set(wide) | {w for t in hist for w in t}, key=str), wave, body, hist)[name][0]
                    cnt_same = r in hist
                    said.append(dict(lines_of(former, r), 前に出した物と同じ=cnt_same))
                    hist.append(r)
                free.append(dict(話=ti, 組み方=name, 初めの行=line_of(None, tris[0]), 続き=said, 同じ物をもう一度出した数=sum(x["前に出した物と同じ"] for x in said)))
        hear(tris, words)
    return dict(相手に返す=reply, 自分で続ける=free, 数え_広い候補=cnt, 広い候補の語の数=len(vocab))


def main():
    import i42_fused_flow as F
    import i49_foundation as FD
    import i49c_read_words as Rw
    import i49m_attr_base as AB
    import i49zzy_form as Y
    import i49zzz_talk_memory_run as Z
    ap = argparse.ArgumentParser()
    for k in ("text", "wave", "content", "cache", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--talks", type=int, default=2)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--sample", type=int, default=300)
    a = ap.parse_args()
    t0 = time.time()
    talks = []
    for f in a.text.split(","):
        talks += [(t[0], t[1]) for t in ZP.talks_of(Z.read(f))[0]]
    wv = Rw.WordVecs.load(a.wave)

    def wave_of(w):
        v = wv.of(*w)
        return wv.of(w[0], "PROPN") if v is None and w[1] == "NOUN" else v

    full = AB.AttrBase(json.load(open(a.content, encoding="utf-8")))
    works = json.load(open(a.cache, encoding="utf-8"))
    works = works["works"] if isinstance(works, dict) and "works" in works else works
    former = Y.Former(F.items(works))
    print("部品を作った", round(time.time() - t0, 1), flush=True)
    seq = [talks[i] for i in ZP.order(len(talks), a.k)]
    res = dict(話=len(talks), 手で置いた値=dict(DIMS=W.DIMS, SCALE=W.SCALE, KAPPA=KAPPA, 組み方=FORM),
               **run(seq[:len(seq) // 2], seq[len(seq) // 2:], full.of, FD.body(wave_of, full.of), former, a.talks, a.steps, a.sample))
    res["秒"] = round(time.time() - t0, 1)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res["数え_広い候補"], ensure_ascii=False), res["広い候補の語の数"], res["秒"])
    print("自分で続ける: 同じ 3 つ組をもう 1 度出した数", [(x["話"], x["組み方"], x["同じ物をもう一度出した数"]) for x in res["自分で続ける"]])


if __name__ == "__main__":
    main()
