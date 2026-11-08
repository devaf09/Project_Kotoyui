"""介入の研究 実験I27w: 語ごとの「像の偏り」（どの像がよく使う語か）を束ね、作品をまたいで同じかを確かめる

持ち主（2026-11-08）: 「語板、機能板、品詞板にそれぞれの像が優先的に選ぶ属性か、尊敬語と謙譲語、現代語などの像がよく使う属性を追加するのはどう思う？」→「始めてOK」
I27v で、像（台詞の語尾と一人称で束ねた人の型）は「誰が話したか当てる」より「どう話すかを作る・確かめる」向きとわかった。
そこで、像から語へ向かう属性を作る:
  1. I27t（語尾から学び、介入で広げる）を全作品の正解で回し、台詞ごとの話し手の確からしさを出す。
  2. 人の板を k 平均で束ねて像にし（I27u・I27v と同じ）、人ごとに像の混ぜ方（cos の softmax）を決める。
  3. 台詞の中の語を、話し手の確からしさ × 像の混ぜ方の重みで束ね、語 × 像 の板にする（束ねる）。
  4. 語の像の偏り = その語での像の割合 ÷ 全部の語での像の割合（期待）の対数（差分: + は発見 = その像がよく使う）。
  5. 作品を半分ずつに分けて 4 を 2 回作り、語ごとの偏りの相関を見る。高ければ汎用板の候補（作品によらない属性）。
尊敬語・謙譲語などの名前は手で決めない。偏りから浮かんだ語の束に、最後に名前を付けて読むだけ（確かめの語は表示だけで、作り方には使わない）。
結果と表は非公開の置き場だけ。本文はコードに書かない。
実行: python intervention/i27w_word_images.py --quotes <quotes_words_wn.jsonl> --stats <jsonl> --out <md> --table <tsv> --jobs 10
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i27t_first_person as T  # noqa: E402
import i27u_character_images as U  # noqa: E402
import i27v_images_1808 as V  # noqa: E402

LOOK = ("いらっしゃる", "おっしゃる", "申す", "参る", "存じる", "くださる", "召し上がる", "ござる", "致す", "伺う",
        "やがる", "じゃん", "マジ", "ヤバい", "めっちゃ", "わらわ", "そなた", "貴様", "おぬし", "候")
GROUPS = (("動詞・形容詞", {"VERB", "ADJ"}), ("助動詞・助詞", {"AUX", "PART", "ADP", "SCONJ", "CCONJ"}),
          ("名詞・代名詞", {"NOUN", "PRON", "PROPN", "NUM"}), ("副詞・感動詞・連体詞", {"ADV", "INTJ", "DET"}))


def job(args):
    qs_list, narr, rounds, rates = args
    posts = []
    _, _, boards, _ = T.run_fold(qs_list, -1, U.USE, rounds, narr, rates=rates, posts=posts)
    return [(posts[i], U.person_vecs(P, min_n=0.0)) for i, P in enumerate(boards)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quotes", required=True)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--beta", type=float, default=20.0)
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--min_count", type=int, default=20)
    ap.add_argument("--kseed", type=int, default=0, help="k 平均の種（像の顔ぶれを選び直すとき）")
    ap.add_argument("--stats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--table", required=True)
    a = ap.parse_args()
    t0 = time.time()
    works = V.load(a.quotes)
    allq = [w[1] for w in works]
    rates = T.flow_rates(allq, T.known_of(allq, -1))
    from multiprocessing import Pool
    idx = list(range(len(works)))
    parts = [idx[i::a.jobs * 4] for i in range(a.jobs * 4)]
    parts = [p for p in parts if p]
    out = {}
    with Pool(a.jobs) as pool:
        tasks = [([works[i][1] for i in p], [works[i][2] for i in p], a.rounds, rates) for p in parts]
        for p, res in zip(parts, pool.map(job, tasks)):
            for i, r in zip(p, res):
                out[i] = r
    print("広げた", f"{time.time() - t0:.0f}s", flush=True)
    # 像（板の厚い人だけで作る）
    vecs3 = [{c: v for c, v in out[i][1].items() if v[0] >= 3.0} for i in idx]
    im = U.Images(vecs3, a.K, a.beta, seed=a.kseed)
    C, M, asg, rs = im.for_work(-1)
    K = len(C)
    allm = M.mean(0)
    names = []
    for j in range(K):
        d = M[j] - allm
        top = [im.vocab[i][1] for i in np.argsort(-d)[:10] if d[i] > 0 and im.vocab[i][0] == "尾1"][:4]
        names.append(f"像{j + 1}（{'・'.join(top)}）")
    # 語 × 像 の板（作品を半分ずつ）
    rng = random.Random(0)
    half = {i: rng.random() < 0.5 for i in idx}
    WI = [defaultdict(lambda: np.zeros(K)), defaultdict(lambda: np.zeros(K))]
    IJ = [np.zeros(K), np.zeros(K)]
    n_q = 0
    for i in idx:
        posts, pv = out[i]
        mix = {}
        for c, (n, sh) in pv.items():
            v = im.vec(sh)
            nv = np.linalg.norm(v)
            if nv == 0:
                continue
            s = C @ (v / nv)
            e = np.exp(a.beta * (s - s.max()))
            mix[c] = e / e.sum()
        h = int(half[i])
        for k, q in enumerate(works[i][1]):
            pr = posts.get(k)
            if not pr or "w" not in q:
                continue
            d = np.zeros(K)
            for c, p in pr.items():
                if c in mix:
                    d += p * mix[c]
            if d.sum() <= 0:
                continue
            n_q += 1
            for x, v in q["w"].items():
                WI[h][x] += v * d
                IJ[h] += v * d
    print("語を束ねた", n_q, f"{time.time() - t0:.0f}s", flush=True)

    def bias(W, I):
        base = I / I.sum()
        out = {}
        for x, v in W.items():
            tot = v.sum()
            out[x] = (tot, np.log((v + a.alpha * base) / (tot + a.alpha) / base))
        return out
    BA, BB = bias(WI[0], IJ[0]), bias(WI[1], IJ[1])
    Wall = defaultdict(lambda: np.zeros(K))
    for h in (0, 1):
        for x, v in WI[h].items():
            Wall[x] += v
    BALL = bias(Wall, IJ[0] + IJ[1])
    common = [x for x in BA if x in BB and BA[x][0] >= a.min_count and BB[x][0] >= a.min_count]
    stab = {}
    for j in range(K):
        xa = np.array([BA[x][1][j] for x in common])
        xb = np.array([BB[x][1][j] for x in common])
        stab[names[j]] = round(float(np.corrcoef(xa, xb)[0, 1]), 3) if len(common) > 2 else None
    pooled = float(np.corrcoef(np.concatenate([[BA[x][1][j] for x in common] for j in range(K)]),
                               np.concatenate([[BB[x][1][j] for x in common] for j in range(K)]))[0, 1])
    # 語ごとの安定さ（両方の半分で偏りの向きが同じか）
    agree = {x: float(np.corrcoef(BA[x][1], BB[x][1])[0, 1]) for x in common}
    rec = dict(exp="I27w", kseed=a.kseed, 作品=len(works), 束ねた台詞=n_q, 像=names, 語=len(Wall),
               両方の半分で数の足りた語=len(common), 像ごとの半分どうしの相関=stab, まとめた相関=round(pooled, 3),
               語ごとの向きが合う割合=round(sum(1 for v in agree.values() if v > 0.5) / max(1, len(agree)), 3),
               秒=round(time.time() - t0))
    print(json.dumps(rec, ensure_ascii=False, indent=1), flush=True)
    with open(a.stats, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    # 表（語の属性の候補）
    os.makedirs(os.path.dirname(os.path.abspath(a.table)), exist_ok=True)
    with open(a.table, "w", encoding="utf-8") as f:
        f.write("pos\tlemma\tcount\tstable\t" + "\t".join(names) + "\n")
        for x, (tot, b) in sorted(BALL.items(), key=lambda kv: -kv[1][0]):
            if tot < a.min_count:
                continue
            st = agree.get(x)
            f.write(f"{x[0]}\t{x[1]}\t{tot:.1f}\t{'' if st is None else round(st, 3)}\t" + "\t".join(f"{v:.3f}" for v in b) + "\n")
    # 読む用の md
    lines = ["# I27w 語ごとの像の偏り（非公開）", "",
             f"台詞 {n_q:,} を、話し手の確からしさ × 像の混ぜ方で束ねた。偏り = その語での像の割合 ÷ 全体での像の割合 の対数（+ はその像がよく使う）。",
             f"作品を半分ずつに分けた偏りの相関（{len(common):,} 語）: まとめて {pooled:.3f}。", ""]
    for j in range(K):
        lines += [f"## {names[j]}（半分どうしの相関 {stab[names[j]]}）", ""]
        for gname, ps in GROUPS:
            cand = [(BALL[x][1][j], x) for x in BALL if x[0] in ps and BALL[x][0] >= max(30, a.min_count) and agree.get(x, 0) > 0.5]
            cand.sort(reverse=True)
            lines.append(f"- {gname}: " + "、".join(f"{x[1]}（{b:+.2f}）" for b, x in cand[:12]))
        lines.append("")
    lines += ["## 確かめの語（作り方には使っていない。どの像に偏ったかだけ）", ""]
    for word in LOOK:
        hits = [(BALL[x][0], x) for x in BALL if x[1] == word]
        if not hits:
            lines.append(f"- {word}: （数が足りない）")
            continue
        tot, x = max(hits)
        b = BALL[x][1]
        j = int(np.argmax(b))
        lines.append(f"- {word}（{x[0]}、{tot:.0f} 回）: いちばん偏った像 {names[j]} {b[j]:+.2f}、向きの安定 {agree.get(x, float('nan')):.2f}")
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
