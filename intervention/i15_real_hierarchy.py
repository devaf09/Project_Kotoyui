"""介入の研究 実験I15: 本物の語板（語板 v1）で、親（上位語のクラス）から知らない語の性質・動きを確率つきで想定する

I10・I11 の手作りの世界（科 → 品種）を、語板のスレッドが本物の文から作った語板 v1 でやり直す。
材料（非公開の置き場 ../Project_Kotoyui_private/data/boards/v1/）:
  - content_boards.json: 内容語の語板（帯 A 文法・B 性質・C 動き（する／される）・D クラス（日本語 WordNet の上位語）・E 関係）。数えた割合。
  - light_v1_bright_s0.npy: 先頭 5,000 語を光の器で学んだ出口の明るさ（5,000 × 属性 1,790。行 = 属性のある語板の順、列 = 属性の名札を並べ替えた順）。
方法:
  - 名詞のうち、D 帯でいちばん明るいクラスを親とする（語板のスレッドの「機能」と同じ決め方）。子が 8 つ以上のクラスだけ使う。
  - クラスごとに子の 2 割を隠す。隠した語は「どのクラスか」だけ分かるとして、B 帯（性質）と C 帯（動き）を想定する。
  - 親の想定 = (知っている子の明るさの和 + α × 名詞全体の平均) / (子の数 + α)（光の融合 = 明るさの和。I11 と同じ薄め）。帯の中で 1 に正規化。
    α は知っている子の 1 つ抜きで、本当の割合の対数確率が最も大きいものを選ぶ（隠した語は使わない）。
  - 比べ: 名詞全体から（祖先）、同じ大きさの乱数の組から（親子の向きが無いとき）、誤ったクラスから。
  - 物差し（隠した語ごと、帯ごと）: 本当の割合（数えた語板）に付けた対数確率の平均（Σ q log p、大きいほど良い）、
    想定の 1 位の属性が本当の上位 3 つに入るか、1 位の確率と当たりの較正（きっと ≥ 0.5、たぶん 0.2〜0.5、わからない < 0.2。帯の属性が多いので I10 より低い区切り）。
結果: ../Project_Kotoyui_private/intervention/results/i15.jsonl
実行: python intervention/i15_real_hierarchy.py --seed 0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("KOTOYUI_BOARDS", os.path.join(HERE, "..", "..", "Project_Kotoyui_private", "data", "boards", "v1"))
BANDS = ["B", "C"]
ALPHAS = [0.0, 1.0, 4.0, 16.0, 64.0]


def load():
    boards = json.load(open(os.path.join(DATA, "content_boards.json"), encoding="utf-8"))[:5000]
    boards = [b for b in boards if b["attrs"]]                       # lexicon/content_light.py と同じ並び
    S = sorted({(v, x) for b in boards for v, xs in b["attrs"].items() for x in xs})
    L = np.load(os.path.join(DATA, "light_v1_bright_s0.npy")).astype(np.float64)
    assert L.shape == (len(boards), len(S)), (L.shape, len(boards), len(S))
    return boards, S, L


def word_of(p):
    return "きっと" if p >= 0.5 else ("たぶん" if p >= 0.2 else "わからない")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_kids", type=int, default=8)
    ap.add_argument("--hold", type=float, default=0.2)
    ap.add_argument("--out", default="../Project_Kotoyui_private/intervention/results/i15.jsonl")
    a = ap.parse_args()
    t0 = time.perf_counter()
    rng = random.Random(a.seed)
    boards, S, L = load()
    band_ids = {v: [i for i, (vv, _) in enumerate(S) if vv == v] for v in BANDS}
    nouns = [i for i, b in enumerate(boards) if b["pos"] == "名詞" and b["attrs"].get("D") and all(b["attrs"].get(v) for v in BANDS)]
    cls = {i: max(boards[i]["attrs"]["D"], key=boards[i]["attrs"]["D"].get) for i in nouns}
    groups = defaultdict(list)
    for i in nouns:
        groups[cls[i]].append(i)
    groups = {k: v for k, v in groups.items() if len(v) >= a.min_kids}
    hidden, known = set(), {}
    for k, v in groups.items():
        v = v[:]
        rng.shuffle(v)
        h = max(1, int(round(a.hold * len(v))))
        hidden |= set(v[:h])
        known[k] = v[h:]
    allk = [i for v in known.values() for i in v]

    def bdist(rows, v):
        x = L[np.ix_(rows, band_ids[v])].sum(0) if len(rows) else np.zeros(len(band_ids[v]))
        return x

    def norm(x):
        x = x + 1e-9
        return x / x.sum()

    root = {v: norm(bdist(allk, v)) for v in BANDS}

    def predict(kids, v, alpha):
        s = sum(norm(L[i, band_ids[v]]) for i in kids) if kids else np.zeros(len(band_ids[v]))
        return norm((s + alpha * root[v]) / (len(kids) + alpha + 1e-9))

    def truth(i, v):
        q = np.zeros(len(band_ids[v]))
        col = {S[j]: k for k, j in enumerate(band_ids[v])}
        for x, p in boards[i]["attrs"][v].items():
            if (v, x) in col:
                q[col[(v, x)]] = p
        return q / max(q.sum(), 1e-12)

    # α を知っている子の 1 つ抜きで決める（クラスごとに最大 10 語）
    best = None
    for al in ALPHAS:
        ll = []
        for k, ks in known.items():
            for i in ks[:10]:
                others = [j for j in ks if j != i]
                for v in BANDS:
                    ll.append(float((truth(i, v) * np.log(predict(others, v, al))).sum()))
        m = float(np.mean(ll))
        if best is None or m > best[0]:
            best = (m, al)
    alpha = best[1]
    # 乱数の組（親子の向きが無いとき）: 知っている名詞を、クラスと同じ大きさの組に乱数で分ける
    perm = allk[:]
    rng.shuffle(perm)
    rand_groups, p0 = {}, 0
    for k, ks in known.items():
        rand_groups[k] = perm[p0:p0 + len(ks)]
        p0 += len(ks)
    keys = list(known)
    rows = defaultdict(list)
    for i in sorted(hidden):
        k = cls[i]
        wrong = keys[(keys.index(k) + 1 + rng.randrange(len(keys) - 1)) % len(keys)]
        for v in BANDS:
            q = truth(i, v)
            top3 = set(np.argsort(-q)[:3])
            for name, p in (("親（クラス）", predict(known[k], v, alpha)), ("親（薄めなし）", predict(known[k], v, 0.0)),
                            ("祖先（名詞全体）", root[v]), ("乱数の組", predict(rand_groups[k], v, alpha)), ("誤ったクラス", predict(known[wrong], v, alpha))):
                top = int(np.argmax(p))
                rows[name].append((v, float((q * np.log(p)).sum()), top in top3, float(p[top])))
    summary = {}
    for name, rs in rows.items():
        by = defaultdict(list)
        for v, ll, ok, pt in rs:
            by[v].append((ll, ok))
        words = defaultdict(list)
        for v, ll, ok, pt in rs:
            words[word_of(pt)].append(ok)
        summary[name] = dict(logp={v: float(np.mean([x[0] for x in r])) for v, r in by.items()},
                             top1_in_top3={v: float(np.mean([x[1] for x in r])) for v, r in by.items()},
                             words={k: (float(np.mean(v)), len(v)) for k, v in words.items()})
    ex = []
    for i in sorted(hidden)[:200:25]:
        k = cls[i]
        p = predict(known[k], "B", alpha)
        top = np.argsort(-p)[:3]
        ex.append(dict(word=boards[i]["name"], cls=k, pred=[(S[band_ids["B"][j]][1], round(float(p[j]), 3)) for j in top],
                       true=sorted(boards[i]["attrs"]["B"].items(), key=lambda x: -x[1])[:3]))
    rec = dict(exp="I15", seed=a.seed, n_classes=len(groups), n_hidden=len(hidden), n_known=len(allk), alpha=alpha, summary=summary, examples=ex,
               total_s=time.perf_counter() - t0)
    print(json.dumps(rec, ensure_ascii=False), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
