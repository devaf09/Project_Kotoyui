"""介入の研究 実験I10: 属性に向き（親子）を作り、知らない子（新しい品種）の属性を親から確率つきで想定する

持ち主の案（2026-11-05）:
  「属性に向きが作れるといいかも。リンゴとリンゴの品種のような親と子。文章では『○○の△△』とあれば △△ は ○○ を親に持つ。
   住所なら日本 → 宮崎 → 日南。日本に文化:寿司があれば、宮崎にも多少の関係がある。
   リンゴ科（フジ: 赤・甘い、紅玉: 赤・酸っぱい、王林: 緑・甘酸っぱい）→ リンゴ科は知っている限り甘い・酸っぱい・甘酸っぱい・赤・緑。
   矛盾を解消しつつ、別のリンゴ科の想定で誤反応を防ぐ。自覚や不安は確率で持つ（何でも『たぶん』にしない）」。
方法:
  - 世界: 科（親）4 つ × 品種（子）5 つ。品種の属性（色・味・形）は実在の品種を参考にした手作りの簡略版（正確な品種の性質ではない）。
    文: 「リンゴ の フジ は 赤い 。」「フジ は 甘い 。」など。「X の Y」で X が科の語なら、Y の親を X とする（手で与える規則）。品種 1 つあたり --n 文。
  - 子の語板: 光の器（口 1 つ、I2 の marg と同じ学び方）。出口の明るさ = 属性の割合。
  - 親の語板: 子の器の明るさを足し合わせる（光の融合 = 明るさの和。Sin-CNN3 の融合と同じ）。親は子の属性の幅を持つ（甘い・酸っぱい・赤・緑…）。
    祖先（果物全体）: 全部の科の和。
  - 知らない子: 科ごとに品種を 1 つ隠し（その品種の文は「リンゴ の シナノ」のような親を示す文 1 つだけ）、親の明るさから属性を確率で想定する。
    比べ: 果物全体から（祖先）、別の科から（誤った親）、一様。
  - 物差し: 観点ごとに、本当の値に付けた確率（平均）、いちばん確率の高い値の正解率、確率の較正（「きっと / たぶん / わからない」の区切りごとの正解率）。
    言い方: 確率 ≥ 0.7 → きっと、0.4〜0.7 → たぶん、< 0.4 → わからない（持ち主の「何でもたぶんにしない」）。
  - 住所の例（日本 → 宮崎 → 日南）: 親から子へ、深さごとに薄めて（× 0.5）受け継ぐ例を表で示す（数字の実験ではなく形の確かめ）。
結果: results/i10.jsonl
実行: CUDA_VISIBLE_DEVICES= python intervention/i10_hierarchy.py --seed 0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from i02_reverse import PortDevice, VAL0  # noqa: E402

VIEWS = {"色": ["赤", "緑", "黄", "紫", "橙"], "味": ["甘い", "酸っぱい", "渋い"], "形": ["丸い", "長い", "小さい"]}
SLOTS = [(v, x) for v, xs in VIEWS.items() for x in xs]
SLOT = {s: i for i, s in enumerate(SLOTS)}
TOK = {("色", "赤"): "赤い", ("色", "緑"): "緑の", ("色", "黄"): "黄色い", ("色", "紫"): "紫の", ("色", "橙"): "橙の",
       ("味", "甘い"): "甘い", ("味", "酸っぱい"): "酸っぱい", ("味", "渋い"): "渋い",
       ("形", "丸い"): "丸い", ("形", "長い"): "長い", ("形", "小さい"): "小さい"}
FAMILIES = {   # 科 → 品種 → 属性（簡略化した手作り）
    "リンゴ": {"フジ": {"色": ["赤"], "味": ["甘い"], "形": ["丸い"]},
             "紅玉": {"色": ["赤"], "味": ["酸っぱい"], "形": ["丸い"]},
             "王林": {"色": ["緑"], "味": ["甘い", "酸っぱい"], "形": ["丸い"]},
             "ジョナゴールド": {"色": ["赤", "黄"], "味": ["甘い", "酸っぱい"], "形": ["丸い"]},
             "シナノ": {"色": ["黄"], "味": ["甘い", "酸っぱい"], "形": ["丸い"]}},
    "ブドウ": {"巨峰": {"色": ["紫"], "味": ["甘い"], "形": ["丸い"]},
             "マスカット": {"色": ["緑"], "味": ["甘い"], "形": ["丸い"]},
             "デラウェア": {"色": ["紫"], "味": ["甘い", "酸っぱい"], "形": ["小さい"]},
             "ピオーネ": {"色": ["紫"], "味": ["甘い"], "形": ["丸い"]},
             "甲州": {"色": ["紫"], "味": ["酸っぱい"], "形": ["小さい"]}},
    "カンキツ": {"ウンシュウ": {"色": ["橙"], "味": ["甘い"], "形": ["丸い"]},
              "レモン": {"色": ["黄"], "味": ["酸っぱい"], "形": ["長い"]},
              "ユズ": {"色": ["黄"], "味": ["酸っぱい"], "形": ["丸い"]},
              "スダチ": {"色": ["緑"], "味": ["酸っぱい"], "形": ["小さい"]},
              "デコポン": {"色": ["橙"], "味": ["甘い", "酸っぱい"], "形": ["丸い"]}},
    "カキ": {"富有": {"色": ["橙"], "味": ["甘い"], "形": ["丸い"]},
            "次郎": {"色": ["橙"], "味": ["甘い"], "形": ["丸い"]},
            "平核無": {"色": ["橙"], "味": ["渋い"], "形": ["丸い"]},
            "西条": {"色": ["橙"], "味": ["渋い"], "形": ["長い"]},
            "太秋": {"色": ["橙", "緑"], "味": ["甘い"], "形": ["丸い"]}},
}


def make_sentences(child, fam, attrs, n, rng):
    out = []
    for _ in range(n):
        v = rng.choice([v for v in attrs])
        x = rng.choice(attrs[v])
        if v == "味":
            toks = [child, "は"] + (["甘くて", "酸っぱい"] if len(attrs["味"]) == 2 and rng.random() < 0.5 else [TOK[(v, x)]]) + ["。"]
        else:
            toks = [TOK[(v, x)], child, "は", "おいしい", "。"]
        if rng.random() < 0.3:
            toks = [fam, "の"] + toks                                 # 「リンゴ の フジ」親を示す
        out.append(toks)
    return out


def read(toks, families):
    """文から（子の語, 親の語 or None, 属性のリスト）。属性は手の対応表、親は「X の Y」の規則。"""
    inv = {t: s for s, t in TOK.items()}
    attrs = [inv[t] for t in toks if t in inv]
    if "甘くて" in toks:
        attrs.append(("味", "甘い"))
    parent = None
    if len(toks) > 2 and toks[1] == "の" and toks[0] in families:
        parent, rest = toks[0], toks[2:]
    else:
        rest = toks
    child = next(t for t in rest if t not in inv and t not in ("は", "の", "。", "おいしい", "甘くて"))
    return child, parent, attrs


def fit_board(counts, seed, steps):
    torch.manual_seed(seed)
    dev = PortDevice(1, seed=seed)
    q = torch.tensor(counts / counts.sum(), dtype=torch.float32)
    opt = torch.optim.Adam(dev.parameters(), lr=0.02)
    for _ in range(steps):
        W = (dev.forward_field(dev.ports).abs()[:, VAL0:VAL0 + len(SLOTS)] ** 2)[0]
        p = torch.cat([W, (1 - W.sum()).clamp(min=1e-9)[None]])
        loss = -(torch.cat([q, torch.zeros(1)]) * torch.log(p + 1e-9)).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return (dev.forward_field(dev.ports).abs()[:, VAL0:VAL0 + len(SLOTS)] ** 2)[0].numpy()


def view_prob(b, v):
    ids = [SLOT[(v, x)] for x in VIEWS[v]]
    p = b[ids] + 1e-6
    return p / p.sum()


def word_of(p):
    return "きっと" if p >= 0.7 else ("たぶん" if p >= 0.4 else "わからない")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=10, help="品種 1 つあたりの文")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--out", default="intervention/results/i10.jsonl")
    a = ap.parse_args()
    torch.set_num_threads(2)
    rng = random.Random(a.seed)
    t0 = time.perf_counter()
    fams = list(FAMILIES)
    res = defaultdict(list)                                         # 方法 → [(観点, 本当の値の確率, 1 位が正解か, 1 位の確率)]
    examples = []
    for hold_i in range(5):                                         # 科ごとに i 番目の品種を隠す（5 回）
        boards, parent_of = {}, {}
        hidden = {f: list(FAMILIES[f])[hold_i] for f in fams}
        for f in fams:
            for c, at in FAMILIES[f].items():
                if c == hidden[f]:
                    parent_of[c] = f                                # 隠した品種は「リンゴ の シナノ」だけ見る
                    continue
                cnt = np.zeros(len(SLOTS))
                for toks in make_sentences(c, f, at, a.n, rng):
                    ch, par, attrs = read(toks, fams)
                    if par:
                        parent_of[ch] = par
                    for s in attrs:
                        cnt[SLOT[s]] += 1
                boards[c] = fit_board(cnt, a.seed * 100 + hold_i, a.steps)
        # 親 = 子の明るさの和（光の融合）。親を示す文が無かった子は、親に入らない
        par_b = {f: sum((boards[c] for c in boards if parent_of.get(c) == f), np.zeros(len(SLOTS))) for f in fams}
        root = sum(par_b.values())
        for f in fams:
            c = hidden[f]
            if parent_of.get(c) != f:
                continue
            wrong = fams[(fams.index(f) + 1) % len(fams)]
            for v in VIEWS:
                truth = FAMILIES[f][c][v]
                n_kids = sum(1 for c2 in boards if parent_of.get(c2) == f)
                shrink = par_b[f] + root / len(fams) * (1.0 / max(n_kids, 1)) * 4   # 子が少ないほど祖先で薄める（子 4 つなら果物 1 科ぶん）
                for name, b in (("親", par_b[f]), ("親 + 祖先で薄める", shrink), ("祖先（果物全体）", root), ("誤った親", par_b[wrong]), ("一様", np.ones(len(SLOTS)))):
                    p = view_prob(b, v)
                    pt = sum(p[VIEWS[v].index(x)] for x in truth)
                    top = int(np.argmax(p))
                    res[name].append((v, float(pt), VIEWS[v][top] in truth, float(p[top])))
                p = view_prob(par_b[f], v)
                top = int(np.argmax(p))
                examples.append(dict(fam=f, child=c, view=v, truth=truth, pred={VIEWS[v][j]: round(float(p[j]), 2) for j in np.argsort(-p)[:3]},
                                     say=f"{word_of(float(p[top]))} {VIEWS[v][top]}"))
    summary = {}
    for name, rows in res.items():
        by = defaultdict(list)
        for v, pt, ok, ptop in rows:
            by[v].append((pt, ok))
        calib = defaultdict(list)
        for v, pt, ok, ptop in rows:
            calib[word_of(ptop)].append(ok)
        summary[name] = dict(p_true={v: float(np.mean([x[0] for x in r])) for v, r in by.items()},
                             top1={v: float(np.mean([x[1] for x in r])) for v, r in by.items()},
                             calib={k: (float(np.mean(v)), len(v)) for k, v in calib.items()})
    # 住所の例: 親から子へ、深さごとに 0.5 倍で受け継ぐ（形の確かめ）
    tree = {"日本": {"文化:寿司": 1.0}, "宮崎": {"料理:チキン南蛮": 1.0}, "日南": {"魚:カツオ": 1.0}}
    chain = ["日本", "宮崎", "日南"]
    inherit = {}
    for i, place in enumerate(chain):
        acc = defaultdict(float)
        for j, anc in enumerate(chain[:i + 1]):
            for k, w in tree[anc].items():
                acc[k] += w * 0.5 ** (i - j)
        inherit[place] = dict(acc)
    rec = dict(exp="I10", seed=a.seed, n=a.n, summary=summary, examples=examples[:24], address=inherit, total_s=time.perf_counter() - t0)
    print(json.dumps({k: v for k, v in rec.items() if k != "examples"}, ensure_ascii=False), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
