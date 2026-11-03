"""見積もり: 隠し味で会話文の「口調」を変えられるか（数えた表で文を作り、2DWIG で話者の組を当て直す。M1 の口調の当たり、2026-10-24）

着想: est_flavor_counts.py で、隠し味は次の単位の正解率を +0.4 点しか上げなかった（CE は −0.12 改善）。正解率は中身（何を言うか）に引っ張られるので、
      会話の目的により近い「決めた話者の口調で話せるか」を別に測る。
方法: est_flavor_counts.py と同じデータ・サブワード（語彙 4096）・3-gram。話者の組 g5 ごとに、soft の数え方の表を使う。
      文の作り方: 直前 2 単位から次の 1 単位を、確率の上位 20 個から確率に比例して選ぶ（同じ単位の 3 連続は選ばない）。終わり（」）か 40 単位で止める。
      条件: none = 隠し味なし（全部の会話文の表だけ）、mix λ = λ × 組の表 + (1 − λ) × 全体の表（λ = 0.5・0.8・1.0）。各組 100 文 × seed 3。
      当て直し: 作った文の一人称を隠し、2DWIG（数えた表のナイーブベイズ、語尾 1〜3 + 2 つ組、本物の訓練の会話文で学んだもの）で 5 組のどれかを当てる。
指標: 口調の当たり = 決めた組と当て直した組が同じ割合（%）。比べる: none（決めた組に関係なく作るので、組の多さの偏りで当たる分）、
      本物の会話文での当て直しの正解率（judge_real_acc、当て直しの上限の目安）。
      あわせて、作った文にその組の一人称が出る割合、文が終わり（」）で閉じた割合、同じ単位の繰り返し（3-gram の重なり）。
結果: results/est_flavor_style.txt（1 条件 = 1 行の JSON）、例は docs/samples/m1_est_flavor_style.md
実行: python experiments/est_flavor_style.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aozora  # noqa: E402
import est_flavor_counts as efc  # noqa: E402

FIRST_OF = {"私系": ("私", "わたくし", "わたし", "我々", "私たち"), "僕": ("僕", "ぼく"), "俺系": ("俺", "おれ", "おいら", "おら", "あっし", "わっし"),
            "女ことば": ("あたし", "あたい", "わらわ", "妾", "わちき"), "古風": ("わし", "儂", "拙者", "それがし", "身共", "吾輩")}


def generate(sp, P_all, P_g, lam, rng, topk=20, max_len=40):
    """1 単位ずつ選んで会話文を 1 つ作る。(文字列, 終わりで閉じたか)。"""
    seq = [1, 1]
    for _ in range(max_len):
        p = P_all.prob(seq[-2], seq[-1])
        if P_g is not None:
            p = lam * P_g.prob(seq[-2], seq[-1]) + (1 - lam) * p
        p = p.copy()
        p[[0, 1]] = 0
        if len(seq) >= 4 and seq[-1] == seq[-2]:
            p[seq[-1]] = 0
        top = np.argsort(-p)[:topk]
        y = int(rng.choice(top, p=p[top] / p[top].sum()))
        if y == 2:
            return sp.decode(seq[2:]), True
        seq.append(y)
    return sp.decode(seq[2:]), False


def rep3(s):
    """文字の 3-gram のうち、同じ文の中で 2 回目以降に出たものの割合（繰り返しの目安）。"""
    g = [s[i:i + 3] for i in range(len(s) - 2)]
    return 1 - len(set(g)) / len(g) if g else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--V", type=int, default=4096)
    ap.add_argument("--n", type=int, default=100, help="組ごと・条件ごと・seed ごとの文の数")
    ap.add_argument("--set", default="1000", choices=["1000", "all"], help="青空文庫の作品の組（1000 = 最初の 1,000 作品、all = 全 4,477 作品）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--device", default="cpu", help="使わない（約束の引数）")
    ap.add_argument("--out", default="results/est_flavor_style.txt")
    ap.add_argument("--samples", default="docs/samples/m1_est_flavor_style.md")
    a = ap.parse_args()
    t0 = time.perf_counter()
    d = aozora.load(a.set)
    tr_w, _, te_w = efc.split_works(d)
    tr = efc.utter_items(d, tr_w)
    sp = efc.train_spm([u for u, *_ in tr], a.V, a.set)
    classes = aozora.CLASSES
    gmap = efc.GROUPS["g5"]
    groups = sorted(set(gmap.values()))
    nb = efc.NB([(u, c or s) for u, c, s, _ in tr if (c or s)], classes)
    nb.T = 8
    # 当て直し用の 2DWIG は、本物の一人称つきの会話文だけで学ぶ（広げたラベルを使わない、温度なし）
    judge = efc.NB([(u, c) for u, c, _, _ in tr if c], classes)
    to_g = np.zeros((len(classes), len(groups)))
    for ci, c in enumerate(classes):
        to_g[ci, groups.index(gmap[c])] = 1
    # 当て直しの上限の目安: 本物のテストの一人称つき会話文を、同じ 2DWIG で 5 組に当てた正解率
    te_lab = [(u, c) for u, c, _, _ in efc.utter_items(d, te_w) if c]
    judge_real = 100.0 * float(np.mean([groups[int(np.argmax(judge.post(u) @ to_g))] == gmap[c] for u, c in te_lab]))
    enc = {u: [1, 1] + sp.encode(u) + [2] for u, *_ in tr}
    P_all = efc.Tri(sp.get_piece_size())
    P_s = {g: efc.Tri(sp.get_piece_size()) for g in groups}
    for u, c, s, _ in tr:
        P_all.add(enc[u])
        w = nb.post(u) if not c else np.eye(len(classes))[classes.index(c)]
        for ci, wc in enumerate(w):
            if wc > 0.01:
                P_s[gmap[classes[ci]]].add(enc[u], float(wc))
    for t in [P_all, *P_s.values()]:
        t.finish()
    print(f"表を作った（{time.perf_counter() - t0:.0f} 秒）", flush=True)
    samples = {}
    for cond, lam in (("none", 0.0), ("mix0.5", 0.5), ("mix0.8", 0.8), ("mix1.0", 1.0)):
        per_seed = []
        for s in range(a.seed, a.seed + a.seeds):
            rng = np.random.default_rng(s)
            hit = n = closed = has_fp = 0
            reps = []
            per_group = {}
            for g in groups:
                hg = 0
                for i in range(a.n):
                    text, ok = generate(sp, P_all, None if cond == "none" else P_s[g], lam, rng)
                    pg = groups[int(np.argmax(judge.post(text) @ to_g))]
                    hg += pg == g
                    closed += ok
                    has_fp += any(f in text for f in FIRST_OF[g])
                    reps.append(rep3(text))
                    if s == a.seed and i < 8:
                        samples.setdefault((cond, g), []).append((text, pg, ok))
                per_group[g] = 100.0 * hg / a.n
                hit += hg
                n += a.n
            per_seed.append(dict(acc=100.0 * hit / n, closed=100.0 * closed / n, first_person=100.0 * has_fp / n,
                                 rep3=100.0 * float(np.mean(reps)), per_group=per_group))
        m = lambda k: float(np.mean([r[k] for r in per_seed]))
        se = lambda k: float(np.std([r[k] for r in per_seed], ddof=1) / np.sqrt(len(per_seed))) if len(per_seed) > 1 else None
        rec = dict(exp="est_flavor_style", set=a.set, cond=cond, lam=lam, V=a.V, groups="g5", n_per_group=a.n, seed=a.seed, n_seeds=a.seeds, out=a.out,
                   style_acc=m("acc"), style_acc_se=se("acc"), closed=m("closed"), first_person=m("first_person"), rep3=m("rep3"),
                   per_group={g: float(np.mean([r["per_group"][g] for r in per_seed])) for g in groups},
                   chance=100.0 / len(groups), judge_real_acc=judge_real, time_s=time.perf_counter() - t0)
        print(json.dumps(rec, ensure_ascii=False), flush=True)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    lines = ["# 隠し味で口調を変えた会話文の例（数えた表、est_flavor_style.py）", "",
             f"作り方: 青空文庫 1,000 作品の会話文で数えた 3-gram（サブワード 4096）から、上位 20 個を確率に比例して選ぶ。seed {a.seed}、条件ごと・組ごとに先頭の 8 文（選び直していない）。",
             "「当て直し」は、作った文の一人称を隠して 2DWIG（数えた表）が当てた組。✓ は決めた組と同じ。（閉じず）は 40 単位で止めた文。", ""]
    for cond in ("none", "mix0.8", "mix1.0"):
        lines += [f"## {cond}", ""]
        for g in groups:
            lines += [f"### 決めた組: {g}", ""]
            for text, pg, ok in samples[(cond, g)]:
                lines.append(f"- 「{text}」{'' if ok else '（閉じず）'} — 当て直し: {pg}{' ✓' if pg == g else ''}")
            lines.append("")
    with open(a.samples, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
