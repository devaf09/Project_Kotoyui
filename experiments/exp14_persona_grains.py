"""実験14（非公開の材料）: 一人称の類ごとの語板を、共通の語板と明るさで足して切り替える（持ち主の案 ①、2026-11-03）

着想: 持ち主「一人称と相手の立ち位置から会話の流れを作るには、学習段階で一人称キャラに合わせたバイアス付きで学習するのが良い」。
      焼き込まず、融合（明るさの重み付きの足し算、sincnn3 の FusedGrain・実験09 の語板）で、キャラクタごとに重みを付けて足す形にする。
方法: 材料 = build_persona_pairs.py の組（RealPersonaChat + 持ち主が集めた Web 小説の会話文。教えだけで数える）。
      共通の表 = 全部の組（重み付き）、類 c の表 = 返事の一人称が c の組だけ。どちらも正の PMI（回数 2 以上）を、語ごとに上位 61 語に切る。
      語板 = kotoyui.LightResonance（1 語 = 1 台の光の器、縦 64・板 16 枚）を共通と類ごとに学ぶ。
      融合: R_w(x, y) = (1 − w) × 共通の器の明るさ + w × 類 c の器の明るさ（どちらも語ごとの和を掛け戻した PMI の単位）。
測る: 試しの組のうち、返事の一人称が c の問で 10 択。間違いの 9 つは (S) 同じ類の返事から、(O) ほかの類の返事から。
      方法: 共通だけ・融合（w = 0.3, 0.5, 1.0）・違う類で融合（対照）、表と光の両方。
例: 試しの前の発言 100 個に、3 人のキャラクタ（私・丁寧／俺・くだけた／僕・丁寧）で返事を選ぶ（候補 = 教えの返事のうち、その類で口調の合う物）。
結果: results/exp14.txt、報告 docs/exp14_report.md（非公開。材料の文を含む）
実行（Project_Kotoyui の直下で）: CUDA_VISIBLE_DEVICES= python experiments/exp14_persona_grains.py
"""
import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict

KOTO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PRIV = os.environ.get("KOTOYUI_PRIVATE", os.path.join(KOTO_ROOT, "..", "Project_Kotoyui_private"))   # 非公開の置き場（資料・結果）
KOTO = KOTO_ROOT
sys.path.insert(0, KOTO)
import kotoyui as K  # noqa: E402

PAIRS = os.path.join(PRIV, "data", "persona", "pairs.jsonl")
CLASSES = ["私", "僕", "俺", "あたし", "わし"]


class Table:
    """数えた表（LightResonance.build に渡す形: cpair と pmi(x, y)）。"""

    def __init__(self, pairs, min_x=10):
        cp, cx, cy, n = Counter(), Counter(), Counter(), 0.0
        for xs, ys, w in pairs:
            n += w
            for x in xs:
                cx[x] += w
            for y in ys:
                cy[y] += w
            for x in xs:
                for y in ys:
                    cp[(x, y)] += w
        self.n, self.cx, self.cy = n, cx, cy
        self.cpair = {k: v for k, v in cp.items() if v >= 2 and cx[k[0]] >= min_x}

    def pmi(self, x, y):
        c = self.cpair.get((x, y), 0)
        return max(0.0, math.log(c * self.n / (self.cx[x] * self.cy[y]))) if c >= 2 else 0.0


def topk(table, k=61):
    """語ごとに上位 k 語（回数 × PMI の大きい順）に切った PMI の表 {x: {y: pmi}}（実験09 の表の版）。"""
    by = defaultdict(list)
    for (x, y), c in table.cpair.items():
        p = table.pmi(x, y)
        if p > 0:
            by[x].append((p * c, y, p))
    return {x: {y: p for _, y, p in sorted(v, reverse=True)[:k]} for x, v in by.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_q", type=int, default=1000, help="類ごとの試しの問の数（上限）")
    ap.add_argument("--max_train", type=int, default=1200000, help="共通の表に使う教えの組の数（上限、類つきの組は全部使う）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(PRIV, "results", "exp14.txt"))
    ap.add_argument("--report", default=os.path.join(PRIV, "docs", "exp14_report.md"))
    a = ap.parse_args()
    rng = random.Random(a.seed)
    t0 = time.perf_counter()
    rows = [json.loads(l) for l in open(PAIRS, encoding="utf-8")]
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    rng.shuffle(train)
    base_rows = [r for r in train if r["cls"]] + [r for r in train if not r["cls"]][:a.max_train]
    print(f"組 教え {len(train)}（共通の表に {len(base_rows)}）・試し {len(test)}（{time.perf_counter() - t0:.0f} 秒）", flush=True)
    aw = lambda t: K.analyze(t)[0]
    tr = [(aw(r["prev"]), aw(r["reply"]), r["w"], r["cls"]) for r in base_rows]
    print(f"解析 {time.perf_counter() - t0:.0f} 秒", flush=True)
    tabs = {"共通": Table([(x, y, w) for x, y, w, _ in tr], min_x=20)}
    for c in CLASSES:
        tabs[c] = Table([(x, y, w) for x, y, w, cc in tr if cc == c], min_x=5)
    tops = {k: topk(t) for k, t in tabs.items()}
    for k in tabs:
        print(k, "組", sum(1 for *_, cc in tr if k == "共通" or cc == k), "語", len(tops[k]), flush=True)
    grains = {}
    for k, t in tabs.items():                                          # 語板（光）を学ぶ
        grains[k] = K.LightResonance().build(t, n=64, plates=16, steps=500, log=lambda *_: None)
        print(f"語板 {k}: {len(grains[k].words)} 台（{time.perf_counter() - t0:.0f} 秒）", flush=True)

    def lift(kind, k, x):
        return grains[k].lift(x) if kind == "light" else tops[k].get(x, {})

    def res(uw, cw_, kind, c, w):
        if not uw or not cw_:
            return 0.0
        s = 0.0
        for x in uw:
            b = lift(kind, "共通", x)
            p = lift(kind, c, x) if (c and w > 0) else {}
            for y in cw_:
                s += (1 - w) * b.get(y, 0.0) + w * p.get(y, 0.0)
        return s / math.sqrt(len(uw) * len(cw_))

    # 10 択
    by_cls = defaultdict(list)
    for r in test:
        if r["cls"] in CLASSES:
            by_cls[r["cls"]].append(r)
    rec = dict(exp="14", **{k: v for k, v in vars(a).items() if k not in ("out", "report")}, runs=[],
               n_pairs=dict(train=len(train), base=len(base_rows), test=len(test)),
               n_class_train={c: sum(1 for *_, cc in tr if cc == c) for c in CLASSES},
               grains={k: len(g.words) for k, g in grains.items()}, version=K.__version__)
    other = {"私": "俺", "僕": "俺", "俺": "私", "あたし": "俺", "わし": "私"}
    for c in CLASSES:
        qs = by_cls[c][:]
        rng.shuffle(qs)
        qs = qs[:a.n_q]
        if len(qs) < 50:
            continue
        pool_same = [r["reply"] for r in by_cls[c]]
        pool_other = [r["reply"] for cc in CLASSES if cc != c for r in by_cls[cc]]
        sets = {}
        for tag, pool in (("S", pool_same), ("O", pool_other)):
            sets[tag] = [(aw(q["prev"]), [aw(q["reply"])] + [aw(t) for t in rng.sample([p for p in pool if p != q["reply"]], 9)]) for q in qs]
        for kind in ("table", "light"):
            for name, cc, w in (("共通だけ", None, 0.0), ("融合 w0.3", c, 0.3), ("融合 w0.5", c, 0.5), ("類だけ w1.0", c, 1.0), (f"違う類（{other[c]}）で融合 w0.5", other[c], 0.5)):
                r = dict(cls=c, kind=kind, name=name, w=w, n=len(qs))
                for tag, qq in sets.items():
                    ok = 0
                    for uw, cands in qq:
                        s = [res(uw, cw_, kind, cc, w) + 1e-6 * rng.random() for cw_ in cands]
                        ok += all(s[0] > v for v in s[1:])
                    r[tag] = ok / len(qq)
                rec["runs"].append(r)
                print(c, kind, name, {t: round(r[t], 3) for t in ("S", "O")}, flush=True)
    # 例 100 通り
    personas = [("私・丁寧", "私", True), ("俺・くだけた", "俺", False), ("僕・丁寧", "僕", True)]
    pools = {}
    for name, c, polite in personas:
        cand = list(dict.fromkeys(r["reply"] for r in train if r["cls"] == c and 4 <= len(r["reply"]) <= 60 and K.analyze(r["reply"])[1] == polite))
        inv = defaultdict(list)
        for i, t in enumerate(cand):
            for wd in aw(t):
                inv[wd].append(i)
        pools[name] = (cand, inv)
        print("候補", name, len(cand), flush=True)
    prompts = [r["prev"] for r in test if r["src"] == "rpc" and 6 <= len(r["prev"]) <= 50]
    prompts_web = [r["prev"] for r in test if r["src"] == "web" and 6 <= len(r["prev"]) <= 50]
    rng.shuffle(prompts)
    rng.shuffle(prompts_web)
    prompts = prompts[:50] + prompts_web[:50]
    ex = []
    for u in prompts:
        uw = aw(u)
        row = [u]
        for name, c, polite in personas:
            cand, inv = pools[name]
            ids = Counter()
            for wd in uw:
                for i in inv.get(wd, [])[:2000]:
                    ids[i] += 1
            for x in uw:                                              # 共鳴で光った語を持つ候補も
                for y, _ in sorted(grains[c].lift(x).items(), key=lambda kv: -kv[1])[:5]:
                    for i in inv.get(y, [])[:300]:
                        ids[i] += 0
            best = max(ids, key=lambda i: K.overlap(uw, aw(cand[i])) + res(uw, aw(cand[i]), "light", c, 0.5), default=None)
            row.append(cand[best] if best is not None else "（候補なし）")
        ex.append(row)
    rec["total_s"] = time.perf_counter() - t0
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.makedirs(os.path.dirname(a.report), exist_ok=True)
    with open(a.report.replace(".md", "_examples.json"), "w", encoding="utf-8") as f:
        json.dump(dict(personas=[p[0] for p in personas], rows=ex), f, ensure_ascii=False, indent=1)
    print("終わり", f"{rec['total_s']:.0f} 秒", flush=True)


if __name__ == "__main__":
    main()
