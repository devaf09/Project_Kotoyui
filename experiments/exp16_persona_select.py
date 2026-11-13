"""実験16（非公開の材料）: 光の 2DWIG の判定で、一人称の無い返事も人格の候補にして返事を選ぶ（a の続き、2026-11-03）

方法: 候補 = 教えの返事（4〜60 文字、重なりは 1 回）から 20 万。実験15 の光の 2DWIG（wig_persona.pt・wig_polite.pt）で、
      一人称を隠した返事が「どの類の人が言いそうか」の確率と、丁寧さの確率を出しておく。
      人格 c（一人称 c・口調）の候補 = 一人称の表記が無いか c のもの、かつ 2DWIG の丁寧さが口調と合うもの。
      点 = 語の重なり + 共鳴（実験14 の共通の語板、光） + λ × log P(c)（2DWIG の人格の確率）。
比べる: 実験14 の選び方（候補 = 一人称 c を含む返事だけ、点 = 重なり + 融合した共鳴）。
測る: 選んだ返事のうち一人称の無い返事の割合、2DWIG の P(c) の平均。例 100 通り（実験14 と同じ相手の発言）を報告に。
結果: ../results/exp16.txt、../docs/exp16_report.md（非公開）
実行（Project_Kotoyui の直下で）: CUDA_VISIBLE_DEVICES= python experiments/exp16_persona_select.py
"""
import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

import torch

KOTO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PRIV = os.environ.get("KOTOYUI_PRIVATE", os.path.join(KOTO_ROOT, "..", "Project_Kotoyui_private"))   # 非公開の置き場（資料・結果）
KOTO = KOTO_ROOT
sys.path.insert(0, KOTO)
sys.path.insert(0, os.path.join(KOTO, "experiments"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kotoyui as K  # noqa: E402
from exp01_wig_bag import L, bag, p2, to_field  # noqa: E402
from exp14_persona_grains import CLASSES, Table, PAIRS  # noqa: E402
from exp15_persona_wig import hide, HIDE  # noqa: E402
from build_persona_pairs import cls_of  # noqa: E402


def load_wig(task):
    d = torch.load(os.path.join(PRIV, "models", f"wig_{task}.pt"))
    net = p2.Net2D(L, len(d["classes"]), ["xy"] * d["layers"], patches=d["patches"], seed=d["seed"])
    net.load_state_dict(d["state"])
    net.eval()
    return net, d["classes"]


@torch.no_grad()
def probs(net, texts, bs=2000):
    out = []
    for i in range(0, len(texts), bs):
        out.append(torch.softmax(net(to_field(bag([hide(t) for t in texts[i:i + bs]]))), 1))
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=200000)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    t0 = time.perf_counter()
    rows = [json.loads(l) for l in open(PAIRS, encoding="utf-8")]
    train = [r for r in rows if r["split"] == "train"]
    # 共鳴（実験14 と同じ共通の表・語板。材料が同じなので作り直す）
    rng2 = random.Random(0)
    tr = train[:]
    rng2.shuffle(tr)
    base_rows = [r for r in tr if r["cls"]] + [r for r in tr if not r["cls"]][:1200000]
    aw = lambda t: K.analyze(t)[0]
    tab = Table([(aw(r["prev"]), aw(r["reply"]), r["w"]) for r in base_rows], min_x=20)
    grain = K.LightResonance().build(tab, n=64, plates=16, steps=500, log=lambda *_: None)
    print(f"共通の語板 {len(grain.words)} 台（{time.perf_counter() - t0:.0f} 秒）", flush=True)

    def res(uw, cw_):
        if not uw or not cw_:
            return 0.0
        return sum(grain.lift(x).get(y, 0.0) for x in uw for y in cw_) / math.sqrt(len(uw) * len(cw_))
    cand = list(dict.fromkeys(r["reply"] for r in train if 4 <= len(r["reply"]) <= 60))
    rng.shuffle(cand)
    cand = cand[:a.pool]
    wp, pcls = load_wig("persona")
    wq, qcls = load_wig("polite")
    P = probs(wp, cand)
    Q = probs(wq, cand)[:, qcls.index("丁寧")]
    explicit = [cls_of(t) for t in cand]
    words = [aw(t) for t in cand]
    inv = defaultdict(list)
    for i, ws in enumerate(words):
        for w in ws:
            inv[w].append(i)
    print(f"候補 {len(cand)}、一人称あり {sum(e is not None for e in explicit)}（{time.perf_counter() - t0:.0f} 秒）", flush=True)
    ex14 = json.load(open(os.path.join(PRIV, "docs", "exp14_report_examples.json"), encoding="utf-8"))
    prompts = [r[0] for r in ex14["rows"]]
    personas = [("私・丁寧", "私", True), ("俺・くだけた", "俺", False), ("僕・丁寧", "僕", True)]
    out_rows, stats = [], {p[0]: Counter() for p in personas}
    for u in prompts:
        uw = aw(u)
        ids = set()
        for w in uw:
            ids.update(inv.get(w, [])[:3000])
        row = [u]
        for name, c, polite in personas:
            ci = pcls.index(c)
            best, bs = None, -1e9
            for i in ids:
                if explicit[i] not in (None, c):
                    continue
                if (Q[i] > 0.5) != polite:
                    continue
                s = K.overlap(uw, words[i]) + res(uw, words[i]) + a.lam * math.log(float(P[i, ci]) + 1e-6)
                if s > bs:
                    best, bs = i, s
            if best is None:
                row.append("（候補なし）")
                stats[name]["none"] += 1
            else:
                row.append(cand[best])
                stats[name]["explicit" if explicit[best] else "implicit"] += 1
                stats[name]["p_sum"] += float(P[best, ci])
        out_rows.append(row)
    rec = dict(exp="16", **vars(a), n_pool=len(cand), stats={k: dict(v) for k, v in stats.items()}, total_s=time.perf_counter() - t0)
    print(rec["stats"], flush=True)
    with open(os.path.join(PRIV, "results", "exp16.txt"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(os.path.join(PRIV, "docs", "exp16_report_examples.json"), "w", encoding="utf-8") as f:
        json.dump(dict(personas=[p[0] for p in personas], rows=out_rows), f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
