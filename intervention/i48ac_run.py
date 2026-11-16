"""I48ac の測り（i48ac_center_grain）。Tatoeba、学び／試しは I48ab と同じ分け方（文の id の割り算、試し 1/20）。学びの最後の 2,000 文は温度を選ぶ取りおき。

測る物（窓 W ごと）:
  詰め物（要点 → 文の間）: 焦点 = 内容語 i と i+1 の間の詰め物。まわり = 要点の内容語（左 −1 が内容語 i、右 +1 が内容語 i+1 …）と、文末なら意味の印（+1）。
  隠した機能語: 焦点 = 文の中の機能語 1 つ（原形|品詞の細目）。まわり = 文の語（内容語も機能語も、語の位置の数え）。
  隠した品詞: 焦点 = 文の語 1 つの品詞。まわり = 文の語。
基準: 隣の語そのものを鍵にした表（NeighborTable）、I48ab の 2 語の組の波・語の表・混ぜ（同じ試しの文で、i48ab_run.py の数）。

python intervention/i48ac_run.py --conllu ... --content ... --func ... --out out.json [--train N] [--test N] [--W 1 2 3 8] [--D 2048] [--tasks 詰め物 機能語 品詞]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i48ab_long_grain as G  # noqa: E402
import i48ac_center_grain as CG  # noqa: E402


def fill_items(S, s):
    """要点の間ごとに (まわり, 答え, 左の原形, 右の原形か印, 位置, 内容語の数)。"""
    cont, gaps, _ = G.split(s)
    if not cont:
        return []
    mk = S.func.marks(s["toks"])
    A = [S.attrs(l, u) for l, _, u in cont]
    marks = [S.attrs("印:" + m, "印") for m in sorted(mk)] + [S.attrs("印:文末", "印")]
    out = []
    n = len(cont)
    for i in range(n):
        ctx = [(-(i - j + 1), A[j]) for j in range(0, i + 1)]       # 左: 内容語 i が −1、i−1 が −2 …
        if i + 1 < n:
            ctx += [(j - i, A[j]) for j in range(i + 1, n)]           # 右: 内容語 i+1 が +1 …
            right = cont[i + 1][0]
        else:
            ctx += [(1, m) for m in marks]
            right = ("文末",) + tuple(sorted(mk))
        out.append((ctx, gaps[i], cont[i][0], right, i, n))
    return out


def tok_items(S, s, which):
    T = s["toks"]
    W_ = [S.tok(t) for t in T]
    out = []
    for i, t in enumerate(T):
        if which == "機能語":
            if G.is_content(t) or t["upos"] == "PUNCT":
                continue
            y = t["lemma"] + "|" + t["xpos"]
        else:
            y = t["upos"]
        ctx = [(j - i, W_[j]) for j in range(len(T)) if j != i]
        l = T[i - 1]["lemma"] if i > 0 else "^"
        r = T[i + 1]["lemma"] if i + 1 < len(T) else "$"
        if which == "品詞":
            l = T[i - 1]["upos"] if i > 0 else "^"
            r = T[i + 1]["upos"] if i + 1 < len(T) else "$"
        out.append((ctx, y, l, r, i, len(T)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conllu", required=True)
    ap.add_argument("--content", required=True)
    ap.add_argument("--func", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, default=0)
    ap.add_argument("--test", type=int, default=3000)
    ap.add_argument("--W", type=int, nargs="*", default=[1, 2, 3, 8])
    ap.add_argument("--D", type=int, default=2048)
    ap.add_argument("--tasks", nargs="*", default=["詰め物", "機能語", "品詞"])
    ap.add_argument("--n_cls", type=int, default=300)
    a = ap.parse_args()
    t0 = time.time()
    sents = [s for s in G.read_conllu(a.conllu) if s["toks"]]
    te = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 == 0][: a.test]
    tr = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 != 0]
    hold, tr = tr[-2000:], tr[:-2000]
    if a.train:
        tr = tr[: a.train]
    cb = json.load(open(a.content, encoding="utf-8"))
    func = G.Func(json.load(open(a.func, encoding="utf-8")))
    S = G.AttrSpace(cb, func, a.D, bands=("B", "C", "D", "E", "G"))
    C = CG.codes(a.D)
    train_lemmas = {t["lemma"] for s in tr for t in s["toks"]}
    res = dict(学びの文=len(tr), 試しの文=len(te), D=a.D, 手で置いた物=dict(窓=a.W, 温度の候補=[1, 3, 10, 30, 100, 300], 板の数=a.n_cls), 課題={})
    print(f"読み込み {time.time() - t0:.0f} 秒、学び {len(tr)} 文", flush=True)
    for task in a.tasks:
        items = (lambda s: fill_items(S, s)) if task == "詰め物" else (lambda s, w=task: tok_items(S, s, w))
        cnt = Counter()
        nt = CG.NeighborTable()
        for s in tr:
            for ctx, y, l, r, i, n in items(s):
                cnt[y] += 1
                nt.add(l, r, y)
        classes = [y for y, _ in cnt.most_common(a.n_cls)]
        T_items = [x for s in te for x in items(s)]
        H_items = [(x[0], x[1]) for s in hold for x in items(s)][:3000]
        base = cnt.most_common(1)[0][0]
        o = dict(板の数=len(classes), 試しの所=len(T_items), いちばん多い物=[base, round(sum(x[1] == base for x in T_items) / max(1, len(T_items)), 3)],
                 隣の語の表=round(sum(nt.predict(x[2], x[3]) == x[1] for x in T_items) / max(1, len(T_items)), 3), 窓ごと={})
        for W in a.W:
            t1 = time.time()
            cg = CG.CenterGrain(S, W, classes, C)
            for s in tr:
                for ctx, y, l, r, i, n in items(s):
                    cg.add(y, ctx)
            cg.build()
            temp = cg.fit_temp(H_items)
            hit = Counter()
            for ctx, y, l, r, i, n in T_items:
                p = cg.predict(ctx)
                hit["n"] += 1
                hit["当たり"] += p == y
                if task == "詰め物":
                    lb = "1〜3" if n <= 3 else ("4〜6" if n <= 6 else "7 以上")
                else:
                    lb = "1〜8" if n <= 8 else ("9〜14" if n <= 14 else "15 以上")
                hit[lb + ":n"] += 1
                hit[lb + ":当たり"] += p == y
                if task == "詰め物" and (l not in train_lemmas or (isinstance(r, str) and r not in train_lemmas)):
                    hit["無:n"] += 1
                    hit["無:当たり"] += p == y
                    hit["無:表"] += nt.predict(l, r) == y
                if task == "詰め物" and isinstance(r, tuple):
                    hit["文末:n"] += 1
                    hit["文末:当たり"] += p == y
                    hit["文末:表"] += nt.predict(l, r) == y
            ow = dict(秒=round(time.time() - t1), 温度=temp, 当たり=round(hit["当たり"] / max(1, hit["n"]), 3),
                      長さごと={k.split(":")[0]: round(hit[k.split(":")[0] + ":当たり"] / max(1, hit[k]), 3) for k in hit if k.endswith(":n") and k[0].isdigit()})
            if task == "詰め物":
                ow["学びに無い語を含む所"] = dict(件数=hit["無:n"], 中心のグレイン=round(hit["無:当たり"] / max(1, hit["無:n"]), 3), 隣の語の表=round(hit["無:表"] / max(1, hit["無:n"]), 3))
                ow["文末"] = dict(件数=hit["文末:n"], 中心のグレイン=round(hit["文末:当たり"] / max(1, hit["文末:n"]), 3), 隣の語の表=round(hit["文末:表"] / max(1, hit["文末:n"]), 3))
                ex = []
                for keys, mk in (((("昨日", "NOUN"), ("犬", "NOUN"), ("散歩", "NOUN"), ("行く", "VERB")), {"過去"}),
                                 ((("昨日", "NOUN"), ("友達", "NOUN"), ("散歩", "NOUN"), ("行く", "VERB")), {"過去"}),
                                 ((("昨日", "NOUN"), ("公園", "NOUN"), ("犬", "NOUN"), ("散歩", "NOUN"), ("する", "VERB")), {"過去"})):
                    s_ = dict(toks=[dict(lemma=l, form=l, upos=u, xpos="", dep="") for l, u in keys])
                    A = [S.attrs(l, u) for l, u in keys]
                    marks = [S.attrs("印:" + m, "印") for m in sorted(mk)] + [S.attrs("印:文末", "印")]
                    fills = []
                    for i in range(len(keys)):
                        ctx = [(-(i - j + 1), A[j]) for j in range(0, i + 1)]
                        ctx += [(j - i, A[j]) for j in range(i + 1, len(keys))] if i + 1 < len(keys) else [(1, m) for m in marks]
                        fills.append(cg.predict(ctx))
                    ex.append(dict(要点=[k for k, _ in keys] + sorted(mk), 詰め物=fills))
                ow["持ち主の例"] = ex
            o["窓ごと"][str(W)] = ow
            print(task, W, json.dumps(ow, ensure_ascii=False), flush=True)
        res["課題"][task] = o
        print(task, json.dumps({k: v for k, v in o.items() if k != "窓ごと"}, ensure_ascii=False), flush=True)
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
