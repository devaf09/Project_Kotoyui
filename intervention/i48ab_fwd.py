"""I48ab: 入口の波の置き方（i48ab_place）と D（圧縮を含む）の比べ。順方向（文の波 → 位置ごとの語・意味の印）だけを測る。

python intervention/i48ab_fwd.py --conllu tatoeba_ginza.conllu --content v2_3b/content_boards.json --func func_boards.json --out out.json
  [--test 300] [--lex 5000] [--D 512 1024 2048 4096 8192] [--modes 頭から 拡散 ...] [--topk 4]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i48ab_long_grain as G  # noqa: E402
import i48ab_place as PL  # noqa: E402

POS_MAP = {"名詞": "NOUN", "動詞": "VERB", "形容詞": "ADJ"}


def lbin(n):
    return "1〜8" if n <= 8 else ("9〜14" if n <= 14 else "15 以上")


def pbin(k):
    return str(k) if k < 10 else "10 以上"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conllu", required=True)
    ap.add_argument("--content", required=True)
    ap.add_argument("--func", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--lex", type=int, default=5000)
    ap.add_argument("--D", type=int, nargs="*", default=[512, 1024, 2048, 4096, 8192])
    ap.add_argument("--modes", nargs="*", default=list(PL.MODES))
    ap.add_argument("--topk", type=int, default=G.TOPK)
    a = ap.parse_args()
    sents = [s for s in G.read_conllu(a.conllu) if s["toks"]]
    te = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 == 0][: a.test]
    tr = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 != 0]
    pos_n = Counter()
    for s in tr:
        for k in range(len(s["toks"])):
            pos_n[pbin(k)] += 1
    cb = json.load(open(a.content, encoding="utf-8"))
    func = G.Func(json.load(open(a.func, encoding="utf-8")))
    lex = [dict(lemma=w["name"], upos=POS_MAP.get(w["pos"], "NOUN"), xpos="", form=w["name"])
           for w in sorted(cb, key=lambda x: -x.get("freq", 0))[: a.lex]]
    seen = Counter()
    for s in tr[:50000]:
        for t in s["toks"]:
            if t["upos"] not in POS_MAP.values() or not G.is_content(t):
                seen[(t["lemma"], t["upos"], t["xpos"], t["form"] if t["upos"] == "PUNCT" else t["lemma"])] += 1
    lex += [dict(lemma=l, upos=u, xpos=x, form=f) for (l, u, x, f), _ in seen.most_common(1500)]
    lexset = {t["lemma"] for t in lex}
    tot = sum(pos_n.values())
    res = dict(試しの文=len(te), 候補の語=len(lex), 学びの位置ごとの語の割合={k: round(v / tot, 3) for k, v in sorted(pos_n.items(), key=lambda z: (len(z[0]), z[0]))},
               試しの内容語のうち候補にある割合=None, 版={})
    cov = Counter()
    for s in te:
        for t in s["toks"]:
            if G.is_content(t):
                cov["n"] += 1
                cov["in"] += t["lemma"] in lexset
    res["試しの内容語のうち候補にある割合"] = round(cov["in"] / max(1, cov["n"]), 3)
    for D in a.D:
        S = G.AttrSpace(cb, func, D, bands=("A", "B", "C", "D", "E", "G"), topk=a.topk)
        for mode in a.modes:
            t0 = time.time()
            P = PL.Placer(S, mode)
            L = np.stack([P.word(*S.tok(t)) for t in lex])
            c = Counter()
            for s in te:
                T = s["toks"]
                n = len(T)
                X = P.encode([P.word(*S.tok(t)) for t in T])
                gs = P.decode(X, n, L, rounds=2)
                c["使う割合:" + lbin(n)] += P.usage(X)
                c["使う割合n:" + lbin(n)] += 1
                for r, g in enumerate(gs):
                    for k, (i, t) in enumerate(zip(g, T)):
                        if not G.is_content(t):
                            continue
                        hit = lex[i]["lemma"] == t["lemma"]
                        c[f"読み{r}:n"] += 1
                        c[f"読み{r}:当たり"] += hit
                        if r == len(gs) - 1:
                            for key in ("位置:" + pbin(k), "長さ:" + lbin(n), "長さ×位置:" + lbin(n) + "・" + ("頭 3 語" if k < 3 else "それより後")):
                                c[key + ":n"] += 1
                                c[key + ":当たり"] += hit
                dt = [dict(lemma=lex[i]["lemma"], upos=lex[i]["upos"], xpos=lex[i]["xpos"], form=lex[i]["form"], dep="") for i in gs[-1]]
                c["印:n"] += 1
                c["印:同じ"] += func.marks(dt) == func.marks(T)
            rate = lambda k: round(c[k + ":当たり"] / max(1, c[k + ":n"]), 3)
            o = dict(秒=round(time.time() - t0), 内容語_1回で読む=rate("読み0"), 内容語_読み直し2回=rate("読み2"), 印の組が同じ=round(c["印:同じ"] / max(1, c["印:n"]), 3),
                     位置ごと={k[3:]: rate(k) for k in sorted({x.rsplit(":", 1)[0] for x in c if x.startswith("位置:")}, key=lambda z: (len(z), z))},
                     長さごと={k[3:]: rate(k) for k in sorted({x.rsplit(":", 1)[0] for x in c if x.startswith("長さ:")})},
                     長さと位置={k[6:]: rate(k) for k in sorted({x.rsplit(":", 1)[0] for x in c if x.startswith("長さ×位置:")})},
                     Dの使う割合={k[5:]: round(c[k] / max(1, c["使う割合n:" + k[5:]]), 3) for k in c if k.startswith("使う割合:")})
            res["版"][f"D={D}・{mode}"] = o
            print(D, mode, json.dumps({k: v for k, v in o.items() if k not in ("位置ごと",)}, ensure_ascii=False), flush=True)
            json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
