"""I48ab の測り（i48ab_long_grain）。Tatoeba の GiNZA の解析で、学び／試しを文の id の割り算で分ける（試し 1/20）。

python intervention/past/i48ab_run.py --conllu tatoeba_ginza.conllu --content v2_3b/content_boards.json --func func_boards.json --out out.json
  [--train N] [--test N] [--fwd N] [--D 2048 4096 8192]
"""
from __future__ import annotations
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする

import argparse
import json
import os
import sys
import time
import zlib
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import i48ab_long_grain as G  # noqa: E402

POS_MAP = {"名詞": "NOUN", "動詞": "VERB", "形容詞": "ADJ"}
GROUPS = {"生き物": ("四つ脚", "哺乳", "動物", "鳥"), "人": ("個人", "人間", "成人"), "物": ("アーチファクト", "道具", "器具", "容器"),
          "場所": ("地域", "地区", "場所", "建物", "行政区画")}   # 入れ替えの試しのクラスの組（手で置いた）
SWAP = {"生き物": "人", "人": "生き物", "物": "場所", "場所": "物"}


def top_class(w):
    D = (w or {}).get("attrs", {}).get("D", {})
    return max(D.items(), key=lambda z: z[1])[0].split(":")[-1].split("(")[0] if D else None


def group_of(cls):
    if not cls:
        return None
    for g, ks in GROUPS.items():
        if any(k in cls for k in ks):
            return g
    return None


def word_rows(S, sents):
    wr = []
    for s in sents:
        cont, gaps, _ = G.split(s)
        if not cont:
            continue
        mk = S.func.marks(s["toks"])
        for i in range(len(cont)):
            wr.append((cont[i][0], cont[i + 1][0] if i + 1 < len(cont) else ("文末",) + tuple(sorted(mk)), gaps[i]))
    return wr


def rows_of(S, sents):
    """逆回しの学びの行（属性の波の特徴, 詰め物）を順に出す。"""
    for s in sents:
        cont, gaps, _ = G.split(s)
        if not cont:
            continue
        mk = S.func.marks(s["toks"])
        A = [S.attrs(l, u) for l, _, u in cont]
        for i in range(len(cont)):
            f, v = S.pair(A[i], A[i + 1]) if i + 1 < len(cont) else S.pair(A[i], None, mk)
            yield f, v, gaps[i]


def rebuild(S, pg, wt, surf, cont, mk, use="wave"):
    """要点（内容語の原形・品詞の並びと印）→ 文。use: wave（属性の波）か word（語の表）。"""
    A = [S.attrs(l, u) for l, u in cont]
    out, fills = [], []
    for i in range(len(cont)):
        last = i + 1 == len(cont)
        b_ = ("文末",) + tuple(sorted(mk)) if last else cont[i + 1][0]
        if use == "wave" or (use == "mix" and sum(wt.t2.get((cont[i][0], b_), {}).values()) < 3):   # mix: 語の組の経験が 3 件未満なら属性の波
            f, v = S.pair(A[i], None if last else A[i + 1], mk)
            g = pg.predict(f, v)[0]
        else:
            g = wt.predict(cont[i][0], ("文末",) + tuple(sorted(mk)) if last else cont[i + 1][0])
        fills.append(g)
        out.append(surf.form(cont[i][0], cont[i][1], g) + g)
    return "".join(out), fills


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conllu", required=True)
    ap.add_argument("--content", required=True)
    ap.add_argument("--func", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, default=0)
    ap.add_argument("--test", type=int, default=3000)
    ap.add_argument("--fwd", type=int, default=500)
    ap.add_argument("--lex", type=int, default=10000)
    ap.add_argument("--D", type=int, nargs="*", default=[2048, 4096, 8192])
    a = ap.parse_args()
    t0 = time.time()
    sents = [s for s in G.read_conllu(a.conllu) if s["toks"]]
    te = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 == 0]
    tr = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 != 0]
    if a.train:
        tr = tr[: a.train]
    te = te[: a.test]
    cb = json.load(open(a.content, encoding="utf-8"))
    fb = json.load(open(a.func, encoding="utf-8"))
    func = G.Func(fb)
    print(f"読み込み {time.time() - t0:.0f} 秒、学び {len(tr)} 文、試し {len(te)} 文", flush=True)
    surf = G.Surface()
    train_lemmas = set()
    for s in tr:
        cont, gaps, _ = G.split(s)
        for (l, f, u), g in zip(cont, gaps):
            surf.add(l, f, u, g)
            train_lemmas.add(l)
    res = dict(学びの文=len(tr), 試しの文=len(te), 手で置いた物=dict(CONTENT=G.CONTENT, FUNC_DEP=G.FUNC_DEP, MARKS=G.MARKS, TOPK=G.TOPK,
                                                       ALPHA=G.ALPHA, GROUPS=GROUPS, SWAP=SWAP), 逆回し={}, 順方向={})
    cbn = {}
    for w in cb:
        if w["name"] not in cbn or w.get("freq", 0) > cbn[w["name"]].get("freq", 0):
            cbn[w["name"]] = w
    # 入れ替えの候補: クラスの組ごとに、よく出る名詞 30
    cand = defaultdict(list)
    for w in sorted(cb, key=lambda x: -x.get("freq", 0)):
        if w["pos"] == "名詞":
            g = group_of(top_class(w))
            if g and len(cand[g]) < 30 and w["name"] in train_lemmas:
                cand[g].append(w["name"])
    # 人の資料での、(名詞のクラスの組, 次の原形) → 詰め物（入れ替えの答え合わせ）
    ref = defaultdict(Counter)
    for s in tr:
        cont, gaps, _ = G.split(s)
        for i in range(len(cont) - 1):
            if cont[i][2] == "NOUN":
                g = group_of(top_class(cbn.get(cont[i][0])))
                if g:
                    ref[(g, cont[i + 1][0])][gaps[i]] += 1
    rng = np.random.default_rng(0)
    for D in a.D:
        for bands_name, bands in (("A を除く", ("B", "C", "D", "E", "G")), ("A を含む（参考）", ("A", "B", "C", "D", "E", "G"))):
            if bands_name.startswith("A を含む") and D != a.D[-1]:
                continue
            S = G.AttrSpace(cb, func, D, bands=bands)
            t1 = time.time()
            wr = word_rows(S, tr)
            pg = G.PairGrain(D).fit(rows_of(S, tr), Counter(y for _, _, y in wr))
            wt = G.WordTable().fit(wr)
            most = Counter(y for _, b_, y in wr if not isinstance(b_, tuple)).most_common(1)[0][0]
            c = Counter()
            bylen = defaultdict(Counter)
            unseen = Counter()
            swap = Counter()
            examples = []
            for s in te:
                cont, gaps, head = G.split(s)
                if not cont:
                    continue
                mk = func.marks(s["toks"])
                key = [(l, u) for l, _, u in cont]
                L = len(cont)
                lb = "1〜3 語" if L <= 3 else ("4〜6 語" if L <= 6 else "7 語以上")
                for use in ("wave", "word", "mix"):
                    sent, fills = rebuild(S, pg, wt, surf, key, mk, use)
                    ok = G.norm(head + sent) == G.norm(s["text"])
                    c[use + ":文"] += ok
                    bylen[lb][use + ":文"] += ok
                    for i, (g, y) in enumerate(zip(fills, gaps)):
                        if i + 1 < L:
                            c[use + ":間"] += g == y
                            bylen[lb][use + ":間"] += g == y
                            if cont[i][0] not in train_lemmas or cont[i + 1][0] not in train_lemmas:
                                unseen[use] += g == y
                                if use == "wave":
                                    unseen["n"] += 1
                        else:
                            c[use + ":文末"] += g == y
                            pf = surf.form(cont[i][0], cont[i][2], g) + g
                            c[use + ":述語の形"] += pf == cont[i][1] + y
                    if use == "wave" and len(examples) < 8 and L >= 3:
                        examples.append(dict(元=s["text"], 要点=[l for l, _, _ in cont] + sorted(mk), 戻し=head + sent))
                c["文"] += 1
                bylen[lb]["文"] += 1
                c["間"] += L - 1
                bylen[lb]["間"] += L - 1
                c["most:間"] += sum(g == most for g in gaps[:-1])
                # 入れ替えの試し
                A = [S.attrs(l, u) for l, u in key]
                for i in range(L - 1):
                    if cont[i][2] != "NOUN":
                        continue
                    g0 = group_of(top_class(cbn.get(cont[i][0])))
                    if not g0 or not cand[SWAP[g0]]:
                        continue
                    nw = cand[SWAP[g0]][int(rng.integers(len(cand[SWAP[g0]])))]
                    f, v = S.pair(A[i], A[i + 1])
                    before = pg.predict(f, v)[0]
                    f, v = S.pair(S.attrs(nw, "NOUN"), A[i + 1])
                    after = pg.predict(f, v)[0]
                    swap["n"] += 1
                    swap["変わった"] += after != before
                    r = ref.get((SWAP[g0], cont[i + 1][0]))
                    if r and sum(r.values()) >= 5:
                        swap["答えあり"] += 1
                        swap["人の資料の多い詰め物と同じ"] += after == r.most_common(1)[0][0]
                        swap["元の詰め物のまま_答えあり"] += after == before
                        r0 = ref.get((g0, cont[i + 1][0]))
                        swap["人の資料でも変わる"] += bool(r0) and r0.most_common(1)[0][0] != r.most_common(1)[0][0]
            n, m = c["文"], max(1, c["間"])
            o = dict(学びの時間_秒=round(time.time() - t1), 文が元と同じ_波=round(c["wave:文"] / n, 3), 文が元と同じ_語の表=round(c["word:文"] / n, 3),
                     混ぜ_語の組が3件以上なら語の表=dict(文=round(c["mix:文"] / n, 3), 間=round(c["mix:間"] / m, 3), 文末=round(c["mix:文末"] / n, 3),
                                                述語の形=round(c["mix:述語の形"] / n, 3), 学びに無い語を含む間=round(unseen["mix"] / max(1, unseen["n"]), 3)),
                     間の詰め物_波=round(c["wave:間"] / m, 3), 間の詰め物_語の表=round(c["word:間"] / m, 3), 間の詰め物_いちばん多い物=[most, round(c["most:間"] / m, 3)],
                     文末_波=round(c["wave:文末"] / n, 3), 文末_語の表=round(c["word:文末"] / n, 3),
                     述語の形_波=round(c["wave:述語の形"] / n, 3), 述語の形_語の表=round(c["word:述語の形"] / n, 3),
                     学びに無い語を含む間=dict(件数=unseen["n"], 波=round(unseen["wave"] / max(1, unseen["n"]), 3), 語の表=round(unseen["word"] / max(1, unseen["n"]), 3)),
                     長さごと={k: dict(文=v["文"], 文が元と同じ_波=round(v["wave:文"] / max(1, v["文"]), 3), 文が元と同じ_語の表=round(v["word:文"] / max(1, v["文"]), 3),
                                       間_波=round(v["wave:間"] / max(1, v["間"]), 3), 間_語の表=round(v["word:間"] / max(1, v["間"]), 3)) for k, v in sorted(bylen.items())},
                     入れ替え=dict(件数=swap["n"], 詰め物が変わった=round(swap["変わった"] / max(1, swap["n"]), 3), 答え合わせできた=swap["答えあり"],
                                  人の資料の多い詰め物と同じ=round(swap["人の資料の多い詰め物と同じ"] / max(1, swap["答えあり"]), 3),
                                  人の資料でも組で詰め物が変わる割合=round(swap["人の資料でも変わる"] / max(1, swap["答えあり"]), 3)),
                     例=examples)
            res["逆回し"][f"D={D}・{bands_name}"] = o
            print(D, bands_name, json.dumps({k: v for k, v in o.items() if k not in ("例", "長さごと")}, ensure_ascii=False), flush=True)
            if bands_name == "A を除く":
                for kind, keys in (("持ち主の例", [("昨日", "NOUN"), ("犬", "NOUN"), ("散歩", "NOUN"), ("行く", "VERB")]),
                                   ("持ち主の例（入れ替え）", [("昨日", "NOUN"), ("友達", "NOUN"), ("散歩", "NOUN"), ("行く", "VERB")])):
                    o.setdefault("持ち主の例", []).append(dict(要点=[k for k, _ in keys] + ["過去"], 戻し=rebuild(S, pg, wt, surf, keys, {"過去"}, "wave")[0],
                                                            戻し_語の表=rebuild(S, pg, wt, surf, keys, {"過去"}, "word")[0]))
        # ---- 順方向: 入口の波から要点を読み出す ----
        S = G.AttrSpace(cb, func, D, bands=("A", "B", "C", "D", "E", "G"))
        lex = []
        for w in sorted(cb, key=lambda x: -x.get("freq", 0))[: a.lex]:
            lex.append(dict(lemma=w["name"], upos=POS_MAP.get(w["pos"], "NOUN"), xpos="", form=w["name"]))
        seen_f = Counter()
        for s in tr[:50000]:
            for t in s["toks"]:
                if not (t["upos"] in POS_MAP.values()) or not G.is_content(t):
                    seen_f[(t["lemma"], t["upos"], t["xpos"], t["form"] if t["upos"] == "PUNCT" else t["lemma"])] += 1
        for (l, u, x, f), _ in seen_f.most_common(1500):
            lex.append(dict(lemma=l, upos=u, xpos=x, form=f))
        M = np.stack([S.dense(*S.tok(t)) for t in lex])
        Mraw = np.stack([S.raw(*S.tok(t)) for t in lex])
        fc = Counter()
        for s in te[: a.fwd]:
            T = s["toks"]
            X = S.encode(T)
            gs = S.decode(X, len(T), M, Mraw, rounds=2)
            gold_c0 = [t["lemma"] for t in T if G.is_content(t)]
            for r_, g_ in enumerate(gs):
                pc_ = [lex[i]["lemma"] for i, t in zip(g_, T) if G.is_content(t)]
                fc[f"読み{r_}:n"] += len(gold_c0)
                fc[f"読み{r_}:当たり"] += sum(x == y for x, y in zip(gold_c0, pc_))
            got = [lex[i] for i in gs[-1]]
            gold_c = [(t["lemma"], t["upos"]) for t in T if G.is_content(t)]
            pred_c = [g["lemma"] for g, t in zip(got, T) if G.is_content(t)]
            for (gl, gu), pl in zip(gold_c, pred_c):
                k = "名詞・動詞・形容詞" if gu in POS_MAP.values() else "そのほかの内容語"
                fc[k + ":n"] += 1
                fc[k + ":当たり"] += gl == pl
            dt = [dict(lemma=g["lemma"], upos=g["upos"], xpos=g["xpos"], form=g["form"], dep="") for g in got]
            fc["印:n"] += 1
            fc["印:組が同じ"] += func.marks(dt) == func.marks(T)
            for mname in ("過去", "否定", "丁寧", "進行", "問い"):
                gm, pm = mname in func.marks(T), mname in func.marks(dt)
                fc[mname + ":gold"] += gm
                fc[mname + ":pred"] += pm
                fc[mname + ":both"] += gm and pm
            fc["語の数:" + ("1〜8" if len(T) <= 8 else ("9〜14" if len(T) <= 14 else "15 以上")) + ":n"] += sum(1 for t in T if G.is_content(t))
            fc["語の数:" + ("1〜8" if len(T) <= 8 else ("9〜14" if len(T) <= 14 else "15 以上")) + ":当たり"] += sum(gl == pl for (gl, _), pl in zip(gold_c, pred_c))
        o = {k: round(fc[k + ":当たり"] / max(1, fc[k + ":n"]), 3) for k in ("名詞・動詞・形容詞", "そのほかの内容語")}
        o["印の組が同じ"] = round(fc["印:組が同じ"] / max(1, fc["印:n"]), 3)
        o["印ごと（再現率／適合率）"] = {m: [round(fc[m + ":both"] / max(1, fc[m + ":gold"]), 3), round(fc[m + ":both"] / max(1, fc[m + ":pred"]), 3)]
                                  for m in ("過去", "否定", "丁寧", "進行", "問い")}
        o["文の長さ（語の数）ごとの内容語の当たり"] = {k: round(fc[f"語の数:{k}:当たり"] / max(1, fc[f"語の数:{k}:n"]), 3) for k in ("1〜8", "9〜14", "15 以上")}
        o["読み直しの回ごとの内容語の当たり（0 = 1 回で読む）"] = [round(fc[f"読み{r_}:当たり"] / max(1, fc[f"読み{r_}:n"]), 3) for r_ in range(3)]
        o["候補の語"] = len(lex)
        res["順方向"][f"D={D}"] = o
        print(D, "順方向", json.dumps(o, ensure_ascii=False), flush=True)
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
