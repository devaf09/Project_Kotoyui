"""介入の研究 実験I35a: 指摘の板（相手の指摘 → 言った側がどう受けるか）を、隣り合う台詞で数える

持ち主（2026-11-09）:「違和感や会話の中で指摘があれば、自覚と介入に回す、そういう判断もグレインの学習と判断で出来ればいい」
振り分けのグレイン（I35b、chat_bot3.py）の、相手の手がかりの側の材料。続く台詞 A → B → C（多くは 話し手 甲 → 乙 → 甲）で、
B が A を指摘しているか（手がかりの型）と、そのあと C で甲がどう受けたか（回し先）を数える。
手がかりの型（どれも A と B の関係で決める。語の一覧は使わない）:
  聞き返し   A の名詞が B で「？」「って」の前に出る（I34 と同じ）。
  否定の繰り返し  A の述語を B が否定で繰り返す。
  言い直し   A と B が同じ (述語, 助詞) の枠を持ち、名詞が違う（「X を読む」→「Y を読む」）。
  短い問い返し  B が内容語を持たず「？」で終わる（「え？」の形）。
  否定の始め  B の始めの感動詞が、会話の板で否定の答えに多く出る語（P(否定|語) > P(肯定|語)、板から数える）。
回し先（C の形で決める。物差しのための名前）:
  聞き返す  C が「？」で終わる。
  介入      C が A の内容語を言い直す（A の内容語を持ち、A と同じ行ではない）= 自分の行を作り直す。
  自覚      C が B の新しい内容語（A に無い語）を取り込む = 相手の言い分を自分の板に入れる。
  そのまま  上のどれでもない（話を先へ進める）。
くらべ: 手がかりの無い B のあとの C（同じ数え方）。手がかりの型ごとに P(回し先 | 型) と、指摘された A の語の印
  （本の板での珍しさ・述語に合わない助詞の差・裸の名詞）を、指摘されなかった語の印と比べる（何を暗くするかの手がかり）。
Web 小説は _wn。結果は非公開の置き場だけ。本文はコードに書かない。
実行: python intervention/i35_route_grain.py --cache <解析済み JSON> --stats <jsonl>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i34_dialogue_board as D  # noqa: E402

NOUN = ("NOUN", "PROPN")
CW = ("NOUN", "PROPN", "VERB", "ADJ")
ROUTES = ("そのまま", "聞き返す", "自覚", "介入")
CUES = ("聞き返し", "否定の繰り返し", "言い直し", "短い問い返し", "否定の始め")


def neg_intj(works):
    """会話の板（I34a と同じ数え方）から、否定の答えに多く出る感動詞。"""
    c = defaultdict(Counter)
    for w in works:
        Ls = w["lines"]
        for i in range(len(Ls) - 1):
            if not (D.isq(Ls[i]["text"]) and D.isq(Ls[i + 1]["text"])):
                continue
            qt = D.q_type(Ls[i]["toks"], Ls[i]["text"]).split(":")[0]
            if qt == "問いでない":
                continue
            at = Ls[i + 1]["toks"]
            first = next((t for t in at if t["pos"] not in ("PUNCT", "SYM")), None)
            qp = {t["lemma"] for t in Ls[i]["toks"] if t["pos"] in ("VERB", "ADJ")}
            ap_ = {t["lemma"] for t in at if t["pos"] in ("VERB", "ADJ")}
            neg = any(t["lemma"] in ("ない", "ぬ") and t["pos"] == "AUX" for t in at)
            if first is not None and first["pos"] == "INTJ" and qp & ap_:
                c[first["text"]]["否定" if neg else "肯定"] += 1
    return {x: v for x, v in c.items() if (v["否定"] + 0.5) / (v["否定"] + v["肯定"] + 1) > 0.5}


def slots(toks):
    """(述語, 助詞) → 名詞。"""
    out = {}
    for t in toks:
        if t["pos"] not in NOUN:
            continue
        h = toks[t["head"]] if 0 <= t["head"] < len(toks) else None
        cs = [k["text"] for k in toks if k["head"] == t["i"] and k["dep"] == "case"]
        if h is not None and h["pos"] in ("VERB", "ADJ") and cs:
            out[(h["lemma"], cs[0])] = t["lemma"]
    return out


def cues(a, b, negx):
    """B が A を指摘しているか: 手がかりの型と、指摘された A の語（lemma）。"""
    at, bt = a["toks"], b["toks"]
    bb = D.body(b["text"])
    out = []
    for t in at:
        if t["pos"] in NOUN and len(t["text"]) >= 2 and t["text"] in bb:
            rest = bb.split(t["text"], 1)[1]
            if rest.startswith(("って", "？", "?")):
                out.append(("聞き返し", t["lemma"]))
                break
    ap_ = {t["lemma"] for t in at if t["pos"] in ("VERB", "ADJ")}
    neg = any(t["lemma"] in ("ない", "ぬ") and t["pos"] == "AUX" for t in bt)
    rep = [t["lemma"] for t in bt if t["pos"] in ("VERB", "ADJ") and t["lemma"] in ap_]
    if rep and neg:
        out.append(("否定の繰り返し", rep[0]))
    sa, sb = slots(at), slots(bt)
    for k, n in sa.items():
        if k in sb and sb[k] != n:
            out.append(("言い直し", n))
            break
    if not any(t["pos"] in CW for t in bt) and bb.rstrip("…。 ").endswith(("？", "?")):
        out.append(("短い問い返し", None))
    first = next((t for t in bt if t["pos"] not in ("PUNCT", "SYM")), None)
    if first is not None and first["pos"] == "INTJ" and first["text"] in negx:
        out.append(("否定の始め", None))
    return out


def route_of(a, b, c):
    cb = D.body(c["text"]).rstrip("…。 ")
    if cb.endswith(("？", "?")):
        return "聞き返す"
    ca = {t["lemma"] for t in a["toks"] if t["pos"] in CW}
    cbb = {t["lemma"] for t in b["toks"] if t["pos"] in CW} - ca
    cc = {t["lemma"] for t in c["toks"] if t["pos"] in CW}
    if cc & ca and D.body(c["text"]) != D.body(a["text"]):
        return "介入"
    if cc & cbb:
        return "自覚"
    return "そのまま"


def word_marks(works):
    """A の語の印: 作品をまたぐ珍しさ（その語が出る作品の割合）。"""
    df = Counter()
    for w in works:
        df.update({t["lemma"] for ln in w["lines"] for t in ln["toks"] if t["pos"] in CW})
    return df, len(works)


def count_raw(works, negx, df, W, tab=None, tgt=None, n=None):
    """作品ごとに足せる数え上げ（1,808 作品で回すとき i35c から呼ぶ）。tgt は珍しさの和と数。"""
    tab = tab if tab is not None else defaultdict(Counter)        # 型 → 回し先
    tgt = tgt if tgt is not None else defaultdict(Counter)        # 指摘された語 / されなかった語 の珍しさ
    n = n if n is not None else Counter()
    for w in works:
        Ls = w["lines"]
        for i in range(len(Ls) - 2):
            if not (D.isq(Ls[i]["text"]) and D.isq(Ls[i + 1]["text"]) and D.isq(Ls[i + 2]["text"])):
                continue
            a, b, c = Ls[i], Ls[i + 1], Ls[i + 2]
            cs = cues(a, b, negx)
            r = route_of(a, b, c)
            n["組"] += 1
            if not cs:
                tab["手がかり無し"][r] += 1
                continue
            n["指摘"] += 1
            tab["指摘（どれか）"][r] += 1
            for k, word in cs:
                tab[k][r] += 1
            hit = {word for k, word in cs if word}
            if hit and df is not None:
                for t in a["toks"]:
                    if t["pos"] in CW:
                        rr = (df[t["lemma"]] - 1) / max(1, W - 1)
                        key = "指摘された語" if t["lemma"] in hit else "指摘されなかった語"
                        tgt[key]["n"] += 1
                        tgt[key]["和"] += rr
    return tab, tgt, n


def summarize(tab, tgt, n, negx):
    return dict(組=n["組"], 指摘=n["指摘"], 否定の始めの語=sorted(negx),
                回し先={k: dict(n=sum(v.values()), **{r: round(v[r] / max(1, sum(v.values())), 3) for r in ROUTES}) for k, v in tab.items()},
                語の珍しさ={k: dict(n=v["n"], ほかの作品に出る割合=round(v["和"] / v["n"], 3)) for k, v in tgt.items() if v["n"]})


def route_board(tab):
    """振り分けのグレインに渡す、手がかりの型 → 回し先 の数（手がかり無しとの比べも中で使う）。"""
    return {k: dict(v) for k, v in tab.items()}


def count(works, negx=None):
    negx = negx if negx is not None else neg_intj(works)
    df, W = word_marks(works)
    tab, tgt, n = count_raw(works, negx, df, W)
    return summarize(tab, tgt, n, negx), route_board(tab)


# ---- I35b: 振り分けのグレイン（自分の行の側）----
# 自分の行の 名詞+助詞+述語 の所（「所」）ごとに、気づきの器の差を記号にし、相手の指摘（あり/なし）の経験から
# P(指摘 | 記号) を経験の器（I33 の CountGrain・回帰）で持つ。指摘ありと見込む所が回し先「介入」（その所を暗くして作り直す）、
# 同じ所が繰り返し指摘されれば「自覚」（自覚の板に残し、会話をまたいで暗くする）。

ASK = D.ASK


def _bin(x, edges):
    for i, e in enumerate(edges):
        if x < e:
            return i
    return len(edges)


class PieceBoards:
    """本の行から数える板: (述語, 助詞)・(名詞, 助詞, 述語)・(名詞, 助詞)・名詞の修飾の割合・述語が取る名詞のばらつき。"""

    def __init__(self, works):
        self.pp, self.pn, self.tri, self.nc, self.nn, self.mod = Counter(), Counter(), Counter(), Counter(), Counter(), Counter()
        pv = defaultdict(set)
        for w in works:
            for ln in w["lines"]:
                T = ln["toks"]
                kids = defaultdict(list)
                for t in T:
                    if 0 <= t["head"] < len(T) and t["head"] != t["i"]:
                        kids[t["head"]].append(t)
                for t in T:
                    if t["pos"] not in NOUN + ("PRON",):
                        continue
                    self.nn[t["lemma"]] += 1
                    if any(k["dep"] in ("acl", "amod", "nmod", "det") and k["i"] < t["i"] for k in kids[t["i"]]):
                        self.mod[t["lemma"]] += 1
                    cs = [k["text"] for k in kids[t["i"]] if k["dep"] == "case"]
                    if not cs:
                        continue
                    self.nc[(t["lemma"], cs[0])] += 1
                    h = T[t["head"]] if 0 <= t["head"] < len(T) else None
                    if h is not None and h["pos"] in ("VERB", "ADJ"):
                        self.pp[(h["lemma"], cs[0])] += 1
                        self.pn[h["lemma"]] += 1
                        self.tri[(t["lemma"], cs[0], h["lemma"])] += 1
                        pv[h["lemma"]].add(t["lemma"])
        self.top = Counter()
        for (q, _), v in self.pp.items():
            self.top[q] = max(self.top[q], v)
        self.pvar = {q: len(v) / self.pn[q] for q, v in pv.items()}

    def pieces(self, toks, text=""):
        """行の所: (名詞, 助詞, 述語, 名詞の表記, 述語の表記, 裸か)。"""
        kids = defaultdict(list)
        for t in toks:
            if 0 <= t["head"] < len(toks) and t["head"] != t["i"]:
                kids[t["head"]].append(t)
        out = []
        for t in toks:
            if t["pos"] not in NOUN + ("PRON",):
                continue
            h = toks[t["head"]] if 0 <= t["head"] < len(toks) else None
            cs = [k["text"] for k in kids[t["i"]] if k["dep"] == "case"]
            if h is None or h["pos"] not in ("VERB", "ADJ") or not cs:
                continue
            bare = not any(k["dep"] in ("acl", "amod", "nmod", "det") and k["i"] < t["i"] for k in kids[t["i"]])
            out.append(dict(n=t["lemma"], c=cs[0], p=h["lemma"], ns=t["text"], ps=h["text"], bare=bare))
        return out

    def symbols(self, pc, question=False):
        n, c, p = pc["n"], pc["c"], pc["p"]
        top = self.top.get(p, 0)
        a = 1.0 - self.pp.get((p, c), 0) / top if top else 1.0
        tri = self.tri.get((n, c, p), 0)
        nn = self.nn.get(n, 0)
        pnc = self.nc.get((n, c), 0) / nn if nn else 0.0
        m = self.mod[n] / nn if nn else 0.0
        sy = [f"a:{_bin(a, (0.5, 0.8, 0.95))}",                 # 気づきの器 5(a): 述語に合わない助詞
              f"t:{_bin(tri, (1, 2, 4, 10))}",                  # 名詞+助詞+述語 の組が本にあるか
              f"nc:{_bin(pnc, (0.02, 0.1, 0.3))}",              # その名詞がその助詞を取る割合
              f"nf:{_bin(nn, (1, 10, 100, 1000))}",             # 名詞の多さ
              f"pv:{_bin(self.pvar.get(p, 1.0), (0.05, 0.2, 0.5))}"]   # 述語が取る名詞のばらつき（中身の薄さ）
        if pc["bare"]:
            sy.append(f"b:{_bin(m, (0.3, 0.6, 0.85))}")         # 気づきの器 5(b): 修飾を取るはずの名詞が裸
        if n in ASK and not question:
            sy.append("wh")                                     # 問いでない行の問いの語
        return sy


def i34d_flag(pb, pc, h_gen, h_book, bins=10):
    """I34d の決め方を所ごとに: 不自然さ（a と b の大きい方）の区切りで、自分の行に出る割合 > 本の行に出る割合 なら違和感。"""
    p, c, n = pc["p"], pc["c"], pc["n"]
    top = pb.top.get(p, 0)
    a = 1.0 - pb.pp.get((p, c), 0) / top if top else 0.0
    nn = pb.nn.get(n, 0)
    m = pb.mod[n] / nn if (nn and pc["bare"]) else 0.0
    d = max(a, m)
    b = min(bins - 1, int(d * bins))
    ng, nb = sum(h_gen.values()), sum(h_book.values())
    return (h_gen[b] + 0.5) / (ng + 0.5 * bins) > (h_book[b] + 0.5) / (nb + 0.5 * bins), d


def eval_self(works, labels, own_toks, folds=5, seed=0):
    """相手の指摘（labels）を 5 つに分け、ほかの 4 つの経験で残りを当てる。I34d の決め方と比べる。"""
    import random
    import i33_predicate_grain as G
    pb = PieceBoards(works)
    items = labels["items"]
    sc = [pb.symbols(dict(x, bare=x.get("bare", True))) for x in items]
    y = ["指摘" if x["odd"] else "そのまま" for x in items]
    # I34d の分かれ方: 自分の行（own_toks）と本の行の、所の不自然さ
    rng = random.Random(seed)
    lines = [ln for w in works for ln in w["lines"] if ln["toks"]]
    rng.shuffle(lines)
    h_book, h_gen = Counter(), Counter()
    for ln in lines[:2000]:
        for pc in pb.pieces(ln["toks"]):
            h_book[min(9, int(i34d_flag(pb, pc, Counter(), Counter())[1] * 10))] += 1
    for T in own_toks:
        for pc in pb.pieces(T):
            h_gen[min(9, int(i34d_flag(pb, pc, Counter(), Counter())[1] * 10))] += 1
    order = list(range(len(items)))
    rng.shuffle(order)
    res = defaultdict(Counter)
    for f in range(folds):
        te = set(order[f::folds])
        tr = [i for i in order if i not in te]
        cg = G.CountGrain()
        for i in tr:
            cg.add(sc[i], y[i])
        lr = G.LRGrain()
        lr.fit([sc[i] for i in tr], [y[i] for i in tr])
        for i in te:
            for name, g in (("経験の器", cg), ("回帰", lr)):
                s = g.scores(sc[i])
                res[name][(max(s, key=s.get), y[i])] += 1
            fl = i34d_flag(pb, dict(items[i], bare=items[i].get("bare", True)), h_gen, h_book)[0]
            res["I34d の決め方"][("指摘" if fl else "そのまま", y[i])] += 1
            res["いつも そのまま"][("そのまま", y[i])] += 1
    out = {}
    for name, c in res.items():
        tp, fp, fn = c[("指摘", "指摘")], c[("指摘", "そのまま")], c[("そのまま", "指摘")]
        tn = c[("そのまま", "そのまま")]
        out[name] = dict(当たり=round((tp + tn) / sum(c.values()), 3), 拾えた指摘=f"{tp}/{tp + fn}", 拾いすぎ=f"{fp}/{fp + tn}")
    return out, pb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--board", help="手がかりの型 → 回し先 の数（JSON）。chat_bot3.py の --route に渡す")
    a = ap.parse_args()
    works = json.load(open(a.cache, encoding="utf-8"))
    summ, board = count(works)
    rec = dict(exp="I35a", 作品=len(works), **summ)
    if a.board:
        json.dump(board, open(a.board, "w", encoding="utf-8"), ensure_ascii=False)
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    with open(a.stats, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
