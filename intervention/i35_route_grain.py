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
