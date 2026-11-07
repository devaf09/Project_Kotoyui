"""介入の研究 実験I26: 曖昧な係り先を判断の板で見直す — 「近い方に係る」を基本に、手がかりがあるときだけ遠い方へ替える

持ち主（2026-11-06）の案 2: 「優しい漁師であった浦島太郎は…」の A（浦島太郎: 漁師、優しい）/ B（漁師: 優しい）を介入で確かめる。
I24 で、曖昧な例文 120 個のうち係り先を選ぶもの 75 個に、修飾する語・意図の係り先・誤読の係り先を付けた（data/tales/i24_ambig_labels.json）。
  - 文法解析グレインの素の読み: 近い方に係る（I24 で 63 / 75）。
  - 判断の板（介入）: 近い方で読んだあと、手がかりが遠い方を強く推すときだけ、その一手を遠い方に替える。
    手がかり (a) 一般常識の語板 v2: 修飾する語と係り先の語の合い方
         動詞 V → 名詞 N: N の帯 C の「V/する・される・に・で」+ N のクラス（帯 D）と V の「主語:・目的語:クラス」（帯 C）の重なり
         形容詞 A → N: N の帯 B の「性質:A」+ N のクラスと A の「修飾先:・主語:クラス」の重なり
         名詞 M の → N: N の帯 E（部分を持つ・材料など）と M のクラスの重なり、M の帯 E（の部分）と N のクラスの重なり
    手がかり (b) 話の文脈: その話の文（4 段の長さすべて）で、修飾する語と候補の語が係り受けでつながっていた回数
  - 語板に無い語: 複合語の後ろの部分（白雪姫 → 姫、一寸法師 → 法師）で引き、それも無ければ話の文から作る（その語が出る文の語板の和）。
物差し: 意図を選んだ割合（75 個）、近い方が誤る 12 個を直せた数、近い方が正しい 63 個を壊した数。
結果: ../Project_Kotoyui_private/intervention/results/i26.jsonl
実行: python intervention/i26_ambig_judge.py --tales <tales_v1_parsed.json> --labels <i24_ambig_labels.json> --boards <content_boards.json>
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict

ROLES = ("する", "される", "に", "で", "を")


def compound(toks, i):
    """i の語に、前に付いた複合語の部分を足した表記（お ばあ さん → おばあさん）。"""
    parts = [toks[j]["text"] for j in range(i) if toks[j]["dep"] == "compound" and toks[j]["head"] == i]
    return "".join(parts) + toks[i]["text"]


class Boards:
    def __init__(self, path, tales):
        self.b = {x["name"]: x["attrs"] for x in json.load(open(path, encoding="utf-8"))}
        self.ctx = defaultdict(Counter)                                # 話ごとの、知らない語の文脈の板
        self.tales = tales
        self.how = Counter()

    def get(self, word, lemma=None, tale=None):
        for w in (word, lemma):
            if w and w in self.b:
                self.how["そのまま"] += 1
                return self.b[w]
        for k in range(1, len(word)):                                 # 後ろの部分（長い方から）
            if word[k:] in self.b:
                self.how["後ろの部分"] += 1
                return self.b[word[k:]]
        if tale is not None:
            self.how["文脈から"] += 1
            return self.from_context(word, tale)
        self.how["無い"] += 1
        return {}

    def from_context(self, word, tale):
        acc = defaultdict(Counter)
        n = 0
        for ss in self.tales[tale]["parsed"].values():
            for s in ss:
                if word not in s["text"]:
                    continue
                for w in s["toks"]:
                    if w["pos"] in ("NOUN", "VERB", "ADJ") and w["lemma"] in self.b and w["lemma"] != word:
                        for band, d in self.b[w["lemma"]].items():
                            acc[band].update(d)
                        n += 1
        return {band: {k: v / n for k, v in d.items()} for band, d in acc.items()} if n else {}


def cls(b):
    return {k.split(":", 1)[1]: v for k, v in b.get("D", {}).items()}


def compat(bm, m_tok, bn, lem=None, cand_pos=None):
    """修飾する語の板 bm と、係り先の候補の板 bn の合い方（大きいほど合う）。候補が動詞（連用の係り）は、まだ測る手が無いので 0。"""
    if cand_pos == "VERB":
        return 0.0
    lem, pos = lem or m_tok["lemma"], m_tok["pos"]
    cm, cn = cls(bm), cls(bn)
    C_m, C_n = bm.get("C", {}), bn.get("C", {})
    if pos == "VERB":
        s = sum(C_n.get(f"{lem}/{r}", 0.0) for r in ROLES)
        s += sum(v * (C_m.get("主語:" + c, 0.0) + C_m.get("目的語:" + c, 0.0)) for c, v in cn.items())
        return s
    if pos in ("ADJ",) or m_tok["dep"] in ("amod",):
        s = bn.get("B", {}).get("性質:" + lem, 0.0)
        s += sum(v * (C_m.get("修飾先:" + c, 0.0) + C_m.get("主語:" + c, 0.0)) for c, v in cn.items())
        return s
    if pos in ("NOUN", "PROPN"):
        E_n, E_m = bn.get("E", {}), bm.get("E", {})
        s = sum(v * cm.get(k.split(":", 1)[1], 0.0) for k, v in E_n.items())
        s += sum(v * cn.get(k.split(":", 1)[1], 0.0) for k, v in E_m.items())
        return s
    return 0.0


def context_links(tale, m_lemma, n_word):
    """話の文で、修飾する語（見出し）から係り受けを 2 歩までたどって候補の語に着いた回数。"""
    c = 0
    for ss in tale["parsed"].values():
        for s in ss:
            toks = s["toks"]
            for i, w in enumerate(toks):
                if w["lemma"] != m_lemma:
                    continue
                h = i
                for _ in range(2):
                    h = toks[h]["head"]
                    if compound(toks, h).endswith(n_word) or toks[h]["lemma"] == n_word:
                        c += 1
                        break
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tales", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--boards", required=True)
    ap.add_argument("--out", default="../Project_Kotoyui_private/intervention/results/i26.jsonl")
    a = ap.parse_args()
    tales = json.load(open(a.tales, encoding="utf-8"))
    labels = json.load(open(a.labels, encoding="utf-8"))
    B = Boards(a.boards, tales)
    items = []
    for ti, t in enumerate(tales):
        for k, it in enumerate(t["ambig"]):
            lab = labels.get(f"{ti}.{k}")
            if not lab or lab[0] != "attach":
                continue
            toks = it["ginza"]
            tx = [w["text"] for w in toks]
            m = tx.index(lab[1])
            ia = [i for i, w in enumerate(tx) if w == lab[2] and i != m][0]
            ib = [i for i, w in enumerate(tx) if w == lab[3] and i != m][0]
            mt = toks[m]
            lem = mt["lemma"]
            if lem == "する":                                         # 「掃除をする」は 掃除 で引く（する は軽い動詞）
                objs = [w for w in toks if w["head"] == m and w["dep"] == "obj"]
                if objs:
                    lem = objs[0]["lemma"]
            bm = B.get(lem, lem, ti) if lem != mt["lemma"] else B.get(compound(toks, m), mt["lemma"], ti)
            sc, cx = {}, {}
            for name, j in (("意図", ia), ("誤読", ib)):
                word = compound(toks, j)
                sc[name] = compat(bm, mt, B.get(word, toks[j]["lemma"], ti), lem, toks[j]["pos"])
                cx[name] = context_links(t, mt["lemma"], word)
            near = "意図" if abs(ia - m) < abs(ib - m) else "誤読"
            items.append(dict(id=f"{ti}.{k}", near=near, compat=sc, ctx=cx, m_pos=mt["pos"], m_lemma=lem,
                              kind="連用（〜ながら・〜て）" if toks[ia]["pos"] == "VERB" and toks[ib]["pos"] == "VERB" else "連体"))
    far = {"意図": "誤読", "誤読": "意図"}

    def decide(x, use_common, use_ctx, delta):
        n, f = x["near"], far[x["near"]]
        if use_ctx and x["ctx"][f] > x["ctx"][n]:
            return f
        if use_common and x["compat"][f] > x["compat"][n] + delta:
            return f
        return n

    def common_only(x):
        s = x["compat"]
        if abs(s["意図"] - s["誤読"]) < 1e-9:
            return "決まらない"
        return max(s, key=s.get)

    def ctx_only(x):
        s = x["ctx"]
        if s["意図"] == s["誤読"]:
            return "決まらない"
        return max(s, key=s.get)

    res = dict(n=len(items), boards_lookup=dict(B.how))
    res["近い方"] = Counter(x["near"] for x in items)
    res["常識だけ"] = Counter(common_only(x) for x in items)
    res["文脈だけ"] = Counter(ctx_only(x) for x in items)
    for name, uc, ux in (("近い方 + 常識で替える", True, False), ("近い方 + 文脈で替える", False, True), ("近い方 + 常識 + 文脈", True, True)):
        for delta in (0.0, 0.02, 0.05, 0.1):
            if not uc and delta > 0:
                continue
            ch = [decide(x, uc, ux, delta) for x in items]
            fixed = sum(1 for x, c in zip(items, ch) if x["near"] == "誤読" and c == "意図")
            broke = sum(1 for x, c in zip(items, ch) if x["near"] == "意図" and c == "誤読")
            res[f"{name} δ={delta}"] = dict(acc=sum(c == "意図" for c in ch) / len(items), fixed=fixed, broke=broke)
    res["種類"] = Counter(x["kind"] for x in items)
    res["種類ごとの近い方"] = {k: sum(1 for x in items if x["kind"] == k and x["near"] == "意図") for k in res["種類"]}
    res["items"] = items
    rec = dict(exp="I26", **{k: (dict(v) if isinstance(v, Counter) else v) for k, v in res.items()})
    print(json.dumps({k: v for k, v in rec.items() if k != "items"}, ensure_ascii=False, indent=1))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
