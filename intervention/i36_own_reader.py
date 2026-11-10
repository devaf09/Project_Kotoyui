"""介入の研究 実験I36: GiNZA に頼らない読み手（板と経験だけで読む）

持ち主（2026-11-10）:「GiNZAに頼らないグレイン化について、進めてOKです。多分、会話には流れが必要だから、ロジック的な文法解析は足かせになってくると思う。」
読み手が会話で要るのは 4 つだけ: 語の切れ目、語の役（品詞）、原形、名詞+助詞がどの述語に係るか（所）。木全体は作らない。
  1. 語の板（字の並び → 品詞・原形 の数）と、品詞の流れの板（前の品詞 → 次の品詞 の数）。読んだ行から数える。
  2. 切れ目と品詞: 字の並びを左から読み、語の板の出やすさ × 品詞の流れの出やすさ がいちばん大きい並びを選ぶ（流れで読む）。
     板に無い字の並びは、字の種類（漢字・ひらがな・カタカナ・数字・記号）の続きを 1 語とみる。
  3. 係り先: 名詞+助詞のあとに出る述語のうち、何番目の述語に係るかを経験の器で持つ
     （記号: 助詞、何番目か、間に読点があるか、その述語の後ろ（名詞に続く・て形などで続く・言い切り））。
  初めの板と経験は、前に GiNZA で読んだ本の行から数える（読む時には GiNZA を使わない）。会話で読んだ行を足していける。
GiNZA の読みとの一致を測る: --cache <解析済み JSON> --stats <jsonl>（本の行の 2 割を試し、8 割で数える）。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import unicodedata
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PRED = ("VERB", "ADJ")
NOUNISH = ("NOUN", "PROPN", "PRON", "NUM")
CASES = ("が", "を", "に", "で", "へ", "と", "から", "まで", "より", "は", "も")
MAXLEN = 8


def ctype(ch):
    if ch.isdigit():
        return "数"
    n = unicodedata.name(ch, "")
    if "CJK UNIFIED" in n or ch in "々〆":
        return "漢"
    if "HIRAGANA" in n:
        return "ひ"
    if "KATAKANA" in n or ch == "ー":
        return "カ"
    if ch.isalpha():
        return "英"
    return "記"


class OwnReader:
    def __init__(self):
        self.lex = defaultdict(Counter)                     # 字の並び → (品詞, 原形) の数
        self.flow = defaultdict(Counter)                    # 前の品詞 → 次の品詞 の数
        self.pos_n = Counter()
        self.att = defaultdict(Counter)                     # 係り先の経験: 記号 → (係る, 係らない) の数
        self.te_aux = defaultdict(Counter)                  # 「て」のあとの述語 → (補助, 補助でない) の数
        self.N = 0

    # ---- 板を数える（読んだ行から） ----
    def learn(self, toks):
        prev = "始"
        for t in toks:
            self.lex[t["text"]][(t["pos"], t["lemma"])] += 1
            self.flow[prev][t["pos"]] += 1
            self.pos_n[t["pos"]] += 1
            prev = t["pos"]
            self.N += 1
        self.flow[prev]["終"] += 1
        for t in toks:                                      # 「て」のあとの述語が、前の述語の補助（aux）になる割合（述語ごと）
            if t["i"] > 0 and toks[t["i"] - 1]["text"] in ("て", "で") and toks[t["i"] - 1]["pos"] == "SCONJ":
                self.te_aux[t["lemma"]][t["dep"] in ("aux", "fixed") or t["pos"] == "AUX"] += 1
        for t, case, cands in self._att_cands(toks):        # 係り先の経験（読んだ行の本当の係り先から）
            for k, c in enumerate(cands):
                self.att[self._att_sym(toks, t, case, k, c)][c["i"] == t["head"]] += 1

    def learn_words(self, entries, weight=1):
        """語板（語の名前・品詞・数）を、語の板に足す（本の行に無い語・複合語を知るため）。"""
        P = {"名詞": "NOUN", "動詞": "VERB", "形容詞": "ADJ"}
        for e in entries:                                   # 漢字かカタカナを含む名前だけ（ひらがなだけの名前は機能語の並びと紛れる。「した」「遅く」）
            w = e["name"]                                   # 動詞・形容詞は原形（言い切りの形）だけが入る
            if e["pos"] in P and w not in self.lex and any(ctype(ch) in ("漢", "カ") for ch in w):
                self.lex[w][(P[e["pos"]], w)] += weight

    # ---- 切れ目と品詞（流れで読む） ----
    def _cands(self, s, i):
        out = []
        for j in range(i + 1, min(len(s), i + MAXLEN) + 1):
            w = s[i:j]
            if w in self.lex:
                tot = sum(self.lex[w].values())
                for (pos, lem), v in self.lex[w].items():
                    out.append((j, w, pos, lem, math.log(v / tot)))
        j = i + 1                                           # 板に無い: 字の種類の続きを 1 語に
        ct = ctype(s[i])
        while j < len(s) and ctype(s[j]) == ct and ct != "記" and (j - i < MAXLEN or ct == "カ"):
            j += 1
        w = s[i:j]
        if ct == "カ" and w not in self.lex:                 # カタカナの続きは、知っている語に分けずに 1 つの名詞とみる（名詞のふつうの出やすさで）
            out.append((j, w, "NOUN", w, math.log(max(self.pos_n["NOUN"], 1) / max(self.N, 1) / max(len(self.lex), 1)) + 5))
        elif w not in self.lex:
            pos = {"漢": "NOUN", "カ": "NOUN", "英": "NOUN", "数": "NUM", "記": "SYM", "ひ": "NOUN"}[ct]
            out.append((j, w, pos, w, math.log(0.5 / (self.N + 1)) * (1 if ct != "ひ" else 2)))
        return out

    def _flow_lp(self, a, b):
        c = self.flow.get(a, {})
        tot = sum(c.values())
        return math.log((c.get(b, 0) + 0.1) / (tot + 0.1 * 20))

    def segment(self, s):
        best = {(0, "始"): (0.0, None)}                    # (位置, 前の品詞) → (点, 戻り)
        for i in range(len(s)):
            here = [(k, v) for k, v in best.items() if k[0] == i]
            if not here:
                continue
            cands = self._cands(s, i)
            for (pi, pp), (sc, _) in here:
                for j, w, pos, lem, lp in cands:
                    v = sc + lp + self._flow_lp(pp, pos)
                    key = (j, pos)
                    if key not in best or v > best[key][0]:
                        best[key] = (v, ((pi, pp), w, pos, lem))
        ends = [(v[0] + self._flow_lp(k[1], "終"), k) for k, v in best.items() if k[0] == len(s)]
        if not ends:
            return []
        k = max(ends)[1]
        out = []
        while best[k][1] is not None:
            prev, w, pos, lem = best[k][1]
            out.append((w, pos, lem))
            k = prev
        return out[::-1]

    # ---- 係り先 ----
    def _att_cands(self, toks):
        for t in toks:
            if t["pos"] not in NOUNISH:
                continue
            nxt = toks[t["i"] + 1] if t["i"] + 1 < len(toks) else None
            if nxt is None or nxt["pos"] != "ADP" or nxt["text"] not in CASES:
                continue
            cands = [c for c in toks[t["i"] + 2:] if c["pos"] in PRED]
            if cands:
                yield t, nxt["text"], cands

    def _after(self, toks, c):
        """述語の後ろ: その述語の助動詞などを飛ばして、次に来るもの。"""
        j = c["i"] + 1
        while j < len(toks) and toks[j]["pos"] in ("AUX", "PART"):
            j += 1
        if j >= len(toks) or toks[j]["pos"] == "PUNCT" and toks[j]["text"] in ("。", "！", "？", "!", "?", "」", "』"):
            return "言い切り"
        x = toks[j]
        if x["pos"] in NOUNISH:
            return "名詞へ"
        if x["pos"] == "SCONJ":
            return "つなぎ:" + x["text"]
        return x["pos"]

    def _att_sym(self, toks, t, case, k, c):
        between = toks[t["i"] + 2:c["i"]]
        comma = any(x["text"] in ("、", ",", "，") for x in between)
        same = any(x["text"] == case and x["pos"] == "ADP" for x in between)          # 間に同じ助詞（別の名詞の枠）
        rel = sum(1 for x in between if x["pos"] in PRED and self._after(toks, x) == "名詞へ")   # 間の、名詞に続く述語
        return (case, min(k - rel, 3), min(rel, 2), comma, same, self._after(toks, c))

    def attach(self, toks):
        """名詞+助詞ごとに、係る述語を選ぶ: 何番目かの記号で、係る割合のいちばん大きい述語。"""
        heads = {}
        for t, case, cands in self._att_cands(toks):
            best, bv = cands[-1], -1.0
            for k, c in enumerate(cands):                   # 左から読み、係る見込みが係らない見込みより大きい最初の述語に係る（流れで決める）
                v = self.att.get(self._att_sym(toks, t, case, k, c), Counter())
                p = (v[True] + 0.1) / (v[True] + v[False] + 1.0)
                if p > 0.5:
                    best = c
                    break
                if p > bv:
                    best, bv = c, p
            heads[t["i"]] = best["i"]
        return heads

    def _is_aux(self, toks, t):
        """「て」のあとの述語が補助か: 本の行で、その述語が「て」のあとで補助になった割合が、ならない割合より大きいか。"""
        if not (t["i"] > 1 and toks[t["i"] - 1]["text"] in ("て", "で") and toks[t["i"] - 1]["pos"] == "SCONJ"):
            return False
        c = self.te_aux.get(t["lemma"], Counter())
        return c[True] > c[False]

    # ---- 読む（chat_bot の toks と同じ形にする） ----
    def read(self, text):
        seg = self.segment(text)
        toks = [dict(i=i, text=w, lemma=lem, pos=pos, dep="dep", head=i) for i, (w, pos, lem) in enumerate(seg)]
        # 言い切りの述語: 「て」のあとの述語（いる・しまう・くる）は前の述語の続き（補助）なので、その前の述語を根にする
        preds = [t for t in toks if t["pos"] in PRED and not self._is_aux(toks, t)]
        root = preds[-1] if preds else (toks[-1] if toks else None)
        if root is not None:
            root["dep"] = "ROOT"
        for t in toks:                                      # 助詞・助動詞は前の語に、述語は次の述語に（流れのとおり）
            if t["pos"] == "ADP" and t["i"] > 0:
                t["dep"], t["head"] = "case", t["i"] - 1
            elif t["pos"] in ("AUX", "PART", "SCONJ") and t["i"] > 0:
                h = next((x for x in reversed(toks[:t["i"]]) if x["pos"] in PRED + ("NOUN",)), toks[t["i"] - 1])
                if h["dep"] == "fixed":                     # 「迷って しまっ た」: 補助の述語のあとは、もとの述語に付ける
                    h = toks[h["head"]]
                t["dep"], t["head"] = ("mark" if t["pos"] == "SCONJ" else "aux"), h["i"]
            elif t["pos"] == "PUNCT":
                t["dep"], t["head"] = "punct", root["i"] if root else t["i"]
            elif t["pos"] in PRED and t is not root and self._is_aux(toks, t) and toks[t["i"] - 2]["pos"] in PRED:
                t["dep"], t["head"] = "fixed", t["i"] - 2       # 「持って いる」: 前の述語の補助（GiNZA の fixed と同じ）
            elif t["pos"] in PRED and t is not root:
                nx = next((x for x in toks[t["i"] + 1:] if x["pos"] in PRED + NOUNISH), None)
                t["head"] = nx["i"] if nx else (root["i"] if root else t["i"])
                t["dep"] = "acl" if nx is not None and nx["pos"] in NOUNISH else "advcl"
        for n, h in self.attach(toks).items():
            c = toks[n + 1]["text"]
            toks[n]["head"] = h
            toks[n]["dep"] = {"が": "nsubj", "は": "nsubj", "を": "obj"}.get(c, "obl")
        for t in toks:                                      # 名詞+の+名詞
            if t["pos"] in NOUNISH and t["dep"] == "dep" and t["i"] + 2 < len(toks) and toks[t["i"] + 1]["text"] == "の" \
                    and toks[t["i"] + 2]["pos"] in NOUNISH:
                t["dep"], t["head"] = "nmod", t["i"] + 2
        return toks


def _spans(toks):
    out, p = set(), 0
    for t in toks:
        out.add((p, p + len(t["text"]), t["pos"]))
        p += len(t["text"])
    return out


def compare(reader, lines, pb):
    c = Counter()
    for ln in lines:
        g = ln["toks"]
        text = "".join(t["text"] for t in g)
        o = reader.read(text)
        sg, so = _spans(g), _spans(o)
        bg, bo = {(a, b) for a, b, _ in sg}, {(a, b) for a, b, _ in so}
        c["語_一致"] += len(bg & bo)
        c["語_GiNZA"] += len(bg)
        c["語_自分"] += len(bo)
        c["品詞_一致"] += len(sg & so)
        pg = {(x["n"], x["c"], x["p"]) for x in pb.pieces(g)}
        po = {(x["n"], x["c"], x["p"]) for x in pb.pieces(o)}
        c["所_一致"] += len(pg & po)
        c["所_GiNZA"] += len(pg)
        c["所_自分"] += len(po)
        c["行"] += 1
    f = lambda a, b, d: round(2 * a / (b + d), 3) if b + d else 0.0
    return dict(行=c["行"], 語の切れ目F1=f(c["語_一致"], c["語_GiNZA"], c["語_自分"]),
                品詞まで一致F1=f(c["品詞_一致"], c["語_GiNZA"], c["語_自分"]),
                所F1=f(c["所_一致"], c["所_GiNZA"], c["所_自分"]), 所_GiNZA=c["所_GiNZA"], 所_自分=c["所_自分"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--stats")
    ap.add_argument("--extra", nargs="*", help="会話の行（GiNZA で読んだ JSON の toks の並び）も試しに")
    ap.add_argument("--words", help="語板（content_boards.json）。語の板に足す")
    a = ap.parse_args()
    import i35_route_grain as R
    works = json.load(open(a.cache, encoding="utf-8"))
    train = [ln for i, w in enumerate(works) if i % 5 for ln in w["lines"] if ln["toks"]]
    test = [ln for i, w in enumerate(works) if not i % 5 for ln in w["lines"] if ln["toks"]][:2000]
    rd = OwnReader()
    for ln in train:
        rd.learn(ln["toks"])
    if a.words:
        rd.learn_words(json.load(open(a.words, encoding="utf-8")))
    pb = R.PieceBoards([])
    rec = dict(exp="I36", 数えた行=len(train), 本の行=compare(rd, test, pb))
    for p in a.extra or []:
        rec[os.path.basename(p)] = compare(rd, [dict(toks=t) for t in json.load(open(p, encoding="utf-8"))], pb)
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    if a.stats:
        with open(a.stats, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()


# ---- spaCy の代わりに差し込む形（chat_bot の self.nlp(text)・nlp.pipe・CB.to_toks がそのまま使える） ----
class _Tok:
    def __init__(self, d, doc):
        self.i, self.text, self.lemma_, self.pos_, self.dep_ = d["i"], d["text"], d["lemma"], d["pos"], d["dep"]
        self.norm_ = d["lemma"]
        self._h, self._doc = d["head"], doc

    @property
    def head(self):
        return self._doc[self._h]


class OwnNLP:
    """GiNZA の代わりの読み手（I36）。読む時に GiNZA を使わない。"""

    def __init__(self, works, words=None):
        self.reader = OwnReader()
        for w in works:
            for ln in w["lines"]:
                if ln["toks"]:
                    self.reader.learn(ln["toks"])
        if words:
            self.reader.learn_words(words)

    def __call__(self, text):
        doc = []
        for d in self.reader.read(text):
            doc.append(_Tok(d, doc))
        return doc

    def pipe(self, texts, batch_size=None):
        for x in texts:
            yield self(x)
