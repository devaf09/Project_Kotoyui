"""文・要点・間: CoNLL-U の読み・内容語と詰め物への分け方・機能語の板（意味の印）・内容語の表記（活用）。

実装は i48ab_long_grain から移した物（i48ab_long_grain は、ここから読み直して同じ名前で見せる）。
手で置く物: CONTENT（内容語の品詞）、FUNC_DEP（内容語の品詞でも詰め物にする係り）、MARKS（機能語の板の機能 → 意味の印）。
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

CONTENT = ("NOUN", "PROPN", "PRON", "NUM", "VERB", "ADJ", "ADV", "DET")
FUNC_DEP = ("fixed", "aux")
MARKS = {"テンス": "過去", "否定": "否定", "丁寧": "丁寧", "アスペクト": "進行"}   # 機能語の板の「機能」 → 意味の印（か・？ は問い）


def read_conllu(path, limit=None):
    out, cur, meta = [], [], {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if cur:
                    out.append(dict(id=meta.get("sent_id", str(len(out))), text=meta.get("text", ""), toks=cur))
                    if limit and len(out) >= limit:
                        return out
                cur, meta = [], {}
                continue
            if line.startswith("#"):
                m = re.match(r"# (\w+) = (.*)", line)
                if m:
                    meta[m.group(1)] = m.group(2)
                continue
            c = line.split("\t")
            if "-" in c[0] or "." in c[0]:
                continue
            cur.append(dict(form=c[1], lemma=c[2], upos=c[3], xpos=c[4], dep=c[7]))
    if cur:
        out.append(dict(id=meta.get("sent_id", str(len(out))), text=meta.get("text", ""), toks=cur))
    return out


def is_content(t):
    return t["upos"] in CONTENT and t["dep"] not in FUNC_DEP


def split(sent):
    """文 → (内容語の並び [(原形, 表記, 品詞)], 詰め物の並び（内容語 i の後ろ、最後は文末まで）, 頭の詰め物)。"""
    T = sent["toks"]
    ci = [i for i, t in enumerate(T) if is_content(t)]
    if not ci:
        return [], [], ""
    head = "".join(t["form"] for t in T[:ci[0]])
    cont = [(T[i]["lemma"], T[i]["form"], T[i]["upos"]) for i in ci]
    gaps = []
    for k, i in enumerate(ci):
        j = ci[k + 1] if k + 1 < len(ci) else len(T)
        gaps.append("".join(t["form"] for t in T[i + 1:j]))
    return cont, gaps, head


class Func:
    def __init__(self, func_boards):
        self.by = defaultdict(list)
        for w in func_boards:
            self.by[w["word"]].append(w)

    def board(self, t):
        c = self.by.get(t["lemma"]) or self.by.get(t["form"]) or []
        for w in c:
            if t["xpos"] in w["attrs"].get("品詞", {}):
                return w
        return c[0] if c else None

    def marks(self, toks):
        m = set()
        for t in toks:
            if is_content(t):
                continue
            if t["form"] in ("？", "?") or (t["lemma"] == "か" and t["xpos"].endswith("終助詞")):
                m.add("問い")
            w = self.board(t)
            for f in (w["attrs"].get("機能", {}) if w else {}):
                if f in MARKS:
                    m.add(MARKS[f])
        return m


class Surface:
    """内容語の表記（活用）: (原形, 次の詰め物の頭の字) → 表記。無い原形は (原形の最後の字, 品詞, 頭の字) → (削る字数, 足す字)。"""

    def __init__(self):
        self.t, self.r = defaultdict(Counter), defaultdict(Counter)

    @staticmethod
    def _head(g):
        return (g or "")[:1]

    def add(self, lemma, form, upos, nxt):
        h = self._head(nxt)
        self.t[(lemma, h)][form] += 1
        n = 0
        while n < min(len(lemma), len(form)) and lemma[n] == form[n]:
            n += 1
        self.r[(lemma[-1:], upos, h)][(len(lemma) - n, form[n:])] += 1

    def form(self, lemma, upos, nxt):
        c = self.t.get((lemma, self._head(nxt)))
        if c:
            return c.most_common(1)[0][0]
        c = self.r.get((lemma[-1:], upos, self._head(nxt)))
        if c:
            cut, add = c.most_common(1)[0][0]
            return lemma[: len(lemma) - cut] + add
        return lemma


def norm(s):
    return re.sub(r"[\s　]", "", s)


_LAZY = {  # 文・要点・間のうち、まだ試しのファイルに実装がある物（読み込みの輪を避けて、使う時に読む）
    "MARK_NAMES": "i48ae_marks", "MODES": "i48ae_marks", "clause_marks": "i48ae_marks", "feats": "i48ae_marks", "items": "i48ae_marks",
    "keys_of": "i48ak_teach", "made_items": "i48ak_teach", "made_sentence": "i48ak_teach",
    "FUNC_UPOS": "i48aj_resonance", "PRED": "i48aj_resonance", "next_pred": "i48aj_resonance",
}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(name)
