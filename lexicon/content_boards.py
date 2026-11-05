"""語板 v1: 内容語（名詞・動詞・形容詞）の属性を、例文の係り受けと日本語 WordNet から数える

持ち主の案（2026-11-05）を直した形（content_attr_design.md）。属性は帯（観点）ごとに分け、帯の中で割合を持つ:
  A 文法の振る舞い（約 200）: 名詞 = 付く格助詞・取り立て助詞、文の中の役（主語・目的語・連体…）、コピュラ。
                              動詞・形容詞 = 付く助動詞、接続の形（て・ば・ので…）、文の中の役。
  B 静的（約 300）: 名詞 = 修飾する形容詞・形状詞（「赤いリンゴ」）と、述語の形容詞（「リンゴは赤い」）。動詞 = 様子の副詞。
                     数値（大きさ・重さ）は桁の帯にする予定だが、例文からは取りにくいので v1 では形容詞（大きい・重い）で代わりにする。
  C 動的（約 1,100）: 名詞 = 「動詞/する」（主語）、「動詞/される」（目的語）、「動詞/で」「動詞/に」（場所・道具・着点）。
                      動詞 = 主語・目的語のクラス（「主語:人」「目的語:食べ物」）。形容詞 = 主語・修飾先のクラス。
  D クラス（約 100）: 日本語 WordNet の上位の概念（名詞 80・動詞 20）。語のすべての意味の上位を、意味の数で割って持つ。
  E 関係（約 200）: WordNet の部分・成員・材料（部分を持つ / 〜の部分）× 行き先のクラス。
例文: UD Japanese-GSD と Tatoeba の文に GiNZA で品詞・係り受けを付けたもの（lexicon/parse_tatoeba.py。同じ解析器でそろえる）。
試しの文（崩した文を見分ける試し）に使う文は、属性を数えるのに使わない（--test_frac）。
出す物（非公開の置き場 ../Project_Kotoyui_private/data/boards/v1/）:
  content_boards.json（1 語 = {name, pos, freq, attrs: {帯: {値: 割合}}}）、slots.json（帯ごとの名札）、test_triples.json（試しの文の 名詞・動詞・役）
実行: python lexicon/content_boards.py --conllu a.conllu b.conllu --wordnet wnjpn.db --out <書き出す先>
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PRIV = os.environ.get("KOTOYUI_PRIVATE", os.path.join(HERE, "..", "..", "Project_Kotoyui_private"))
BUDGET = dict(A=200, B=300, C=1100, D=100, E=200)
CONTENT = {"NOUN": "名詞", "PROPN": "名詞", "VERB": "動詞", "ADJ": "形容詞"}
CASES = {"が", "を", "に", "で", "と", "から", "へ", "まで", "より", "の"}
ROLE = {"nsubj": "主語", "obj": "目的語", "obl": "補語", "nmod": "連体", "root": "文末", "compound": "複合", "acl": "連体修飾",
        "advcl": "従属節", "ccomp": "補文", "amod": "連体修飾", "advmod": "連用修飾", "iobj": "補語", "dislocated": "提示", "csubj": "主語節"}
REL = {"hprt": "部分を持つ", "mprt": "の部分", "hmem": "成員を持つ", "mmem": "の成員", "hsub": "材料を持つ", "msub": "の材料"}


def read_conllu(path):
    sent = []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            if sent:
                yield sent
            sent = []
            continue
        if line.startswith("#"):
            continue
        c = line.split("\t")
        if "-" in c[0] or "." in c[0]:
            continue
        sent.append(dict(i=int(c[0]), form=c[1], lemma=c[2], upos=c[3], xpos=c[4], head=int(c[6]), dep=c[7].split(":")[0]))
    if sent:
        yield sent


class WordNet:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.syn_of = defaultdict(list)
        for lemma, syn, pos in self.db.execute("SELECT word.lemma, sense.synset, synset.pos FROM word JOIN sense ON word.wordid = sense.wordid "
                                               "JOIN synset ON synset.synset = sense.synset WHERE word.lang = 'jpn'"):
            self.syn_of[lemma].append(syn)
        self.anc = defaultdict(dict)
        for s1, s2, h in self.db.execute("SELECT synset1, synset2, hops FROM ancestor"):
            self.anc[s1][s2] = h
        self.jname = {}
        for syn, lemma in self.db.execute("SELECT sense.synset, word.lemma FROM sense JOIN word ON word.wordid = sense.wordid WHERE word.lang = 'jpn' ORDER BY sense.synset, sense.rank"):
            self.jname.setdefault(syn, lemma)
        self.ename = dict(self.db.execute("SELECT synset, name FROM synset"))
        self.links = defaultdict(list)
        for s1, s2, link in self.db.execute("SELECT synset1, synset2, link FROM synlink WHERE link IN ('hprt','mprt','hmem','mmem','hsub','msub')"):
            self.links[s1].append((link, s2))

    def name(self, syn):
        """クラスの名前 = 日本語の見出し（英語の名前）。日本語の見出しだけだと分かりにくいものがある（「全般」= whole）。"""
        j, e = self.jname.get(syn), self.ename.get(syn, syn)
        return f"{j}({e})" if j and j != e else e

    def depth(self, syn):
        return max(self.anc[syn].values(), default=0)

    def ancestors(self, syn):
        return {syn} | set(self.anc[syn])


def choose_classes(wn, words, freq, n_noun=80, n_verb=20, min_depth=4):
    """クラス = 候補の語を多く覆う上位の概念。名詞は深さ 4 以上（実体・物・全体のような根に近いものは除く）から、覆う語の回数の多い順。"""
    cover = defaultdict(float)
    for w in words:
        syns = wn.syn_of.get(w, [])
        if not syns:
            continue
        seen = set()
        for s in syns:
            seen |= wn.ancestors(s)
        for a in seen:
            cover[a] += freq[w]
    noun = [s for s in sorted(cover, key=lambda s: -cover[s]) if s.endswith("-n") and wn.depth(s) >= min_depth][:n_noun]
    verb = [s for s in sorted(cover, key=lambda s: -cover[s]) if s.endswith("-v") and wn.depth(s) >= 1][:n_verb]
    return noun + verb


def classes_of(wn, w, classes):
    """語のクラスの割合（意味ごとに等分し、意味の上位にあるクラスに配る）。"""
    syns = wn.syn_of.get(w, [])
    c = Counter()
    for s in syns:
        anc = wn.ancestors(s) & classes
        for a in anc:
            c[a] += 1 / len(syns)
    return c


def best_class(wn, w, classes, cache):
    """語のいちばん細かい（深い）クラス 1 つ（動詞の主語・目的語のクラスに使う）。"""
    if w not in cache:
        c = classes_of(wn, w, classes)
        cache[w] = max(c, key=lambda s: (c[s], wn.depth(s))) if c else None
    return cache[w]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conllu", nargs="+", required=True)
    ap.add_argument("--wordnet", required=True)
    ap.add_argument("--out", default=os.path.join(PRIV, "data", "boards", "v1"))
    ap.add_argument("--n_words", type=int, default=20000)
    ap.add_argument("--min_freq", type=int, default=5)
    ap.add_argument("--test_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)
    sents = [s for p in a.conllu for s in read_conllu(p)]
    rng.shuffle(sents)
    n_test = int(len(sents) * a.test_frac)
    test, train = sents[:n_test], sents[n_test:]
    print("文", len(sents), "学び", len(train), "試し", len(test), flush=True)

    freq = Counter()
    pos_of = defaultdict(Counter)
    for s in train:
        for t in s:
            if t["upos"] in CONTENT:
                freq[t["lemma"]] += 1
                pos_of[t["lemma"]][CONTENT[t["upos"]]] += 1
    words = [w for w, c in freq.most_common() if c >= a.min_freq][:a.n_words]
    W = set(words)
    print("候補の語", len(words), flush=True)

    wn = WordNet(a.wordnet)
    classes = choose_classes(wn, words, freq)
    CL = set(classes)
    cname = {s: wn.name(s) for s in classes}
    bc_cache = {}

    raw = defaultdict(lambda: defaultdict(Counter))                  # 語 → 帯 → 値 → 回数
    for s in train:
        by_i = {t["i"]: t for t in s}
        kids = defaultdict(list)
        for t in s:
            kids[t["head"]].append(t)
        for t in s:
            w = t["lemma"]
            if w not in W:
                continue
            P = CONTENT.get(t["upos"])
            R = raw[w]
            R["A"][f"役:{ROLE.get(t['dep'], 'その他')}"] += 1
            head = by_i.get(t["head"])
            for k in kids[t["i"]]:
                if k["dep"] == "case" and k["upos"] == "ADP":
                    tag = "格" if k["lemma"] in CASES else "取り立て"
                    R["A"][f"{tag}:{k['lemma']}"] += 1
                elif k["dep"] == "cop":
                    R["A"]["述語:コピュラ"] += 1
                elif k["dep"] == "aux":
                    R["A"][f"助動詞:{k['lemma']}"] += 1
                elif k["dep"] == "mark":
                    R["A"][f"接続:{k['lemma']}"] += 1
                if P == "名詞" and k["upos"] == "ADJ" and k["dep"] in ("amod", "acl"):
                    R["B"][f"性質:{k['lemma']}"] += 1
                if P == "動詞" and k["upos"] == "ADV" and k["dep"] == "advmod":
                    R["B"][f"様子:{k['lemma']}"] += 1
                if P in ("動詞", "形容詞") and k["dep"] in ("nsubj", "obj") and k["upos"] in ("NOUN", "PROPN"):
                    c = best_class(wn, k["lemma"], CL, bc_cache)
                    if c:
                        R["C"][f"{'主語' if k['dep'] == 'nsubj' else '目的語'}:{cname[c]}"] += 1
            if P == "名詞" and head is not None:
                if head["upos"] == "ADJ" and t["dep"] == "nsubj":
                    R["B"][f"性質:{head['lemma']}"] += 1
                if head["upos"] == "VERB":
                    if t["dep"] == "nsubj":
                        R["C"][f"{head['lemma']}/する"] += 1
                    elif t["dep"] == "obj":
                        R["C"][f"{head['lemma']}/される"] += 1
                    elif t["dep"] == "obl":
                        cs = [k["lemma"] for k in kids[t["i"]] if k["dep"] == "case" and k["lemma"] in ("で", "に")]
                        if cs:
                            R["C"][f"{head['lemma']}/{cs[0]}"] += 1
            if P == "形容詞" and head is not None and head["upos"] in ("NOUN", "PROPN") and t["dep"] in ("amod", "acl"):
                c = best_class(wn, head["lemma"], CL, bc_cache)
                if c:
                    R["C"][f"修飾先:{cname[c]}"] += 1
    for w in words:
        if pos_of[w].most_common(1)[0][0] == "名詞":
            for s, p in classes_of(wn, w, CL).items():
                raw[w]["D"][f"クラス:{cname[s]}"] += p
            for syn in wn.syn_of.get(w, []):
                for link, s2 in wn.links[syn]:
                    c = best_class(wn, wn.name(s2), CL, bc_cache) or next(iter(wn.ancestors(s2) & CL), None)
                    if c:
                        raw[w]["E"][f"{REL[link]}:{cname[c]}"] += 1
        else:
            for s, p in classes_of(wn, w, CL).items():
                raw[w]["D"][f"クラス:{cname[s]}"] += p

    # 帯ごとに名札を選ぶ（回数の多い順に予算まで）
    tot = defaultdict(Counter)
    for w in words:
        for b, c in raw[w].items():
            for x, n in c.items():
                tot[b][x] += n
    slots = {b: [x for x, _ in tot[b].most_common(BUDGET[b])] for b in BUDGET}
    S = {b: set(v) for b, v in slots.items()}
    boards = []
    for w in words:
        attrs = {}
        for b in BUDGET:
            c = {x: n for x, n in raw[w][b].items() if x in S[b]}
            t = sum(c.values())
            if t > 0:
                attrs[b] = {x: round(n / t, 4) for x, n in sorted(c.items(), key=lambda kv: -kv[1])}
        boards.append(dict(name=w, pos=pos_of[w].most_common(1)[0][0], freq=freq[w], n_obs={b: round(sum(raw[w][b].values()), 2) for b in raw[w]}, attrs=attrs))

    # 試しの文: 名詞・動詞・役（主語 / 目的語）の組
    triples = []
    for s in test:
        by_i = {t["i"]: t for t in s}
        for t in s:
            h = by_i.get(t["head"])
            if t["lemma"] in W and h is not None and h["lemma"] in W and h["upos"] == "VERB" and t["dep"] in ("nsubj", "obj") and t["upos"] in ("NOUN", "PROPN"):
                triples.append(dict(noun=t["lemma"], verb=h["lemma"], role="する" if t["dep"] == "nsubj" else "される"))
    json.dump(boards, open(os.path.join(a.out, "content_boards.json"), "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(dict(slots=slots, classes={cname[s]: s for s in classes}), open(os.path.join(a.out, "slots.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(triples, open(os.path.join(a.out, "test_triples.json"), "w", encoding="utf-8"), ensure_ascii=False)
    cover = {b: sum(1 for x in boards if b in x["attrs"]) for b in BUDGET}
    print(json.dumps(dict(語=len(boards), 名札={b: len(v) for b, v in slots.items()}, 帯を持つ語=cover, 試しの組=len(triples),
                          品詞=dict(Counter(b["pos"] for b in boards))), ensure_ascii=False))


if __name__ == "__main__":
    main()
