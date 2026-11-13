"""語板 v2: 文章を使って語板を深める（v1 = content_boards.py からの変更、2026-11-05）

1. 文から語義を絞る（クラス D の外れを減らす）
   - 固有名詞は UniDic の細かい品詞で決める: 人名 → individual（人）、地名 → location。WordNet の語義は使わない（トム → uncle_tom を防ぐ）。
   - 普通名詞は、出てくる文ごとに語義の重みを付け直す（EM を数回）:
       重み(語義) ∝ 事前（語の語義の割合。初めは英語 WordNet の語義の回数）
                   × 係る動詞がその役（主語・目的語）にそのクラスを取りやすいか（前の回の数え）
                   × 同じ文と前後 2 文の名詞のクラスとの相性（クラスどうしが一緒に出る比 = lift、前の回の数え）
     語義のクラス = その語義の上位にある、選んだクラスのうちいちばん深いもの。
   - D の帯 = 出てくる文ごとの重みの和。動詞の「主語:クラス」「目的語:クラス」と形容詞の「修飾先:クラス」も、絞った語義の重みで数える。
   - クラスの選び方も、絞った語義の重みで選び直す（v1 は トム が多いので uncle_tom・negroid などが選ばれていた）。
2. 話題の帯 G（約 100）: 語が出るたびに、同じ文と前後 2 文の名詞のクラス・動詞のクラスを数えて「話題:クラス」の割合にする。
   前後の文があるのは文書の文（--docs）だけ。1 文ずつの文（Tatoeba・GSD）は同じ文の中だけ。
3. 試しの文書（--doc_test_frac、学びに使わない）を書き出す: 文ごとの内容語の並び（的はずれな文を見分ける試し、content_light_v2.py）。
1 文ずつの文の試しの分け方は v1 と同じ（同じ順に読んで同じ乱数で並べ替え、先頭 5%）。v1 と同じ試しの組（test_triples）になる。
実行: python lexicon/content_boards_v2.py --conllu tatoeba.conllu gsd.conllu --docs docs_ginza.conllu --wordnet wnjpn.db --out <書き出す先>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from content_boards import CASES, CONTENT, PRIV, REL, ROLE, WordNet  # noqa: E402

BUDGET = dict(A=200, B=300, C=1100, D=100, E=200, G=100)
PERSON, LOCATION = "00007846-n", "00027167-n"


def read_conllu_docs(path):
    """(文書の名前 or None, 文) を順に返す。文書の切れ目は「# newdoc id = …」。"""
    sent, doc = [], None
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            if sent:
                yield doc, sent
            sent = []
            continue
        if line.startswith("# newdoc id = "):
            doc = line.split("= ", 1)[1]
            continue
        if line.startswith("#"):
            continue
        c = line.split("\t")
        if "-" in c[0] or "." in c[0]:
            continue
        sent.append(dict(i=int(c[0]), form=c[1], lemma=c[2], upos=c[3], xpos=c[4], head=int(c[6]), dep=c[7].split(":")[0]))
    if sent:
        yield doc, sent


class Senses:
    """語義の候補と、その事前の割合・クラス。"""

    def __init__(self, wn, words):
        self.wn = wn
        eng = defaultdict(float)
        for syn, f in wn.db.execute("SELECT synset, SUM(freq) FROM sense WHERE lang = 'eng' AND freq IS NOT NULL GROUP BY synset"):
            eng[syn] = f or 0.0
        self.prior, self.nprior = {}, {}       # 語義の事前の割合（全部の品詞 / 名詞の語義だけ。名詞のほうを文で絞る）
        for w in words:
            syns = wn.syn_of.get(w, [])
            if syns:
                p = [eng[s] + 0.5 for s in syns]
                t = sum(p)
                self.prior[w] = {s: x / t for s, x in zip(syns, p)}
                ns = [(s, x) for s, x in zip(syns, p) if s.endswith("-n")]
                if ns:
                    t = sum(x for _, x in ns)
                    self.nprior[w] = {s: x / t for s, x in ns}

    def cands(self, t):
        """その出現の語義の候補と事前の割合。"""
        x = t["xpos"]
        if t["upos"] == "PROPN" or "固有名詞" in x:
            if "人名" in x:
                return {PERSON: 1.0}
            if "地名" in x:
                return {LOCATION: 1.0}
        return self.nprior.get(t["lemma"], {})

    def set_classes(self, classes):
        self.CL = set(classes)
        self.leaf_cache, self.anc_cache = {}, {}

    def leaf(self, s):
        """語義のクラス = 上位にある選んだクラスのうちいちばん深いもの。"""
        if s not in self.leaf_cache:
            a = self.wn.ancestors(s) & self.CL
            self.leaf_cache[s] = max(a, key=lambda c: (self.wn.depth(c), c)) if a else None
        return self.leaf_cache[s]

    def anc(self, s):
        if s not in self.anc_cache:
            self.anc_cache[s] = self.wn.ancestors(s) & self.CL
        return self.anc_cache[s]


DEDUP = 0.0   # main で --dedup から入れる
USE_SEEDS = False   # main で --seed_classes から入れる
# 必ず入れる具体的なクラス（v2.1、童話や日常の文に要る物の種類。回数の多い抽象的なクラスに枠を取られないように）
SEEDS = ["00017222-n", "13104059-n", "11669921-n", "12205694-n",               # 植物・木・花・草
         "00015388-n", "01861778-n", "01503061-n", "02512053-n", "02159955-n", "01661091-n",   # 動物・哺乳・鳥・魚・虫・爬虫類
         "00007846-n", "10235549-n",                                             # 人・親族
         "07555863-n", "07705931-n", "07707451-n", "07881800-n",                 # 食べ物・果物・野菜・飲み物
         "04451818-n", "03094503-n", "03051540-n", "02913152-n", "03259505-n", "04524313-n", "04565375-n",
         "03405725-n", "03183080-n", "03621049-n", "03964744-n", "13384557-n",   # 道具・器・衣類・建物・住まい・乗物・武器・家具・機器・台所道具・玩具・お金
         "05220461-n", "00019128-n", "09225146-n", "09239740-n", "09334396-n", "14580897-n", "14625458-n",
         "11524662-n",                                                           # 身体部位・自然物・水・天体・陸地・材料・金属・天気
         "00027167-n", "15113229-n", "00026192-n"]                               # 場所・時代（期間）・気持ち


def choose_classes(wn, weights, n_noun=80, n_verb=20, min_depth=4):
    """weights: {語義: 重み}（出現の回数 × 語義の割合）。上位の概念ごとに重みを集め、多い順に選ぶ。
    DEDUP > 0 のとき、すでに選んだクラスの祖先か子孫で、覆う重みがその DEDUP 倍以上（ほぼ同じ語の集まり）のものは飛ばす
    （生物 / 生き物、脊椎動物 / 脊索動物 / 哺乳 のような鎖で枠を使わず、植物・道具などの深めのクラスに枠を回す）。"""
    cover = defaultdict(float)
    for s, w in weights.items():
        for a in wn.ancestors(s):
            cover[a] += w
    order = sorted(cover, key=lambda s: -cover[s])

    def pick(cands, n, out=None):
        out = list(out or [])                                        # out に先に入れたもの（種）は飛ばさない
        for s in cands:
            if len(out) >= n:
                break
            if DEDUP > 0:
                anc = wn.ancestors(s)
                if any((b in anc or s in wn.ancestors(b)) and min(cover[s], cover[b]) >= DEDUP * max(cover[s], cover[b]) for b in out):
                    continue
            out.append(s)
        return out
    seeds = [s for s in SEEDS if cover.get(s, 0) > 0] if USE_SEEDS else []
    rest = [s for s in order if s.endswith("-n") and wn.depth(s) >= min_depth and s not in seeds]
    noun = pick(rest, n_noun, seeds[:n_noun])
    verb = pick([s for s in order if s.endswith("-v") and wn.depth(s) >= 1], n_verb)
    return noun + verb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conllu", nargs="+", required=True, help="1 文ずつの文（v1 と同じ順に）")
    ap.add_argument("--docs", nargs="*", default=[], help="前後の文がある文書（parse_docs.py）")
    ap.add_argument("--docs_extra", nargs="*", default=[], help="学びにだけ使う文書（青空文庫など。試しの文書には入れない）")
    ap.add_argument("--private_docs", nargs="*", default=[], help="公開しない資料（Web 小説など）。学びにだけ使う。使うと出力の置き場の名前は _wn で終わる必要がある")
    ap.add_argument("--extra_weight", default="1", help="学びだけの文書（--docs_extra・--private_docs）の文の重み。auto = 内容語の延べ数がほかの文と同じになる重み（v2.3）")
    ap.add_argument("--wordnet", required=True)
    ap.add_argument("--out", default=os.path.join(PRIV, "data", "boards", "v2"))
    ap.add_argument("--n_words", type=int, default=20000)
    ap.add_argument("--min_freq", type=int, default=5)
    ap.add_argument("--test_frac", type=float, default=0.05)
    ap.add_argument("--doc_test_frac", type=float, default=0.1)
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--em", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dedup", type=float, default=0.0, help="クラスの鎖を飛ばす（0 = v2 と同じ、0.7 など = v2.1）")
    ap.add_argument("--seed_classes", action="store_true", help="具体的なクラス（SEEDS）を必ず入れる（v2.1）")
    a = ap.parse_args()
    if a.private_docs and not a.out.rstrip("/\\").endswith("_wn"):
        raise SystemExit("公開しない資料（--private_docs）を使うときは、出力の置き場の名前を _wn で終わらせる（例: v2_3_wn）")
    global DEDUP, USE_SEEDS
    DEDUP = a.dedup
    USE_SEEDS = a.seed_classes
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)
    # 1 文ずつの文: v1 と同じ分け方
    sents = [s for p in a.conllu for _, s in read_conllu_docs(p)]
    rng.shuffle(sents)
    n_test = int(len(sents) * a.test_frac)
    test, train1 = sents[:n_test], sents[n_test:]
    # 文書: 文書ごとに分ける
    docs = defaultdict(list)
    for p in a.docs:
        for d, s in read_conllu_docs(p):
            docs[d].append(s)
    names = sorted(docs)
    random.Random(a.seed + 1).shuffle(names)
    n_dt = int(len(names) * a.doc_test_frac)
    test_docs, train_docs = names[:n_dt], names[n_dt:]
    extra = defaultdict(list)
    for p in a.docs_extra + a.private_docs:
        for d, s in read_conllu_docs(p):
            extra[d].append(s)
    # 学びの文の並び: (文, 前後の文の番号の範囲)。1 文ずつの文は自分だけ
    train = [(s, None) for s in train1]
    docs.update(extra)
    for d in train_docs + sorted(extra):
        base = len(train)
        for k, s in enumerate(docs[d]):
            train.append((s, (base + max(0, k - a.window), base + min(len(docs[d]), k + a.window + 1))))
    n_main = len(train) - sum(len(v) for v in extra.values())          # 先頭 n_main 文がふつうの文、残りが学びだけの文書
    ntok = lambda ss: sum(1 for s, _ in ss for t in s if t["upos"] in CONTENT)  # noqa: E731
    if a.extra_weight == "auto":
        ew = ntok(train[:n_main]) / max(ntok(train[n_main:]), 1) if n_main < len(train) else 1.0
    else:
        ew = float(a.extra_weight)
    swt = [1.0] * n_main + [ew] * (len(train) - n_main)              # 文の重み（数えるときに掛ける）
    print("学びだけの文書の重み", round(ew, 4), flush=True)
    print("文", len(sents), "学び", len(train1), "試し", len(test), "文書", len(names), "学びの文書", len(train_docs), "文書の文", len(train) - len(train1), flush=True)

    freq, wfreq, pos_of = Counter(), Counter(), defaultdict(Counter)
    for (s, _), wt in zip(train, swt):
        for t in s:
            if t["upos"] in CONTENT:
                freq[t["lemma"]] += 1
                wfreq[t["lemma"]] += wt
                pos_of[t["lemma"]][CONTENT[t["upos"]]] += 1
    # 語の並び = 重みを掛けた回数の順（先頭の語ほど光の器に学ばせる）。語に入るかは素の回数（--min_freq）で決める
    words = [w for w, c in wfreq.most_common() if freq[w] >= a.min_freq][:a.n_words]
    W = set(words)
    print("候補の語", len(words), flush=True)
    wn = WordNet(a.wordnet)
    SE = Senses(wn, W)
    is_noun = lambda t: t["upos"] in ("NOUN", "PROPN")  # noqa: E731

    # 出現ごとの語義の重み（名詞だけ）。key = (文の番号, 語の番号)
    post = {}
    for j, (s, _) in enumerate(train):
        for t in s:
            if is_noun(t) and (t["lemma"] in W or "固有名詞" in t["xpos"]):
                c = SE.cands(t)
                if c:
                    post[(j, t["i"])] = dict(c)

    def sense_weights():
        acc = defaultdict(float)
        for p in post.values():
            for s, x in p.items():
                acc[s] += x
        for w in W:                                                  # 動詞のクラスも選べるように、動詞の出現 × 動詞の語義の割合
            nv = pos_of[w]["動詞"]
            if nv:
                vs = {s: x for s, x in SE.prior.get(w, {}).items() if s.endswith("-v")}
                tv_ = sum(vs.values())
                for s, x in vs.items():
                    acc[s] += nv * x / tv_
        return acc

    def verb_classes(lemma):
        """動詞のクラスの割合（語義を英語の回数で重み付け）。"""
        out = Counter()
        for s, x in SE.prior.get(lemma, {}).items():
            if s.endswith("-v"):
                for c in SE.anc(s):
                    out[c] += x
        return out

    SE.set_classes(choose_classes(wn, sense_weights()))
    for it in range(a.em + 1):
        # 出現ごとのクラスの割合（語義の重みをクラスへ）
        leafp = {k: Counter() for k in post}
        for k, p in post.items():
            for s, x in p.items():
                c = SE.leaf(s)
                if c:
                    leafp[k][c] += x
        # 数え: 動詞 × 役 → クラス、クラスどうしが一緒に出る回数（文と前後 2 文）
        arg = defaultdict(Counter)
        argall = defaultdict(Counter)
        sent_cls = []
        for j, (s, _) in enumerate(train):
            by_i = {t["i"]: t for t in s}
            sc = Counter()
            for t in s:
                k = (j, t["i"])
                if k in leafp:
                    sc.update(leafp[k])
                    h = by_i.get(t["head"])
                    if h is not None and h["upos"] == "VERB" and t["dep"] in ("nsubj", "obj"):
                        arg[(h["lemma"], t["dep"])].update(leafp[k])
                        argall[t["dep"]].update(leafp[k])
            sent_cls.append(sc)
        ctx_cls = []
        for j, (s, rngj) in enumerate(train):
            if rngj is None:
                ctx_cls.append(sent_cls[j])
            else:
                c = Counter()
                for m in range(*rngj):
                    c.update(sent_cls[m])
                ctx_cls.append(c)
        if it == a.em:
            break
        pair, single = Counter(), Counter()
        for j in range(len(train)):
            c = ctx_cls[j]
            tot = sum(c.values())
            if tot <= 0:
                continue
            for x, v in c.items():
                single[x] += v
                for y, u in c.items():
                    if x != y:
                        pair[(x, y)] += v * u / tot
        Ns, Np = sum(single.values()), sum(pair.values()) + 1e-9
        lift = {xy: (v / Np) / ((single[xy[0]] / Ns) * (single[xy[1]] / Ns)) for xy, v in pair.items()}
        # 重みを付け直す
        new_prior = defaultdict(lambda: defaultdict(float))
        for k, p in post.items():
            j, i = k
            s = train[j][0]
            t = next(x for x in s if x["i"] == i)
            if len(p) <= 1:
                new_prior[t["lemma"]].update(p)
                continue
            by_i = {x["i"]: x for x in s}
            h = by_i.get(t["head"])
            pri = SE.nprior.get(t["lemma"], p)
            own = leafp[k]
            ctx = ctx_cls[j] - own
            ctot = sum(ctx.values())
            sc = {}
            for sy in p:
                c = SE.leaf(sy)
                v = math.log(pri.get(sy, 1e-6))
                if c is not None and h is not None and h["upos"] == "VERB" and t["dep"] in ("nsubj", "obj"):
                    A, G = arg[(h["lemma"], t["dep"])], argall[t["dep"]]
                    gp = (G[c] + 1) / (sum(G.values()) + len(SE.CL))
                    v += math.log((A[c] - own[c] + 5 * gp) / (sum(A.values()) - sum(own.values()) + 5) / gp)
                if c is not None and ctot > 0:
                    v += sum(u * math.log(lift.get((c, y), 0.5) + 0.05) for y, u in ctx.items()) / ctot
                sc[sy] = v
            mx = max(sc.values())
            z = {sy: math.exp(v - mx) for sy, v in sc.items()}
            tz = sum(z.values())
            post[k] = {sy: v / tz for sy, v in z.items()}
            for sy, v in post[k].items():
                new_prior[t["lemma"]][sy] += v
        for w, p in new_prior.items():
            if w in SE.nprior and len(SE.nprior[w]) > 1:
                tz = sum(p.values())
                SE.nprior[w] = {s: 0.5 * SE.nprior[w][s] + 0.5 * p.get(s, 0) / tz for s in SE.nprior[w]}
        if it == 0:
            SE.set_classes(choose_classes(wn, sense_weights()))   # 絞った重みでクラスを選び直す
        print("語義の絞り込み", it + 1, "回", flush=True)

    classes = sorted(SE.CL, key=lambda c: (c[-1], -wn.depth(c), c))
    cname = {c: wn.name(c) for c in classes}
    leafp = {k: Counter() for k in post}
    for k, p in post.items():
        for s, x in p.items():
            c = SE.leaf(s)
            if c:
                leafp[k][c] += x
    vcls = {}

    raw = defaultdict(lambda: defaultdict(Counter))
    for j, (s, rngj) in enumerate(train):
        wt = swt[j]
        by_i = {t["i"]: t for t in s}
        kids = defaultdict(list)
        for t in s:
            kids[t["head"]].append(t)
        # 話題: 同じ文と前後の文の、名詞のクラスと動詞のクラス
        topic = Counter()
        span = range(*rngj) if rngj else [j]
        for m in span:
            for t in train[m][0]:
                k = (m, t["i"])
                if k in leafp:
                    topic.update({c: v for c, v in leafp[k].items()})
                elif t["upos"] == "VERB" and t["lemma"] in W:
                    if t["lemma"] not in vcls:
                        vc = verb_classes(t["lemma"])
                        tv_ = sum(vc.values())
                        vcls[t["lemma"]] = {c: v / tv_ for c, v in vc.items()} if tv_ else {}
                    topic.update(vcls[t["lemma"]])
        for t in s:
            w = t["lemma"]
            if w not in W:
                continue
            P = CONTENT.get(t["upos"])
            R = raw[w]
            R["A"][f"役:{ROLE.get(t['dep'], 'その他')}"] += wt
            head = by_i.get(t["head"])
            own = leafp.get((j, t["i"]), Counter())
            for c, v in topic.items():
                x = v - own.get(c, 0)
                if x > 1e-6:
                    R["G"][f"話題:{cname[c]}"] += x * wt
            if P == "名詞":
                for sy, x in post.get((j, t["i"]), {}).items():
                    for c in SE.anc(sy):
                        R["D"][f"クラス:{cname[c]}"] += x * wt
            for kk in kids[t["i"]]:
                if kk["dep"] == "case" and kk["upos"] == "ADP":
                    tag = "格" if kk["lemma"] in CASES else "取り立て"
                    R["A"][f"{tag}:{kk['lemma']}"] += wt
                elif kk["dep"] == "cop":
                    R["A"]["述語:コピュラ"] += wt
                elif kk["dep"] == "aux":
                    R["A"][f"助動詞:{kk['lemma']}"] += wt
                elif kk["dep"] == "mark":
                    R["A"][f"接続:{kk['lemma']}"] += wt
                if P == "名詞" and kk["upos"] == "ADJ" and kk["dep"] in ("amod", "acl"):
                    R["B"][f"性質:{kk['lemma']}"] += wt
                if P == "動詞" and kk["upos"] == "ADV" and kk["dep"] == "advmod":
                    R["B"][f"様子:{kk['lemma']}"] += wt
                if P in ("動詞", "形容詞") and kk["dep"] in ("nsubj", "obj") and is_noun(kk):
                    for c, x in leafp.get((j, kk["i"]), {}).items():
                        R["C"][f"{'主語' if kk['dep'] == 'nsubj' else '目的語'}:{cname[c]}"] += x * wt
            if P == "名詞" and head is not None:
                if head["upos"] == "ADJ" and t["dep"] == "nsubj":
                    R["B"][f"性質:{head['lemma']}"] += wt
                if head["upos"] == "VERB":
                    if t["dep"] == "nsubj":
                        R["C"][f"{head['lemma']}/する"] += wt
                    elif t["dep"] == "obj":
                        R["C"][f"{head['lemma']}/される"] += wt
                    elif t["dep"] == "obl":
                        cs = [kk["lemma"] for kk in kids[t["i"]] if kk["dep"] == "case" and kk["lemma"] in ("で", "に")]
                        if cs:
                            R["C"][f"{head['lemma']}/{cs[0]}"] += wt
            if P == "形容詞" and head is not None and is_noun(head) and t["dep"] in ("amod", "acl"):
                for c, x in leafp.get((j, head["i"]), {}).items():
                    R["C"][f"修飾先:{cname[c]}"] += x * wt
    # 名詞で D が付かなかった語（文に出ても語義の候補がない）と、動詞・形容詞の D・E は語の事前の割合（絞ったもの）から
    for w in words:
        P = pos_of[w].most_common(1)[0][0]
        pri = SE.nprior.get(w, {}) if P == "名詞" else {s: x for s, x in SE.prior.get(w, {}).items() if P == "形容詞" or s.endswith("-v")}
        if P != "名詞" or not raw[w]["D"]:
            for sy, x in pri.items():
                for c in SE.anc(sy):
                    raw[w]["D"][f"クラス:{cname[c]}"] += x
        if P == "名詞":
            for sy, x in pri.items():
                for link, s2 in wn.links[sy]:
                    c = SE.leaf(s2)
                    if c:
                        raw[w]["E"][f"{REL[link]}:{cname[c]}"] += x

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
                attrs[b] = {x: round(n / t, 4) for x, n in sorted(c.items(), key=lambda kv: -kv[1]) if n / t >= 5e-5}
        boards.append(dict(name=w, pos=pos_of[w].most_common(1)[0][0], freq=freq[w],
                           n_obs={b: round(sum(raw[w][b].values()), 2) for b in raw[w]}, attrs=attrs))

    triples = []
    for s in test:
        by_i = {t["i"]: t for t in s}
        for t in s:
            h = by_i.get(t["head"])
            if t["lemma"] in W and h is not None and h["lemma"] in W and h["upos"] == "VERB" and t["dep"] in ("nsubj", "obj") and is_noun(t):
                triples.append(dict(noun=t["lemma"], verb=h["lemma"], role="する" if t["dep"] == "nsubj" else "される"))
    tdocs = [dict(doc=d, sents=[[dict(w=t["lemma"], pos=CONTENT[t["upos"]], dep=t["dep"]) for t in s if t["upos"] in CONTENT] for s in docs[d]])
             for d in test_docs]
    json.dump(boards, open(os.path.join(a.out, "content_boards.json"), "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(dict(公開してよいか=not a.private_docs, 公開しない資料=a.private_docs, 資料=dict(文=a.conllu, 文書=a.docs, 学びだけの文書=a.docs_extra),
                   名前の決まり="公開しない資料を混ぜた語板は、置き場の名前が _wn で終わる（例: v2_3_wn）。そこから作った器・明るさも同じ置き場に置く"),
              open(os.path.join(a.out, "sources.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(dict(slots=slots, classes={cname[c]: c for c in classes}), open(os.path.join(a.out, "slots.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(triples, open(os.path.join(a.out, "test_triples.json"), "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(tdocs, open(os.path.join(a.out, "test_docs.json"), "w", encoding="utf-8"), ensure_ascii=False)
    # 語義の例（外れが減ったかを見る）
    ex = {}
    for w in ["トム", "メアリー", "犬", "本", "リンゴ", "手", "木", "目", "東京", "先生", "電話", "家", "竹", "猫", "包丁", "着物"]:
        b = next((x for x in boards if x["name"] == w), None)
        if b and "D" in b["attrs"]:
            ex[w] = list(b["attrs"]["D"].items())[:4]
    cover = {b: sum(1 for x in boards if b in x["attrs"]) for b in BUDGET}
    print(json.dumps(dict(語=len(boards), 名札={b: len(v) for b, v in slots.items()}, 帯を持つ語=cover, 試しの組=len(triples), 試しの文書=len(tdocs),
                          クラス=[cname[c] for c in classes], 語義の例=ex), ensure_ascii=False))


if __name__ == "__main__":
    main()
