"""新しい違和感 4 つの物差しを、数えだけで確かめる（持ち主の案、未確定。重い学びは無し）。

1. 経験からの違和感（属性・並び順）: Tatoeba の学びの文で、内容語の並びの数え（品詞の 2 つ組・語の 2 つ組・属性の 2 つ組。属性は語板の D の帯のいちばん強い物）を取る。
   試しの文の行の「いちばん珍しい所」（位置ごとの −log P(x_k | x_{k−1}) の最大）と平均で、元の行と、
   (a) 語順を入れ替えた行（隣り合う内容語 2 つ）、(b) 意味の合わない語に替えた行（同じ品詞で、D の帯の属性が違う語）を見分ける AUC。
2. 展開①「一つの行では説明ができない」: 会話の資料（rpc）の話し手の 1 回の発言で、人が 1 文で言ったか、2 文以上に分けたかを、
   発言の内容語の数・述語の数（数えの帯）から見分ける AUC（学び 8 割で P(分けた | 帯) を数え、残り 2 割で試す）。
   あわせて、同じ発言の中の内容語の組のうち、別々の文に分かれて書かれた組の割合。
3. 展開②「これで終わっていいの？」: 同じ発言の中の文ごとに、「同じ話し手が続けて言う」（最後の文でない）か「ここで相手に渡る」（最後の文）かを、
   文の終わりの形（記号を除いた最後の語の原形と品詞。数えから出し、手で並べない。少ない物は品詞に戻す）・問いかどうかから見分ける AUC。
4. 相手の返事から後で付く違和感: bot の会話の記録（md）で、bot の行に先生が付けた種類（言い方・足りない・つながり・文法・なし＝褒め）を答えにし、
   相手の次の行の印（聞き返し・問い・感謝・長さの帯・bot の行と同じ内容語があるか・新しい内容語の数の帯・頭の語）から、種類を言い当てる割合
   （学び 8 割で数え、残り 2 割。いちばん多い種類を言い続けた時と比べる）と、「直す系か褒めか」の AUC。
どれも、値 = −log P（期待の確からしさ）の形で、「期待との差」と同じ形に入る。
手で置く物: 帯の境（内容語・述語の数は 0・1・2・3・4・5〜7・8 以上、長さは 0〜5・6〜10・11〜20・21〜40・41 以上の字）、少ない語の境（10 回）、
  学びと試しの分け方（8 割・2 割、名前の crc32）、乱数の種、ならし 0.5。
python tools/i48_odd_new_kinds.py --conllu ... --content ... --rpc ... --logs "記録の md の glob" --out out.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
import zlib
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "intervention"))
import i48ab_long_grain as G  # noqa: E402

PUNCT_END = "。！？!?"
RARE = 10
A = 0.5


def auc(pos, neg):
    """pos が neg より大きい確からしさ（同点は半分）。"""
    a = np.array(list(neg) + list(pos), dtype=float)
    r = a.argsort().argsort() + 1.0
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r)
    r = (sums / cnt)[inv]
    return float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def bucket(n, edges=(0, 1, 2, 3, 4, 5, 8)):
    b = 0
    for i, e in enumerate(edges):
        if n >= e:
            b = i
    return b


class Bigram:
    def __init__(self):
        self.c, self.u = Counter(), Counter()

    def add(self, seq):
        seq = ["<s>"] + seq + ["</s>"]
        for a, b in zip(seq, seq[1:]):
            self.c[(a, b)] += 1
            self.u[a] += 1
        self.V = None

    def nlp(self, seq):
        """位置ごとの −log P(x_k | x_{k−1})（ならし A）。"""
        if self.V is None:
            self.V = len({b for _, b in self.c}) + 1
        seq = ["<s>"] + seq + ["</s>"]
        return [-math.log((self.c[(a, b)] + A) / (self.u[a] + A * self.V)) for a, b in zip(seq, seq[1:])]


def part1(a, res):
    cb = {}
    for w in json.load(open(a.content, encoding="utf-8")):
        if w["name"] not in cb or w.get("freq", 0) > cb[w["name"]].get("freq", 0):
            cb[w["name"]] = w

    def attr(k):
        w = cb.get(k[0])
        D = (w or {}).get("attrs", {}).get("D", {}) if w else {}
        return max(D, key=D.get) if D else "D:無し/" + k[1]
    sents = [s for s in G.read_conllu(a.conllu) if s["toks"]]
    te = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 == 0][: a.test]
    tr = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 != 0][:-2000][: a.train]
    del sents
    kinds = {"品詞": Bigram(), "語": Bigram(), "属性": Bigram()}
    kc = Counter()
    for s in tr:
        keys = [(l, u) for l, _, u in G.split(s)[0]]
        if not keys:
            continue
        kc.update(keys)
        kinds["品詞"].add([u for _, u in keys])
        kinds["語"].add([f"{l}/{u}" for l, u in keys])
        kinds["属性"].add([attr(k) for k in keys])
    rng = np.random.default_rng(5)
    by_pos = defaultdict(list)
    for k, n in kc.items():
        if n >= RARE:
            by_pos[k[1]].append(k)
    vals = {m: {"元": [], "語順": [], "属性": []} for m in kinds}

    def score(keys):
        out = {}
        for m, bg in kinds.items():
            seq = [u for _, u in keys] if m == "品詞" else ([f"{l}/{u}" for l, u in keys] if m == "語" else [attr(k) for k in keys])
            v = bg.nlp(seq)
            out[m] = (max(v), float(np.mean(v)))
        return out
    n_used = 0
    for s in te:
        keys = [(l, u) for l, _, u in G.split(s)[0]]
        if len(keys) < 2:
            continue
        sw = [k for k in range(len(keys) - 1) if keys[k] != keys[k + 1]]
        p = int(rng.integers(len(keys)))
        cand = [k for k in by_pos.get(keys[p][1], []) if attr(k) != attr(keys[p])]
        if not sw or not cand:
            continue
        k = sw[int(rng.integers(len(sw)))]
        k2 = list(keys)
        k2[k], k2[k + 1] = k2[k + 1], k2[k]
        k3 = list(keys)
        k3[p] = cand[int(rng.integers(len(cand)))]
        for nm, ks in (("元", keys), ("語順", k2), ("属性", k3)):
            sc = score(ks)
            for m in kinds:
                vals[m][nm].append(sc[m])
        n_used += 1
    out = dict(試しの行=n_used)
    for m in kinds:
        for nm in ("語順", "属性"):
            out[f"{m}の並び_{nm}を替えた行"] = dict(
                いちばん珍しい所のAUC=round(auc([x[0] for x in vals[m][nm]], [x[0] for x in vals[m]["元"]]), 4),
                平均のAUC=round(auc([x[1] for x in vals[m][nm]], [x[1] for x in vals[m]["元"]]), 4))
    res["1_経験からの違和感"] = out
    print("1", json.dumps(out, ensure_ascii=False), flush=True)


def sentences(toks):
    out, cur = [], []
    for t in toks:
        cur.append(t)
        if t["text"] in PUNCT_END:
            out.append(cur)
            cur = []
    if any(t["pos"] != "PUNCT" for t in cur):
        out.append(cur)
    return [x for x in out if any(t["pos"] != "PUNCT" for t in x)]


CONTENT = ("NOUN", "PROPN", "VERB", "ADJ", "ADV")
PRED = ("VERB", "ADJ")


def part23(a, res):
    data = json.load(open(a.rpc, encoding="utf-8"))
    tr = [d for d in data if zlib.crc32(d["name"].encode()) % 5 != 0]
    te = [d for d in data if zlib.crc32(d["name"].encode()) % 5 == 0]

    def turns(ds):
        for d in ds:
            for ln in d["lines"]:
                S = sentences(ln["toks"])
                if S:
                    yield S
    # 2. 展開①
    cnt, tot = Counter(), Counter()
    pair_all = pair_split = 0
    for S in turns(tr):
        cw = [t for x in S for t in x if t["pos"] in CONTENT]
        b = (bucket(len(cw)), bucket(sum(t["pos"] in PRED for t in cw)))
        tot[b] += 1
        cnt[b] += len(S) >= 2
    for S in turns(te):
        where = defaultdict(set)
        for si, x in enumerate(S):
            for t in x:
                if t["pos"] in CONTENT:
                    where[t["lemma"]].add(si)
        L = sorted(where)
        for i in range(len(L)):
            for j in range(i + 1, len(L)):
                pair_all += 1
                pair_split += not (where[L[i]] & where[L[j]])
    pos, neg, pos_c, neg_c, pos_p, neg_p = [], [], [], [], [], []
    base = sum(cnt.values()) / max(1, sum(tot.values()))
    for S in turns(te):
        cw = [t for x in S for t in x if t["pos"] in CONTENT]
        npred = sum(t["pos"] in PRED for t in cw)
        b = (bucket(len(cw)), bucket(npred))
        p = (cnt[b] + A * base * 2) / (tot[b] + A * 2) if tot[b] else base
        (pos if len(S) >= 2 else neg).append(p)
        (pos_c if len(S) >= 2 else neg_c).append(len(cw))
        (pos_p if len(S) >= 2 else neg_p).append(npred)
    res["2_展開①_一つの行では説明できない"] = dict(
        試しの発言=len(pos) + len(neg), 二文以上に分けた割合=round(len(pos) / max(1, len(pos) + len(neg)), 4),
        内容語と述語の帯で見分けるAUC=round(auc(pos, neg), 4), 内容語の数だけのAUC=round(auc(pos_c, neg_c), 4), 述語の数だけのAUC=round(auc(pos_p, neg_p), 4),
        同じ発言の内容語の組が別々の文に分かれた割合=round(pair_split / max(1, pair_all), 4), 組の数=pair_all)
    print("2", json.dumps(res["2_展開①_一つの行では説明できない"], ensure_ascii=False), flush=True)
    # 3. 展開②
    def end_key(x):
        T = [t for t in x if t["pos"] != "PUNCT"]
        last = T[-1] if T else None
        punct = next((t["text"] for t in reversed(x) if t["pos"] == "PUNCT"), "")
        return (last["lemma"] + "/" + last["pos"]) if last else "無し", (last["pos"] if last else "無し"), punct
    ce, te_, cp, tp, cq, tq = Counter(), Counter(), Counter(), Counter(), Counter(), Counter()
    n_cont = n_all = 0
    for S in turns(tr):
        for si, x in enumerate(S):
            cont = si < len(S) - 1
            k, ps, pu = end_key(x)
            te_[k] += 1
            ce[k] += cont
            tp[(ps, pu)] += 1
            cp[(ps, pu)] += cont
            n_cont += cont
            n_all += 1
    base = n_cont / max(1, n_all)
    pos, neg, pos_q, neg_q = [], [], [], []
    for S in turns(te):
        for si, x in enumerate(S):
            k, ps, pu = end_key(x)
            if te_[k] >= RARE:
                p = (ce[k] + A) / (te_[k] + 2 * A)
            elif tp[(ps, pu)]:
                p = (cp[(ps, pu)] + A) / (tp[(ps, pu)] + 2 * A)
            else:
                p = base
            q = float(pu in ("？", "?"))
            if si < len(S) - 1:
                pos.append(p)
                pos_q.append(-q)
            else:
                neg.append(p)
                neg_q.append(-q)
    res["3_展開②_これで終わっていいの"] = dict(
        試しの文=len(pos) + len(neg), 続けて言う文の割合=round(len(pos) / max(1, len(pos) + len(neg)), 4),
        文の終わりの形で見分けるAUC=round(auc(pos, neg), 4), 問いかどうかだけのAUC=round(auc(pos_q, neg_q), 4),
        終わりの形の種類=len([k for k in te_ if te_[k] >= RARE]))
    # 続けやすい・渡しやすい終わり方は、品詞の組だけ数で出す（語そのものは会話の資料なので出さない）
    pp = sorted(((cp[k] / tp[k], k, tp[k]) for k in tp if tp[k] >= 200), reverse=True)
    res["3_展開②_これで終わっていいの"]["終わりの品詞と記号ごとの続ける割合"] = {f"{k[0]}+{k[1] or '記号なし'}": dict(続ける=round(r, 3), 数=n) for r, k, n in pp[:12]}
    print("3", json.dumps(res["3_展開②_これで終わっていいの"], ensure_ascii=False), flush=True)


def part4(a, res):
    import spacy
    nlp = spacy.load("ja_ginza")
    rows = []
    for f in sorted(glob.glob(a.logs)):
        L = open(f, encoding="utf-8").read().split("\n")
        bot, kind = None, None
        for ln in L:
            m = re.match(r"- ボット: \*\*(.*)\*\*", ln)
            if m:
                bot, kind = m.group(1), None
                continue
            m = re.match(r"\s+- 先生（段 \d）［(.*?)］", ln)
            if m and bot is not None:
                # 褒めの行は、記録の［］に前の指摘の種類が残ることがあるので、「相手の次の行を見てから」で見分ける
                kind = "なし" if "相手の次の行を見てから" in ln else m.group(1)
                if kind not in ("文法", "言い方", "つながり", "足りない", "なし", "判定できない", "そのまま"):
                    kind = "捨てた"
                continue
            if re.match(r"\s+- 先生（段 \d）: 見立てを捨てた", ln):
                kind = "捨てた"
                continue
            m = re.match(r"- 相手: (.*)", ln)
            if m and bot is not None:
                if kind not in (None, "捨てた", "そのまま", "判定できない"):
                    rows.append((f, bot, kind, m.group(1)))
                bot, kind = None, None
    docs = list(nlp.pipe([r[1] for r in rows] + [r[3] for r in rows], batch_size=256))
    nb = len(rows)

    def feats(bot_doc, nxt_doc, nxt):
        bc = {t.lemma_ for t in bot_doc if t.pos_ in CONTENT}
        nc = [t.lemma_ for t in nxt_doc if t.pos_ in CONTENT]
        first = next((t.lemma_ + "/" + t.pos_ for t in nxt_doc if t.pos_ != "PUNCT"), "無し")
        L = len(nxt)
        lb = 0 if L <= 5 else 1 if L <= 10 else 2 if L <= 20 else 3 if L <= 40 else 4
        return {"聞き返し": bool(re.search(r"って[？?]", nxt)), "問い": nxt.rstrip("」 ").endswith(("？", "?")),
                "感謝": any(t.lemma_ in ("ありがとう", "感謝") for t in nxt_doc), "長さ": lb,
                "同じ内容語": bool(bc & set(nc)), "新しい内容語": min(3, len(set(nc) - bc)), "頭の語": first}
    F = [feats(docs[i], docs[nb + i], rows[i][3]) for i in range(nb)]
    split = [zlib.crc32((r[0] + str(i)).encode()) % 5 != 0 for i, r in enumerate(rows)]
    K = Counter(r[2] for r, s in zip(rows, split) if s)
    fc = defaultdict(Counter)
    head_n = Counter()
    for (r, s, f) in zip(rows, split, F):
        if s:
            for k, v in f.items():
                fc[(k, v)][r[2]] += 1
            head_n[f["頭の語"]] += 1
    kinds = sorted(K)
    ntr = sum(K.values())

    def predict(f):
        sc = {}
        for c in kinds:
            lp = math.log((K[c] + A) / (ntr + A * len(kinds)))
            for k, v in f.items():
                if k == "頭の語" and head_n[v] < RARE:
                    continue
                lp += math.log((fc[(k, v)][c] + A) / (K[c] + A * 8))
            sc[c] = lp
        return sc
    hit = n = maj_hit = 0
    pos, neg = [], []
    major = K.most_common(1)[0][0] if K else None
    for r, s, f in zip(rows, split, F):
        if s:
            continue
        sc = predict(f)
        hit += max(sc, key=sc.get) == r[2]
        maj_hit += r[2] == major
        n += 1
        fix = [c for c in kinds if c != "なし"]
        if fix and "なし" in sc:
            v = np.logaddexp.reduce([sc[c] for c in fix]) - sc["なし"]
            (pos if r[2] != "なし" else neg).append(v)
    res["4_相手の返事から後で付く違和感"] = dict(
        記録のファイル=len(glob.glob(a.logs)), 行=nb, 種類の数={k: int(v) for k, v in Counter(r[2] for r in rows).items()},
        試しの行=n, 言い当てた割合=round(hit / max(1, n), 4), いちばん多い種類を言い続けた時=round(maj_hit / max(1, n), 4),
        直す系か褒めかのAUC=round(auc(pos, neg), 4) if pos and neg else None,
        印ごとの直す系の割合={f"{k}={v}": round(1 - fc[(k, v)]["なし"] / max(1, sum(fc[(k, v)].values())), 3)
                         for (k, v) in sorted(fc) if k != "頭の語" and sum(fc[(k, v)].values()) >= 30})
    print("4", json.dumps(res["4_相手の返事から後で付く違和感"], ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    for k in ("conllu", "content", "rpc", "logs", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--train", type=int, default=50000)
    ap.add_argument("--test", type=int, default=1000)
    a = ap.parse_args()
    res = {}
    for f in (part1, part23, part4):
        f(a, res)
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
