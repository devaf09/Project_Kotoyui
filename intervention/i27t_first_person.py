"""介入の研究 実験I27t（A: 一人称と個人の板）: 台詞の語尾から話し手を学び、介入で広げる

持ち主（2026-11-08）: 「一人称を特定する方法だけど、語尾から学び、次に介入で広げる、とかでお願いしたい。」
一人称（台詞の中の 俺・私 など）が誰かは、台詞の話し手が誰かと同じ。語りの一人称は ＜語り手＞ という 1 人として持つ。
  0. 正解: 話し手がはっきりしている台詞（同じ行か次の行に「X は言った」の形: 話す動詞の主語が固有名詞か一人称の代名詞）。
  1. 語尾から学ぶ: 話し手ごとに、台詞の文の終わり（最後の 1 語）の板を束ねる（board_ops.bundle）。
     全作品の共有の板では、語尾 → 一人称の類（俺・私・僕…）も学ぶ（一人称の語がある台詞から。自分の作品は引く）。
     台詞の語尾（文の最後の 1 語）が、作品全体での割合より、その人で多い（発見 = 期待より明るい）ほど、その人らしい。
  2. 介入で広げる: 話し手がわからない台詞にも、当てた話し手の確からしさの重みで板を足し（束ねる）、
     その台詞を前後の台詞の「流れ」（何個前の台詞と同じ人か、を正解の台詞から数えた割合）にも使う。4 回くり返す。
     広げた板には台詞の中の一人称の語（俺・私…）も入り、人ごとの一人称の板になる。
物差し: 正解の台詞を半分ずつ隠し（作品の中で交互）、隠した台詞の話し手を当てた割合。候補は、その作品で隠していない正解に出る話し手。
Web 小説は _wn。結果は非公開の置き場だけ。本文はコードに書かない。
実行: python intervention/i27t_first_person.py --cache <解析済み JSON> --stats <jsonl> --out <人ごとの板の md>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from board_ops import Board, bundle, light, winner  # noqa: E402

SPEECH = set("言う 叫ぶ 答える 呟く つぶやく 尋ねる 訊く 聞く 告げる 続ける 返す 問う 応える 話す 怒鳴る 囁く ささやく 漏らす "
             "笑う 言い放つ 口走る 促す 頷く うなずく 呼ぶ 謝る 喚く 唸る 吐く 零す こぼす 切り出す 遮る 叱る 頼む 命じる".split())
FP = set("俺 おれ オレ 私 わたし ワタシ 僕 ぼく ボク あたし アタシ わたくし うち 拙者 我 われ 吾輩 自分 儂 わし".split())
NARR = "＜語り手＞"
CLS = {"俺": "俺", "おれ": "俺", "オレ": "俺", "私": "私", "わたし": "私", "ワタシ": "私", "あたし": "あたし", "アタシ": "あたし",
       "僕": "僕", "ぼく": "僕", "ボク": "僕", "わたくし": "わたくし", "我": "古風", "われ": "古風", "儂": "古風", "わし": "古風",
       "拙者": "古風", "吾輩": "古風", "うち": "うち"}
END_PUNCT = set("。！？!?…‥、ー〜～♪」』）)")
MAXD = 4


def spans(line):
    """語ごとの文字の位置。"""
    out, pos, text = [], 0, line["text"]
    for t in line["toks"]:
        j = text.find(t["text"], pos)
        if j < 0:
            j = pos
        out.append((j, j + len(t["text"])))
        pos = j + len(t["text"])
    return out


def quote_end(text):
    if not text.startswith("「"):
        return -1
    depth = 0
    for k, ch in enumerate(text):
        if ch == "「":
            depth += 1
        elif ch == "」":
            depth -= 1
            if depth == 0:
                return k
    return -1


def speaker_in(toks, idx, beat=False):
    """idx の語の中で、話す動詞の主語（beat=True なら、根の述語の主語 = 台詞のあとの動き）。
    固有名詞はその語、一人称の代名詞は ＜語り手＞。"""
    ids = set(idx)
    for t in (toks[i] for i in idx):
        if (t["dep"] == "ROOT") if beat else (t["lemma"] in SPEECH and t["pos"] == "VERB"):
            for s in (toks[i] for i in idx):
                if s["dep"] in ("nsubj", "dislocated") and s["head"] == t["i"] and s["i"] in ids:
                    if s["pos"] == "PROPN":
                        return s["lemma"]
                    if s["pos"] == "PRON" and s["lemma"] in FP:
                        return NARR
    return None


def feats(toks, idx):
    """台詞の特徴: 文の終わりの 1 語（語尾）と、一人称の語。"""
    f = Board()
    sent = []
    for i in list(idx) + [None]:
        t = toks[i] if i is not None else None
        if t is None or t["text"] in ("。", "！", "？", "!", "?"):
            core = [x for x in sent if not all(ch in END_PUNCT for ch in x["text"])]
            if core:
                f[("尾1", core[-1]["text"])] += 1      # 最後の 2 語の組は、板がまばらになり当たりを下げた（I27t の試し）
            sent = []
        else:
            sent.append(t)
    fp = Board()
    for i in idx:
        t = toks[i]
        if t["pos"] == "PRON" and t["lemma"] in FP:
            fp[("一", t["lemma"])] += 1
    return f, fp


def quotes_of(w):
    """作品の台詞: [(行, 語尾の板, 一人称の板, 正解の話し手か None)]。"""
    out = []
    L = w["lines"]
    for li, ln in enumerate(L):
        e = quote_end(ln["text"])
        if e < 0:
            continue
        sp = spans(ln)
        inside = [k for k, (a, b) in enumerate(sp) if a >= 1 and b <= e]
        after = [k for k, (a, b) in enumerate(sp) if a > e]
        if not inside:
            continue
        f, fp = feats(ln["toks"], inside)
        who = speaker_in(ln["toks"], after) if after else None
        if who is None and not after and li + 1 < len(L) and quote_end(L[li + 1]["text"]) < 0:
            nx = L[li + 1]
            sp2 = spans(nx)
            cut = next((a for a in range(len(nx["text"])) if nx["text"][a] == "。"), len(nx["text"]))
            first = [k for k, (a, b) in enumerate(sp2) if b <= cut + 1]
            who = speaker_in(nx["toks"], first)
            kind = "話す動詞"
            if who is None:
                who = speaker_in(nx["toks"], first, beat=True)
                kind = "動き"
        else:
            kind = "話す動詞"
        out.append(dict(li=li, f=f, fp=fp, gold=who, kind=kind if who else None))
    return out


def fp_class(fp):
    c = Counter()
    for (_, lem), v in fp.items():
        if lem in CLS:
            c[CLS[lem]] += v
    return c


def end_class_boards(works_q):
    """1. 語尾から学ぶ（全作品の共有の板）: 一人称の語がある台詞から、語尾 → 一人称の類 の板を束ねる。作品ごとの板も返す（自分の作品を引くため）。"""
    own = []
    for qs in works_q:
        rows = []
        for q in qs:
            c = fp_class(q["fp"])
            if not c:
                continue
            k = max(c, key=lambda x: (c[x], x))
            b = Board({(f, k): v for f, v in q["f"].items()})
            b[("類", k)] += 1
            rows.append((0, b))
        own.append(bundle(rows))
    G = Board()
    for b in own:
        G.update(b)
    classes = sorted({k[1] for k in G if k[0] == "類"})
    return G, own, classes


def class_post(G, own, classes, f, alpha=200.0):
    """語尾から、一人称の類の確からしさ（ほかの作品の板で）。類の割合 × 語尾の差分（その類での割合 ÷ 全体での割合）。"""
    n = {c: light(G, ("類", c), minus=own) for c in classes}
    N = sum(n.values()) or 1.0
    sc = {}
    for c in classes:
        v = math.log((n[c] + 1) / (N + len(classes)))
        for k, x in f.items():
            tot = sum(light(G, (k, c2), minus=own) for c2 in classes)
            e = (tot + 0.5) / (N + 1)
            o = (light(G, (k, c), minus=own) + alpha * e) / (n[c] + alpha)
            v += x * math.log(o / e)
        sc[c] = v
    return softmax(sc)


class People:
    """1 作品の人ごとの板。鍵: (人, 特徴) と (人, "数")。作品全体: ("全", 特徴) と ("全", "数")。"""

    def __init__(self):
        self.B = Board()

    def add(self, who, f, wt):
        if wt <= 0:
            return
        self.B.update(bundle([(0, Board({(who, k): v for k, v in f.items()}))], weight=lambda _: wt))
        self.B.update(bundle([(0, Board({("全", k): v for k, v in f.items()}))], weight=lambda _: wt))
        self.B[(who, "数")] += wt
        self.B[("全", "数")] += wt

    def ending_score(self, c, f, alpha=20.0, back=None):
        """語尾の差分: その人での割合 ÷ 作品全体での割合（期待）の対数の和。期待より明るい = 発見（+）、暗い = 違和感（−）。
        back（I27u の共通の像: 特徴 → 割合）があれば、その人の台詞が少ないときは、作品全体ではなく像の割合に寄せる。"""
        nc, nall = light(self.B, (c, "数")), light(self.B, ("全", "数"))
        if nall <= 0:
            return 0.0
        s = 0.0
        for k, v in f.items():
            e = (light(self.B, ("全", k)) + 0.5) / (nall + 1.0)
            b = e if back is None else 0.5 * e + 0.5 * back.get(k, 0.0)
            o = (light(self.B, (c, k)) + alpha * b) / (nc + alpha)
            s += v * math.log(o / e)
        return s


def flow_rates(works_q, known):
    """何個前（1〜MAXD）の台詞と同じ話し手か、の割合（正解どうしの組から数える = 流れの板）。"""
    same, tot = Counter(), Counter()
    for wi, qs in enumerate(works_q):
        for k, q in enumerate(qs):
            a = known[wi].get(k)
            if a is None:
                continue
            for d in range(1, MAXD + 1):
                if k - d >= 0 and known[wi].get(k - d) is not None and qs[k]["li"] - qs[k - d]["li"] <= 3 * d:
                    tot[d] += 1
                    same[d] += known[wi][k - d] == a
    return {d: (same[d] + 1) / (tot[d] + 2) for d in range(1, MAXD + 1)}


def flow_score(c, k, qs, post, rates, K):
    """前後の台詞の話し手（正解か、広げて当てた確からしさ）から、c らしさの対数。"""
    s = 0.0
    for d in range(1, MAXD + 1):
        for j in (k - d, k + d):
            if 0 <= j < len(qs) and abs(qs[j]["li"] - qs[k]["li"]) <= 3 * d and j in post:
                p = post[j].get(c, 0.0)
                r = rates[d]
                s += math.log(p * r + (1 - p) * (1 - r) / max(1, K - 1) + 1e-6) - math.log(1.0 / K)
    return s


def softmax(sc):
    m = max(sc.values())
    ex = {c: math.exp(v - m) for c, v in sc.items()}
    z = sum(ex.values())
    return {c: v / z for c, v in ex.items()}


def narr_class(w):
    """語りの行（台詞でない行）の一人称の類。語り手の一人称の板の材料。"""
    c = Counter()
    for ln in w["lines"]:
        if quote_end(ln["text"]) >= 0 or ln["text"].startswith("「"):
            continue
        for t in ln["toks"]:
            if t["pos"] == "PRON" and t["lemma"] in CLS:
                c[CLS[t["lemma"]]] += 1
    return c


def run_fold(works_q, fold, use, rounds, narr=None, cprior=None, image=None):
    """fold の正解を隠して当てる。use: 使う手の集まり。返り値: 当たり・数・作品ごとの人の板。"""
    known = []
    for qs in works_q:
        kd = {}
        g = 0
        for k, q in enumerate(qs):
            if q["gold"] is not None:
                if g % 2 != fold:
                    kd[k] = q["gold"]
                g += 1
        known.append(kd)
    rates = flow_rates(works_q, known)
    hit, n = Counter(), 0
    boards = []
    for wi, qs in enumerate(works_q):
        cands = sorted(set(known[wi].values()))
        P = People()
        for k, a in known[wi].items():
            P.add(a, qs[k]["f"] + (qs[k]["fp"] if "一人称" in use else Board()), 1.0)
        prior = Counter(known[wi].values())
        CB = defaultdict(Counter)                       # 人 → 一人称の類（正解の台詞の一人称の語と、語りの一人称）
        for k, a2 in known[wi].items():
            for c2, v in fp_class(qs[k]["fp"]).items():
                CB[a2][c2] += v
        if narr is not None and narr[wi]:
            tot = sum(narr[wi].values())
            for c2, v in narr[wi].items():
                CB[NARR][c2] += 3.0 * v / tot           # 語りの一人称は、台詞 3 つぶんの重み
        post = {k: {a: 1.0} for k, a in known[wi].items()}
        hidden = [k for k, q in enumerate(qs) if k not in known[wi]]
        if len(cands) < 2:
            boards.append(P)
            continue
        K = len(cands)

        IMG = {}

        def scores(k, P=P, post=post):
            f = qs[k]["f"] + (qs[k]["fp"] if "一人称" in use else Board())
            sc = {}
            for c in cands:
                v = 0.0
                if "多い人" in use:
                    v += math.log((prior[c] + 1) / (sum(prior.values()) + K))
                if "語尾" in use:
                    v += P.ending_score(c, f, back=IMG.get(c))
                if "流れ" in use:
                    v += flow_score(c, k, qs, post, rates, K)
                if "類" in use:
                    pq = qs[k]["pc"]
                    tb = sum(CB[c].values())
                    pc = {x: (CB[c][x] + cprior[x]) / (tb + 1.0) for x in cprior}
                    v += math.log(sum(pq.get(x, 0) * pc[x] for x in cprior) + 1e-6) - math.log(sum(pq.get(x, 0) * cprior[x] for x in cprior) + 1e-6)
                sc[c] = v
            return sc
        CB0 = {c: Counter(v) for c, v in CB.items()}
        if image is not None:
            IMG.update(image(wi, P, cands))
        for _ in range(rounds if "広げる" in use else 0):
            newP = People()
            for k, a in known[wi].items():
                newP.add(a, qs[k]["f"] + (qs[k]["fp"] if "一人称" in use else Board()), 1.0)
            newpost = dict((k, v) for k, v in post.items() if k in known[wi])
            newCB = defaultdict(Counter, {c: Counter(v) for c, v in CB0.items()})
            for k in hidden:
                pr = softmax(scores(k, P, post))
                newpost[k] = pr
                for c, p in pr.items():
                    newP.add(c, qs[k]["f"] + (qs[k]["fp"] if "一人称" in use else Board()), p)
                    if "類" in use:
                        for c2, v in fp_class(qs[k]["fp"]).items():
                            newCB[c][c2] += p * v
            P, post = newP, newpost
            CB.clear()
            CB.update(newCB)
            if image is not None:
                IMG.clear()
                IMG.update(image(wi, P, cands))
        for k in hidden:
            q = qs[k]
            if q["gold"] is None or q["gold"] not in cands:
                continue
            n += 1
            sc = scores(k, P, post)
            ok = winner([(v, c) for c, v in sc.items()]) == q["gold"]
            hit["当たり"] += ok
            hit[("当", q["kind"])] += ok
            hit[("数", q["kind"])] += 1
        boards.append(P)
    return hit, n, boards, rates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    works = json.load(open(a.cache, encoding="utf-8"))
    works_q = [quotes_of(w) for w in works]
    G, own, classes = end_class_boards(works_q)
    narr = [narr_class(w) for w in works]
    tot_c = Counter({c: light(G, ("類", c)) for c in classes})
    cprior = {c: v / sum(tot_c.values()) for c, v in tot_c.items()}
    s1 = Counter()
    for wi, qs in enumerate(works_q):
        for q in qs:
            c = fp_class(q["fp"])
            pr = class_post(G, own[wi], classes, q["f"])
            if c:
                k = max(c, key=lambda x: (c[x], x))
                s1["数"] += 1
                s1["語尾から当たり"] += winner([(v, x) for x, v in pr.items()]) == k
                s1["いちばん多い類で当たり"] += k == max(cprior, key=cprior.get)
                q["pc"] = {x: v / sum(c.values()) for x, v in c.items()}
            else:
                q["pc"] = pr
    step1 = dict(数=s1["数"], 語尾から当たり=round(s1["語尾から当たり"] / s1["数"], 4), いちばん多い類=round(s1["いちばん多い類で当たり"] / s1["数"], 4),
                 類の割合={c: round(v, 3) for c, v in cprior.items()})
    print("1. 語尾から一人称の類", step1, flush=True)
    nq = sum(len(q) for q in works_q)
    ng = sum(1 for qs in works_q for q in qs if q["gold"] is not None)
    gold_who = Counter(q["gold"] == NARR for qs in works_q for q in qs if q["gold"] is not None)
    rec_kind = Counter(q["kind"] for qs in works_q for q in qs if q["gold"] is not None)
    print(rec_kind, flush=True)
    print("台詞", nq, "正解", ng, "語り手", gold_who[True], flush=True)
    variants = {
        "多い人": {"多い人"},
        "流れ（前後の正解の話し手）": {"流れ"},
        "語尾だけ": {"語尾"},
        "語尾 + 多い人": {"語尾", "多い人"},
        "語尾 + 流れ + 多い人": {"語尾", "流れ", "多い人"},
        "語尾 + 流れ + 多い人 + 一人称": {"語尾", "流れ", "多い人", "一人称"},
        "語尾 + 流れ + 多い人 → 介入で広げる": {"語尾", "流れ", "多い人", "広げる"},
        "語尾 + 流れ + 多い人 + 一人称 → 介入で広げる": {"語尾", "流れ", "多い人", "一人称", "広げる"},
        "多い人 + 一人称の類（語尾から学んだ）": {"多い人", "類"},
        "多い人 + 流れ + 一人称の類": {"多い人", "流れ", "類"},
        "多い人 + 流れ + 一人称の類 → 介入で広げる": {"多い人", "流れ", "類", "広げる"},
        "語尾 + 多い人 + 流れ + 一人称の類 → 介入で広げる": {"語尾", "多い人", "流れ", "類", "広げる"},
    }
    rec = dict(exp="I27t", 語尾から一人称の類=step1, 台詞=nq, 正解の台詞=ng, 正解のうち語り手=gold_who[True], 結果={})
    last_boards = None
    for name, use in variants.items():
        h, m = Counter(), 0
        for fold in (0, 1):
            x, y, boards, rates = run_fold(works_q, fold, use, a.rounds, narr, cprior)
            h.update(x)
            m += y
            if name == "語尾 + 流れ + 多い人 + 一人称 → 介入で広げる" and fold == 0:
                last_boards = boards
        rec["結果"][name] = dict(当たり=round(h["当たり"] / max(1, m), 4), 数=m,
                               **{f"当たり（{k}）": round(h[("当", k)] / max(1, h[("数", k)]), 4) for k in ("話す動詞", "動き")},
                               **{f"数（{k}）": h[("数", k)] for k in ("話す動詞", "動き")})
        rec["流れの割合"] = {d: round(r, 3) for d, r in rates.items()}
        print(name, rec["結果"][name], flush=True)
    # 人ごとの一人称の板（広げた後）
    fp_rows = []
    agree = Counter()
    for wi, P in enumerate(last_boards):
        people = Counter()
        for (who, k), v in P.B.items():
            if who != "全" and k == "数":
                people[who] = v
        for who, cnt in people.most_common(6):
            fps = Counter({k[1]: v for (w2, k), v in P.B.items() if w2 == who and isinstance(k, tuple) and k[0] == "一"})
            ends = Counter({k[1]: v for (w2, k), v in P.B.items() if w2 == who and isinstance(k, tuple) and k[0] == "尾1"})
            tot_e = sum(ends.values()) or 1
            all_e = {k[1]: v for (w2, k), v in P.B.items() if w2 == "全" and isinstance(k, tuple) and k[0] == "尾1"}
            tot_all = sum(all_e.values()) or 1
            disc = sorted(ends, key=lambda e: -(ends[e] / tot_e - all_e.get(e, 0) / tot_all))[:4]
            fp_rows.append((works[wi]["name"], who, round(cnt, 1), fps.most_common(3), disc))
            if fps and sum(fps.values()) >= 2:
                top_fp, tv = fps.most_common(1)[0]
                agree["一人称が 1 つに決まる（7 割以上）"] += tv / sum(fps.values()) >= 0.7
                agree["一人称がある人"] += 1
    rec["一人称の板"] = dict(agree)
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    with open(a.stats, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out = ["# I27t 人ごとの板（語尾と一人称、介入で広げた後）（非公開）", "",
           "| 作品 | 人 | 台詞の重み | 一人称 | その人らしい語尾（作品全体より多い） |", "|---|---|---|---|---|"]
    for nm, who, cnt, fps, disc in fp_rows:
        out.append(f"| {nm[-12:-4]} | {who} | {cnt} | {'、'.join(f'{k} {v:.1f}' for k, v in fps) or '—'} | {'、'.join(disc)} |")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(out))


if __name__ == "__main__":
    main()
