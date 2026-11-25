"""I48p: 言い切りだけの返事に終助詞を付けるための板（--tail）。手で置く値は足さない（数える表と資料の割合だけ）。

行の最後の述語: 行の終わり（句読点を除く）から機能語をさかのぼり、その前が動詞・形容詞の時の、その語と後ろの機能語の並び。
  並び = E.chain の形、終助詞 = 並びの最後に続く終助詞（PART）の原形（I48o の tail_of）、芯 = 並びから終助詞を外した物。
数える行: 丁寧（芯に です・ます・ござる）でなく、問い（I48e の決まり: ？ で終わる、終わりが か）でない行。
  終わりの助詞（資料で半分より多くが句の終わりに出る終助詞）で終わる行と、終助詞の無い行だけ（「たり」「って」などで終わる行は外す）。問いの「の・か」は今のまま（bot でも触らない）。
  付ける割合 = 終助詞のある行の割合。述語の語（行が min_count 以上）→ 品詞 → 全体 の順に下がる。
  どの終助詞か = 終助詞のある行の、終助詞の数え。同じく 語 → 品詞 → 全体（候補は全体で min_count 以上の物）。

測る物（外した対話）: 付ける割合（読み）と本当の割合、どの終助詞かの 1 番目の当たり（語→品詞、品詞だけ、いちばん多い終助詞）。
実行: python intervention/past/i48p_tail_board.py --cache <rpc.json> [--extra_cache <青空文庫>] --out i48p_tail.json [--board <名前>.json]
"""
from __future__ import annotations
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import i48e_relation_split as E  # noqa: E402
import i48j_situation_slide as J  # noqa: E402
import i48o_slide_board as O  # noqa: E402

NONE = "∅"


def core_of(chain, tail):
    """並び（E.chain）から、最後に続く終助詞 tail を外した芯。"""
    es = J.elems(chain)
    k = len(tail.split("+")) if tail else 0
    return "+".join(es[:len(es) - k]) or NONE


def final_pred(toks):
    """行の最後の述語 (語, 品詞, 並び, 終助詞, 芯)。無ければ None。"""
    T = [t for t in toks if t["pos"] != "PUNCT"]
    j = len(T)
    while j > 0 and T[j - 1]["pos"] in E.FUNC:
        j -= 1
    if j == 0 or T[j - 1]["pos"] not in E.PREDS:
        return None
    after = T[j:]
    chain, tail = E.chain(after), O.tail_of(after)
    return T[j - 1]["lemma"], T[j - 1]["pos"], chain, tail, core_of(chain, tail) if tail else chain


def rows(work):
    out = []
    for ln in work["lines"]:
        f = final_pred(ln["toks"])
        if f is None or E.is_question(ln):                     # 問い（？ か、終わりの か）は数えない（I48e の決まり）
            continue
        w, pos, chain, tail, core = f
        if J.situation(core)[2]:
            continue
        out.append(dict(w=w, pos=pos, tail=tail, core=core))
    return out


def final_parts(works):
    """終わりの助詞: 資料で、出た回数の半分より多くが、句の終わり（後ろが終助詞だけで、その後が句読点か行の終わり）にある終助詞（PART）の原形。
    「たり」「って」のように後ろに語が続くことが多い物は、行の終わりに来ても語尾の終助詞として数えない。"""
    tot, end = Counter(), Counter()
    for w in works:
        for ln in w["lines"]:
            T = ln["toks"]
            for i, t in enumerate(T):
                if t["pos"] == "PART":
                    j = i + 1
                    while j < len(T) and T[j]["pos"] == "PART":
                        j += 1
                    tot[t["lemma"]] += 1
                    end[t["lemma"]] += j == len(T) or T[j]["pos"] == "PUNCT"
    return {p for p, n in tot.items() if end[p] * 2 > n}


class TailBoard:
    def __init__(self, min_count=20):
        self.min_count = min_count
        self.rw, self.rp, self.rg = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0]), [0, 0]
        self.dw, self.dp, self.dg = defaultdict(Counter), defaultdict(Counter), Counter()
        self.cands = set()
        self.finals = None

    def fit(self, R, finals=None):
        self.finals = set(finals) if finals is not None else None
        for r in R:
            if self.finals is not None and r["tail"] and not all(e in self.finals for e in r["tail"].split("+")):
                continue                                       # 終わりの助詞でない物で終わる行は数えない
            has = int(bool(r["tail"]))
            for d in (self.rw[r["w"]], self.rp[r["pos"]], self.rg):
                d[0] += 1
                d[1] += has
            if has:
                self.dw[r["w"]][r["tail"]] += 1
                self.dp[r["pos"]][r["tail"]] += 1
                self.dg[r["tail"]] += 1
        self.cands = {t for t, n in self.dg.items() if n >= self.min_count}
        return self

    def rate(self, w, pos):
        for d in (self.rw.get(w), self.rp.get(pos), self.rg):
            if d and d[0] >= self.min_count:
                return d[1] / d[0], d[0]
        return self.rg[1] / max(1, self.rg[0]), self.rg[0]

    def dist(self, w, pos, use_word=True):
        for d in ((self.dw.get(w) if use_word else None), self.dp.get(pos), self.dg):
            if d and sum(d.values()) >= self.min_count:
                return Counter({t: n for t, n in d.items() if t in self.cands})
        return Counter({t: n for t, n in self.dg.items() if t in self.cands})

    def save(self, path):
        json.dump(dict(kind="i48p_tail", min_count=self.min_count, rw=dict(self.rw), rp=dict(self.rp), rg=self.rg,
                       dw={k: dict(v) for k, v in self.dw.items()}, dp={k: dict(v) for k, v in self.dp.items()}, dg=dict(self.dg),
                       cands=sorted(self.cands), finals=sorted(self.finals) if self.finals is not None else None), open(path, "w", encoding="utf-8"), ensure_ascii=False)

    @classmethod
    def load(cls, path):
        d = json.load(open(path, encoding="utf-8"))
        b = cls(d["min_count"])
        b.rw.update({k: list(v) for k, v in d["rw"].items()})
        b.rp.update({k: list(v) for k, v in d["rp"].items()})
        b.rg = list(d["rg"])
        b.dw.update({k: Counter(v) for k, v in d["dw"].items()})
        b.dp.update({k: Counter(v) for k, v in d["dp"].items()})
        b.dg = Counter(d["dg"])
        b.cands = set(d["cands"])
        b.finals = set(d["finals"]) if d.get("finals") is not None else None
        return b


def dim(dist, used):
    """受けの板と同じ薄れ: この会話で使った割合が見込みを越えた分だけ、見込み ÷ 使った割合 を掛ける。"""
    tot = sum(dist.values())
    if not tot:
        return dist
    q = {t: v / tot for t, v in dist.items()}
    n = sum(used.values())
    if n:
        for t, k in used.items():
            if t in q and q[t] > 0:
                q[t] *= min(1.0, q[t] / (k / n))
    return q


def measure(tb, te):
    if tb.finals is not None:
        te = [r for r in te if not r["tail"] or all(e in tb.finals for e in r["tail"].split("+"))]
    has = [r for r in te if r["tail"]]
    out = dict(行=len(te), 終助詞のある行=len(has), 本当の割合=round(len(has) / max(1, len(te)), 4),
               読んだ割合の平均=round(sum(tb.rate(r["w"], r["pos"])[0] for r in te) / max(1, len(te)), 4))
    top = tb.dg.most_common(1)[0][0]
    for label, f in (("いちばん多い終助詞", lambda r: top), ("品詞の表", lambda r: tb.dist(r["w"], r["pos"], False).most_common(1)[0][0]),
                     ("語→品詞の表", lambda r: tb.dist(r["w"], r["pos"]).most_common(1)[0][0])):
        out[label] = round(sum(f(r) == r["tail"] for r in has) / max(1, len(has)), 4)
    rng = random.Random(0)                                     # 表の比で選んだ時（bot と同じ選び方。薄れなし）
    hit = 0
    for r in has:
        d = tb.dist(r["w"], r["pos"])
        ks = list(d)
        hit += rng.choices(ks, weights=[d[k] for k in ks])[0] == r["tail"] if ks else 0
    out["語→品詞の表の比で選ぶ"] = round(hit / max(1, len(has)), 4)
    out["多い終助詞"] = [(t, round(n / max(1, len(has)), 3)) for t, n in Counter(r["tail"] for r in has).most_common(8)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--extra_cache")
    ap.add_argument("--extra_name", default="混ぜた資料")
    ap.add_argument("--test_works", type=int, default=500)
    ap.add_argument("--min_count", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--board", help="学びの対話で作った板の保存先（.json）")
    ap.add_argument("--out")
    a = ap.parse_args()
    works = json.load(open(a.cache, encoding="utf-8"))
    works = works["works"] if isinstance(works, dict) else works
    random.Random(a.seed).shuffle(works)
    te = {"RPC": works[: a.test_works]}
    tr_w = works[a.test_works:]
    if a.extra_cache:
        ex = json.load(open(a.extra_cache, encoding="utf-8"))
        ex = ex["works"] if isinstance(ex, dict) else ex
        random.Random(a.seed).shuffle(ex)
        k = len(ex) // 5
        te[a.extra_name] = ex[:k]
        tr_w = tr_w + ex[k:]
    tb = TailBoard(a.min_count).fit([r for w in tr_w for r in rows(w)], final_parts(tr_w))
    if a.board:
        tb.save(a.board)
    res = dict(学び=dict(行=tb.rg[0], 割合=round(tb.rg[1] / max(1, tb.rg[0]), 4), 候補=sorted(tb.cands, key=lambda t: -tb.dg[t])), 試し={})
    for name, ws in te.items():
        res["試し"][name] = measure(tb, [r for w in ws for r in rows(w)])
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
