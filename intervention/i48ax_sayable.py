"""I48ax: 採点の 2 本目「言えるか」（持ち主の不安点 5: 採点が厳しすぎて、その知性なりの言い方を削っていないか）。

元の文と同じ詰め物か（今まで）に加えて、選んだ詰め物が「言えるか」を、学びの数えだけで決める（手で置く線なし。線は「出たか出ないか」だけ）。
  (a) 言える: 選んだ詰め物が、学びの中で同じ (左の内容語, 右の内容語) の間に 1 回でも出ている。文末は (最後の内容語, 意味の印の組)。
  (b) 組が学びに無い時: (左の語, 何でも) と (何でも, 右の語) の両方に出ていれば「言えそう」、どちらかでも無ければ「分からない」。
      組が学びにあるのに、選んだ詰め物が出ていなければ「言えない」。
測る物:
  - 元と同じ・言える・言えそう・言えない・分からない の割合（間・文末ごと）
  - 外れのうち、言える・言えそうの割合（その知性なりの別の言い方の量）
  - 外れなのに言える（元, 選び）の組の多い順
  - 自然さ: 選んだ詰め物の助詞と右の述語の PMI（I48at (三) の Natural）が、元の詰め物の PMI 以上の割合
  - 外れなのに言える例 30 件（Tatoeba の文の要点だけ）
物差しの文: 本体が学んでいない文（学びの 5 万・取りおき 2,000・試し 1,000 を除いた残り）から数える（自分の数えで自分を採点しない。不安点 4）。
  比べのために、学びの 5 万から数えた物差しでの割合も横に出す（「自分の物差し」）。自然さの PMI も、物差しの文と学びの文の両方で数える。
比べる型: 同じ重さ、−log p（I48at (二)）、その他を選べる形（I48at (四)）。筋の無い共振（想定外の反応の見張り）も出す。
python intervention/i48ax_sayable.py --conllu ... --content ... --func ... --out out.json [--train 50000] [--test 1000]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i48ab_long_grain as G  # noqa: E402
import i48ad_lookahead as AD  # noqa: E402
import i48ae_marks as AE  # noqa: E402
import i48aj_resonance as AJ  # noqa: E402
import i48ak_teach as AK  # noqa: E402
import i48at_recheck as RC  # noqa: E402
from i48core import learn as LN  # noqa: E402
from i48core import measure as ME  # noqa: E402


class Sayable:
    """学びの文の (左, 右) → 詰め物 の数え。右は次の内容語の原形、文末は意味の印の組。"""

    def __init__(self, S, tr):
        self.pair, self.left, self.right = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
        for s in tr:
            for x in AE.items(S, s, "節"):
                r = self.rkey(x)
                self.pair[(x["l"], r)][x["y"]] += 1
                self.left[x["l"]][x["y"]] += 1
                self.right[r][x["y"]] += 1

    @staticmethod
    def rkey(x):
        return ("印", tuple(sorted(x["mk"]))) if x["last"] else x["r"]

    def judge(self, x, g):
        """選んだ詰め物 g の判定: 言える・言えない・言えそう・分からない。"""
        r = self.rkey(x)
        c = self.pair.get((x["l"], r))
        if c:
            return "言える" if c.get(g, 0) > 0 else "言えない"
        if self.left.get(x["l"], {}).get(g, 0) > 0 and self.right.get(r, {}).get(g, 0) > 0:
            return "言えそう"
        return "分からない"


def score(m, te, say, nat, n_ex=30):
    kinds = ("言える", "言えない", "言えそう", "分からない")
    out = {w: Counter() for w in ("間", "文末")}
    miss_say, pairs, ex = Counter(), Counter(), []
    nat_n = nat_ge = 0
    for s in te:
        xs = AE.items(m.S, s, "節")
        if not xs:
            continue
        keys = AK.keys_of(s)
        rs, _ = m.walk(keys, xs)
        for x, r in zip(xs, rs):
            g = m.pg.cls[r["選び"]]
            where = "文末" if x["last"] else "間"
            ok = g == x["y"]
            k = "その他" if g is None else say.judge(x, g)
            out[where]["数"] += 1
            out[where]["元と同じ"] += ok
            out[where][k] += 1
            if not ok:
                out[where]["外れ"] += 1
                miss_say[where] += k in ("言える", "言えそう")
                if k == "言える":
                    pairs[(x["y"], g)] += 1
                    if len(ex) < n_ex:
                        ex.append(dict(要点="・".join(kk[0] for kk in keys), 間=x["i"], 元=x["y"], 選び=g))
            if not x["last"] and g is not None:
                pred = AJ.next_pred(keys, x["i"])
                p = pred[0] if pred else None
                if p is not None:
                    nat_n += 1
                    nat_ge += nat.score(g, p) >= nat.score(x["y"], p)
    res = {}
    for where, c in out.items():
        n = max(1, c["数"])
        res[where] = dict(数=c["数"], 元と同じ=round(c["元と同じ"] / n, 4), **{k: round(c[k] / n, 4) for k in kinds},
                          その他=round(c["その他"] / n, 4), 外れのうち言える_言えそう=round(miss_say[where] / max(1, c["外れ"]), 4))
    res["外れなのに言える組"] = [dict(元=a, 選び=b, 回数=v) for (a, b), v in pairs.most_common(15)]
    res["自然さが元以上"] = round(nat_ge / max(1, nat_n), 4)
    res["例"] = ex
    return res


def main():
    ap = argparse.ArgumentParser()
    for k in ("conllu", "content", "func", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--train", type=int, default=50000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--D", type=int, default=8192)
    ap.add_argument("--r1", default="架空・前")
    ap.add_argument("--kinds", nargs="*", default=["同じ", "logp", "その他"])
    a = ap.parse_args()
    t0 = time.time()
    sents = [s for s in G.read_conllu(a.conllu) if s["toks"]]
    te = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 == 0][: a.test]
    tr_all = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 != 0]
    hold, tr = tr_all[-2000:], tr_all[:-2000][: a.train]
    ref = tr_all[:-2000][a.train:]                              # 物差しの文: 本体が学んでいない文
    cb = json.load(open(a.content, encoding="utf-8"))
    func = G.Func(json.load(open(a.func, encoding="utf-8")))
    S = G.AttrSpace(cb, func, a.D, bands=("B", "C", "D", "E", "G"))
    fill = AD.Fill(S)
    surf = G.Surface()
    for s in tr:
        fill.see(s)
        cont, gaps, _ = G.split(s)
        for (l, f_, u), g in zip(cont, gaps):
            surf.add(l, f_, u, g)
    fill_ref = AD.Fill(S)
    for s in ref:
        fill_ref.see(s)
    rulers = {"物差し（学んでいない文）": (Sayable(S, ref), RC.Natural(fill_ref, ref)), "自分の物差し（学びの文）": (Sayable(S, tr), RC.Natural(fill, tr))}
    res = dict(学びの文=len(tr), 試しの文=len(te), 物差しの文=len(ref), 組の数={k: len(v[0].pair) for k, v in rulers.items()}, 型={})

    def save():
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"数え {time.time() - t0:.0f} 秒", json.dumps(res["組の数"], ensure_ascii=False), flush=True)
    for kind in a.kinds:
        t1 = time.time()
        m = LN.WModel(S, fill, a.r1)
        m.fit_weighted(tr, hold, "-logp" if kind == "logp" else "同じ")
        if kind == "その他":
            m.other = True
            m.refresh()
        r = ME.measure(m, te, surf)
        r["言えるか"] = {nm: score(m, te, say, nat) for nm, (say, nat) in rulers.items()}
        r["筋の無い共振"] = LN.monitor(m, te)
        res["型"][kind] = r
        print(kind, round(time.time() - t1), "秒", json.dumps({k: r[k] for k in ("間", "文末", "文")}, ensure_ascii=False),
              json.dumps({nm: {k: v[k] for k in ("間", "文末", "自然さが元以上")} for nm, v in r["言えるか"].items()}, ensure_ascii=False), flush=True)
        save()
    res["秒"] = round(time.time() - t0)
    save()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
