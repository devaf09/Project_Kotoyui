"""I48am: 介入で教える時の重みを、述語の語板で決める（持ち主の方針: 属性で広げる。どの属性を広げるかは、
「行く」の板に人や散歩といった強い属性があれば、それを重点にする）。I48al（別の板）の続き。

- 述語の「左に来る内容語の属性」の板: 学びの文で、内容語 w と、その右のいちばん近い述語 p（VERB・ADJ）の組ごとに、
  p の属性 × w の属性 の共起を数える（語そのものではなく属性で持つ。属性の重さはそれぞれ和 1）。
  p の板で w の属性 a が「いつもより何倍鳴るか」:
    P(a | p) = (Σ_pa p の重さ(pa) × 共起[pa, a] + P(a)) / (Σ_pa p の重さ(pa) × 共起[pa, ·] の和 + 1)   （ならしは全体の P(a) の 1 回分）
    比(a | p) = P(a | p) / P(a)      P(a) は、述語の左に来た内容語の属性の全体の出やすさ
- 教える時の重み: 教える間の左の語の属性の重さ × 比（左の語の重さは和 1 に戻す）。比が 1 より小さい属性は薄く、強い属性に重点。手のしきい値は置かない。
  型: (0) 今の形（全属性に同じ重さ、I48al）、(1) 述語の板で左の語に重み、(2) 述語の板で左の語と右の語（右の語が述語でない時）に重み、
      (3) 詰め物の板の重点: 教える特徴の帯ごとに、本体で その詰め物の板が「いつもより何倍鳴るか」（p(帯|詰め物) / p(帯)）を掛ける。
- 介入の板は本体に融合しない（I48al の決まり）。記録に重みの決め方（型・述語・重点の属性の上位）を残す。
- 測る物: 持ち主の例、広がってほしい例・広がってほしくない例（自作）の「と」の順位、試しの文の元が「と」の間の当たり・上位 3、害、間全体。
  「行く」「遊ぶ」「買う」「言う」の板で、左に来る語の属性として強く鳴る属性の上位 10（寄与 P(a|p) log 比 の順。[名前, 比, P(a|p), P(a)]）。
手で置く物: 教える文 5 つ（I48al から）、例の文（自作）、ならし（全体の P(a) の 1 回分）。
python intervention/i48am_attr_weight.py --conllu ... --content ... --func ... --out out.json [--train 50000] [--test 1000]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i48ab_long_grain as G  # noqa: E402
import i48ad_lookahead as AD  # noqa: E402
import i48aj_resonance as AJ  # noqa: E402
import i48ak_teach as AK  # noqa: E402
import i48al_teach_one as AL  # noqa: E402

N, V, P_ = "NOUN", "VERB", "PRON"
TEACH5 = ("友達と散歩に行った", "先生と学校に行った", "友達と遊んだ", "傘と帽子を買った", "行くと言った")
WANT = [("広げたい", "昨日・友達・散歩・行く", [("昨日", N), ("友達", N), ("散歩", N), ("行く", V)], 1),
        ("広げたい", "先生・学校・行く", [("先生", N), ("学校", N), ("行く", V)], 0),
        ("広げたい", "母・買い物・行く", [("母", N), ("買い物", N), ("行く", V)], 0),
        ("広げたい", "彼・映画・行く", [("彼", P_), ("映画", N), ("行く", V)], 0),
        ("広げたい", "友達・公園・遊ぶ", [("友達", N), ("公園", N), ("遊ぶ", V)], 0),
        ("広げたくない", "傘・帽子・買う の 2 つ目", [("傘", N), ("帽子", N), ("買う", V)], 1),
        ("広げたくない", "昨日・公園・犬・散歩・する の「で」の間", [("昨日", N), ("公園", N), ("犬", N), ("散歩", N), ("する", V)], 1),
        ("広げたくない", "本・机・置く の 1 つ目", [("本", N), ("机", N), ("置く", V)], 0),
        ("広げたくない", "本・机・置く の 2 つ目", [("本", N), ("机", N), ("置く", V)], 1)]
SHOW_PRED = (("行く", V), ("遊ぶ", V), ("買う", V), ("言う", V))
TYPES = ("0 同じ重さ", "1 述語で左の語", "2 述語で左と右の語", "3 詰め物の板の重点")


def attr_names(S, lemma, upos):
    """AttrSpace.attrs と同じ並びの属性の名前（内容語）。"""
    out = ["品詞:" + upos]
    w = S.cb.get(lemma)
    if w is not None and upos in G.CONTENT:
        for b in S.bands:
            for a, _ in sorted(w["attrs"].get(b, {}).items(), key=lambda z: -z[1])[: S.topk]:
                out.append(b + "|" + a)
    return out


class LeftAttr:
    """述語の属性 × 左に来る内容語の属性 の共起（属性の名前の番号で持つ）。"""

    def __init__(self, S):
        self.S, self.idx = S, {}
        self.rows = []

    def ids(self, key, pred=False):
        """属性の番号と重さ（和 1）。pred=True（述語の側）は「品詞:」を外す（どの述語にも同じに付き、述語ごとの違いを消すため）。"""
        names = attr_names(self.S, *key)
        f, v = self.S.attrs(*key)
        assert len(names) == len(v)
        keep = [k for k, n in enumerate(names) if not (pred and n.startswith("品詞:"))] or list(range(len(names)))
        names = [names[k] for k in keep]
        v = np.asarray(v, dtype=np.float64)[keep]
        for n in names:
            self.idx.setdefault(n, len(self.idx))
        return np.array([self.idx[n] for n in names]), v / max(1e-9, float(np.sum(v)))

    def fit(self, sents):
        pairs = []
        for s in sents:
            keys = AK.keys_of(s)
            for i in range(len(keys)):
                p = AJ.next_pred(keys, i)
                if p and keys[i][1] in G.CONTENT:
                    pairs.append((self.ids(p, pred=True), self.ids(keys[i])))
        n = len(self.idx)
        self.C = np.zeros((n, n), dtype=np.float32)
        for (pi, pv), (wi, wv) in pairs:
            self.C[np.ix_(pi, wi)] += np.outer(pv, wv).astype(np.float32)
        self.Pa = self.C.sum(0) / max(1e-9, float(self.C.sum()))
        self.n_pairs = len(pairs)
        return self

    def ratio(self, pred, word):
        """word の属性（S.attrs の並び）ごとの、pred の板での比。"""
        pi, pv = self.ids(pred, pred=True)
        wi, _ = self.ids(word)
        pi_ok = pi < len(self.C)
        row = pv[pi_ok] @ self.C[pi[pi_ok]]
        tot = float(row.sum())
        ok = wi < len(self.C)
        r = np.ones(len(wi))
        pa = self.Pa[wi[ok]]
        r[ok] = ((row[wi[ok]] + pa) / (tot + 1)) / np.maximum(pa, 1e-12)
        return r

    def top(self, pred, k=10):
        pi, pv = self.ids(pred, pred=True)
        ok = pi < len(self.C)
        row = pv[ok] @ self.C[pi[ok]]
        tot = float(row.sum())
        q = (row + self.Pa) / (tot + 1)
        r = q / np.maximum(self.Pa, 1e-12)
        contrib = np.where(self.Pa > 0, q * np.log(np.maximum(r, 1e-12)), -np.inf)   # 寄与 = P(a|p) log 比（まれな属性の大きな比より、よく鳴る属性を上に。しきい値なし）
        names = {v: k_ for k_, v in self.idx.items()}
        o = np.argsort(-contrib)[:k]
        return [[names[i], round(float(r[i]), 2), round(float(q[i]), 4), round(float(self.Pa[i]), 4)] for i in o]


def reweight(LA, keys, x, typ):
    """型に合わせて、教える間の語の属性の重さを替えた x と、重点の属性（記録用）。"""
    i = x["i"]
    A = list(x["A"])
    info = {}
    p = AJ.next_pred(keys, i)
    if typ in (TYPES[1], TYPES[2]) and p:
        targets = [i] + ([i + 1] if typ == TYPES[2] and i + 1 < len(keys) and keys[i + 1][1] not in AJ.PRED else [])
        for j in targets:
            f, v = A[j]
            r = LA.ratio(p, keys[j])
            v2 = v * r
            v2 = v2 / max(1e-9, float(np.linalg.norm(v2)))       # S.attrs と同じく長さ 1 に戻す
            A[j] = (f, v2.astype(np.float32))
            names = attr_names(LA.S, *keys[j])
            o = np.argsort(-r)[:3]
            info[keys[j][0]] = [[names[k], round(float(r[k]), 2)] for k in o]
        info["述語"] = p[0]
    return dict(x, A=A), info


def sep_add(m, keys, x, c, typ, text, info):
    f, v = m.feat(x)
    v = np.asarray(v, dtype=np.float64)
    if typ == TYPES[3]:                                     # 詰め物の板の重点: 本体で p(帯|c) / p(帯)
        w = np.exp(m.pg.prior[:-1])
        pd = (w / w.sum()) @ np.exp(m.pg.logp[:-1])          # 全体の p(帯)（詰め物の出やすさで重みをつけた平均）
        v = v * (np.exp(m.pg.logp[c, f]) / np.maximum(pd[f], 1e-12))
    np.add.at(m.Wi[c], f, (v / max(1e-9, float(v.sum()))).astype(np.float32))
    m.ni[c] += 1
    m.log.append(dict(番号=len(m.log), 時刻=time.strftime("%F %T"), 文=text, 要点=[list(k) for k in keys], 間=x["i"], 印=list(x["mk"]),
                      最後=x["last"], 詰め物=m.pg.cls[c], 重さ=1, 重みの型=typ, 重点=info, 取り消し=False))


def teach(m, LA, item, typ):
    axis, text, keys, gaps = item
    xs = AK.made_items(m.S, keys, gaps, () if text.endswith("好きだ") else ("過去",))
    m.sep = True
    for x in xs:
        c = m.cid(x["y"])
        if c == m.nc - 1:
            continue
        x2, info = reweight(LA, keys, x, typ)
        sep_add(m, keys, x2, c, typ, text, info)
    m.refresh()


def ranks_of(m):
    out = {}
    wc = m.cid("と")
    for kind, name, keys, gi in WANT:
        res, _ = m.walk(keys, AK.made_items(m.S, keys, None))
        sc = res[gi]["今×三つ"]
        out[name] = dict(種類=kind, 順位=int(AJ.ranks(sc)[wc]) + 1, 点の差=round(float(sc[wc] - np.max(np.delete(sc, wc))), 3), 選び=m.pg.cls[res[gi]["選び"]])
    return out


def main():
    ap = argparse.ArgumentParser()
    for k in ("conllu", "content", "func", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--train", type=int, default=50000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--D", type=int, default=8192)
    ap.add_argument("--r1", default="架空・前", choices=AJ.R1_VARS)
    a = ap.parse_args()
    t0 = time.time()
    sents = [s for s in G.read_conllu(a.conllu) if s["toks"]]
    te = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 == 0][: a.test]
    tr_all = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 != 0]
    hold, tr = tr_all[-2000:], tr_all[:-2000][: a.train]
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
    LA = LeftAttr(S).fit(tr)
    print(f"左の属性の板 {time.time() - t0:.0f} 秒、組 {LA.n_pairs}、属性 {len(LA.idx)}", flush=True)
    base = AK.Model(S, fill, a.r1).fit(tr, hold)
    res = dict(学びの文=len(tr), 試しの文=len(te), R1の型=a.r1, 左の属性の組=LA.n_pairs,
               述語の板の上位=[dict(述語=p[0], 上位10=LA.top(p)) for p in SHOW_PRED], 土台=dict(例=ranks_of(base), 試し=AL.measure(base, te, surf)), 文={})
    print("土台", round(time.time() - t0), "秒", json.dumps(res["土台"], ensure_ascii=False), flush=True)
    items = {it[1]: it for it in AL.TEACH_SET}
    for text in TEACH5:
        res["文"][text] = {}
        for typ in TYPES:
            t1 = time.time()
            m = base.clone()
            teach(m, LA, items[text], typ)
            o = dict(例=ranks_of(m), 試し=AL.measure(m, te, surf), 重点=[e["重点"] for e in m.log])
            res["文"][text][typ] = o
            print(text, typ, round(time.time() - t1), "秒", json.dumps(o, ensure_ascii=False), flush=True)
            json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
