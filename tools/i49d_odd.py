"""I49d の下ごしらえ: 違和感を 1 文ごとに値で返す口（持ち主が選んだ違和感: 自信のなさ（選び）・迷い・組の珍しさ、相手の返事から、展開②）。

ループ本体（intervention/i49d_*、クラウド側）は、ここを import して使う。正解は使わない（確かめの数えの時だけ使う）。

1. 行づくりの型（I48bi と同じ: UModel・型 C、Tatoeba 5 万文で学ぶ）を、作って保存し、読み込む口:
     OddScorer.build(conllu, content, func, ...)   学ぶ（15〜25 分）。比べる並び（学んだ文の先頭 recheck 文の二回り目の値の並び）も作る
     scorer.save(path) / OddScorer.load(path)       pickle で保存・読み込み（保存先は引数で渡す。リポジトリの外に置く）
2. 1 文の違和感: scorer.line(keys, marks=()) → 間ごとの dict
     keys  [(原形, 品詞)] の並び（内容語だけ、順番つき）。marks は、その文の意味の印（過去・否定・丁寧・進行・問い。I49d の読み手の印と同じ名前）
     返す物: 間の番号、選んだ詰め物（文字列）、種類ごとの値（迷い・自信のなさ_選び・筋と波の食い違い・筋の無い共振・組の珍しさ・手持ちに無い。
     どれも大きいほど違和感）、種類ごとの比べる並びの中の順位の割合（0〜1）、名札（いちばん大きい割合の種類）。値の計算は I48bi の rows_of と同じ。
   scorer.items_line(xs, keys) は、I48ae の items（青空・Tatoeba の conllu から作る間）をそのまま渡す口（確かめ用）。
3. 展開②（文の終わりの形から「まだ続ける」見込み）: ContinueTable
     ContinueTable.build(rpc_json)   会話の資料の発言を文に分け、文の終わりの形（最後の内容語でない語も含む最後の語の原形/品詞と記号）ごとに、
                                     同じ話し手が続けた割合を数える（tools/i48_odd_new_kinds.py の 3 と同じ数え方）。少ない形は（品詞, 記号）に戻す
     table.save(path) / ContinueTable.load(path)   数えの表は会話の資料（非公開）から作るので、リポジトリの外に置く
     table.value(toks)  toks は [(原形, 品詞, 表記)] の並び（記号を含む）→ 続ける見込み（0〜1）
手で置く物: 上位 1 割（確かめの数え）、比べる並びの文の数 1 万、展開②の少ない形の境 10・ならし 0.5、学びと試しの分け方（I48bi と同じ）。
python tools/i49d_odd.py build --conllu ... --content ... --func ... --save <型の置き場.pkl>
python tools/i49d_odd.py check --conllu ... --load <型の置き場.pkl> --out <stats.json>
python tools/i49d_odd.py cont --rpc <会話の資料> --save <表の置き場.json>
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
import zlib
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
import i48ab_long_grain as G  # noqa: E402
import i48ad_lookahead as AD  # noqa: E402
import i48ae_marks as AE  # noqa: E402
import i48aj_resonance as AJ  # noqa: E402
import i48ak_teach as AK  # noqa: E402
import i48ap_units as AP  # noqa: E402
import i48at_recheck as RC  # noqa: E402
import i48bi_odd_kinds as BI  # noqa: E402
from i48core import model as MD  # noqa: E402

KINDS = BI.LABEL_KINDS                                      # 正解を使わない 6 種類
TIME_WORDS = AE.TIME_WORDS if hasattr(AE, "TIME_WORDS") else ()


def split_ids(sents, test=1000, train=50000):
    te = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 == 0][:test]
    tr_all = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 != 0]
    return te, tr_all[-2000:], tr_all[:-2000][:train]


class OddScorer:
    def __init__(self, m, nat, ref):
        self.m, self.nat, self.ref = m, nat, ref         # ref: 種類 → いつもの値の並び（昇順）

    # ---- 作る・保存・読み込み ----
    @classmethod
    def build(cls, conllu, content, func, train=50000, recheck=10000, D=8192, r1="架空・前", log=print):
        t0 = time.time()
        sents = [s for s in G.read_conllu(conllu) if s["toks"]]
        _, hold, tr = split_ids(sents, train=train)
        del sents
        cb = json.load(open(content, encoding="utf-8"))
        fn = G.Func(json.load(open(func, encoding="utf-8")))
        S = G.AttrSpace(cb, fn, D, bands=("B", "C", "D", "E", "G"))
        fill = AD.Fill(S)
        for s in tr:
            fill.see(s)
        m = AP.UModel(S, fill, r1).fit(tr, hold).init_units("C")
        m.refresh()
        log(f"学び {time.time() - t0:.0f} 秒")
        nat = RC.Natural(fill, tr)
        rows = BI.rows_of(m, tr[:recheck], nat)
        ref = {kd: np.sort(np.array([r[kd] for r in rows], dtype=np.float64)) for kd in KINDS}
        log(f"比べる並び {len(rows)} 間、{time.time() - t0:.0f} 秒")
        return cls(m, nat, ref)

    def save(self, path):
        feat = self.m.feat
        self.m.feat = None                                # 型の中の lambda は pickle できないので外して保存し、読み込みで作り直す
        try:
            with open(path, "wb") as f:
                pickle.dump(dict(m=self.m, nat=self.nat, ref=self.ref), f, protocol=pickle.HIGHEST_PROTOCOL)
        finally:
            self.m.feat = feat

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        m = d["m"]
        S = m.S
        m.feat = lambda x: AE.feats(S, x, MD.MODE)
        return cls(m, d["nat"], d["ref"])

    # ---- 1 文の違和感 ----
    def pct(self, kd, v):
        r = self.ref[kd]
        return float(np.searchsorted(r, v, side="right") / max(1, len(r)))

    def line(self, keys, marks=()):
        """keys: [(原形, 品詞)]（内容語、順番つき）。marks: 文の意味の印。→ 間ごとの dict の並び。"""
        keys = [tuple(k) for k in keys]
        if not keys:
            return []
        mk = tuple(sorted(marks))
        S = self.m.S
        A = [S.attrs(l, u) for l, u in keys]
        n = len(keys)
        has_time = any(l in TIME_WORDS for l, _ in keys)
        xs = [dict(A=A, i=i, y=None, l=keys[i][0], r=keys[i + 1][0] if i + 1 < n else None, mk=mk, mk_s=mk, last=i + 1 == n, time=has_time)
              for i in range(n)]
        return self.items_line(xs, keys)

    def items_line(self, xs, keys):
        m, nat = self.m, self.nat
        fic = m.who.fiction(keys)
        ups = {u for _, u in keys}
        placed, out = [], []
        for x in xs:
            g = m.gap_scores(keys, x, fic)
            r3 = m.gram.score(m.gram.state(ups, placed, x["last"]))
            logic, wave = g["R2"] + r3, g["R1"]
            three = wave + logic
            both = g["今"] + three
            p_other = float(np.exp(g["今"][-1] - np.logaddexp.reduce(g["今"])))
            three[-1] = both[-1] = -1e9
            cand = (set(np.argsort(-g["今"])[:AK.K]) | set(np.argsort(-three)[:AK.K])) - {m.nc - 1}
            pick = max(cand, key=lambda c: both[c])
            top2 = np.sort(both)[-2:]
            p = np.exp(both - both.max())
            p /= p.sum()
            lg, wv = logic.copy(), wave.copy()
            lg[-1] = wv[-1] = -1e9
            pred = AJ.next_pred(keys, x["i"])
            v = dict(迷い=-float(top2[1] - top2[0]), 自信のなさ_選び=-float(p[pick]),
                     筋と波の食い違い=float(AJ.ranks(lg)[pick] + AJ.ranks(wv)[pick]),
                     筋の無い共振=float(lg[pick] <= np.median(lg[:-1]) and int(np.argmax(wv)) == pick),
                     組の珍しさ=-nat.score(m.pg.cls[pick], pred[0] if pred else None), 手持ちに無い=p_other)
            pc = {kd: self.pct(kd, v[kd]) for kd in KINDS}
            out.append(dict(間=int(x["i"]), 詰め物=m.pg.cls[pick] or "", 選び=int(pick), 値=v, 割合=pc, 名札=max(KINDS, key=lambda k: pc[k])))
            placed.append(m.pg.cls[pick])
        return out


class ContinueTable:
    """展開②: 文の終わりの形 → 同じ話し手が続けた割合（会話の資料から数える）。"""

    RARE, A = 10, 0.5

    def __init__(self, ce, te, cp, tp, base):
        self.ce, self.te, self.cp, self.tp, self.base = ce, te, cp, tp, base

    @staticmethod
    def end_key(toks):
        T = [t for t in toks if t[1] != "PUNCT"]
        last = T[-1] if T else None
        punct = next((t[2] for t in reversed(toks) if t[1] == "PUNCT"), "")
        return ((last[0] + "/" + last[1]) if last else "無し"), (last[1] if last else "無し"), punct

    @classmethod
    def build(cls, rpc_json):
        data = json.load(open(rpc_json, encoding="utf-8"))
        tr = [d for d in data if zlib.crc32(d["name"].encode()) % 5 != 0]   # tools/i48_odd_new_kinds.py と同じ分け方の学びの側
        ce, te, cp, tp = Counter(), Counter(), Counter(), Counter()
        n_c = n = 0
        for d in tr:
            for ln in d["lines"]:
                S = split_sents([(t["lemma"], t["pos"], t["text"]) for t in ln["toks"]])
                for si, x in enumerate(S):
                    cont = si < len(S) - 1
                    k, ps, pu = cls.end_key(x)
                    te[k] += 1
                    ce[k] += cont
                    tp[f"{ps}|{pu}"] += 1
                    cp[f"{ps}|{pu}"] += cont
                    n_c += cont
                    n += 1
        return cls(dict(ce), dict(te), dict(cp), dict(tp), n_c / max(1, n))

    def value(self, toks):
        k, ps, pu = self.end_key(list(toks))
        if self.te.get(k, 0) >= self.RARE:
            return (self.ce.get(k, 0) + self.A) / (self.te[k] + 2 * self.A)
        b = f"{ps}|{pu}"
        if self.tp.get(b):
            return (self.cp.get(b, 0) + self.A) / (self.tp[b] + 2 * self.A)
        return self.base

    def save(self, path):
        json.dump(dict(ce=self.ce, te=self.te, cp=self.cp, tp=self.tp, base=self.base), open(path, "w", encoding="utf-8"), ensure_ascii=False)

    @classmethod
    def load(cls, path):
        d = json.load(open(path, encoding="utf-8"))
        return cls(d["ce"], d["te"], d["cp"], d["tp"], d["base"])


def split_sents(toks, ends="。！？!?"):
    """[(原形, 品詞, 表記)] → 文の並び（終わりの記号で切る。記号だけの文は捨てる）。"""
    out, cur = [], []
    for t in toks:
        cur.append(t)
        if t[2] in ends:
            out.append(cur)
            cur = []
    if any(t[1] != "PUNCT" for t in cur):
        out.append(cur)
    return [x for x in out if any(t[1] != "PUNCT" for t in x)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("build", "check", "cont"))
    for k in ("conllu", "content", "func", "save", "load", "out", "rpc"):
        ap.add_argument("--" + k, default=None)
    ap.add_argument("--train", type=int, default=50000)
    ap.add_argument("--recheck", type=int, default=10000)
    a = ap.parse_args()
    t0 = time.time()
    if a.what == "build":
        sc = OddScorer.build(a.conllu, a.content, a.func, train=a.train, recheck=a.recheck, log=lambda s: print(s, flush=True))
        sc.save(a.save)
        print("保存", os.path.getsize(a.save), "バイト", f"{time.time() - t0:.0f} 秒", flush=True)
    elif a.what == "check":
        t1 = time.time()
        sc = OddScorer.load(a.load)
        load_s = time.time() - t1
        sents = [s for s in G.read_conllu(a.conllu) if s["toks"]]
        te, _, _ = split_ids(sents)
        rows, secs, secs_keys = [], [], []
        for j, s in enumerate(te):
            xs = AE.items(sc.m.S, s, "節")
            if not xs:
                continue
            keys = AK.keys_of(s)
            t1 = time.perf_counter()
            out = sc.items_line(xs, keys)
            secs.append(time.perf_counter() - t1)
            t1 = time.perf_counter()
            sc.line(keys, xs[0]["mk_s"])                 # 印を外から渡す口（文の印を、その文の詰め物から読んだ物で代わりにする）
            secs_keys.append(time.perf_counter() - t1)
            for x, o in zip(xs, out):
                rows.append(dict(o["値"], gold=int(sc.m.cid(x["y"])), pick=o["選び"], last=bool(x["last"])))
        n = len(rows)
        wrong = np.array([r["pick"] != r["gold"] for r in rows])
        k = max(1, int(n * BI.TOP))
        res = dict(間=n, 外れの割合_全体=round(float(wrong.mean()), 4), 読み込みの秒=round(load_s, 1),
                   一文の呼び出しのミリ秒_中央値=round(float(np.median(secs)) * 1000, 2),
                   一文の呼び出しのミリ秒_印を外から=round(float(np.median(secs_keys)) * 1000, 2), 上位1割の外れの割合={})
        for kd in KINDS:
            v = np.array([r[kd] for r in rows])
            idx = np.nonzero(v > 0)[0] if kd == "筋の無い共振" else np.argsort(-v, kind="stable")[:k]
            res["上位1割の外れの割合"][kd] = round(float(wrong[idx].mean()), 4) if len(idx) else None
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(json.dumps(res, ensure_ascii=False), flush=True)
    else:
        tb = ContinueTable.build(a.rpc)
        tb.save(a.save)
        print("表", len(tb.te), "形", f"{time.time() - t0:.0f} 秒", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
