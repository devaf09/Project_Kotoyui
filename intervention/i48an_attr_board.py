"""I48an: 介入の板を 2 枚に分ける。(P) 並びの板（I48al の別の板）と、(Q) 属性の板（語の属性の帯だけ。位置のずれ・積なし）。

- (Q) は役ごとに別の帯の組: 左（間の左の語）・右（間の右の語）・述語（右のいちばん近い述語）。教えた詰め物 c に、その役の語の属性の帯を載せる。
  点: 試す間の、役ごとの語の属性 x（和 1）と (Q) の重なりを、本体の同じ役の板（i48aj の rR・rL・rP、p(帯|c)）との比で測る
      役の点 = Σ_帯 x × log( ((Q の数え + 本体の p(帯|c)) / (Q の回数 + 1)) / 本体の p(帯|c) )、教えていない帯・詰め物は 0（I48ak の sep_ratio と同じ形）
  役の合わせ方: 掛け算（役の点の和 = 比の積の対数。どの役も鳴る時に高い）と、足し算（log 比の平均 = log( Σ_役 exp(役の点) / 役の数 )）。
- (Q) に載せる重み:
    Q0 同じ重さ
    Q1 述語 × 詰め物で絞った比（左の役だけ）: 学びの文で「詰め物 c でその述語に掛かる左の語」の属性の出やすさ ÷ 「その述語の左の語」全体の出やすさ
       P_c(a|p) = (Σ p の重さ × 共起_c + P(a|p)) / (総量_c + 1)、P(a|p) は I48am の述語の左の属性の板（ならしは 1 回分）。右・述語の役は Q0
    Q2 詰め物だけで絞った比（3 つの役）: 本体の役の板の比 p(帯|c) / p(帯)（rR・rL・rP の L、β は取りおきで選んだ物）
  比をそのまま重みにして和 1 にならす。しきい値は置かない。
- 比べ: P だけ（I48al）、Q だけ、P＋Q（今の点に両方を足す）。Q は Q0・Q1・Q2 × 掛け算・足し算。
- 介入の板は本体に融合しない。記録に (Q) の重みの上位の属性を残す。
- 測る物: I48am と同じ（例 9 つの「と」の順位と点の差、試しの文の「と」の間の当たり・上位 3・害・間）。
  さらに、元が「と」の間を、左の語が人のクラス（HUMAN）を持つ間と持たない間に分けた当たり・上位 3。
手で置く物: HUMAN（人のクラスの名前 8 つ）、教える文 5 つ、例の文（I48am と同じ）。
python intervention/i48an_attr_board.py --conllu ... --content ... --func ... --out out.json [--train 50000] [--test 1000] [--teach 文 ...]
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
import i48ae_marks as AE  # noqa: E402
import i48aj_resonance as AJ  # noqa: E402
import i48ak_teach as AK  # noqa: E402
import i48al_teach_one as AL  # noqa: E402
import i48am_attr_weight as AM  # noqa: E402

ROLES = ("左", "右", "述語")
HUMAN = ("個人(individual)", "成人(adult)", "女(female)", "男性(male)", "肉縁(relation)", "祖(ancestor)", "domestic_partner", "吾れ(self)")
QW = ("Q0", "Q1", "Q2")
VARIANTS = [("P", None, None)] + [(pq, q, how) for pq in ("Q", "P+Q") for q in QW for how in ("掛け算", "足し算")]


def role_keys(keys, i):
    """間 i の役ごとの語（無い役は None）。"""
    last = i + 1 == len(keys)
    return {"左": keys[i], "右": None if last else keys[i + 1], "述語": None if last else AJ.next_pred(keys, i)}


class FillerLeft:
    """述語 × 左の語の属性の共起を、間の詰め物ごとに絞った物（Q1 用）。"""

    def __init__(self, LA, tr):
        self.LA, self.pairs, self.cache = LA, [], {}
        for s in tr:
            keys = AK.keys_of(s)
            gaps = G.split(s)[1]
            for i in range(len(keys) - 1):
                p = AJ.next_pred(keys, i)
                if p and keys[i][1] in G.CONTENT:
                    self.pairs.append((LA.ids(p, pred=True), LA.ids(keys[i]), gaps[i]))

    def C(self, y):
        if y not in self.cache:
            n = len(self.LA.C)
            C = np.zeros((n, n), dtype=np.float32)
            for (pi, pv), (wi, wv), g in self.pairs:
                if g == y and (pi < n).all() and (wi < n).all():
                    C[np.ix_(pi, wi)] += np.outer(pv, wv).astype(np.float32)
            self.cache[y] = C
        return self.cache[y]

    def ratio(self, pred, word, y):
        LA = self.LA
        pi, pv = LA.ids(pred, pred=True)
        wi, _ = LA.ids(word)
        n = len(LA.C)
        okp = pi < n
        row = pv[okp] @ LA.C[pi[okp]]
        tot = float(row.sum())
        rowc = pv[okp] @ self.C(y)[pi[okp]]
        totc = float(rowc.sum())
        r = np.ones(len(wi))
        ok = wi < n
        pa = LA.Pa[wi[ok]]
        pap = (row[wi[ok]] + pa) / (tot + 1)                    # P(a|p)
        r[ok] = ((rowc[wi[ok]] + pap) / (totc + 1)) / np.maximum(pap, 1e-12)
        return r


class QModel(AK.Model):
    """AK.Model に (Q) 属性の板を足した物。"""

    def init_q(self, how="掛け算"):
        self.Q = {r: np.zeros_like(self.rR.W, dtype=np.float32) for r in ROLES}
        self.nq = {r: np.zeros(self.nc) for r in ROLES}
        self.Lq, self.how = None, how
        return self

    def board(self, r):
        return {"左": self.rR, "右": self.rL, "述語": self.rP}[r]

    def clone(self):
        m = super().clone()
        m.Q = {r: q.copy() for r, q in self.Q.items()}
        m.nq = {r: n.copy() for r, n in self.nq.items()}
        m.Lq = None
        return m

    def refresh(self):
        super().refresh()
        if getattr(self, "Q", None) is None:
            return
        self.Lq = {}
        for r in ROLES:
            if self.nq[r].any():
                W = self.board(r).W
                logpm = np.log((W + G.ALPHA) / (W.sum(1, keepdims=True) + G.ALPHA * W.shape[1])).astype(np.float32)
                self.Lq[r] = AK.sep_ratio(self.Q[r], self.nq[r], logpm)
        self.Lq = self.Lq or None

    def q_score(self, keys, i):
        if not self.Lq:
            return 0
        parts = []
        for r, key in role_keys(keys, i).items():
            if key is None or r not in self.Lq:
                continue
            f, v = self.S.attrs(*key)
            parts.append(self.Lq[r][:, f] @ (v / max(1e-9, float(v.sum()))))
        if not parts:
            return 0
        P = np.stack(parts)
        if self.how == "掛け算":
            return P.sum(0)
        return np.log(np.exp(P).mean(0))

    def gap_scores(self, keys, x, fic):
        g = super().gap_scores(keys, x, fic)
        g["今"] = g["今"] + self.q_score(keys, x["i"])
        return g

    def teach_q(self, keys, x, c, qw, FL, text):
        info = {}
        for r, key in role_keys(keys, x["i"]).items():
            if key is None:
                continue
            f, v = self.S.attrs(*key)
            w = np.asarray(v, dtype=np.float64)
            if qw == "Q1" and r == "左" and role_keys(keys, x["i"])["述語"] is not None:
                w = w * FL.ratio(role_keys(keys, x["i"])["述語"], key, self.pg.cls[c])
            elif qw == "Q2":
                w = w * np.exp(self.board(r).L[c, f].astype(np.float64))
            w = w / max(1e-9, float(w.sum()))
            np.add.at(self.Q[r][c], f, w.astype(np.float32))
            self.nq[r][c] += 1
            names = AM.attr_names(self.S, *key)
            o = np.argsort(-w)[:3]
            info[r] = [key[0]] + [[names[k], round(float(w[k]), 3)] for k in o]
        self.log.append(dict(番号=len(self.log), 時刻=time.strftime("%F %T"), 文=text, 間=x["i"], 詰め物=self.pg.cls[c], 板="Q", 重み=qw, 重点=info, 取り消し=False))


def teach(m, item, variant, FL):
    pq, qw, how = variant
    axis, text, keys, gaps = item
    xs = AK.made_items(m.S, keys, gaps, () if text.endswith("好きだ") else ("過去",))
    m.sep = True
    m.how = how or "掛け算"
    fic = m.who.fiction(keys)
    for x in xs:
        c = m.cid(x["y"])
        if c == m.nc - 1:
            continue
        if "P" in pq:
            m.cur_text = text
            m.add_boards(keys, x, c, 1.0, fic, gaps[:x["i"]], teach=True)
        if "Q" in pq:
            m.teach_q(keys, x, c, qw, FL, text)
    m.refresh()


def is_human(S, key):
    return any(n.startswith("D|") and n.split(":", 1)[-1] in HUMAN for n in AM.attr_names(S, *key))


def measure(m, te):
    """AL.measure と同じ数え ＋ 元が「と」の間を、左の語が人のクラスか否かで分けた当たり・上位 3。"""
    keys0, gi, want = AK.EXAMPLES[0]
    wc = m.cid("と")
    c = dict(と=0, と当たり=0, と上位3=0, 害=0, 間=0, 間当たり=0, 人=0, 人当たり=0, 人上位3=0, 人以外=0, 人以外当たり=0, 人以外上位3=0)
    for s in te:
        xs = AE.items(m.S, s, "節")
        if not xs:
            continue
        gaps = G.split(s)[1]
        keys = AK.keys_of(s)
        rs, _ = m.walk(keys, xs)
        for i, (y, r) in enumerate(zip(gaps, rs)):
            if i + 1 == len(gaps):
                continue
            pk = m.pg.cls[r["選び"]]
            c["間"] += 1
            c["間当たり"] += pk == y
            if y == "と":
                top3 = AJ.ranks(r["今×三つ"])[wc] < 3
                c["と"] += 1
                c["と当たり"] += pk == "と"
                c["と上位3"] += top3
                h = "人" if is_human(m.S, keys[i]) else "人以外"
                c[h] += 1
                c[h + "当たり"] += pk == "と"
                c[h + "上位3"] += top3
            elif pk == "と":
                c["害"] += 1
    d = lambda a, b: round(c[a] / max(1, c[b]), 3)
    return dict(元がとの間=c["と"], との当たり=d("と当たり", "と"), との上位3=d("と上位3", "と"), 害_とにした数=c["害"], 間=round(c["間当たり"] / max(1, c["間"]), 4),
                人の間=c["人"], 人の当たり=d("人当たり", "人"), 人の上位3=d("人上位3", "人"), 人以外の間=c["人以外"], 人以外の当たり=d("人以外当たり", "人以外"), 人以外の上位3=d("人以外上位3", "人以外"))


def main():
    ap = argparse.ArgumentParser()
    for k in ("conllu", "content", "func", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--train", type=int, default=50000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--D", type=int, default=8192)
    ap.add_argument("--r1", default="架空・前", choices=AJ.R1_VARS)
    ap.add_argument("--teach", nargs="*", default=list(AM.TEACH5))
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
    for s in tr:
        fill.see(s)
    base = QModel(S, fill, a.r1).fit(tr, hold).init_q()
    LA = AM.LeftAttr(S).fit(tr)
    FL = FillerLeft(LA, tr)
    print(f"学び {time.time() - t0:.0f} 秒、述語と左の語の組 {LA.n_pairs}、詰め物つき {len(FL.pairs)}", flush=True)
    res = dict(学びの文=len(tr), 試しの文=len(te), R1の型=a.r1, HUMAN=HUMAN, 土台=dict(例=AM.ranks_of(base), 試し=measure(base, te)), 文={})
    print("土台", round(time.time() - t0), "秒", json.dumps(res["土台"], ensure_ascii=False), flush=True)
    items = {it[1]: it for it in AL.TEACH_SET}
    for text in a.teach:
        res["文"][text] = {}
        for var in VARIANTS:
            t1 = time.time()
            m = base.clone()
            teach(m, items[text], var, FL)
            nm = "・".join(v for v in var if v)
            o = dict(例=AM.ranks_of(m), 試し=measure(m, te), 重点=[e.get("重点") for e in m.log if e.get("板") == "Q"])
            res["文"][text][nm] = o
            print(text, nm, round(time.time() - t1), "秒", json.dumps(dict(例={k: v["順位"] for k, v in o["例"].items()}, 試し=o["試し"]), ensure_ascii=False), flush=True)
            json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
