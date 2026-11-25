"""I49d を本物の単語で試す（思考のループと感情・状態のループを分けた最小版。持ち主の案、未確定）。数えだけ。ループ本体は変えない。

読み手: 青空文庫の作品 1（I49b と同じ選び方）の初めの N 文を 1 語ずつ（tools/i49c_read_words.WordReader、skip、印は mark_mode="sent"）。
  語ごとの印は tools/i49d_assoc.token_marks（過去・丁寧・否定・進行）。文の最後の語に、展開②の値（tools/i49d_odd.ContinueTable.value、
  文の終わりの形から「まだ続けそうか」）を「展開②」の名前で float として付ける。pos_of（語の名前 → 品詞）は、その作品で最初に出た品詞。
組み合わせ（部品を 1 つずつ止める比べ）:
  a  連想なし ＋ Plain（仮の組立）
  b  連想なし ＋ Scored（行づくりで詰め物を入れる。tools/i49d_odd.OddScorer）
  c  波の連想 ＋ Scored（tools/i49d_assoc.Assoc.from_wordvecs。1 回が重いので、文を減らしてよい）
  d  属性の連想 ＋ Scored（語板の属性の向きで引き、返すベクトルは同じ語の波の語の波。問いは入れ物の語の属性の和。
     器と入れ物は波の向きで動くので、属性の向きのままでは「材料が不足」と比べられないため）
  e  連想なし ＋ Scored ＋ 選ぶ口（Pick、k=3、重さは拾われやすさ tools/i49d_pickup.PickupTable.value(名前, pos_of[名前])）
  f2・f5  e の k を 2・5 にした物
  g  連想なし ＋ Scored ＋ 選ぶ口（k=3、重さは全部同じ 0。Pick は同じ重さなら聞いた順の前から残すので、聞いた順の初めの k 個。乱れは無し）
数える物（数だけ）:
  した事の内訳、出力の数、出力が文の終わりの直後（その文の最後の語を読んだ後、次の語を読む前）にある割合、文 1 つあたりの出力の数、
  出力の行の長さ（内容語の数。入れ物にあった数と、選ぶ口の後に行に出た数）の分布、聞いた語だけの行の割合、連想で入った語を含む行の割合、連想で入った語のうち聞いていない語の割合、
  「もう言った」で休んだ回数、待つの連続の長さの分布、tally() の 2 つの表、
  ずらした数え（違和感の元の強さが上がった歩の 1・2・4・8 歩後の、した事の割合と器の波ごとの動きの量の、上がらなかった歩との差）、
  行づくりの違和感（迷い・自信のなさ_選び・組の珍しさ）の割合の分布（聞いた順のままの行と、連想の語が入った行で分けて）、
  見張り（張り付き・死んだ刻み）、時間。行そのもの・語の名前は数えに入れない。
手で置く物: 読み終わった後に回す歩 200、歩の上限（語の数の 30 倍 ＋ 1,000）、ずらす歩 1・2・4・8、作品の選び方（I49b と同じ）。
python tools/past/i49d_trial.py --src ... --conllu ... --content ... --func ... --wave <波の板> --scorer <型.pkl> --cont <展開②の表.json>
    --n_sent 300 --combos a,b,c,d --out <stats.json>   （e・f2・f5・g は --pickup <拾われやすさの表.json> も渡す）
"""
from __future__ import annotations
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする

import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, HERE)

AFTER = 200
LAGS = (1, 2, 4, 8)
ODD3 = ("迷い", "自信のなさ_選び", "組の珍しさ")


class AttrNear:
    """属性の連想を、波の向きで返す口。問いは入れ物の語（無ければ直前に読んだ語）の属性の和。返すのは、波の板にもある語の波の向き。"""

    def __init__(self, attr_assoc, wv):
        self.A, self.wv = attr_assoc, wv
        self.th = None

    def near(self, vec, k):
        names = [it["名前"] for it in self.th.tray.vecs()] if self.th is not None else []
        q = [self.A.vec(n) for n in names if self.A.vec(n) is not None]
        if not q:
            return []
        out = []
        for name, _, sim in self.A.near(np.sum(q, axis=0), int(k) + 10):
            i = self.wv.ix.get(name)
            if i is not None:
                out.append((name, self.wv.W[i], sim))
            if len(out) >= int(k):
                break
        return out


class Watch:
    """1 歩ごとの記録（数えのためだけ）。"""

    def __init__(self):
        self.acts, self.end_read, self.assoc_names, self.lines, self.felt_build, self.builds = [], [], [], [], [], []
        self.heard = set()
        self.snap = None
        self.rest_said = 0

    def after(self, th, rec):
        a = rec["した事"]
        self.acts.append(a)
        self.end_read.append(a == "読む" and bool(rec["印"].get("文の終わり")))
        if a == "読む" and rec["語"] is not None:
            self.heard.add(rec["語"])
        if a == "連想":
            self.assoc_names.append(rec["足した物"][0])
        if a == "組立":
            self.snap = [(it["名前"], it["種類"]) for it in th.tray.vecs()]
            self.builds.append(dict(違和感=sorted(rec["違和感"]), 感じた物=dict(rec.get("感じた物", {})),
                                    連想あり=any(k == "連想" for _, k in self.snap)))
            self.rest_said += "もう言った" in rec["違和感"]
        if a == "出力":
            mats = self.snap or []
            names = {n for n, _ in mats}
            ln = list(rec["行"])
            self.lines.append(dict(n=len(mats), 出した内容語=sum(w in names for w in ln), 出した語=len(ln), 連想=[n for n, k in mats if k == "連想"],
                                   聞いた語だけ=all(k == "語" for _, k in mats), 歩=rec["歩"]))


def make_reader(docs, wv, table):
    import i49c_read_words as R
    import i49d_assoc as AS
    sents, pos_of, cont = [], {}, []
    for d, m in docs:
        toks = [t for t in d if t.pos_ != "PUNCT"]
        mk = AS.token_marks(toks)
        sents.append(([(t.lemma_, t.pos_, mk[i]) for i, t in enumerate(toks)], m))
        for t in toks:
            pos_of.setdefault(t.lemma_, t.pos_)
        cont.append(float(table.value([(t.lemma_, t.pos_, t.text) for t in d])))
    rd = R.WordReader.from_tokens(sents, wv.of, "skip", "sent")
    k = 0                                                  # 文の最後の語に展開②を付ける（丸ごと飛んだ文の値は付かない）
    s_has = [any(wv.of(w, p) is not None for w, p, _ in toks) for toks, _ in sents]
    js = [j for j, h in enumerate(s_has) if h]
    for w, v, mk in rd.items:
        if mk.get("文の終わり"):
            mk["展開②"] = cont[js[k]]
            k += 1
    return rd, pos_of, sum(not h for h in s_has)


def lagged(acts, R, kinds, va):
    """違和感の元の強さが上がった歩（前の歩より大きい）の L 歩後の、した事の割合と器の波ごとの動きの量の、上がらなかった歩との差。"""
    import i49d_two_loops as TL
    A = np.array([TL.ACTS.index(a) for a in acts])
    T = len(A)
    out = {}
    for j, k in enumerate(kinds):
        up = np.zeros(T, dtype=bool)
        up[1:] = R[1:, j] > R[:-1, j]
        d = {}
        for L in LAGS:
            idx_u = np.nonzero(up[: T - L])[0]
            idx_n = np.nonzero(~up[: T - L])[0]
            if len(idx_u) == 0 or len(idx_n) == 0:
                d[str(L)] = None
                continue
            act_diff = {n: round(float((A[idx_u + L] == i).mean() - (A[idx_n + L] == i).mean()), 4) for i, n in enumerate(TL.ACTS)}
            band = np.round(va[idx_u + L].mean(0) - va[idx_n + L].mean(0), 4).tolist()
            d[str(L)] = dict(上がった歩=int(len(idx_u)), した事の割合の差=act_diff, 器の波の動きの量の差=band)
        out[k] = d
    return out


def runs(acts, name):
    c, out = 0, Counter()
    for a in acts + [None]:
        if a == name:
            c += 1
        elif c:
            out[c if c < 5 else (5 if c < 10 else (10 if c < 50 else 50))] += 1
            c = 0
    return {("1" if k == 1 else "2-4" if k < 5 else "5-9" if k == 5 else "10-49" if k == 10 else "50-"): v for k, v in sorted(out.items())}


PICK_K = dict(e=3, f2=2, f5=5, g=3)


def one(combo, rd, pos_of, wv, scorer, assoc_w, assoc_a, n_sent, pickup=None):
    import i49d_two_loops as TL
    from i49a_inner_vessel import Vessel
    near = None
    if combo == "c":
        near = assoc_w
    elif combo == "d":
        near = AttrNear(assoc_a, wv)
    pick = None
    if combo == "g":
        pick = TL.Pick(lambda n: 0.0, k=PICK_K[combo])
    elif combo in PICK_K:
        pick = TL.Pick(lambda n: pickup.value(n, pos_of.get(n)), k=PICK_K[combo])
    builder = TL.Plain() if combo == "a" else TL.Scored(scorer, pos_of, pick=pick)
    th = TL.Thinker(Vessel(2048, bands=20, longest=1024), rd, near, builder)
    if combo == "d":
        near.th = th
    wt = th.attach(Watch())
    t0 = time.time()
    cap = 30 * len(rd) + 1000
    while rd.has_next() and th.n < cap:
        th.step()
    read_steps = th.n
    th.run(AFTER)
    sec = time.time() - t0
    a = th.arrays()
    acts = wt.acts
    outs = [i for i, x in enumerate(acts) if x == "出力"]
    # 文の終わりの直後: 最後に読んだ語が文の終わりの語で、その後まだ次の語を読んでいない
    after_end, last_end = [], False
    for i, x in enumerate(acts):
        if x == "読む":
            last_end = wt.end_read[i]
        after_end.append(last_end)
    lens = Counter(min(x["n"], 10) for x in wt.lines)
    assoc_in = [n for x in wt.lines for n in x["連想"]]
    H, R = th.odd.arrays()
    va = a["act"] / float(th.v.cap)
    tl = th.tally()
    felt = {}
    for grp, sel in (("聞いた順のまま", lambda b: not b["連想あり"]), ("連想の語あり", lambda b: b["連想あり"])):
        bs = [b for b in wt.builds if sel(b) and b["感じた物"]]
        felt[grp] = {k: (dict(数=len(v), 中央値=round(float(np.median(v)), 4), 上位1割の線=round(float(np.quantile(v, 0.9)), 4)) if v else None)
                     for k in ODD3 for v in [[b["感じた物"][k] for b in bs if k in b["感じた物"]]]}
    return dict(
        組=combo, 文=n_sent, 渡す語=len(rd), 歩=int(th.n), 読み終わるまでの歩=int(read_steps), 秒=round(sec, 1),
        一歩のミリ秒=round(sec / max(1, th.n) * 1000, 2), 内訳=th.counts(), 出力=len(outs),
        出力のうち文の終わりの直後=round(float(np.mean([after_end[i] for i in outs])), 4) if outs else None,
        文の終わりの直後の歩の割合=round(float(np.mean(after_end)), 4),
        文あたりの出力=round(len(outs) / max(1, n_sent), 3),
        行の長さ_内容語の数={str(k) if k < 10 else "10-": v for k, v in sorted(lens.items())},
        出した行の内容語の数={str(k) if k < 10 else "10-": v for k, v in sorted(Counter(min(x["出した内容語"], 10) for x in wt.lines).items())},
        出した行の内容語の数の中央値=float(np.median([x["出した内容語"] for x in wt.lines])) if wt.lines else None,
        出した行の語の数の中央値_詰め物込み=float(np.median([x["出した語"] for x in wt.lines])) if wt.lines else None,
        聞いた語だけの行の割合=round(float(np.mean([x["聞いた語だけ"] for x in wt.lines])), 4) if wt.lines else None,
        連想の語を含む行の割合=round(float(np.mean([bool(x["連想"]) for x in wt.lines])), 4) if wt.lines else None,
        行の中の連想の語のうち聞いていない語の割合=round(float(np.mean([n not in wt.heard for n in assoc_in])), 4) if assoc_in else None,
        連想の回数=len(wt.assoc_names), 組立の回数=len(wt.builds), もう言ったで休んだ回数=wt.rest_said,
        待つの連続の長さ=runs(acts, "待つ"),
        tally=dict(下がり方={k: {n: (None if v != v else round(v, 4)) for n, v in d.items()} for k, d in tl["下がり方"].items()},
                   器との動き={k: [None if v != v else round(v, 3) for v in row] for k, row in tl["器との動き"].items()}),
        ずらした数え=lagged(acts, R, list(th.odd.kinds), va),
        行づくりの違和感の割合=felt,
        見張り=dict(張り付き=th.v.watch["張り付き"].tolist(), 死んだ刻み=int(th.v.watch["死んだ"])))


def main():
    import i48ab_long_grain as G
    import i49b_read_long as RL
    import i49c_read_words as R
    import i49d_assoc as AS
    import i49d_odd as OD
    ap = argparse.ArgumentParser()
    for k in ("src", "conllu", "content", "func", "wave", "scorer", "cont", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--pickup", default=None, help="拾われやすさの表（e・f2・f5 で使う）")
    ap.add_argument("--n_sent", type=int, default=300)
    ap.add_argument("--n_sent_c", type=int, default=None, help="波の連想（c）だけ文を減らす時の文の数")
    ap.add_argument("--combos", default="a,b,c,d")
    a = ap.parse_args()
    t0 = time.time()
    RL.MAX_SENT = max(a.n_sent, a.n_sent_c or 0)
    w = RL.pick_works(a.src, a.conllu, n=1)[0][: a.n_sent]
    import spacy
    nlp = spacy.load("ja_ginza")
    docs = list(zip(nlp.pipe([s for s, _ in w], batch_size=256), [m for _, m in w]))
    wv = R.WordVecs.load(a.wave)
    table = OD.ContinueTable.load(a.cont)
    combos = a.combos.split(",")
    scorer = OD.OddScorer.load(a.scorer) if any(c != "a" for c in combos) else None
    assoc_w = AS.Assoc.from_wordvecs(wv) if "c" in combos else None
    assoc_a = None
    pickup = None
    if any(c in ("e", "f2", "f5") for c in combos):
        import i49d_pickup as PU
        pickup = PU.PickupTable.load(a.pickup)
    if "d" in combos:
        cb = json.load(open(a.content, encoding="utf-8"))
        assoc_a = AS.Assoc.from_attrs(G.AttrSpace(cb, G.Func(json.load(open(a.func, encoding="utf-8"))), 2048), cb)
    res = dict(用意の秒=round(time.time() - t0), 試し=[])
    for c in combos:
        n = a.n_sent_c if (c == "c" and a.n_sent_c) else a.n_sent
        rd, pos_of, dropped = make_reader(docs[:n], wv, table)
        known = sum(1 for w_, v, _ in rd.items if pos_of.get(w_))
        r = one(c, rd, pos_of, wv, scorer, assoc_w, assoc_a, n, pickup)
        r.update(丸ごと飛んだ文=dropped, 品詞の分かる語の割合=round(known / max(1, len(rd)), 4))
        if scorer is not None:
            r["型の語板に無い語の割合"] = round(float(np.mean([w_ not in scorer.m.S.cb for w_, _, _ in rd.items])), 4)
        res["試し"].append(r)
        print(c, json.dumps({k: r[k] for k in ("歩", "秒", "内訳", "出力")}, ensure_ascii=False), flush=True)
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
