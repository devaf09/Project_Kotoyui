"""I48k: 主語の名詞に付く「は・が・も」を、前の行までの流れから波で当てる試し（語板 v3 の L3 の続き）。

I48e〜I48f・4.13 で分かったこと: 関係の板（名詞の後ろの助詞、係り方なしで 0.43）を bot につないでも、言い方の指摘は減らなかった。
助詞の選び方の多くは、行の中の語よりも「その名詞が話にもう出ているか」で決まる（新しい話題は が、出ている話題は は、足すなら も）。
ここでは、主語の場所（後ろが は・が・も のどれか）は決まっているとして、3 つのどれかを当てる。bot の行づくりでは、
主語の名詞と述語を決めた後に、この 3 つを選ぶ段に当たる。

鍵（手で置く規則や閾値は足さない。流れの印を鍵にして、学びの資料から明るさを学ぶ）:
  語・品詞       : 主語の名詞
  係             : 係る述語（係り受けの head）
  行             : 行のほかの中身の語
  流れ           : その名詞が どこに出ていたか（相手の前の行 / 自分の前の行 / 3 行以内 / もっと前 / 初めて）、
                   前の行の主語と同じか（同じならその時の助詞も）、2 行前の主語と同じか、相手の前の行の主語の助詞、
                   行の中で初めの中身の語か
  印             : 問いの行か、前の行が問いか、否定があるか（I48e と同じ）
RealPersonaChat は 2 人の対話で、行が交互になるので、1 つ前の行 = 相手、2 つ前 = 自分、として読む。
係り方（dep）は解析器が助詞から決めるので、鍵に入れない。

比べる物: 同じ手がかりの数える表、流れなし（語と係る述語だけ）、いちばん多い助詞。

実行: python intervention/past/i48k_subject_flow.py --cache <parsed.json> --D 512 --seed 0 --out i48k.json [--device cuda:0]
"""
from __future__ import annotations
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from i48_wave_board import Waves  # noqa: E402
from i48e_relation_split import (CONTENT, CUE, FLAGS, FUNC, NEG, NOUNS, NONE, CountTable, Relation, chain,  # noqa: E402
                                 is_question, mark)

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

log = logging.getLogger("i48k")
SUBJ = ("は", "が", "も")
WHERE = ("相手の前の行", "自分の前の行", "3 行以内", "もっと前", "初めて")
FLOWS = ("どこ", "前の主語", "2 行前の主語", "相手の主語の助詞", "行の初め")


def where(w, prev):
    """名詞 w が、前の行（prev[-1] = 相手、prev[-2] = 自分）までのどこに出ていたか。"""
    if prev and w in prev[-1]:
        return WHERE[0]
    if len(prev) >= 2 and w in prev[-2]:
        return WHERE[1]
    if any(w in p for p in prev[-3:]):
        return WHERE[2]
    if any(w in p for p in prev):
        return WHERE[3]
    return WHERE[4]


def samples(work):
    """対話 1 本 → 主語の場面（名詞の後ろのつなぎが は・が・も のどれか）。ほかの名詞の場面は流れの記録にだけ使う。"""
    out, prev, subj, prev_q = [], [], [], False                      # subj[li] = その行の主語の (名詞, 助詞) の並び
    for li, ln in enumerate(work["lines"]):
        T = ln["toks"]
        q = is_question(ln)
        idx = [i for i, t in enumerate(T) if t["pos"] in CONTENT]
        cues = [t["lemma"] for t in T if t["pos"] in CUE]
        here = []
        for k, i in enumerate(idx):
            if T[i]["pos"] not in NOUNS:
                continue
            j = i + 1
            while j < len(T) and T[j]["pos"] in FUNC:
                j += 1
            post = chain(T[i + 1:j])
            if post not in SUBJ:
                continue
            w = T[i]["lemma"]
            own = set(range(i + 1, j))
            neg = any(t["lemma"] in NEG and t["pos"] in ("AUX", "ADJ", "PART") and n not in own for n, t in enumerate(T))
            h = T[i].get("head", i)
            head = T[h]["lemma"] if 0 <= h < len(T) and h != i and T[h]["pos"] in CONTENT else NONE
            s1 = subj[-1] if subj else []
            s2 = subj[-2] if len(subj) >= 2 else []
            same1 = next((p for x, p in s1 if x == w), None)
            flow = {"どこ": where(w, prev),
                    "前の主語": f"同じ:{same1}" if same1 else ("違う" if s1 else "無し"),
                    "2 行前の主語": "同じ" if any(x == w for x, _ in s2) else ("違う" if s2 else "無し"),
                    "相手の主語の助詞": s1[0][1] if s1 else NONE,
                    "行の初め": k == 0}
            out.append(dict(w=w, pos=T[i]["pos"], host="名", post=post, cues=[c for c in cues if c != w], head=head, dep="?",
                            flags=dict(問い=q, 前の行の問い=prev_q, 否定=neg, 既出=flow["どこ"] != WHERE[4], 先頭=k == 0,
                                       末尾=k == len(idx) - 1),
                            flow=flow, nxt=NONE, npost=NONE, li=li))
            here.append((w, post))
        subj.append(here)
        prev.append({t["lemma"] for t in T if t["pos"] in CONTENT})
        prev_q = q
    return out


class FlowRelation(Relation):
    """I48e の関係の板に「流れ」の鍵を足した物。流れの印 5 つは 1 本にまとめ、語 × 流れ の対も入れる。"""

    ROLES = Relation.ROLES + ("流れ", "流れ対")

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.flmark = {}

    def fl(self, f, v):
        x = f"流れ:{f}:{v}"
        if x not in self.flmark:
            self.flmark[x] = mark(x, self.D)
        return self.flmark[x]

    def key(self, s):
        """I48e の鍵と同じ足し方（どの手がかりも 1 つ分の重さ）。流れは、印と同じく 5 つを 1 本にまとめて 1 つ分にし、語 × 流れ の対を足す。"""
        k, R, parts = self.keys, self.role, []
        w = self.w(s["w"])
        if w is not None:
            parts.append(w * R["語"])
        parts.append(mark("品:" + s["pos"], self.D) * R["品詞"])
        if "行" in k:
            b = self.bag(s["cues"])
            if b is not None:
                parts.append(b * R["行"])
                if w is not None:
                    parts.append(w * b * R["対"])
        if "印" in k:
            fv = sum(self.fmark[f"{f}:{s['flags'][f]}"] for f in FLAGS)
            fv = fv / (np.linalg.norm(fv) / math.sqrt(self.D) + 1e-9)
            parts.append(fv * R["印"])
            if w is not None:
                parts.append(w * fv * R["対"])
        if "係" in k:
            h = self.w(s["head"])
            if h is not None:
                parts.append(h * R["係"])
                if w is not None:
                    parts.append(w * h * R["係対"])
        if "流" in k:
            fv = sum(self.fl(f, s["flow"][f]) for f in FLOWS)
            fv = fv / (np.linalg.norm(fv) / math.sqrt(self.D) + 1e-9)
            parts.append(fv * R["流れ"])
            if w is not None:
                parts.append(w * fv * R["流れ対"])                   # 語 × 流れ（「猫」が相手の前の行に出ている）
        v = sum(parts)
        return (v / (np.linalg.norm(v) + 1e-9)).astype(np.complex64)


class FlowTable(CountTable):
    def feats(self, s):
        f = super().feats(s)
        if "流" in self.keys:
            f += [f"流れ:{x}:{s['flow'][x]}" for x in FLOWS]
        return f


def measure(SC, te, cands):
    cix = {c: i for i, c in enumerate(cands)}
    t = np.array([cix[s["post"]] for s in te])
    p = SC.argmax(1)
    r = dict(top1=round(float((p == t).mean()), 4))
    rec = {c: round(float((p[t == i] == i).mean()), 4) if (t == i).any() else None for c, i in cix.items()}
    r["recall"] = rec
    r["macro"] = round(float(np.mean([v for v in rec.values() if v is not None])), 4)
    r["share_pred"] = {c: round(float((p == i).mean()), 4) for c, i in cix.items()}
    return r


VARIANTS = (
    ("流れなし（語・係る述語）", ("語", "係")),
    ("＋行の語", ("語", "係", "行")),
    ("＋印", ("語", "係", "印")),
    ("＋流れ", ("語", "係", "流")),
    ("＋流れ・印", ("語", "係", "流", "印")),
    ("＋印・行の語", ("語", "係", "印", "行")),
    ("＋流れ・印・行の語", ("語", "係", "流", "印", "行")),
)
TABLES = (
    ("数える表（語）", ()),
    ("数える表 ＋係る述語", ("係",)),
    ("数える表 ＋係る述語・行の語", ("係", "行")),
    ("数える表 ＋係る述語・印", ("係", "印")),
    ("数える表 ＋係る述語・流れ", ("係", "流")),
    ("数える表 ＋係る述語・流れ・印", ("係", "流", "印")),
    ("数える表 ＋係る述語・印・行の語", ("係", "印", "行")),
    ("数える表 ＋係る述語・流れ・印・行の語", ("係", "流", "印", "行")),
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="解析済みの会話の資料（works: [{name, lines: [{text, toks}]}]）")
    ap.add_argument("--test_works", type=int, default=500)
    ap.add_argument("--train_works", type=int, default=0, help="0 = 残り全部")
    ap.add_argument("--D", type=int, default=512)
    ap.add_argument("--max_scenes", type=int, default=200000)
    ap.add_argument("--test_n", type=int, default=4000)
    ap.add_argument("--device", default=None, help="cuda:0 など。無ければ numpy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    t0 = time.time()
    works = json.load(open(a.cache, encoding="utf-8"))
    works = works["works"] if isinstance(works, dict) else works
    random.Random(a.seed).shuffle(works)
    te_w, tr_w = works[: a.test_works], works[a.test_works:]
    if a.train_works:
        tr_w = tr_w[: a.train_works]
    tr = [s for w in tr_w for s in samples(w)]
    te_all = [s for w in te_w for s in samples(w)]
    te = random.Random(a.seed + 1).sample(te_all, min(a.test_n, len(te_all)))
    cands = list(SUBJ)
    share = Counter(s["post"] for s in te)
    log.info("主語の場面: 学び %d（%d 対話）、試し %d（%d 対話の %d から）、試しの割合 %s", len(tr), len(tr_w), len(te), len(te_w), len(te_all),
             {c: round(share[c] / len(te), 3) for c in cands})
    log.info("流れ（試し）: %s", dict(Counter(s["flow"]["どこ"] for s in te)))
    by_flow = {}                                                     # 流れの値ごとの は・が・も の割合（学び）。流れに情報があるかの目安
    for f in FLOWS:
        c = {}
        for s in tr:
            c.setdefault(str(s["flow"][f]), Counter())[s["post"]] += 1
        by_flow[f] = {v: dict(n=sum(x.values()), **{p: round(x[p] / sum(x.values()), 3) for p in SUBJ}) for v, x in c.items()}
        log.info("流れ %s: %s", f, by_flow[f])
    vocab = sorted({t["lemma"] for w in tr_w for ln in w["lines"] for t in ln["toks"] if t["pos"] in CUE})
    seq = [[t["lemma"] for t in ln["toks"] if t["pos"] in CUE] for w in tr_w for ln in w["lines"] + [dict(toks=[])]]
    wv = Waves(vocab, a.D, a.seed)
    dev = torch.device(a.device) if (a.device and torch is not None) else None
    wv.learn(seq, dev=dev)
    wv.W, wv.base = wv.W.astype(np.complex64), None
    tr_k = tr[:: max(1, math.ceil(len(tr) / a.max_scenes))]
    rows = {}

    def row(label, SC):
        rows[label] = r = measure(SC, te, cands)
        log.info("%-34s 1 番目 %.3f　平均の再現率 %.3f　（は %.3f・が %.3f・も %.3f）", label, r["top1"], r["macro"],
                 r["recall"]["は"] or 0, r["recall"]["が"] or 0, r["recall"]["も"] or 0)

    pr = np.array([Counter(s["post"] for s in tr)[c] for c in cands], float)
    row("いちばん多い助詞", np.tile(pr, (len(te), 1)))
    for label, keys in TABLES:
        ct = FlowTable(cands, keys).fit(tr)
        row(label, np.stack([ct.scores(s) for s in te]))
    B = 256
    for vi, (label, keys) in enumerate(VARIANTS):
        rel = FlowRelation(wv, cands, a.D, keys, device=dev).fit(tr_k)
        if vi == 0:
            row("波: 語ごとの帯（1 本）", np.stack([rel.band_scores(s) for s in te]))
        row("波: すくう " + label, np.concatenate([rel.scoop_scores(te[i:i + B]) for i in range(0, len(te), B)]))
        del rel
        if dev is not None:
            torch.cuda.empty_cache()
    res = dict(cache=os.path.basename(a.cache), seed=a.seed, D=a.D, n_train_works=len(tr_w), n_test_works=len(te_w), n_train=len(tr),
               n_scenes=len(tr_k), n_test=len(te), test_share={c: round(share[c] / len(te), 4) for c in cands},
               flow_share=dict(Counter(s["flow"]["どこ"] for s in te)), by_flow=by_flow, rows=rows, seconds=round(time.time() - t0))
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log.info("済み %.0f 秒", time.time() - t0)


if __name__ == "__main__":
    main()
