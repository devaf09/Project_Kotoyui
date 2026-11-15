"""I48n: 受けの句の板（測る所まで）。返事の行の始めの句のうち、相手の前の行の語 X を受けて、X 以外がすべて機能語の句
（X＋です＋ね、X＋です＋か、X＋は …）を、相手の前の行から 2 段で当てる。I48m の数えの続き。

句の取り方（一覧は手で置かない）:
  始めの句（I48m と同じ。受けの一言を外した、最初の区切りまで）に中身の語がちょうど 1 つあり、それが相手の前の行の中身の語と同じ原形で、
  ほかの語がすべて機能語（助詞・助動詞・接続助詞・終助詞）の時、その形（X を X に置き換えた語の形の並び）を受けの句とする。
  学びの資料で 20 回以上の形を候補にし、少ない形は「ほかの受けの句」、受けの句でない行は「受けの句なし」。
2 段:
  付けるか : 「受けの句なし」の見込みの低さで、受けの句のある行を見分けられるか（AUROC）。鍵は相手の行（語尾・中身の語・印。I48l と同じ）。
  どの形か : 受けの句のある行だけで、名前のある形から 1 番目・3 番目まで。鍵は相手の行 ＋ X の語の波。
どの語を X にするか:
  相手の行の中身の語（異なる原形）のうち、どれが受けられたか。鍵は語の波・品詞・行の中の位置。
  行の最後の中身の語・最初の中身の語・乱数と比べる。
丁寧さ（資料のまま。数だけ見る）:
  形から です・ます（原形）の語を外して束ねた時に、候補がいくつにまとまるか、どの形かの当たりがどう変わるか。

実行: python intervention/i48n_echo_board.py --cache <parsed.json> --D 512 --seed 0 --out i48n.json [--device cuda:0]
"""
from __future__ import annotations

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from i48_wave_board import Waves  # noqa: E402
from i48e_relation_split import CONTENT, CUE, FUNC, CountTable, Relation, mark  # noqa: E402
import i48l_uke_flow as U  # noqa: E402
from i48m_echo_phrase import first_phrase  # noqa: E402

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

log = logging.getLogger("i48n")
NO_PH = "受けの句なし"
OTHER_PH = "ほかの受けの句"
POLITE = ("です", "ます")


def phrase(prev_toks, reply_toks):
    """受けの句の (形, 丁寧さを外した形, X の原形)。受けの句でなければ None。"""
    P = {t["lemma"] for t in U.body(prev_toks) if t["pos"] in CONTENT}
    F = first_phrase(reply_toks)
    cont = [t for t in F if t["pos"] in CONTENT]
    if len(cont) != 1 or cont[0]["lemma"] not in P or not all(t["pos"] in FUNC for t in F if t is not cont[0]):
        return None
    x = cont[0]
    shape = "+".join("X" if t is x else t["text"] for t in F)
    plain = "+".join("X" if t is x else t["text"] for t in F if t is x or t["lemma"] not in POLITE)
    return shape, plain, x["lemma"]


def samples(work):
    """対話 1 本 → 行ごとの場面（2 行目から）。鍵の材料は I48l と同じ（相手の前の行）。post = 受けの句の形、plain = 丁寧さを外した形、
    x = 受けた語、words = 相手の行の中身の語（異なる原形、出た順に 品詞・位置）。"""
    S = U.samples(work)
    L = work["lines"]
    for s in S:
        li = s["li"]
        ph = phrase(L[li - 1]["toks"], L[li]["toks"])
        s["post"], s["plain"], s["x"] = (ph if ph else (NO_PH, NO_PH, None))
        s["w"] = s["x"] or U.NONE
        seen, words = set(), []
        cont = [t for t in U.body(L[li - 1]["toks"]) if t["pos"] in CONTENT]
        for k, t in enumerate(cont):
            if t["lemma"] not in seen:
                seen.add(t["lemma"])
                words.append(dict(w=t["lemma"], pos=t["pos"], first=k == 0, last=k == len(cont) - 1,
                                  rel="前" if k < len(cont) / 3 else ("後" if k >= 2 * len(cont) / 3 else "中")))
        s["words"] = words
    return S


def to_cands(S, cands, key="post"):
    cs = set(cands)
    for s in S:
        if s[key] not in cs:
            s[key] = OTHER_PH
    return S


class PhraseRelation(U.UkeRelation):
    """I48l の受けの鍵（尾・中・印）に、X の語の波（語・語 × 尾）を足した物。"""

    ROLES = U.UkeRelation.ROLES + ("X", "X尾")

    def key(self, s):
        v = super().key(s)
        if "X" not in self.keys:
            return v
        x = self.w(s["w"])
        if x is None:
            return v
        u = (x * self.role["X"] + x * self.m("尾:" + s["tail"]) * self.role["X尾"]) / math.sqrt(2)
        u = u / (np.linalg.norm(u) / math.sqrt(self.D) + 1e-9)
        n_other = sum(1 for k in ("尾", "中", "印") if k in self.keys)          # ほかの手がかりと同じ 1 つ分の重さで足す
        z = v * math.sqrt(max(1, n_other)) * (np.linalg.norm(u) / (np.linalg.norm(v) * math.sqrt(max(1, n_other)) + 1e-9)) + u
        return (z / (np.linalg.norm(z) + 1e-9)).astype(np.complex64)


class PhraseTable(U.UkeTable):
    def feats(self, s):
        f = super().feats(s)
        if "X" in self.keys and s["x"]:
            f.append("X:" + s["x"])
        return f


def pick_samples(S):
    """どの語を X にするか: 受けの句のある場面の、相手の行の中身の語ごとの場面（受けられた = 1）。"""
    out = []
    for s in S:
        if s["x"] is None or len(s["words"]) < 1:
            continue
        for k, w in enumerate(s["words"]):
            out.append(dict(w=w["w"], pos=w["pos"], host="X", post="受ける" if w["w"] == s["x"] else "受けない", line=id(s), k=k,
                            first=w["first"], last=w["last"], rel=w["rel"], n=len(s["words"]), flags={}, cues=[], head=U.NONE,
                            dep="?", nxt=U.NONE, npost=U.NONE))
    return out


class PickRelation(Relation):
    """語の波・品詞・位置で、その語が受けられるかを当てる板。"""

    ROLES = Relation.ROLES + ("位置", "語位置")

    def key(self, s):
        parts = []
        w = self.w(s["w"])
        pm = mark(f"位置:{s['rel']}:{s['first']}:{s['last']}", self.D)
        if w is not None:
            parts.append(w * self.role["語"])
            parts.append(w * pm * self.role["語位置"])
        parts.append(mark("品:" + s["pos"], self.D) * self.role["品詞"])
        parts.append(pm * self.role["位置"])
        v = sum(parts)
        return (v / (np.linalg.norm(v) + 1e-9)).astype(np.complex64)


class PickTable(CountTable):
    def feats(self, s):
        return [f"位置:{s['rel']}:{s['first']}:{s['last']}", "品:" + s["pos"]]


def pick_measure(P, scores):
    """P = 語ごとの場面、scores = 「受ける」の見込み。行ごとに 1 番の語が受けられた語か。"""
    by = {}
    for s, v in zip(P, scores):
        by.setdefault(s["line"], []).append((v, s))
    hit = [max(g, key=lambda z: z[0])[1]["post"] == "受ける" for g in by.values()]
    multi = [h for h, g in zip(hit, by.values()) if len(g) >= 2]
    return dict(行=len(hit), 当たり=round(float(np.mean(hit)), 4), 語が2つ以上の行=len(multi),
                その中の当たり=round(float(np.mean(multi)), 4) if multi else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--test_works", type=int, default=500)
    ap.add_argument("--D", type=int, default=512)
    ap.add_argument("--min_count", type=int, default=20)
    ap.add_argument("--max_scenes", type=int, default=200000)
    ap.add_argument("--test_n", type=int, default=0, help="試しの行の数（0 = 試しの対話の全部）")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    t0 = time.time()
    works = json.load(open(a.cache, encoding="utf-8"))
    works = works["works"] if isinstance(works, dict) else works
    random.Random(a.seed).shuffle(works)
    te_w, tr_w = works[: a.test_works], works[a.test_works:]
    tr0 = [s for w in tr_w for s in samples(w)]
    te0 = [s for w in te_w for s in samples(w)]
    if a.test_n:
        te0 = random.Random(a.seed + 1).sample(te0, min(a.test_n, len(te0)))
    vocab = sorted({t["lemma"] for w in tr_w for ln in w["lines"] for t in ln["toks"] if t["pos"] in CUE})
    seq = [[t["lemma"] for t in ln["toks"] if t["pos"] in CUE] for w in tr_w for ln in w["lines"] + [dict(toks=[])]]
    wv = Waves(vocab, a.D, a.seed)
    dev = torch.device(a.device) if (a.device and torch is not None) else None
    wv.learn(seq, dev=dev)
    wv.W, wv.base = wv.W.astype(np.complex64), None
    res = dict(cache=os.path.basename(a.cache), seed=a.seed, D=a.D, kinds={})
    B = 256
    for kind in ("形のまま", "丁寧さを外した形"):
        key = "post" if kind == "形のまま" else "plain"
        tr = [dict(s, post=s[key]) for s in tr0]
        te = [dict(s, post=s[key]) for s in te0]
        cnt = Counter(s["post"] for s in tr)
        cands = [c for c, n in cnt.most_common() if n >= a.min_count and c != NO_PH] + [OTHER_PH, NO_PH]
        tr, te = to_cands(tr, cands), to_cands(te, cands)
        tr = [dict(s) for s in tr]
        for s in tr + te:                                                # I48l の measure を使うため、受けの句なし・ほかの受けの句の名前をそろえる
            s["post"] = {NO_PH: U.NO_UKE, OTHER_PH: U.OTHER}.get(s["post"], s["post"])
        uc = [{NO_PH: U.NO_UKE, OTHER_PH: U.OTHER}.get(c, c) for c in cands]
        has_share = sum(s["post"] != U.NO_UKE for s in te) / len(te)
        log.info("== %s: 候補 %d（形 %d）、受けの句のある行（試し）%.3f、ほかの受けの句 %.3f", kind, len(uc), len(uc) - 2, has_share,
                 sum(s["post"] == U.OTHER for s in te) / len(te))
        log.info("多い形: %s", " ".join(f"{c}({cnt[c]})" for c in cands[:20]))
        tr_has = [s for s in tr if s["post"] not in (U.NO_UKE, U.OTHER)]
        tr_k = tr[:: max(1, math.ceil(len(tr) / a.max_scenes))]
        rows = {}

        def row(label, SC, SCw=None):
            rows[label] = r = U.measure(SC, te, uc, SCw)
            log.info("%-30s 付けるか AUROC %.3f　どの形か 1 番目 %.3f（3 番まで %.3f、%d 行）", label, r["有無 AUROC"],
                     r["どの受けか 1 番目"], r["どの受けか 3 番目まで"], r["受けの行の数"])

        pr = np.array([cnt.get(c, 0) if c != OTHER_PH else sum(v for k, v in cnt.items() if k not in cands) for c in cands], float)
        row("いちばん多い形", np.tile(pr, (len(te), 1)))
        blind = lambda S: [dict(s, w=U.NONE) for s in S]                # 1 段目（付けるか）は、受けた語を見ない（数える表は語の表に落ちるので、語を隠す）
        ct = PhraseTable(uc, ("尾", "中", "印")).fit(blind(tr))
        cw = PhraseTable(uc, ("尾", "中", "印", "X")).fit(tr_has)
        row("数える表 相手の行（2 段目は ＋X）", np.stack([ct.scores(s) for s in blind(te)]), np.stack([cw.scores(s) for s in te]))
        rel = PhraseRelation(wv, uc, a.D, ("尾", "中", "印"), device=dev).fit(tr_k)
        S1 = np.concatenate([rel.scoop_scores(te[i:i + B]) for i in range(0, len(te), B)])
        del rel
        for label, keys in (("波 相手の行", ("尾", "中", "印")), ("波 相手の行 ＋X の語の波", ("尾", "中", "印", "X"))):
            rw = PhraseRelation(wv, uc, a.D, keys, device=dev).fit(tr_has[:: max(1, math.ceil(len(tr_has) / a.max_scenes))])
            row(label, S1, np.concatenate([rw.scoop_scores(te[i:i + B]) for i in range(0, len(te), B)]))
            del rw
        if dev is not None:
            torch.cuda.empty_cache()
        res["kinds"][kind] = dict(n_cands=len(uc) - 2, has_share=round(has_share, 4), top=[(c, cnt.get(c, 0)) for c in cands[:30]], rows=rows)
    # どの語を X にするか
    Ptr, Pte = pick_samples(tr0), pick_samples(te0)
    pr_rng = random.Random(a.seed + 2)
    picks = dict(
        最後の中身の語=pick_measure(Pte, [float(s["last"]) for s in Pte]),
        最初の中身の語=pick_measure(Pte, [float(s["first"]) for s in Pte]),
        乱数=pick_measure(Pte, [pr_rng.random() for s in Pte]))
    pt = PickTable(["受ける", "受けない"]).fit(Ptr)
    picks["数える表（語・品詞・位置）"] = pick_measure(Pte, [pt.scores(s)[0] / pt.scores(s).sum() for s in Pte])
    pk = PickRelation(wv, ["受ける", "受けない"], a.D, ("語",), device=dev).fit(Ptr[:: max(1, math.ceil(len(Ptr) / a.max_scenes))])
    SC = np.concatenate([pk.scoop_scores(Pte[i:i + B]) for i in range(0, len(Pte), B)])
    picks["波（語の波・品詞・位置）"] = pick_measure(Pte, list(SC[:, 0] / SC.sum(1)))
    for k, v in picks.items():
        log.info("どの語を X に: %-22s 当たり %.3f（%d 行）　語が 2 つ以上の行で %.3f（%d 行）", k, v["当たり"], v["行"], v["その中の当たり"] or 0,
                 v["語が2つ以上の行"])
    res["X の選び方"] = picks
    res["seconds"] = round(time.time() - t0)
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log.info("済み %.0f 秒", time.time() - t0)


if __name__ == "__main__":
    main()
