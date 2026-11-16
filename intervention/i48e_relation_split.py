"""I48e: 語板 v3 の「関係」（L3）の 2 回目。名詞の後ろ（助詞）と述語の後ろ（語尾）を分けて測る。

I48d（i48d_relation_wave.py）で分かったこと: 語ごとの帯で平均の関係は持てる（数える表と同じ高さ）。行のほかの語を全部まとめた鍵は
効きが小さい（+0.02）。逆回しは効く。候補が 1,069 あって、述語の後ろ（語尾）の自由度が高く、全体に低い。

この試しで変えた所:
- 名詞（NOUN・PROPN・PRON）の後ろと、述語（VERB・ADJ）の後ろを、別の板・別の試しにする。
- 行の語は全部まとめず、学びの資料で「後ろのつなぎ」と結び付きの強い語（相互情報量の上位 K）だけを鍵に入れる。
  一覧は手で決めない（時の語・否定の語・問いの語が上がってくるかは、結果の chosen_cues で見る）。
- 行の印（手で決めた数少ない物。どれも並びと記号から読める）:
    問い     : 行の終わりが ？/? か、終わりの助詞が「か」
    前の行の問い: 1 つ前の行が問い（答えの行か）
    否定     : 自分の後ろのつなぎの外に「ない・ず・ぬ」がある
    既出     : その語が前の 3 行に出ている（は・が・も の流れの印。話題が新しいか）
    先頭・末尾: 行の中の最初・最後の中身の語か
- 係り受け: 係る語（head の語）と係り方（dep）を別の鍵にする。係り方は解析器が助詞から決めるので、助詞の答えが漏れる。
  係る語だけの数が、bot で使える数（述語は決まっているが、役は決まっていない時）。
- 「は・が・も・って」を 1 組に数えた当たりも出す（名詞）。

鍵の組み合わせ（variants）を順に足して、どれが効くかを見る。比べる相手は、同じ手がかりの数える表。

資料はオプションで替える（--cache）。公開の版は RealPersonaChat。非公開の版は PC の中だけで使い、出力に _wn を付ける。

実行: python intervention/i48e_relation_split.py --cache <parsed.json> --D 512 --out i48e.json [--device cuda:0]
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
import zlib
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from i48_wave_board import Waves  # noqa: E402

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

log = logging.getLogger("i48e")
NOUNS = ("NOUN", "PROPN", "PRON")
PREDS = ("VERB", "ADJ")
CONTENT = NOUNS + PREDS
CUE = CONTENT + ("ADV", "NUM", "INTJ")
FUNC = ("ADP", "AUX", "SCONJ", "PART")
NEG = ("ない", "ず", "ぬ", "ん")
TOPIC = ("は", "が", "も", "って")
NONE = "∅"
FLAGS = ("問い", "前の行の問い", "否定", "既出", "先頭", "末尾")


def chain(toks):
    return "+".join(t["lemma"] for t in toks) if toks else NONE


def is_question(ln):
    txt = ln.get("text", "").rstrip("」』）) 　")
    if txt.endswith(("？", "?")):
        return True
    T = [t for t in ln["toks"] if t["pos"] != "PUNCT"]
    return bool(T) and T[-1]["pos"] == "PART" and T[-1]["lemma"] == "か"


def samples(work):
    """対話 1 本 → 中身の語ごとの場面。host = 名 / 述。post = 後ろのつなぎ。"""
    out, prev, prev_q = [], [], False
    for li, ln in enumerate(work["lines"]):
        T = ln["toks"]
        q = is_question(ln)
        idx = [i for i, t in enumerate(T) if t["pos"] in CONTENT]
        cues = [t["lemma"] for t in T if t["pos"] in CUE]
        seen = set(sum(prev[-3:], []))
        line_s = []
        for k, i in enumerate(idx):
            j = i + 1
            while j < len(T) and T[j]["pos"] in FUNC:
                j += 1
            own = set(range(i + 1, j))
            neg = any(t["lemma"] in NEG and t["pos"] in ("AUX", "ADJ", "PART") and n not in own for n, t in enumerate(T))
            h = T[i].get("head", i)
            head = T[h]["lemma"] if 0 <= h < len(T) and h != i and T[h]["pos"] in CONTENT else NONE
            w = T[i]["lemma"]
            flags = dict(問い=q, 前の行の問い=prev_q, 否定=neg, 既出=w in seen, 先頭=k == 0, 末尾=k == len(idx) - 1)
            s = dict(w=w, pos=T[i]["pos"], host="名" if T[i]["pos"] in NOUNS else "述", post=chain(T[i + 1:j]),
                     cues=[c for c in cues if c != w], head=head, dep=T[i].get("dep", "?"), flags=flags,
                     nxt=T[idx[k + 1]]["lemma"] if k + 1 < len(idx) else NONE, npost=NONE, li=li)
            if line_s:
                line_s[-1]["npost"] = s["post"]
            line_s.append(s)
        out += line_s
        prev.append([t["lemma"] for t in T if t["pos"] in CONTENT])
        prev_q = q
    return out


def mark(x, D):
    return np.exp(1j * np.random.default_rng(zlib.crc32(x.encode())).uniform(0, 2 * np.pi, D)).astype(np.complex64)


def pick_cues(S, side, K, min_n=30):
    """行の語のうち、後ろのつなぎとの相互情報量（あり／なし × つなぎ）が大きい K 語。"""
    tot = Counter(s[side] for s in S)
    N = len(S)
    cnt, joint = Counter(), defaultdict(Counter)
    for s in S:
        for c in set(s["cues"]):
            cnt[c] += 1
            joint[c][s[side]] += 1
    p_t = {t: n / N for t, n in tot.items()}
    mi = {}
    for c, n in cnt.items():
        if n < min_n or n > 0.5 * N:
            continue
        pc = n / N
        v = 0.0
        for t, nt in tot.items():
            a = joint[c].get(t, 0)            # c あり・t
            b = nt - a                        # c なし・t
            if a:
                v += a / N * math.log(a / N / (pc * p_t[t]))
            if b:
                v += b / N * math.log(b / N / ((1 - pc) * p_t[t]))
        mi[c] = v
    return [c for c, _ in sorted(mi.items(), key=lambda x: -x[1])[:K]]


class Relation:
    """関係の板（後ろのつなぎ）。keys で鍵の組み合わせを選ぶ。
    語・品詞 / 行（行の語全部）/ 絞（選んだ語だけ）/ 印 / 係（係る語）/ 方（係り方）/ 逆（次の語とその後ろが決まっている）"""

    ROLES = ("語", "品詞", "行", "対", "絞", "絞対", "印", "係", "係対", "係り方", "逆", "逆対", "逆後")

    def __init__(self, wv, cands, D, keys, chosen=(), top=30, device=None):
        self.wv, self.cands, self.D, self.keys, self.top, self.device = wv, cands, D, set(keys), top, device
        self.cix = {c: i for i, c in enumerate(cands)}
        self.chosen = set(chosen)
        self.role = {k: mark("役:" + k, D) for k in self.ROLES}
        self.cmark = np.stack([mark("つなぎ:" + c, D) for c in cands])
        self.fmark = {f"{f}:{v}": mark(f"印:{f}:{v}", D) for f in FLAGS for v in (True, False)}

    def w(self, x):
        j = self.wv.ix.get(x)
        return None if j is None else self.wv.W[j].astype(np.complex64)

    def bag(self, ws):
        ids = [self.wv.ix[x] for x in ws if x in self.wv.ix]
        if not ids:
            return None
        v = self.wv.W[ids].sum(0)
        return (v / (np.linalg.norm(v) / math.sqrt(self.D) + 1e-9)).astype(np.complex64)

    def key(self, s):
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
        if "絞" in k:
            b = self.bag([c for c in s["cues"] if c in self.chosen])
            if b is not None:
                parts.append(b * R["絞"])
                if w is not None:
                    parts.append(w * b * R["絞対"])
        if "印" in k:
            fv = sum(self.fmark[f"{f}:{s['flags'][f]}"] for f in FLAGS)     # 印 6 つは 1 本にまとめて、ほかの鍵 1 つ分の重さにする
            fv = fv / (np.linalg.norm(fv) / math.sqrt(self.D) + 1e-9)
            parts.append(fv * R["印"])
            if w is not None:
                parts.append(w * fv * R["対"])                       # 語 × 印（「歩」が問いの行にある）
        if "係" in k:
            h = self.w(s["head"])
            if h is not None:
                parts.append(h * R["係"])
                if w is not None:
                    parts.append(w * h * R["係対"])
        if "方" in k:                                               # 係り方（dep）。解析器が助詞から決める物なので、助詞の答えが漏れる。別の鍵にして分けて測る
            parts.append(mark("係り方:" + s["dep"], self.D) * R["係り方"])
        if "逆" in k:
            n = self.w(s["nxt"])
            if n is not None:
                parts.append(n * R["逆"])
                if w is not None:
                    parts.append(w * n * R["逆対"])
            parts.append(mark("つなぎ:" + s["npost"], self.D) * R["逆後"])
        v = sum(parts)
        return (v / (np.linalg.norm(v) + 1e-9)).astype(np.complex64)

    def fit(self, S):
        K, P = [], []
        V = len(self.wv.vocab)
        self.band = np.zeros((V, self.D), np.complex64)
        self.band_n = np.zeros(V)
        for s in S:
            j = self.cix.get(s["post"])
            if j is None:
                continue
            i = self.wv.ix.get(s["w"])
            if i is not None:
                self.band[i] += self.cmark[j]
                self.band_n[i] += 1
            K.append(self.key(s))
            P.append(j)
        self.K, self.P = np.stack(K), np.array(P)
        self.prior = np.bincount(self.P, minlength=len(self.cands)).astype(float)
        if self.device and torch is not None:
            self.Kt = torch.from_numpy(self.K).to(self.device)
        return self

    def band_scores(self, s):
        """語の帯から読む。雑音の床（σ√(2 ln C)）より下は 0。"""
        i = self.wv.ix.get(s["w"])
        if i is None or self.band_n[i] == 0:
            return self.prior
        b = self.band[i]
        r = (self.cmark @ np.conj(b)).real / self.D
        floor = np.linalg.norm(b) / (self.D * math.sqrt(2)) * math.sqrt(2 * math.log(len(self.cands)))
        return np.where(r > floor, r, 0.0) + 1e-6 * self.prior / self.prior.sum()

    def scoop_scores(self, S):
        """置き場から読む: 問いと共鳴の強い top 場面のつなぎを、共鳴の 2 乗で重ねる。"""
        Q = np.stack([self.key(s) for s in S])
        if self.device and torch is not None:
            Rt = torch.abs(self.Kt @ torch.from_numpy(np.conj(Q)).to(self.device).T)
            val, o = torch.topk(Rt, self.top, dim=0)
            val, o = val.cpu().numpy(), o.cpu().numpy()
        else:
            R = np.abs(self.K @ np.conj(Q).T)
            o = np.argpartition(-R, self.top, axis=0)[: self.top]
            val = np.take_along_axis(R, o, 0)
        out = np.zeros((len(S), len(self.cands)))
        for c in range(len(S)):
            out[c] = np.bincount(self.P[o[:, c]], weights=val[:, c] ** 2, minlength=len(self.cands))
        return out + 1e-6 * self.prior / self.prior.sum()


class CountTable:
    """数えるだけの表（同じ手がかり）。語 → 品詞 → 全体 で下がり、手がかりごとの表を掛ける。"""

    def __init__(self, cands, keys=(), chosen=()):
        self.cands, self.cix, self.keys, self.chosen = cands, {c: i for i, c in enumerate(cands)}, set(keys), set(chosen)
        self.w_, self.p_, self.f_ = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
        self.all = Counter()

    def feats(self, s):
        k, f = self.keys, []
        if "行" in k:
            f += ["行:" + c for c in set(s["cues"])]
        if "絞" in k:
            f += ["行:" + c for c in set(s["cues"]) if c in self.chosen]
        if "印" in k:
            f += [f"印:{x}:{s['flags'][x]}" for x in FLAGS]
        if "係" in k:
            f.append("係:" + s["head"])
        if "方" in k:
            f.append("係り方:" + s["dep"])
        if "逆" in k:
            f += ["次:" + s["nxt"], "次後:" + s["npost"]]
        return f

    def fit(self, S):
        for s in S:
            c = s["post"]
            if c in self.cix:
                self.w_[s["w"]][c] += 1
                self.p_[s["pos"]][c] += 1
                self.all[c] += 1
                for x in self.feats(s):
                    self.f_[x][c] += 1
        tot = sum(self.all.values())
        self.pr = np.array([self.all[c] / tot for c in self.cands])
        return self

    def dist(self, c, back):
        z = sum(c.values())
        v = np.zeros(len(self.cands))
        for k, n in c.items():
            v[self.cix[k]] = n
        return (v + back) / (z + 1)

    def scores(self, s):
        out = self.dist(self.w_.get(s["w"], Counter()), self.dist(self.p_.get(s["pos"], Counter()), self.pr))
        for x in self.feats(s):
            if x in self.f_:
                out = out * self.dist(self.f_[x], self.pr) / self.pr
        return out


def measure(SC, te, cands, host):
    cix = {c: i for i, c in enumerate(cands)}
    t = np.array([cix[s["post"]] for s in te])
    o = np.argsort(-SC, 1)[:, :3]
    r = dict(top1=round(float((o[:, 0] == t).mean()), 4), top3=round(float((o == t[:, None]).any(1).mean()), 4))
    nz = t != cix.get(NONE, -1)
    r["top1_some"] = round(float((o[nz, 0] == t[nz]).mean()), 4) if nz.any() else None
    if host == "名":
        g = np.array([c in TOPIC for c in cands])
        r["top1_topic_group"] = round(float(((o[:, 0] == t) | (g[o[:, 0]] & g[t])).mean()), 4)
        m = g[t]
        r["topic_share"] = round(float(m.mean()), 4)
        r["top1_in_topic"] = round(float((o[m, 0] == t[m]).mean()), 4) if m.any() else None   # は・が・も・って の場面だけ
    return r


VARIANTS = (
    ("語・品詞", ("語",)),
    ("＋行の語（全部、I48d の形）", ("語", "行")),
    ("＋絞った語", ("語", "絞")),
    ("＋印", ("語", "印")),
    ("＋係る語", ("語", "係")),
    ("＋係る語・係り方", ("語", "係", "方")),
    ("＋印・係る語", ("語", "印", "係")),
    ("＋絞った語・印・係る語", ("語", "絞", "印", "係")),
    ("＋絞った語・印・係る語・係り方", ("語", "絞", "印", "係", "方")),
    ("＋絞った語・印・係る語・逆回し", ("語", "絞", "印", "係", "逆")),
    ("＋絞った語・印・係る語・係り方・逆回し", ("語", "絞", "印", "係", "方", "逆")),
)
TABLES = (
    ("数える表（語）", ()),
    ("数える表 ＋行の語（全部）", ("行",)),
    ("数える表 ＋絞った語・印・係る語", ("絞", "印", "係")),
    ("数える表 ＋絞った語・印・係る語・係り方", ("絞", "印", "係", "方")),
    ("数える表 ＋絞った語・印・係る語・逆回し", ("絞", "印", "係", "逆")),
    ("数える表 ＋絞った語・印・係る語・係り方・逆回し", ("絞", "印", "係", "方", "逆")),
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="解析済みの会話の資料（works: [{name, lines: [{text, toks}]}]）")
    ap.add_argument("--test_works", type=int, default=500)
    ap.add_argument("--train_works", type=int, default=0, help="0 = 残り全部")
    ap.add_argument("--D", type=int, default=512)
    ap.add_argument("--min_count", type=int, default=5)
    ap.add_argument("--max_scenes", type=int, default=200000, help="置き場に入れる場面の上限（板ごと）")
    ap.add_argument("--test_n", type=int, default=4000)
    ap.add_argument("--cues", type=int, default=64, help="絞る行の語の数（相互情報量の上位）")
    ap.add_argument("--hosts", default="名,述")
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
    tr_all = [s for w in tr_w for s in samples(w)]
    te_all = [s for w in te_w for s in samples(w)]
    log.info("場面: 学び %d（%d 対話）、試し %d（%d 対話）", len(tr_all), len(tr_w), len(te_all), len(te_w))
    vocab = sorted({t["lemma"] for w in tr_w for ln in w["lines"] for t in ln["toks"] if t["pos"] in CUE})
    seq = [[t["lemma"] for t in ln["toks"] if t["pos"] in CUE] for w in tr_w for ln in w["lines"] + [dict(toks=[])]]
    wv = Waves(vocab, a.D, a.seed)
    dev = torch.device(a.device) if (a.device and torch is not None) else None
    wv.learn(seq, dev=dev)
    wv.W = wv.W.astype(np.complex64)
    wv.base = None
    res = dict(cache=os.path.basename(a.cache), n_train_works=len(tr_w), n_test_works=len(te_w), D=a.D, tasks={})
    for host in a.hosts.split(","):
        tr_h = [s for s in tr_all if s["host"] == host]
        cnt = Counter(s["post"] for s in tr_h)
        cands = [c for c, n in cnt.most_common() if n >= a.min_count]
        cs = set(cands)
        tr_c = [s for s in tr_h if s["post"] in cs]
        te_h = [s for s in te_all if s["host"] == host]
        te = [s for s in te_h if s["post"] in cs]
        cover = len(te) / max(len(te_h), 1)
        te = random.Random(a.seed + 1).sample(te, min(a.test_n, len(te)))
        tr_k = tr_c[:: max(1, math.ceil(len(tr_c) / a.max_scenes))]
        chosen = pick_cues(tr_c, "post", a.cues)
        name = "名詞の後ろ（助詞）" if host == "名" else "述語の後ろ（語尾）"
        log.info("== %s: 候補 %d、学び %d（置き場 %d）、試し %d（候補に入る割合 %.3f）、∅ %.3f", name, len(cands), len(tr_c), len(tr_k),
                 len(te), cover, sum(s["post"] == NONE for s in te) / len(te))
        log.info("絞った語: %s", " ".join(chosen[:40]))
        rows = {}

        def row(label, SC):
            rows[label] = measure(SC, te, cands, host)
            r = rows[label]
            log.info("%-36s 1 番目 %.3f（3 番まで %.3f）つなぎ有り %.3f%s", label, r["top1"], r["top3"], r["top1_some"] or 0,
                     f"　は・が・も・って 1 組 {r['top1_topic_group']:.3f}　その中 {r['top1_in_topic'] or 0:.3f}" if host == "名" else "")

        pr = np.array([cnt[c] for c in cands], float)
        row("いちばん多いつなぎ", np.tile(pr, (len(te), 1)))
        for label, keys in TABLES:
            ct = CountTable(cands, keys, chosen).fit(tr_c)
            row(label, np.stack([ct.scores(s) for s in te]))
        B = 256
        for vi, (label, keys) in enumerate(VARIANTS):
            rel = Relation(wv, cands, a.D, keys, chosen, device=dev).fit(tr_k)
            if vi == 0:
                row("波: 語ごとの帯（1 本）", np.stack([rel.band_scores(s) for s in te]))
            row("波: すくう " + label, np.concatenate([rel.scoop_scores(te[i:i + B]) for i in range(0, len(te), B)]))
            del rel
            if dev is not None:
                torch.cuda.empty_cache()
        res["tasks"][name] = dict(n_cands=len(cands), n_train=len(tr_c), n_scenes=len(tr_k), n_test=len(te), cover=round(cover, 4),
                                  none_share=round(sum(s["post"] == NONE for s in te) / len(te), 4), chosen_cues=chosen,
                                  top_cands=cands[:40], rows=rows)
    res["seconds"] = round(time.time() - t0)
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log.info("済み %.0f 秒", time.time() - t0)


if __name__ == "__main__":
    main()
