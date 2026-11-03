"""見積もり: 話者の「隠し味」で、会話文の次のサブワードがどれだけ当たりやすくなるか（数えた表、M1 T2 の前、2026-10-24）

着想: 持ち主の案（2DWIG の成分で、話者の組ごとの 2DTG を混ぜる）。SN2-Parts-2D の 150 作品の見積もりでは、次の文字で 26.9 → 27.7%（+0.8 点）。
      光で回す前に、1,000 作品・サブワードで、数えた表（後退つき 3-gram）を使って効きの上限を測る。docs/kickoff.md 3.5 節の判断:
      **+0.5 点に届かなければ、光で回す前に報告してデータを増やす相談をする。**
データ: 青空文庫 1,000 作品の会話文（aozora.py）。作品の 15% がテスト、訓練の作品の 10% を調整用（混ぜる強さ λ・温度を選ぶ）。
        単位: SentencePiece（unigram、語彙 4096、文字の網羅率 0.995。2048 は青空文庫の漢字が多く作れない、バイトに落とす）を訓練の作品の会話文だけで学び直す。
        課題: 会話文の中で、直前 2 単位 → 次の 1 単位（文の終わり = 」も当てる）。
方法:
  none      : 全部の会話文で数えた 3-gram（Witten-Bell の後退）だけ。隠し味なし
  組ごとの表: 話者の組ごとに 3-gram を数える。数え方 hard = 一人称つき + 交互の会話で広げたラベルの会話文だけ、
              soft = 全部の訓練の会話文を、2DWIG（数えた表のナイーブベイズ、語尾 1〜3 + 2 つ組）の組の確率で重みをつけて数える
  混ぜ方    : P = λ Σ_s w_s P_s + (1 − λ) P_all。w は話者の手がかりから
  手がかり  : oracle = 本当の類（一人称つきの会話文。一人称の位置そのものも当てやすくなる点に注意）、
              spread = 一人称のない会話文に交互の会話で写した類（一人称の位置がないので公平な「話者が分かっている」場合）、
              prev = 同じ列の 2 つ前の会話文（同じ話者の見込み 93.6%）に 2DWIG を当てた組の確率、
              prefix = いま書いている会話文の、ここまでの部分に 2DWIG を当てた組の確率
  組の分け方: g5 = 私系（私・わたくし・我々）/ 僕 / 俺系（俺・おいら・あっし）/ 女ことば（あたし・わらわ・わちき）/ 古風（わし・武士）、g12 = 12 類そのまま
指標: 次の単位の正解率、CE（1 単位あたり・1 文字あたり）。効き = 手がかりあり − none。
結果: results/est_flavor_counts.txt（1 条件 = 1 行の JSON）
実行: python experiments/est_flavor_counts.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aozora  # noqa: E402

GROUPS = {
    "g5": {"私": "私系", "わたくし": "私系", "我々": "私系", "僕": "僕", "俺": "俺系", "おいら": "俺系", "あっし": "俺系",
           "あたし": "女ことば", "わらわ": "女ことば", "わちき": "女ことば", "わし": "古風", "武士": "古風"},
    "g12": {c: c for c in aozora.CLASSES},
}
SPM_DIR = os.path.join(aozora.DATA_ROOT, "cache")


# ---------------------------------------------------------------------------------------------
# 1. データ: 会話文の分け方、サブワード、ラベル
# ---------------------------------------------------------------------------------------------

def split_works(d):
    """作品を訓練・調整・テストに分ける（テストは aozora.load と同じ。調整は訓練の作品の 10%）。"""
    tr = [i for i in range(len(d["works"])) if i not in d["test_works"]]
    random.Random(1).shuffle(tr)
    dev = set(tr[:len(tr) // 10])
    return set(tr) - dev, dev, set(d["test_works"])


def utter_items(d, works, k=2):
    """[(会話文, 本当の類 or None, 写した類 or None, 2 つ前の会話文 or None)]。写した類は、一人称のない会話文に、同じ列の k の倍数だけ離れた一番近いラベルつきの類。"""
    out = []
    for wi in sorted(works):
        for r in d["works"][wi]["runs"]:
            c = [aozora.cls(x["fp"]) for x in r]
            for i, x in enumerate(r):
                sp = None
                if c[i] is None and x["fp"] is None:
                    cand = [j for j in range(i % k, len(r), k) if j != i and c[j]]
                    if cand:
                        sp = c[min(cand, key=lambda j: abs(j - i))]
                out.append((x["u"], c[i], sp, r[i - k]["u"] if i >= k else None))
    return out


def train_spm(texts, V, wset="1000"):
    """訓練の会話文だけで SentencePiece を学ぶ（data/cache/ に保存、あれば読む）。"""
    import sentencepiece as spm
    prefix = os.path.join(SPM_DIR, f"spm_dialogue_{V}" + ("" if wset == "1000" else f"_{wset}"))
    if not os.path.exists(prefix + ".model"):
        os.makedirs(SPM_DIR, exist_ok=True)
        with open(prefix + ".txt", "w", encoding="utf-8") as f:
            f.write("\n".join(texts))
        spm.SentencePieceTrainer.train(input=prefix + ".txt", model_prefix=prefix, vocab_size=V, model_type="unigram",
                                       character_coverage=0.995, byte_fallback=True, bos_id=1, eos_id=2, unk_id=0, pad_id=-1,
                                       input_sentence_size=10**7, shuffle_input_sentence=True, minloglevel=2)
    return spm.SentencePieceProcessor(model_file=prefix + ".model")


# ---------------------------------------------------------------------------------------------
# 2. 2DWIG の代わり: 数えた表のナイーブベイズ（語尾 1〜3 文字 + 文字の 2 つ組の袋、SN2 実験71・72 と同じ）
# ---------------------------------------------------------------------------------------------

def feats(u):
    return ["E" + u[-k:] for k in (1, 2, 3)] + [u[i:i + 2] for i in range(len(u) - 1)]


class NB:
    def __init__(self, items, classes):
        self.classes = classes
        cnt = Counter(g for _, g in items)
        Fq = {g: Counter() for g in classes}
        for u, g in items:
            Fq[g].update(feats(u))
        tot = {g: sum(Fq[g].values()) for g in classes}
        Vf = len({f for g in classes for f in Fq[g]})
        self.prior = np.log(np.array([cnt[g] + 1e-9 for g in classes]))
        self.unk = np.log(np.array([0.5 / (tot[g] + 0.5 * Vf) for g in classes]))
        self.lp = {f: np.log(np.array([(Fq[g][f] + 0.5) / (tot[g] + 0.5 * Vf) for g in classes])) for g0 in classes for f in Fq[g0]}
        self.T = 1.0

    def post(self, u):
        """類の確率（温度 T で弱める）。"""
        s = self.prior + sum((self.lp.get(f, self.unk) for f in feats(aozora.mask(u))), np.zeros(len(self.classes)))
        s = s / self.T
        s = np.exp(s - s.max())
        return s / s.sum()


# ---------------------------------------------------------------------------------------------
# 3. 後退つき 3-gram（Witten-Bell。重みつきの数え方もそのまま使える）
# ---------------------------------------------------------------------------------------------

class Tri:
    def __init__(self, V):
        self.V = V
        self.c1 = np.zeros(V)
        self.c2 = defaultdict(Counter)
        self.c3 = defaultdict(Counter)

    def add(self, seq, w=1.0):
        for i in range(2, len(seq)):
            a, b, y = seq[i - 2], seq[i - 1], seq[i]
            self.c1[y] += w
            self.c2[b][y] += w
            self.c3[(a, b)][y] += w

    def finish(self):
        """数えた表を、引きやすい形（番号と数の配列）にする。"""
        N, T1 = self.c1.sum(), (self.c1 > 0).sum()
        self.p1 = (self.c1 + T1 / self.V) / (N + T1)
        self.t2 = {h: (np.fromiter(c.keys(), int), np.fromiter(c.values(), float)) for h, c in self.c2.items()}
        self.t3 = {h: (np.fromiter(c.keys(), int), np.fromiter(c.values(), float)) for h, c in self.c3.items()}
        del self.c2, self.c3

    def prob(self, a, b):
        p = self.p1
        for t, h in ((self.t2, b), (self.t3, (a, b))):
            e = t.get(h)
            if e is None:
                continue
            idx, cnt = e
            tot, T = cnt.sum(), len(cnt)
            p = p * (T / (tot + T))
            p[idx] += cnt / (tot + T)
        return p


# ---------------------------------------------------------------------------------------------
# 4. 評価
# ---------------------------------------------------------------------------------------------

def evaluate(items, sp, P_all, P_s, groups, cue, lam, nb, gmap, subset=None):
    """items の会話文を全部の位置で評価。cue: none / oracle / spread / prev / prefix。
    subset（省くと cue と同じ）: 手がかりのある会話文だけに絞る（oracle = 一人称つき、spread = 写した類つき、prev = 2 つ前がある）。"""
    subset = subset or cue
    n = hit = 0
    ce = 0.0
    chars = 0
    G = len(groups)
    gi = {g: i for i, g in enumerate(groups)}
    to_g = np.zeros((len(nb.classes), G))
    for ci, c in enumerate(nb.classes):
        if c in gmap:
            to_g[ci, gi[gmap[c]]] = 1
    for u, true_c, spread_c, prev in items:
        if subset == "oracle" and true_c is None or subset == "spread" and spread_c is None or subset == "prev" and prev is None:
            continue
        ids = sp.encode(u)
        seq = [1, 1] + ids + [2]
        w = None
        if cue == "oracle":
            w = np.eye(G)[gi[gmap[true_c]]]
        elif cue == "spread":
            w = np.eye(G)[gi[gmap[spread_c]]]
        elif cue == "prev":
            w = nb.post(prev) @ to_g
        pieces = [sp.id_to_piece(i) for i in ids]
        prefix = ""
        for t in range(2, len(seq)):
            a, b, y = seq[t - 2], seq[t - 1], seq[t]
            p = P_all.prob(a, b)
            if cue == "prefix" and prefix:
                w = nb.post(prefix) @ to_g
            if cue != "none" and w is not None:
                pm = sum(w[j] * P_s[g].prob(a, b) for j, g in enumerate(groups) if w[j] > 1e-3)
                p = lam * pm / max(w[w > 1e-3].sum(), 1e-9) + (1 - lam) * p
            hit += int(p.argmax() == y)
            ce -= math.log(max(p[y], 1e-12))
            n += 1
            if t - 2 < len(pieces):
                prefix += pieces[t - 2].replace("▁", "")
        chars += len(u) + 1
    return dict(n_utt_tokens=n, acc=100.0 * hit / max(n, 1), ce=ce / max(n, 1), ce_per_char=ce / max(chars, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--V", type=int, default=4096, help="語彙。2048 は作れない（文字の網羅率 0.995 に 2,388 字が要る）")
    ap.add_argument("--groups", default="g5,g12")
    ap.add_argument("--max_test", type=int, default=0, help="テストの会話文の数の上限（0 = 全部）")
    ap.add_argument("--set", default="1000", choices=["1000", "all"], help="青空文庫の作品の組（1000 = 最初の 1,000 作品、all = 全 4,477 作品）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="使わない（約束の引数）")
    ap.add_argument("--out", default="results/est_flavor_counts.txt")
    a = ap.parse_args()
    t0 = time.perf_counter()
    d = aozora.load(a.set)
    tr_w, dev_w, te_w = split_works(d)
    tr, dev, te = utter_items(d, tr_w), utter_items(d, dev_w), utter_items(d, te_w)
    if a.max_test:
        te = random.Random(a.seed).sample(te, min(a.max_test, len(te)))
    sp = train_spm([u for u, *_ in tr], a.V, a.set)
    print(f"会話文: 訓練 {len(tr)}・調整 {len(dev)}・テスト {len(te)}、語彙 {sp.get_piece_size()}（{time.perf_counter() - t0:.0f} 秒）", flush=True)
    classes = aozora.CLASSES
    lab = [(u, c or s) for u, c, s, _ in tr if (c or s)]
    nb = NB(lab, classes)
    # 温度: 調整用の一人称つき会話文での CE が一番小さいもの
    devlab = [(u, c) for u, c, _, _ in dev if c]
    best = None
    for T in (1, 2, 4, 8, 16):
        nb.T = T
        ce = -np.mean([math.log(nb.post(u)[classes.index(c)] + 1e-12) for u, c in devlab])
        best = min(best or (ce, T), (ce, T))
    nb.T = best[1]
    print(f"2DWIG（数えた表）の温度 {nb.T}（調整用の CE {best[0]:.3f}）", flush=True)
    enc = {u: [1, 1] + sp.encode(u) + [2] for u, *_ in tr}
    P_all = Tri(sp.get_piece_size())
    for s in enc.values():
        P_all.add(s)
    P_all.finish()
    base = dict(exp="est_flavor_counts", set=a.set, V=a.V, seed=a.seed, out=a.out, n_train=len(tr), n_dev=len(dev), n_test=len(te), nb_T=nb.T)
    none = {}
    for subset in ("all", "oracle", "spread", "prev"):
        none[subset] = evaluate(te, sp, P_all, {}, [], "none", 0, nb, {}, subset=subset)
        rec = dict(base, groups=None, count=None, cue="none", subset=subset, lam=0, **none[subset], time_s=time.perf_counter() - t0)
        print(json.dumps(rec, ensure_ascii=False), flush=True)
        open(a.out, "a", encoding="utf-8").write(json.dumps(rec, ensure_ascii=False) + "\n")
    for gname in a.groups.split(","):
        gmap = GROUPS[gname]
        groups = sorted(set(gmap.values()))
        for count in ("hard", "soft"):
            P_s = {g: Tri(sp.get_piece_size()) for g in groups}
            for u, c, s, _ in tr:
                if count == "hard":
                    if c or s:
                        P_s[gmap[c or s]].add(enc[u])
                else:
                    w = nb.post(u) if not c else np.eye(len(classes))[classes.index(c)]
                    for ci, wc in enumerate(w):
                        if wc > 0.01:
                            P_s[gmap[classes[ci]]].add(enc[u], float(wc))
            for g in groups:
                P_s[g].finish()
            for cue in ("oracle", "spread", "prev", "prefix"):
                # 混ぜる強さ λ は調整用の会話文で選ぶ（prefix は計算が重いので prev と同じ λ を使う）
                if cue != "prefix":
                    dv = random.Random(0).sample(dev, min(3000, len(dev)))
                    lam = max((0.2, 0.4, 0.6, 0.8), key=lambda l: -evaluate(dv, sp, P_all, P_s, groups, cue, l, nb, gmap)["ce"])
                    if cue == "prev":
                        lam_prev = lam
                else:
                    lam = lam_prev
                subset = "all" if cue == "prefix" else cue
                r = evaluate(te, sp, P_all, P_s, groups, cue, lam, nb, gmap, subset=subset)
                rec = dict(base, groups=gname, count=count, cue=cue, subset=subset, lam=lam, **r,
                           gain_acc=r["acc"] - none[subset]["acc"], gain_ce=r["ce"] - none[subset]["ce"], time_s=time.perf_counter() - t0)
                print(json.dumps(rec, ensure_ascii=False), flush=True)
                open(a.out, "a", encoding="utf-8").write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
