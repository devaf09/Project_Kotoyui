"""見積もり E6: 現代の雑談で、話者の見分け・隠し味の効き・口調の当たりを測る（数えた表、GPU なし、2026-10-26）

着想: 持ち主の依頼（2026-10-26）「現代の雑談でも見積もる。隠し味は成果が出なさそうなら使わなくてよい」。
      現代の雑談（RealPersonaChat・名大会話コーパス）は話者が分かるが、一人称はほぼ「私」（NOTES.md「データ」）。
      そこで「誰か」の答えを、一人称だけでなく、性別・年代・丁寧さ（です・ます）にも広げて測る。
データ: experiments/dialogue.py と同じ 2 つ（RealPersonaChat 1.0.0 は CC BY-SA 4.0、名大会話コーパスは CC BY-NC-ND 4.0）。
        話者の属性: RealPersonaChat は interlocutors.json の性別・年代、名大会話コーパスは各会話の「＠参加者」の行の性別・年代。
        名大会話コーパスの話者の番号（F128 など）は会話をまたいで同じ人なので、話者は「データ + 番号」で区別する。
分け方: **知らない人で測る**。話者の 12% をテストの話者にし、テストの話者が 1 人でも入る対話をテストにする（訓練に同じ人の発言が入らない）。
        残りの対話の 10% を調整用（混ぜる強さ λ を選ぶ）。
答え（話者の類）:
  gender = 性別（女性・男性。その他・不明は除く）
  age3   = 年代 3 つ（〜20 代・30〜40 代・50 代〜）
  fp4    = 話者が一番多く使った一人称の類（私・あたし・俺・僕。一人称を使わない話者は除く）
  polite = 発言ごとの丁寧さ（文末が です・ます など → 丁寧、それ以外 → くだけた）。話者ではなく発言の性質
方法（どれも est_flavor_counts.py・est_flavor_style.py の部品を使う）:
  A 話者の見分け: 数えた表のナイーブベイズ（語尾 1〜3 + 文字の 2 つ組・3 つ組の袋）。発言 1 つと、同じ話者の発言 5・20 個をまとめた袋で。
  B 隠し味の効き: 単位は SentencePiece（語彙 4096、訓練の発言だけで学ぶ）。後退つき 3-gram、類ごとの表は本当の類で数える（話者が分かるので）。
    手がかり oracle = 本当の類（ことゆいが自分の口調を決めて話す場合）、prev = 同じ話者の同じ対話の前の発言（3 つまで）に A の表を当てた確率。
  C 口調の当たり: 類ごとの表で文を作り（上位 20 個から選ぶ、40 単位まで、各類 100 文 × 3 seed）、A の表で類を当て直す。
指標: A 正解率・類ごとの正解率の平均・いつも一番多い類。B 次の単位の正解率・CE の差（判断の基準は E1〜E4 と同じ +0.5 点）。C 当て直しの割合と、本物の発言での当て直し。
結果: results/est_modern.txt（1 条件 = 1 行の JSON）、例 ../Project_Kotoyui_private/docs/samples/m1_est_modern_style.md（決めた類で作った文。データの文そのものではない）
実行: python experiments/est_modern.py
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aozora  # noqa: E402
import dialogue  # noqa: E402
import est_flavor_counts as efc  # noqa: E402
import est_flavor_style as efs  # noqa: E402

LABELS = ["polite", "gender", "age3", "fp4"]
_POLITE = re.compile(r"(です|ます|でした|ました|ません|でしょう|ましょう|ください|ございます|ませ|まして)[かねよなー〜！？!?。…、♪w笑\s]*$")
_FW = str.maketrans("０１２３４５６７８９", "0123456789")


def feats(u):
    """話者の見分けの袋: 語尾 1〜3 文字 + 文字の 2 つ組 + 3 つ組。"""
    return ["E" + u[-k:] for k in (1, 2, 3)] + [u[i:i + 2] for i in range(len(u) - 1)] + [u[i:i + 3] for i in range(len(u) - 2)]


class NB(efc.NB):
    """est_flavor_counts の NB と同じ（袋だけ 3 つ組まで）。"""

    def __init__(self, items, classes):
        old = efc.feats
        efc.feats = feats
        try:
            super().__init__(items, classes)
        finally:
            efc.feats = old

    def post(self, u):
        s = self.prior + sum((self.lp.get(f, self.unk) for f in feats(aozora.mask(u))), np.zeros(len(self.classes)))
        s = s / self.T
        s = np.exp(s - s.max())
        return s / s.sum()


def age3(a):
    if a is None:
        return None
    return "〜20代" if a < 30 else ("30〜40代" if a < 50 else "50代〜")


def attributes():
    """話者（データ, 番号）→ {"gender", "age"}。"""
    out = {}
    I = json.load(open(os.path.join(dialogue.RPC, "interlocutors.json"), encoding="utf-8"))
    for k, v in I.items():
        d = v["demographic_information"]
        g = {"Female": "女性", "Male": "男性"}.get(d["gender"])
        m = re.match(r"-?(\d+)", d["age"])
        a = int(m.group(1)) if m else None
        if d["age"].startswith("-"):
            a = 15
        out[("rpc", k)] = dict(gender=g, age=a)
    for fn in glob.glob(os.path.join(dialogue.NUCC, "data*.txt")):
        for line in open(fn, encoding="utf-8"):
            m = re.match(r"^＠参加者([A-Z]+\d*)：(女性|男性)?([０-９0-9]+)?", line)
            if m:
                a = int(m.group(3).translate(_FW)) if m.group(3) else None
                out[("nucc", m.group(1))] = dict(gender=m.group(2), age=a)
    return out


def load():
    """発言に、話者（全体で同じ人は同じ番号）と 4 つの答えを付ける。"""
    us = dialogue.load()
    for u in us:
        u["spk"] = (u["src"], u["speaker"].split("-", 1)[1] if u["src"] == "nucc" else u["speaker"])
    att = attributes()
    fpc = defaultdict(Counter)
    for u in us:
        g = aozora.cls(u["fp"])
        if g in ("私", "あたし", "俺", "僕"):
            fpc[u["spk"]][g] += 1
    for u in us:
        a = att.get(u["spk"], {})
        u["gender"] = a.get("gender")
        u["age3"] = age3(a.get("age"))
        u["fp4"] = fpc[u["spk"]].most_common(1)[0][0] if fpc[u["spk"]] else None
        u["polite"] = "丁寧" if _POLITE.search(u["text"]) else "くだけた"
    return us


def split(us, seed=0):
    """知らない人で測る分け方。戻り値: 訓練・調整・テストの対話の集合 (src, dialog)。"""
    spk = sorted({u["spk"] for u in us})
    rng = random.Random(seed)
    test_spk = set(rng.sample(spk, len(spk) * 12 // 100))
    dia = defaultdict(set)
    for u in us:
        dia[(u["src"], u["dialog"])].add(u["spk"])
    te = {d for d, s in dia.items() if s & test_spk}
    rest = sorted(set(dia) - te)
    rng.shuffle(rest)
    dev = set(rest[:len(rest) // 10])
    return set(rest) - dev, dev, te, test_spk


def items_of(us, dset, label):
    """[(発言, 本当の類, None, 同じ話者の前の発言 3 つまで)]。est_flavor_counts.evaluate の形。"""
    out = []
    prev = defaultdict(list)
    for u in us:
        key = (u["src"], u["dialog"])
        if key not in dset:
            continue
        p = prev[(key, u["spk"])]
        out.append((u["text"], u[label], None, "／".join(p[-3:]) if p else None))
        p.append(u["text"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--V", type=int, default=4096)
    ap.add_argument("--labels", default=",".join(LABELS))
    ap.add_argument("--max_test", type=int, default=5000)
    ap.add_argument("--n_style", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--device", default="cpu", help="使わない（約束の引数）")
    ap.add_argument("--out", default="results/est_modern.txt")
    ap.add_argument("--samples", default="../Project_Kotoyui_private/docs/samples/m1_est_modern_style.md")
    a = ap.parse_args()
    t0 = time.perf_counter()
    us = load()
    tr_d, dev_d, te_d, test_spk = split(us, a.seed)
    tr_u = [u for u in us if (u["src"], u["dialog"]) in tr_d]
    te_u = [u for u in us if (u["src"], u["dialog"]) in te_d and u["spk"] in test_spk]
    print(f"発言: 訓練 {len(tr_u)}・テストの話者の発言 {len(te_u)}（テストの話者 {len(test_spk)} 人）", flush=True)
    sp = efc.train_spm([u["text"] for u in tr_u], a.V, "modern")
    base = dict(exp="est_modern", V=a.V, seed=a.seed, out=a.out, n_train=len(tr_u), n_test_spk=len(test_spk))

    def write(rec):
        rec = dict(base, **rec, time_s=time.perf_counter() - t0)
        print(json.dumps({k: v for k, v in rec.items() if k not in ("per_class",)}, ensure_ascii=False), flush=True)
        with open(a.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    enc = {}
    for u in tr_u:
        if u["text"] not in enc:
            enc[u["text"]] = [1, 1] + sp.encode(u["text"]) + [2]
    P_all = efc.Tri(sp.get_piece_size())
    for u in tr_u:
        P_all.add(enc[u["text"]])
    P_all.finish()
    samples = {}
    for label in a.labels.split(","):
        trl = [(u["text"], u[label]) for u in tr_u if u[label]]
        classes = [c for c, _ in Counter(c for _, c in trl).most_common()]
        nb = NB(trl, classes)
        # A: 話者の見分け（テストの話者の発言。発言 1 つと、同じ話者の K 個をまとめた袋）
        tel = [(u["text"], u[label], u["spk"]) for u in te_u if u[label]]
        majority = 100.0 * Counter(c for _, c, _ in tel).most_common(1)[0][1] / len(tel)
        for K in (1, 5, 20):
            if label == "polite" and K > 1:
                continue
            by = defaultdict(list)
            for t, c, s in tel:
                by[(s, c)].append(t)
            groups = []
            for (s, c), ts in by.items():
                for i in range(0, len(ts) - K + 1, K):
                    groups.append(("／".join(ts[i:i + K]), c))
            random.Random(a.seed).shuffle(groups)
            groups = groups[:20000]
            pred = [classes[int(np.argmax(nb.post(t)))] for t, _ in groups]
            gold = [c for _, c in groups]
            acc = 100.0 * sum(p == g for p, g in zip(pred, gold)) / len(gold)
            rec_c = {c: 100.0 * sum(p == g for p, g in zip(pred, gold) if g == c) / max(1, sum(g == c for g in gold)) for c in classes}
            maj_k = 100.0 * Counter(gold).most_common(1)[0][1] / len(gold)
            write(dict(part="A", label=label, K=K, classes=classes, n=len(groups), acc=acc, macro=sum(rec_c.values()) / len(rec_c),
                       majority=maj_k, per_class=rec_c))
        # 温度: 調整用の発言での CE が一番小さいもの
        dvl = [(u["text"], u[label]) for u in us if (u["src"], u["dialog"]) in dev_d and u[label]]
        dvl = random.Random(0).sample(dvl, min(5000, len(dvl)))
        nb.T = min((1, 2, 4, 8, 16), key=lambda T: -np.mean([math.log(NB.post(_with_T(nb, T), t)[classes.index(c)] + 1e-12) for t, c in dvl]))
        # B: 隠し味の効き
        P_s = {c: efc.Tri(sp.get_piece_size()) for c in classes}
        for u in tr_u:
            if u[label]:
                P_s[u[label]].add(enc[u["text"]])
        for t in P_s.values():
            t.finish()
        gmap = {c: c for c in classes}
        te_items = [x for x in items_of(us, te_d, label) if x[1]]
        te_items = [x for x in te_items if True]
        te_items = random.Random(a.seed).sample(te_items, min(a.max_test, len(te_items)))
        dv_items = random.Random(0).sample([x for x in items_of(us, dev_d, label) if x[1]], 3000)
        for cue in ("oracle", "prev"):
            lam = max((0.2, 0.4, 0.6, 0.8, 1.0), key=lambda l: -efc.evaluate(dv_items, sp, P_all, P_s, classes, cue, l, nb, gmap)["ce"])
            r0 = efc.evaluate(te_items, sp, P_all, {}, [], "none", 0, nb, {}, subset=cue)
            r = efc.evaluate(te_items, sp, P_all, P_s, classes, cue, lam, nb, gmap, subset=cue)
            write(dict(part="B", label=label, cue=cue, lam=lam, nb_T=nb.T, classes=classes, none_acc=r0["acc"], none_ce=r0["ce"],
                       acc=r["acc"], ce=r["ce"], gain_acc=r["acc"] - r0["acc"], gain_ce=r["ce"] - r0["ce"], n_tokens=r["n_utt_tokens"]))
        # C: 口調の当たり
        judge = NB(trl, classes)
        judge_real = 100.0 * float(np.mean([classes[int(np.argmax(judge.post(t)))] == c for t, c, _ in tel]))
        for cond, lam in (("none", 0.0), ("mix1.0", 1.0)):
            accs = []
            per = defaultdict(list)
            for s in range(a.seed, a.seed + a.seeds):
                rng = np.random.default_rng(s)
                hit = 0
                for c in classes:
                    hc = 0
                    for i in range(a.n_style):
                        text, _ = efs.generate(sp, P_all, None if cond == "none" else P_s[c], lam, rng)
                        pc = classes[int(np.argmax(judge.post(text)))]
                        hc += pc == c
                        if s == a.seed and i < 6:
                            samples.setdefault((label, cond, c), []).append((text, pc))
                    per[c].append(100.0 * hc / a.n_style)
                    hit += hc
                accs.append(100.0 * hit / (a.n_style * len(classes)))
            write(dict(part="C", label=label, cond=cond, classes=classes, style_acc=float(np.mean(accs)),
                       style_acc_se=float(np.std(accs, ddof=1) / np.sqrt(len(accs))), chance=100.0 / len(classes),
                       judge_real_acc=judge_real, per_class={c: float(np.mean(v)) for c, v in per.items()}))
    lines = ["# 現代の雑談で、類を決めて作った文の例（数えた表、est_modern.py）", "",
             "RealPersonaChat・名大会話コーパスの訓練の発言で数えた 3-gram（サブワード 4096）から作った文。データの文そのものではない。",
             f"seed {a.seed}、類ごとに先頭の 6 文（選び直していない）。「当て直し」は数えた表が当てた類。", ""]
    for label in a.labels.split(","):
        lines += [f"## {label}", ""]
        for cond in ("none", "mix1.0"):
            for (lb, cd, c), xs in samples.items():
                if lb != label or cd != cond:
                    continue
                lines += [f"### {cond} — 決めた類: {c}", ""]
                lines += [f"- 「{t}」 — 当て直し: {pc}{' ✓' if pc == c else ''}" for t, pc in xs]
                lines.append("")
    with open(a.samples, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _with_T(nb, T):
    nb.T = T
    return nb


if __name__ == "__main__":
    main()
