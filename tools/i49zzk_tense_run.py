"""I49zzk: 時制の波を学習から作る形を、作り物の世界で数える（持ち主の案、未確定の試し）。

世界は I49zzj と同じ（時の語 8、だれが 6、動詞 6、世界の決まりも同じ）。形は、動詞の後のつなぎの語の並びで表す。
  未来: だろう / 現在: 無し / 現在進行: て・いる / 過去: た / 過去進行: て・い・た
数える物:
  1. 語ごとの、時制の影響を受ける強さ（平均を引く時と、引かない時）。だれが・動詞は弱く、つなぎの語は強いか。
  2. 出来た形の波どうしの近さ（手で並べていないのに、なだらかな並びが出るか）。
  3. 形を選ぶ: 聞いた文の数ごとに、時の語から選んだ形が世界の決まりに合う割合。学習に出た動詞と、出ていない動詞で分ける。
     つなぎの語で持つ形（Learned）と、動詞ごとにひとまとめで持つ形（Whole）を比べる。
手で置いた物: 世界の決まり、つなぎの語への分け方（作り物）、学習と試しの動詞の分け方。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "intervention")]

import i49zzj_tense_run as ZJ
import i49zzk_tense_learned as TL

AUX = dict(未来=("だろう",), 現在=(), 現在進行=("て", "いる"), 過去=("た",), 過去進行=("て", "い", "た"))
TENSES = tuple(AUX)
MARKS = (8, 16, 32, 64, 128, 256, 512, 768)


def sentences(verbs):
    out = []
    for word, rule in ZJ.RULE.items():
        for tense, w in rule.items():
            for _ in range(w):
                for who in ZJ.WHO:
                    for v in verbs:
                        out.append((word, who, v, AUX[tense]))
    return ZJ.golden(out)


def hits(m, verbs):
    ok = n = 0
    for word, rule in ZJ.RULE.items():
        s = m.state_of(word)
        for v in verbs:
            n += 1
            ok += int(s is not None and m.pick(v, AUX, s) in rule)
    return round(ok / n, 4)


def cosm(vs):
    out = []
    for a in vs:
        row = []
        for b in vs:
            na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
            row.append(round(float(a @ b) / (na * nb), 3) if na > 1e-6 and nb > 1e-6 else 0.0)
        out.append(row)
    return out


def run():
    seq = sentences(ZJ.TRAIN)
    models = {"つなぎの語で持つ": TL.Learned(), "動詞ごとにひとまとめ": TL.Whole(), "つなぎの語で持つ・平均を引かない": TL.Learned(center=False)}
    curve = {k: [] for k in models}
    for i, s in enumerate(seq, 1):
        for m in models.values():
            m.learn(*s)
        if i in MARKS:
            for k, m in models.items():
                curve[k].append(dict(文=i, 学習に出た動詞=hits(m, ZJ.TRAIN), 出ていない動詞=hits(m, ZJ.TEST)))
    m, raw = models["つなぎの語で持つ"], models["つなぎの語で持つ・平均を引かない"]
    units = [("だれが", w) for w in ZJ.WHO] + [("動詞", v) for v in ZJ.TRAIN] + [("つなぎ", a) for a in ("だろう", "て", "いる", "い", "た")]
    strength = {"%s:%s" % u: dict(平均を引く=round(m.strength(u), 4), 引かない=round(raw.strength(u), 4)) for u in units}
    who = [raw.wave(("だれが", w)) for w in ZJ.WHO[:2]] + [raw.wave(("つなぎ", "た"))]
    forms = [m.form_wave(ZJ.TRAIN[0], AUX[t]) for t in TENSES]
    table = {}
    for word, rule in ZJ.RULE.items():
        table[word] = dict(世界の決まり=rule, 選んだ形=m.pick(ZJ.TEST[0], AUX, m.state_of(word)))
    return dict(学習の文=len(seq), 当てずっぽう=round(sum(len(r) for r in ZJ.RULE.values()) / (len(ZJ.RULE) * len(TENSES)), 4),
                時制の影響を受ける強さ=strength,
                平均を引かない時の近さ=dict(だれがどうし=cosm(who)[0][1], だれがとつなぎの語=cosm(who)[0][2]),
                形の波どうしの近さ=dict(時制=list(TENSES), 表=cosm(forms), 波の長さ=[round(float(np.linalg.norm(f)), 4) for f in forms]),
                形を選ぶ=curve, 時の語ごとに選んだ形=table)


def run_spans():
    """分け方を決めず、動詞の形の切り出しを全部数える形（Spans）を、つなぎの語に分けて持つ形（Learned）と比べる。
    試すのは学習に出ていない動詞だけ。「決まりの内」と「決まりの中で一番多い時制」の 2 通りで数える。"""
    stems = set("".join(ZJ.TRAIN + ZJ.TEST)) - set("るくう")
    sp, tk = TL.Spans(), TL.Learned()
    seq_sp = []
    for word, rule in ZJ.RULE.items():
        for tense, w in rule.items():
            for _ in range(w):
                for who in ZJ.WHO:
                    for v in ZJ.TRAIN:
                        seq_sp.append((word, who, v, tense))
    seq_sp = ZJ.golden(seq_sp)

    def score(m, forms_of):
        ok = top = n = 0
        for word, rule in ZJ.RULE.items():
            s = m.state_of(word)
            for v in ZJ.TEST:
                n += 1
                if s is None:
                    continue
                p = m.pick(v, forms_of(v), s)
                ok += int(p in rule)
                top += int(p == max(rule, key=rule.get))
        return dict(決まりの内=round(ok / n, 4), 一番多い時制=round(top / n, 4))

    curve = {"切り出しを全部持つ": [], "つなぎの語に分けて持つ": []}
    for i, (word, who, v, tense) in enumerate(seq_sp, 1):
        sp.learn(word, who, v, ZJ.FORMS[v][tense])
        tk.learn(word, who, v, AUX[tense])
        if i in MARKS:
            curve["切り出しを全部持つ"].append(dict(文=i, **score(sp, lambda v: ZJ.FORMS[v])))
            curve["つなぎの語に分けて持つ"].append(dict(文=i, **score(tk, lambda v: AUX)))
    units = [u for u in sp.count if u[0] == "切り出し"]
    strength = sorted(((round(sp.strength(u), 4), u[1], sum(sp.count[u].values()), bool(set(u[1]) & stems)) for u in units), reverse=True)
    table = {w: {v: sp.pick(v, ZJ.FORMS[v], sp.state_of(w)) for v in ZJ.TEST} for w in ZJ.RULE}
    return dict(学習の文=len(seq_sp), 切り出しの種類=len(units), 形の種類=len(ZJ.TRAIN) * len(TENSES),
                切り出しごとの強さ=[dict(切り出し=s, 強さ=x, 回数=c, 動詞の字を含む=k) for x, s, c, k in strength],
                形を選ぶ=curve, 時の語ごとに選んだ形=table)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--spans", action="store_true")
    a = ap.parse_args()
    res = run_spans() if a.spans else run()
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
