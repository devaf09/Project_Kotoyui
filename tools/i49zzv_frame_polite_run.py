"""I49zzv: 打ち消し・過去などは意味の印で先に決め、状況波には「丁寧を入れるか」だけを選ばせる（持ち主の案 1 と 3-3 の小さい試し。未確定）。

持ち主の案:「応答の3つ組（＋否定・過去などの意味印）を I42（融合の板）で「終止形の骨組み（例：雨が降る＋否定なし）」まで組み、
  最後の「終助詞・丁寧度・問いかけ」の選択重みにだけ SituationWave.rank() のスコアを足し合わせる形ににしてみる。」
流れ（解析済みの会話。作品の 1 割を外して確かめる）:
  1. 外した会話の行のうち、行の終わりが述語（語幹 ＋ 後ろの流れ）で終わる 1 文の行を取る。前の行が有る物だけ。
  2. 述語の意味の組（I42 の meaning: 過去・進行・否定・丁寧 …）から、「丁寧」を抜いた組と、足した組の 2 つを作る。他の印は触らない。
  3. 2 つの組を、I42 の逆引き（表 → 無ければ融合の板の探し）と、形を戻す部品で、行の終わりの形に組む。
  4. 状況波（前の行の終わり → 今の行の終わり。i49_situation.Assoc）で、2 つの行のうち合う方を選ぶ。
     前の行は 2 通り: 1 個前の行（相手の行）と、同じ話し手の 1 個前の行（自分の口調の続き）。話し手は「同じかどうか」だけを見る。
見る物:
  組めた割合／元と同じ意味の組から組んだ形が、元の行と同じか／打ち消しが入れ替わった割合（意味の印で見た時と、文字の印で見た時）／
  状況波が選んだ方が、本物の行の丁寧の有り無しと合う割合（比べ役: いつも多い方、前の行の丁寧の印に合わせる）／
  本物と違う方の形（作り替えた行）が、違和感チェックを通る割合。
手で置いた物: 外す割合、打ち消しの文字の印（読むためだけ）、I42 の中の決まり（意味の表、時の副詞の並び）。乱数は使わない。
`--out` には行の例が出るので、リポジトリには入れない。
引数: --cache 解析済み JSON --out 記録.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "intervention"), os.path.join(HERE, "..", "intervention", "past")]

import i42_fused_flow as F  # noqa: E402
import i49_situation as SIT  # noqa: E402
import i49zzr_air as AIR  # noqa: E402

HOLD = 0.1
NEG = ("ない", "ません", "なかった", "ぬ", "ず")   # 打ち消しの文字の印（読むためだけ）
EXAMPLES = 30


def clean(text):
    return text.strip().strip("「」『』")


def last_pred(line, prev_end):
    """行の終わりの述語。→ (文脈の記号, 語幹, 流れ, 語幹の表記, 流れの表記, 頭) か None。行が「語幹 ＋ 流れ」でちょうど終わる時だけ。"""
    toks = line["toks"]
    text = clean(line["text"])
    fls = F.flows(toks)
    if not fls or len(list(SIT.sentences([text]))) != 1:
        return None
    i, fl, fs, nxt = fls[-1]
    t = toks[i]
    tailform = t["text"] + "".join(fs)
    body = text.rstrip("。！？!?…")
    if not body.endswith(tailform) or any(k["pos"] != "PUNCT" for k in toks[nxt:]):
        return None
    sy = F.context(toks, t, nxt, prev_end, line["text"].startswith(("「", "『")))
    return sy, t["lemma"], fl, t["text"], fs, body[:len(body) - len(tailform)]


def build(rev, surf, fsurf, sy, lem, m):
    pos = F._pos(sy)
    fl = rev.table(lem, m, pos, "台詞:はい" in sy) if m in rev.tab else rev.search(sy, m)
    if fl is None:
        return None, None
    return fl, surf.form(lem, fl, pos) + fsurf.form(fl, sy)


def has_neg(s):
    return any(x in s[-8:] for x in NEG)


def ends(text):
    """行の (初めの文の終わり, 最後の文の終わり)。文が無ければ None。"""
    ss = list(SIT.sentences([text]))
    return (SIT.tail_of(ss[0]), SIT.tail_of(ss[-1])) if ss else None


def walk(work):
    """行ごとに (行, 行の文, 1 個前の行の文, 同じ話し手の 1 個前の行の文)。話し手は「同じかどうか」だけを見る。"""
    prev, own = None, {}
    for line in work["lines"]:
        text = clean(line["text"])
        yield line, text, prev, own.get(line.get("speaker"))
        if text:
            prev = text
            own[line.get("speaker")] = text


def situation_pairs(works, who):
    """状況波の組 (前の行の最後の文の終わり, 今の行の初めの文の終わり)。who = 0 は 1 個前の行、1 は同じ話し手の 1 個前の行。同じ組は 1 回。"""
    out = {}
    for w in works:
        for _, text, prev, own in walk(w):
            p, c = (prev, own)[who], ends(text)
            if p and c and ends(p):
                out.setdefault((ends(p)[1], c[0]), None)
    return list(out)


WHO = ("1 個前の行（相手）", "同じ話し手の 1 個前の行")


def run(works, hold=None):
    order = SIT.golden_order(len(works))
    k = max(int(len(works) * (HOLD if hold is None else hold)), 1)
    held, kept = [works[i] for i in order[:k]], [works[i] for i in order[k:]]
    its = F.items(kept)
    fused, surf, fsurf = F.Fused(its), F.Surface(its), F.FlowSurface(its)
    rev = F.Reverse(its, fused)
    lines = []
    for w in kept:
        lines += [clean(l["text"]) for l in w["lines"]] + [""]
    assoc = [SIT.Assoc(situation_pairs(kept, who)) for who in (0, 1)]
    chk = AIR.OddCheck(SIT.distinct(list(SIT.sentences(lines))))
    c = dict(行=0, 組めた=0, 元と同じ形=0, 丁寧の行=0, 二つの形が違う=0, 意味の印で打ち消しが入れ替わった=0, 文字の印で打ち消しが入れ替わった=0,
             打ち消しの行=0, 作り替えた行=0, 作り替えた行が違和感チェックを通る=0, 本物の行が違和感チェックを通る=0)
    pick_c = {n: dict(問い=0, 状況波が当たり=0, 前の行の丁寧の印に合わせて当たり=0, いつも多い方=0) for n in WHO}
    ex = []
    for w in held:
        prev_end = None
        for line, text, prev_text, own_text in walk(w):
            got = last_pred(line, prev_end) if prev_text else None
            root = [x for x in F.flows(line["toks"]) if line["toks"][x[0]]["dep"] == "ROOT"] if line["toks"] else []
            if got:
                sy, lem, fl, s0, fs, head = got
                m = F.meaning(fl, fs, F._pos(sy))
                base = tuple(t for t in m if t != "丁寧")
                forms = {}
                for name, mm in (("丁寧なし", base), ("丁寧あり", tuple(sorted(base + ("丁寧",))))):
                    f2, form = build(rev, surf, fsurf, sy, lem, mm)
                    if form is not None:
                        forms[name] = (f2, form, mm)
                c["行"] += 1
                if len(forms) == 2:
                    c["組めた"] += 1
                    real = "丁寧あり" if "丁寧" in m else "丁寧なし"
                    other = "丁寧なし" if real == "丁寧あり" else "丁寧あり"
                    c["丁寧の行"] += real == "丁寧あり"
                    c["元と同じ形"] += forms[real][1] == s0 + "".join(fs)
                    neg = "否定" in m
                    c["打ち消しの行"] += neg
                    for name in forms:
                        c["意味の印で打ち消しが入れ替わった"] += ("否定" in F.meaning(forms[name][0])) != neg
                    c["文字の印で打ち消しが入れ替わった"] += has_neg(head + forms[other][1]) != has_neg(head + s0 + "".join(fs))
                    if forms["丁寧あり"][1] != forms["丁寧なし"][1]:
                        c["二つの形が違う"] += 1
                        pick = None
                        for who, name in enumerate(WHO):
                            src = (prev_text, own_text)[who]
                            if not src or not ends(src):
                                continue
                            ptail = ends(src)[1]
                            sc = {n: assoc[who].score(ptail, SIT.tail_of(head + forms[n][1])) for n in forms}
                            got_pick = max(sorted(sc), key=lambda n: sc[n])
                            pick = got_pick if who == 0 else pick
                            pc = pick_c[name]
                            pc["問い"] += 1
                            pc["状況波が当たり"] += got_pick == real
                            pc["前の行の丁寧の印に合わせて当たり"] += ("丁寧あり" if SIT.marks(ptail)["丁寧"] else "丁寧なし") == real
                            pc["いつも多い方"] += real == "丁寧あり"
                        new = head + forms[other][1]
                        c["作り替えた行"] += 1
                        c["作り替えた行が違和感チェックを通る"] += chk.passes(new, len(forms[other][1]))
                        c["本物の行が違和感チェックを通る"] += chk.passes(head + s0 + "".join(fs), len(s0 + "".join(fs)))
                        if len(ex) < EXAMPLES and c["二つの形が違う"] % 7 == 1:
                            ex.append(dict(前の行=prev_text, 元=text, 意味="・".join(m), 作り替え=new, 状況波が選んだ方=pick, 本物=real))
            if root:
                prev_end = root[0][1][0] if root[0][1] else "言い切り"
    r = lambda a, b: round(c[a] / max(c[b], 1), 4)
    rate = dict(組めた割合=r("組めた", "行"), 元と同じ形の割合=r("元と同じ形", "組めた"), 丁寧の行の割合=r("丁寧の行", "組めた"),
                意味の印で打ち消しが入れ替わった割合=round(c["意味の印で打ち消しが入れ替わった"] / max(2 * c["組めた"], 1), 4),
                文字の印で打ち消しが入れ替わった割合=r("文字の印で打ち消しが入れ替わった", "組めた"),
                作り替えた行が違和感チェックを通る割合=r("作り替えた行が違和感チェックを通る", "作り替えた行"),
                本物の行が違和感チェックを通る割合=r("本物の行が違和感チェックを通る", "作り替えた行"))
    for pc in pick_c.values():
        n = max(pc["問い"], 1)
        pc["割合"] = dict(状況波=round(pc["状況波が当たり"] / n, 4), 前の行の丁寧の印に合わせる=round(pc["前の行の丁寧の印に合わせて当たり"] / n, 4),
                         いつも多い方=round(max(pc["いつも多い方"], pc["問い"] - pc["いつも多い方"]) / n, 4))
    return dict(作品=len(works), 外した作品=len(held), 数=c, 割合=rate, 丁寧の有り無しを選ぶ=pick_c, 例=ex)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    res = dict(手で置いた値=dict(HOLD=HOLD, NEG=list(NEG)), **run(json.load(open(a.cache, encoding="utf-8"))))
    res["秒"] = round(time.time() - t0, 1)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "例"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
