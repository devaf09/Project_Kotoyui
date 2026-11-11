"""介入の研究 実験I27d: 介入（話の板で係り先を決める）がなぜ間違ったかを調べ、介入が要る所に光る「注意の板」を積み重ねる

持ち主（2026-11-07）: 「介入でなぜ間違ったのか、そして介入を発見する注意の板を作って、注意の板の積み重ねで間違いが減ると思ってる」。
I27c の誤り 2 か所は、長い版の板の「修飾:M」が、長い版の同じ形の曖昧な句（M + A の B）を GiNZA が近い方に係けた跡だった。
板が解析の誤りを写していた（自分で自分を確かめていた）。そこで:
  - 根拠の分け方: 長い版の板の「修飾:M」を、曖昧でない所から来たもの（きれいな根拠）と、曖昧な句から来たもの（写しの根拠）に分ける。
  - 層 0: I27c と同じ（前の版すべての板で決め、決まらなければ近い方）。
  - 層 1 = 根拠の注意（学ばない）: 写しの根拠だけで決まった所に光る。光ったら、決めたことを取り消して近い方に戻し、聞く順の先頭に置く。
  - 層 2 = 形の注意（誤りから学ぶ）: 場面の形（M の種類、A が「A の X」の形で使われやすい語か、根拠の状態）ごとに、層 1 の読みが誤った数を数える。
          明るさ = (誤り + 0.5) / (数 + 1)。0.5 より明るく、きれいな根拠で決まっていない所は、もう一方の候補に替える（介入）。
  - 層 3 = 介入の注意（誤りから学ぶ）: 層 2 の読みの誤りを、(M の種類, 根拠の状態, 層 2 で替えたか) ごとに数える。明るい所は替える。
  - 学ぶ注意は、その話を除いた残りの話の正解の印から作る（1 話抜き）。印は I27c と同じ手の印（20 行版の印も足した）。
  - A が「A の X」で使われやすいかは、12 話すべての文の解析から数える（正解の印は使わない）。
物差し: 層ごとの合い方（聞かないで）、注意の明るい順に聞いたときの、全部合うまでの聞く数と途中の合い方（でたらめな順と比べる）。
結果: ../Project_Kotoyui_private/intervention/results/i27d.jsonl
実行: python intervention/i27d_attention_boards.py --tales <tales_v1_parsed.json> --labels <i27c_labels.json>
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from i26_ambig_judge import compound  # noqa: E402
from i27a_story_check import story_boards  # noqa: E402
from i27c_compress_units import LEVELS, candidates, decide, modifiers, units  # noqa: E402


def copy_board(tale, levels):
    """曖昧な句（候補が 2 つ以上）の GiNZA の係りから来た「修飾:M」だけを数えた板（story_boards と同じ鍵）。"""
    cb = defaultdict(Counter)
    for lv in levels:
        for s in tale["parsed"].get(lv, []):
            toks = s["toks"]
            tok2u, us = units(toks)
            for ui, gu in modifiers(toks, tok2u, us):
                cand, _ = candidates(toks, tok2u, us, ui)
                if len(cand) < 2:
                    continue
                i = us[ui]["head"]
                h = toks[i]["head"]
                w, hw = toks[i], toks[h]
                ok = ((w["pos"] in ("ADJ", "VERB") and w["dep"] in ("acl", "amod") and hw["pos"] in ("NOUN", "PROPN"))
                      or (w["pos"] in ("NOUN", "PROPN") and w["dep"] == "nmod" and hw["pos"] in ("NOUN", "PROPN")))
                if ok:
                    for k in {compound(toks, h), hw["lemma"]}:
                        cb[k]["修飾:" + w["lemma"]] += 1
    return cb


def clean_board(full, cb):
    out = defaultdict(Counter)
    for k, c in full.items():
        for a, v in c.items():
            r = v - cb.get(k, {}).get(a, 0)
            if r > 0:
                out[k][a] = r
    return out


def kindness(tales):
    """名詞のまとまり → 「A の X」の A として使われた割合と数。"""
    as_mod, total = Counter(), Counter()
    for t in tales:
        for lv in LEVELS:
            for s in t["parsed"].get(lv, []):
                toks = s["toks"]
                tok2u, us = units(toks)
                for u in us:
                    if u["kind"] != "名詞":
                        continue
                    total[u["text"]] += 1
                    nxt = u["ids"][-1] + 1
                    if nxt < len(toks) and toks[nxt]["text"] == "の":
                        as_mod[u["text"]] += 1
    return {k: (as_mod[k] / total[k], total[k]) for k in total}


def mtype(w):
    return {"VERB": "V", "ADJ": "A"}.get(w["pos"], "N")


def other(x, r):
    return x["cand"][-1] if r == x["cand"][0] else x["cand"][0]


def right(x, r):
    return x["gold"] == "両方" or r == x["gold"]


def attention(items, key, read):
    """1 話抜きの注意の明るさ: 同じ形の場面で、その読みが誤った割合（ほかの話の印から）。"""
    tab = defaultdict(lambda: defaultdict(lambda: [0, 0]))     # 話 → 形 → [誤り, 数]
    for x in items:
        tab[x["tale"]][key(x)][0] += 0 if right(x, read(x)) else 1
        tab[x["tale"]][key(x)][1] += 1
    out = {}
    for x in items:
        e = n = 0
        for t, d in tab.items():
            if t != x["tale"] and key(x) in d:
                e += d[key(x)][0]
                n += d[key(x)][1]
        out[x["id"]] = (e + 0.5) / (n + 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tales", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--kind_th", type=float, default=0.5)
    ap.add_argument("--out", default="../Project_Kotoyui_private/intervention/results/i27d.jsonl")
    a = ap.parse_args()
    tales = json.load(open(a.tales, encoding="utf-8"))
    labels = json.load(open(a.labels, encoding="utf-8"))
    kind = kindness(tales)
    items = []
    for ti, t in enumerate(tales):
        for li, lv in enumerate(LEVELS):
            longer = LEVELS[:li]
            full = story_boards(t, longer) if longer else None
            clean = clean_board(full, copy_board(t, longer)) if longer else None
            for si, s in enumerate(t["parsed"].get(lv, [])):
                toks = s["toks"]
                tok2u, us = units(toks)
                for ui, gu in modifiers(toks, tok2u, us):
                    cand, _ = candidates(toks, tok2u, us, ui)
                    if len(cand) < 2:
                        continue
                    pid = f"{ti}.{lv}.{si}.{us[ui]['head']}"
                    g = labels.get(pid)
                    if g is None or g == "対象外":
                        continue
                    mlem = toks[us[ui]["head"]]["lemma"]
                    df, dc = decide(full, toks, us, cand, mlem), decide(clean, toks, us, cand, mlem)
                    state = "きれい" if dc is not None else ("写し" if df is not None else "なし")
                    A = us[cand[0]]["text"]
                    kr, kn = kind.get(A, (0.0, 0))
                    items.append(dict(id=pid, tale=ti, level=lv, gold=g, cand=[us[u]["text"] for u in cand],
                                      full=us[df]["text"] if df is not None else None,
                                      clean=us[dc]["text"] if dc is not None else None, state=state,
                                      mtype=mtype(toks[us[ui]["head"]]), a_kind=kr >= a.kind_th and kn >= 2))
    for x in items:
        x["r0"] = x["full"] or x["cand"][0]
        x["r1"] = x["clean"] or x["cand"][0]                                      # 層 1: 写しの根拠を取り消す
    att2 = attention(items, lambda x: (x["mtype"], x["a_kind"], x["state"]), lambda x: x["r1"])
    for x in items:
        x["att2"] = att2[x["id"]]
        x["flip2"] = att2[x["id"]] > 0.5 and x["state"] != "きれい"
        x["r2"] = other(x, x["r1"]) if x["flip2"] else x["r1"]
    att3 = attention(items, lambda x: (x["mtype"], x["state"], x["flip2"]), lambda x: x["r2"])
    for x in items:
        x["att3"] = att3[x["id"]]
        x["flip3"] = att3[x["id"]] > 0.5 and x["state"] != "きれい"
        x["r3"] = other(x, x["r2"]) if x["flip3"] else x["r2"]

    def summary(xs):
        n = len(xs)
        res = dict(n=n, both=sum(x["gold"] == "両方" for x in xs),
                   near=sum(right(x, x["cand"][0]) for x in xs) / n, far=sum(right(x, x["cand"][-1]) for x in xs) / n)
        for k in ("r0", "r1", "r2", "r3"):
            res[k] = sum(right(x, x[k]) for x in xs) / n
        res["layer1_fired"] = sum(x["state"] == "写し" for x in xs)
        res["layer1_fired_wrong_r0"] = sum(x["state"] == "写し" and not right(x, x["r0"]) for x in xs)
        res["flip2"] = sum(x["flip2"] for x in xs)
        res["flip2_fixed"] = sum(x["flip2"] and right(x, x["r2"]) and not right(x, x["r1"]) for x in xs)
        res["flip2_broke"] = sum(x["flip2"] and not right(x, x["r2"]) and right(x, x["r1"]) for x in xs)
        res["flip3"] = sum(x["flip3"] for x in xs)
        res["flip3_fixed"] = sum(x["flip3"] and right(x, x["r3"]) and not right(x, x["r2"]) for x in xs)
        res["flip3_broke"] = sum(x["flip3"] and not right(x, x["r3"]) and right(x, x["r2"]) for x in xs)
        # 聞く順: 層 1 が光った所 → 根拠なしの所を注意（層 2 / 3）の明るい順。きれいな根拠で決まった所は最後。
        for k, att in (("r1", "att2"), ("r2", "att3"), ("r3", "att3")):
            order = sorted(xs, key=lambda x: (x["state"] == "きれい", -(x["state"] == "写し"), -x[att]))
            wrong = [not right(x, x[k]) for x in order]
            base = n - sum(wrong)
            curve = [(base + sum(wrong[:j])) / n for j in range(n + 1)]
            last = max([j + 1 for j, w in enumerate(wrong) if w], default=0)
            rng = random.Random(0)
            rnd_last, rnd_curve = [], [0.0] * (n + 1)
            for _ in range(200):
                w = wrong[:]
                rng.shuffle(w)
                rnd_last.append(max([j + 1 for j, v in enumerate(w) if v], default=0))
                for j in range(n + 1):
                    rnd_curve[j] += (base + sum(w[:j])) / n / 200
            res[f"ask_{k}"] = dict(wrong=sum(wrong), asks_to_all=last, random_asks_to_all=sum(rnd_last) / 200,
                                   at5=curve[min(5, n)], at10=curve[min(10, n)], rand_at5=rnd_curve[min(5, n)],
                                   rand_at10=rnd_curve[min(10, n)])
        return res
    short = [x for x in items if x["level"] != "20"]
    rec = dict(exp="I27d", kind_th=a.kind_th, short=summary(short), all=summary(items),
               wrong_r0=[dict(id=x["id"], state=x["state"]) for x in short if not right(x, x["r0"])],
               wrong_r3=[dict(id=x["id"], state=x["state"], mtype=x["mtype"], a_kind=x["a_kind"]) for x in short if not right(x, x["r3"])],
               flipped=[dict(id=x["id"], layer=2 if x["flip2"] else 3) for x in short if x["flip2"] or x["flip3"]])
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
