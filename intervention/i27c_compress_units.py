"""介入の研究 実験I27c: I27 の候補の取り方を直し、縮めて曖昧になった所を「縮める前の版」で決める（I27 の次の一手 1・2）

I27 の候補の取り方は粗く、見かけの曖昧さ（数え方・接尾辞・かぎかっこの中）が混じっていた。ここではトークンを「まとまり」に束ねてから数える。
  - 接尾辞（たち・さん・様 など）は前の名詞と 1 つのまとまりにする（子 + たち = 子たち）。
  - 数え方（一 + 羽、三 + 匹、二人、数千）は「数のまとまり」とし、係り先の候補にも、修飾する語にも数えない（名詞句はその先へ続く）。
  - かぎかっこ 「…」 の中は 1 つのまとまり（名前のように扱う）。中で閉じた修飾は数えない。外からの修飾の候補は、かぎかっこ全体の 1 つだけ。
  - 形式名詞（とも・一緒・代わり・おかげ・ため など）は候補にしない。そこで名詞句は終わる。
  - 名詞句が続くのは、まとまりの後ろが の（または という）のときだけ。と・、・や で並ぶ所（並列）は止めて、別に数える。
係り先を決める読み:
  - 近い方、GiNZA、
  - 1 つ前の版（12 行なら 20 行、5 行なら 12 行、1 行なら 5 行）の話の板、
  - 前の版すべて（自分より長い版をまとめた板）、
  - 20 行版だけの板（I27 と同じ）。
  話の板は I27a と同じで、候補のうち 1 つだけに「修飾:M」があれば決まる。決まらなければ近い方のまま。
  正解の印（--labels）があれば、それぞれの読みの合い方と、「前の版で決まらなかった所だけを聞く」ときの聞く数を出す。
物差し: 段ごとの曖昧の数と割合（I27 と比べる）、読みごとの決めた割合と合い方、全部合うまでに聞く数。場所の一覧は非公開の置き場（童話の文を含むため）。
結果: ../Project_Kotoyui_private/intervention/results/i27c.jsonl、場所の一覧: <非公開>/data/tales/i27c_places.md
実行: python intervention/i27c_compress_units.py --tales <tales_v1_parsed.json> [--labels <i27c_labels.json>] [--places <md>] [--dump <json>]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from i27a_story_check import story_boards  # noqa: E402

LEVELS = ("20", "12", "5", "1")
NOUN = ("NOUN", "PROPN", "PRON")
SUFFIX = {"たち", "達", "さん", "様", "さま", "君", "くん", "ちゃん", "ら", "ども", "殿"}
COUNTER = {"人", "匹", "羽", "つ", "本", "個", "枚", "頭", "台", "冊", "回", "度", "粒", "杯", "軒", "日", "年", "歳", "才"}
NUMCH = set("一二三四五六七八九十百千万数何幾0123456789０１２３４５６７８９")
FORMAL = {"とも", "共", "一緒", "ため", "代わり", "かわり", "おかげ", "際", "うち", "こと", "もの", "ところ", "よう", "はず", "まま"}
COORD = {"と", "、", "や", "・", "か"}


def is_quantity(text):
    k = 0
    while k < len(text) and text[k] in NUMCH:
        k += 1
    return k > 0 and (text[k:] == "" or text[k:] in COUNTER)


def units(toks):
    """トークン → まとまりの番号。まとまり: dict(ids, head, text, kind)。kind は 名詞 / 数 / 引用 / 形式 / その他。"""
    n = len(toks)
    uid = list(range(n))

    def root(i):
        while uid[i] != i:
            i = uid[i]
        return i

    def join(a, b):                                           # a を b のまとまりに入れる
        uid[root(a)] = root(b)
    for i, w in enumerate(toks):                                # 複合語の前の部分
        if w["dep"] == "compound" and w["head"] != i and abs(w["head"] - i) <= 3:
            join(i, w["head"])
    for i in range(1, n):                                       # 接尾辞と数え方
        w, p = toks[i], toks[i - 1]
        if w["pos"] in NOUN and w["text"] in SUFFIX and p["pos"] in NOUN:
            join(i - 1, i)
        if w["pos"] in NOUN and p["pos"] == "NUM":
            join(i - 1, i)
    i = 0
    while i < n:                                                # かぎかっこ
        if toks[i]["text"] == "「":
            depth, j = 0, i
            while j < n:
                depth += {"「": 1, "」": -1}.get(toks[j]["text"], 0)
                if depth == 0:
                    break
                j += 1
            inner = list(range(i + 1, min(j, n)))
            if inner:
                out = [k for k in inner if not (i < toks[k]["head"] < j)]
                h = out[-1] if out else inner[-1]
                uid[h] = h
                for k in range(i, min(j + 1, n)):
                    if k != h:
                        uid[k] = h
            i = j + 1
            continue
        i += 1
    groups = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(i)
    tok2u, us = [0] * n, []
    for r in sorted(groups, key=lambda r: min(groups[r])):
        ids = groups[r]
        inside = set(ids)
        outs = [k for k in ids if toks[k]["head"] not in inside or toks[k]["head"] == k]
        h = outs[-1] if outs else ids[-1]
        text = "".join(toks[k]["text"] for k in ids if toks[k]["text"] not in ("「", "」"))
        if toks[ids[0]]["text"] == "「":
            kind = "引用"
        elif is_quantity(text) or toks[h]["pos"] == "NUM":
            kind = "数"
        elif toks[h]["pos"] in NOUN and toks[h]["text"] in FORMAL:
            kind = "形式"
        elif toks[h]["pos"] in NOUN:
            kind = "名詞"
        else:
            kind = "その他"
        for k in ids:
            tok2u[k] = len(us)
        us.append(dict(ids=ids, head=h, text=text, kind=kind))
    return tok2u, us


def modifiers(toks, tok2u, us):
    """修飾する語（まとまりの中心）と、GiNZA の係り先のまとまり。"""
    for ui, u in enumerate(us):
        i = u["head"]
        w = toks[i]
        h = w["head"]
        if h == i or u["kind"] in ("数", "引用", "形式"):
            continue
        hu = tok2u[h]
        if hu == ui or us[hu]["kind"] not in ("名詞", "引用", "数"):
            continue
        if (w["pos"] in ("ADJ", "VERB") and w["dep"] in ("acl", "amod")) or (w["pos"] in NOUN and w["dep"] == "nmod"):
            yield ui, hu


def candidates(toks, tok2u, us, ui):
    """M の右の、の でつながる名詞句のまとまり。返り値: (候補のまとまり, 止まった理由)。"""
    out = []
    j = us[ui]["ids"][-1] + 1
    n = len(toks)
    while j < n:
        u = tok2u[j]
        uu = us[u]
        w = toks[uu["head"]]
        nxt = uu["ids"][-1] + 1
        if uu["kind"] == "その他":
            if w["pos"] in ("VERB", "ADJ") and w["dep"] not in ("compound", "fixed"):
                if w["lemma"] in ("いう", "言う") and j > 0 and toks[j - 1]["text"] == "と" and out:
                    j = nxt                                     # 「A という B」は名詞句が続く
                    continue
                return out, "述語"
            if not out:                                         # 最初の名詞まで（て・い・た・の など）は読み飛ばす
                j = nxt
                continue
            return out, "その他"
        if uu["kind"] == "形式":
            return out, "形式名詞"
        if uu["kind"] != "数":
            out.append(u)
        if nxt < n and toks[nxt]["text"] == "の":
            j = nxt + 1
            continue
        if nxt + 1 < n and toks[nxt]["text"] == "と" and toks[nxt + 1]["lemma"] in ("いう", "言う"):
            j = nxt + 2
            continue
        if uu["kind"] == "数":
            j = nxt
            continue
        if nxt < n and toks[nxt]["text"] in COORD:
            return out, "並列"
        return out, "中心"
    return out, "文末"


def story_has(sb, toks, us, u, mlem):
    h = us[u]["head"]
    keys = {us[u]["text"], toks[h]["text"], toks[h]["lemma"]}
    parts = [toks[k]["text"] for k in us[u]["ids"] if toks[k]["dep"] == "compound" and toks[k]["head"] == h]
    keys.add("".join(parts) + toks[h]["text"])
    return max(sb.get(k, {}).get("修飾:" + mlem, 0) for k in keys)


def decide(sb, toks, us, cand, mlem):
    """話の板で決まれば候補の番号、決まらなければ None。"""
    if sb is None:
        return None
    pos = [u for u in cand if story_has(sb, toks, us, u, mlem) > 0]
    return pos[0] if len(pos) == 1 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tales", required=True)
    ap.add_argument("--labels", default=None, help="場所 → 正しい係り先のまとまりの表記（非公開）")
    ap.add_argument("--places", default=None)
    ap.add_argument("--dump", default=None, help="曖昧な所の一覧を JSON で書く（印付け用、非公開）")
    ap.add_argument("--out", default="../Project_Kotoyui_private/intervention/results/i27c.jsonl")
    a = ap.parse_args()
    tales = json.load(open(a.tales, encoding="utf-8"))
    labels = json.load(open(a.labels, encoding="utf-8")) if a.labels else {}
    res = {lv: Counter() for lv in LEVELS}
    items, lines = [], ["# I27c 縮めると曖昧になる所（まとまりで数え直し。非公開: 童話の文を含む）", ""]
    for ti, t in enumerate(tales):
        lines += [f"## {t['name']}", ""]
        for li, lv in enumerate(LEVELS):
            longer = LEVELS[:li]
            boards = dict(prev=story_boards(t, (longer[-1],)) if longer else None,
                          longer=story_boards(t, longer) if longer else None,
                          b20=story_boards(t, ("20",)))
            for si, s in enumerate(t["parsed"].get(lv, [])):
                toks = s["toks"]
                tok2u, us = units(toks)
                for ui, gu in modifiers(toks, tok2u, us):
                    c = res[lv]
                    c["修飾"] += 1
                    cand, why = candidates(toks, tok2u, us, ui)
                    if gu not in cand:
                        c["候補の外（GiNZA の係り先が名詞句の外）"] += 1
                    if why == "並列":
                        c["並列で止めた"] += 1
                    if len(cand) < 2:
                        continue
                    c["曖昧"] += 1
                    mlem = toks[us[ui]["head"]]["lemma"]
                    pid = f"{ti}.{lv}.{si}.{us[ui]['head']}"
                    rd = dict(near=cand[0], far=cand[-1], ginza=gu if gu in cand else cand[0])
                    for name, sb in boards.items():
                        d = decide(sb, toks, us, cand, mlem)
                        rd[name + "_decided"] = d is not None
                        rd[name] = d if d is not None else cand[0]
                        rd[name + "_far"] = d if d is not None else cand[-1]   # 決まらなければ名詞句の中心（いちばん遠い方）
                        if d is not None:
                            c[f"決めた: {name}"] += 1
                    texts = {k: us[v]["text"] for k, v in rd.items() if not k.endswith("_decided")}
                    it = dict(id=pid, level=lv, m=us[ui]["text"], cand=[us[u]["text"] for u in cand],
                              read=texts, decided={k[:-8]: v for k, v in rd.items() if k.endswith("_decided")})
                    if pid in labels:
                        it["gold"] = labels[pid]
                    items.append(it)
                    if lv != "20":
                        lines.append(f"- [{lv} 行] `{pid}` {s['text']}  \n  「{it['m']}」 候補 = {' / '.join(it['cand'])}、"
                                     + "、".join(f"{k} = {v}" for k, v in texts.items()))
        lines.append("")
    rec = dict(exp="I27c", levels={lv: dict(c) for lv, c in res.items()})
    for lv, c in res.items():
        rec["levels"][lv]["曖昧の割合"] = c["曖昧"] / max(1, c["修飾"])
    short = [x for x in items if x["level"] != "20"]
    gold = [x for x in short if x.get("gold") in x["cand"] or x.get("gold") == "両方"]   # 対象外（修飾ではない・並列の誤り）は除く

    def ok(x, name):
        return x["gold"] == "両方" or x["read"][name] == x["gold"]
    if gold:
        acc = {}
        for name in ("near", "far", "ginza", "prev", "longer", "b20", "prev_far", "longer_far", "b20_far"):
            acc[name] = sum(ok(x, name) for x in gold) / len(gold)
        rec["gold"] = dict(n=len(gold), acc=acc, both_ok=sum(1 for x in gold if x["gold"] == "両方"),
                           excluded=sum(1 for x in short if x.get("gold") == "対象外"),
                           n_no_label=sum(1 for x in short if "gold" not in x))
        # 聞き返し: 前の版で決まった所は聞かない。決まらない所だけを聞く（聞けば正しい）。
        for name in ("prev", "longer"):
            asks = [x for x in gold if not x["decided"][name]]
            wrong_left = sum(1 for x in gold if x["decided"][name] and not ok(x, name))
            rec["gold"][f"ask_{name}"] = dict(asks=len(asks), acc_after=1 - wrong_left / len(gold),
                                             decided=len(gold) - len(asks),
                                             decided_acc=(sum(1 for x in gold if x["decided"][name] and ok(x, name))
                                                          / max(1, len(gold) - len(asks))))
        nf = [x for x in gold if x["gold"] not in ("両方", x["cand"][0])]
        rec["gold"]["far_is_right"] = len(nf)
        rec["gold"]["far_fixed_by"] = {name: sum(1 for x in nf if x["read"][name] == x["gold"]) for name in ("far", "ginza", "prev", "longer", "b20")}
        rec["gold"]["near_is_right"] = sum(1 for x in gold if x["gold"] == x["cand"][0])
        # 聞く順: 前の版すべての板で決まらなかった所を、GiNZA が近い方と食い違う所から先に聞く。比べ: でたらめな順（100 回の平均）。
        und = [x for x in gold if not x["decided"]["longer"]]
        base = sum(ok(x, "longer") for x in gold)
        gain = [0 if ok(x, "longer") else 1 for x in und]
        order = sorted(range(len(und)), key=lambda i: -(und[i]["read"]["ginza"] != und[i]["read"]["near"]))
        rng = random.Random(0)
        curve, rand = {}, {}
        for k in range(len(und) + 1):
            curve[k] = (base + sum(gain[i] for i in order[:k])) / len(gold)
            rand[k] = sum((base + sum(gain[i] for i in rng.sample(range(len(und)), k))) / len(gold) for _ in range(100)) / 100
        rec["gold"]["ask_curve"] = dict(n_undecided=len(und), n_wrong_undecided=sum(gain),
                                        ginza_first=curve, random=rand,
                                        ginza_disagree=sum(1 for x in und if x["read"]["ginza"] != x["read"]["near"]))
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if a.places:
        os.makedirs(os.path.dirname(os.path.abspath(a.places)), exist_ok=True)
        open(a.places, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    if a.dump:
        json.dump(items, open(a.dump, "w", encoding="utf-8"), ensure_ascii=False, indent=0)


if __name__ == "__main__":
    main()
