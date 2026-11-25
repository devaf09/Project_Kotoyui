"""I48al: 「と」を教える文を 1 文ずつ別々に教えて、どの文がいちばん効いたかを比べる（持ち主の問い）。I48ak の続き。

- 土台: 学び 5 万文（I48ak の Model、R1 は 架空・前）。候補の外も教える形。土台から毎回やり直して 1 文だけ教える。
- 介入の入れ方: 別の板（I48ak の sep）。教えた分を本体の数えに混ぜず、介入だけの板に持つ。点は 本体の点 ＋ 介入の板の
  「本体の同じ詰め物の板よりいつもより何倍鳴るか」の対数（ならしは本体の p(帯|詰め物) の 1 回分）。教えていない詰め物は 0。
  基準として、本体に重さ 1 回分で混ぜる形（I48ak）を 1 行だけ。最後に I48ak の自作 1・3・10 文と介入 5,000 文を別の板でやり直す。
- 教える文は全部自作（GiNZA を使わず、要点と詰め物を手で書く）。変える軸:
    基本   : 「X と Y に 行った」（I48ak の 10 文）
    軸1    : 人の語が同じ（友達）で述語が違う
    軸2    : 述語が同じ（行く）で、相手の語が違う（人・人でない）
    軸3    : 間に語が挟まる・挟まらない
    軸4    : 人でない語を並べる「と」
    軸5    : 引用の「と」
- 測る物（1 文ごと）: (a) 例「昨日・友達・散歩・行く・過去」の「と」の順位と点の差（「と」− ほかの最上）、
  (b) 試しの文で元が「と」の間の当たり・上位 3 に入る割合（元が「と」の間の数も）、
  (c) 害: 元が「と」でない間を「と」にした数、間全体の当たり。
  効いた上位 5 文は、同じ文を 1・3・10 回教えた時の動きも。
手で置く物: 教える文（下の TEACH_SET）、上位の数 5、回数 1・3・10。
python intervention/past/i48al_teach_one.py --conllu ... --content ... --func ... --out out.json [--train 50000] [--test 1000]
"""
from __future__ import annotations
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする

import argparse
import json
import os
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import i48ab_long_grain as G  # noqa: E402
import i48ad_lookahead as AD  # noqa: E402
import i48ae_marks as AE  # noqa: E402
import i48aj_resonance as AJ  # noqa: E402
import i48ak_teach as AK  # noqa: E402

N, V, A = "NOUN", "VERB", "ADJ"
TEACH_SET = [("基本", f"{a}と{b}に行った", [(a, N), (b, N), ("行く", V)], ["と", "に", "た。"]) for a, b in AK.TEACH] + [
    ("軸1", "友達と遊んだ", [("友達", N), ("遊ぶ", V)], ["と", "だ。"]),
    ("軸1", "友達と話した", [("友達", N), ("話す", V)], ["と", "た。"]),
    ("軸1", "友達と会った", [("友達", N), ("会う", V)], ["と", "た。"]),
    ("軸1", "友達と食べた", [("友達", N), ("食べる", V)], ["と", "た。"]),
    ("軸1", "友達と勉強した", [("友達", N), ("勉強", N), ("する", V)], ["と", "", "た。"]),
    ("軸2", "先生と行った", [("先生", N), ("行く", V)], ["と", "た。"]),
    ("軸2", "母と行った", [("母", N), ("行く", V)], ["と", "た。"]),
    ("軸2", "犬と行った", [("犬", N), ("行く", V)], ["と", "た。"]),
    ("軸2", "彼と行った", [("彼", "PRON"), ("行く", V)], ["と", "た。"]),
    ("軸3", "友達と行った", [("友達", N), ("行く", V)], ["と", "た。"]),
    ("軸3", "友達と公園へ散歩に行った", [("友達", N), ("公園", N), ("散歩", N), ("行く", V)], ["と", "へ", "に", "た。"]),
    ("軸3", "昨日友達と散歩に行った", [("昨日", N), ("友達", N), ("散歩", N), ("行く", V)], ["", "と", "に", "た。"]),
    ("軸4", "傘と帽子を買った", [("傘", N), ("帽子", N), ("買う", V)], ["と", "を", "た。"]),
    ("軸4", "パンと牛乳を買った", [("パン", N), ("牛乳", N), ("買う", V)], ["と", "を", "た。"]),
    ("軸4", "犬と猫が好きだ", [("犬", N), ("猫", N), ("好き", A)], ["と", "が", "だ。"]),
    ("軸5", "行くと言った", [("行く", V), ("言う", V)], ["と", "た。"]),
    ("軸5", "楽しいと思った", [("楽しい", A), ("思う", V)], ["と", "た。"]),
    ("軸5", "来ると聞いた", [("来る", V), ("聞く", V)], ["と", "た。"]),
]
TOP = 5
TIMES = (1, 3, 10)


def measure(m, te, surf):
    """(a) 例の「と」の順位と点の差、(b) 元が「と」の間、(c) 害と間全体。"""
    keys, gi, want = AK.EXAMPLES[0]
    res, _ = m.walk(list(keys), AK.made_items(m.S, list(keys), None))
    sc = res[gi]["今×三つ"]
    wc = m.cid(want)
    other = float(np.max(np.delete(sc, wc)))
    out = dict(例の順位=int(AJ.ranks(sc)[wc]) + 1, 例の点の差=round(float(sc[wc]) - other, 3), 例の選び=m.pg.cls[res[gi]["選び"]])
    c = dict(と=0, と当たり=0, と上位3=0, 害=0, 間=0, 間当たり=0)
    for s in te:
        xs = AE.items(m.S, s, "節")
        if not xs:
            continue
        gaps = G.split(s)[1]
        rs, _ = m.walk(AK.keys_of(s), xs)
        for i, (y, r) in enumerate(zip(gaps, rs)):
            if i + 1 == len(gaps):
                continue
            pk = m.pg.cls[r["選び"]]
            c["間"] += 1
            c["間当たり"] += pk == y
            if y == "と":
                c["と"] += 1
                c["と当たり"] += pk == "と"
                c["と上位3"] += AJ.ranks(r["今×三つ"])[wc] < 3
            elif pk == "と":
                c["害"] += 1
    out.update(元がとの間=c["と"], との当たり=round(c["と当たり"] / max(1, c["と"]), 3), との上位3=round(c["と上位3"] / max(1, c["と"]), 3),
               害_とにした数=c["害"], 間=round(c["間当たり"] / max(1, c["間"]), 4))
    return out


def main():
    ap = argparse.ArgumentParser()
    for k in ("conllu", "content", "func", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--train", type=int, default=50000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--practice", type=int, default=5000)
    ap.add_argument("--D", type=int, default=8192)
    ap.add_argument("--r1", default="架空・前", choices=AJ.R1_VARS)
    a = ap.parse_args()
    t0 = time.time()
    sents = [s for s in G.read_conllu(a.conllu) if s["toks"]]
    te = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 == 0][: a.test]
    tr_all = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 != 0]
    hold = tr_all[-2000:]
    tr = tr_all[:-2000][: a.train]
    cb = json.load(open(a.content, encoding="utf-8"))
    func = G.Func(json.load(open(a.func, encoding="utf-8")))
    S = G.AttrSpace(cb, func, a.D, bands=("B", "C", "D", "E", "G"))
    fill = AD.Fill(S)
    surf = G.Surface()
    for s in tr:
        fill.see(s)
        cont, gaps, _ = G.split(s)
        for (l, f_, u), g in zip(cont, gaps):
            surf.add(l, f_, u, g)
    base = AK.Model(S, fill, a.r1).fit(tr, hold)
    missing = sorted({g for *_, gaps in TEACH_SET for g in gaps if g not in base.ci})
    res = dict(学びの文=len(tr), 試しの文=len(te), R1の型=a.r1, 詰め物の組に無い=missing, 土台=measure(base, te, surf), 文={}, 回数={})
    print(f"土台 {time.time() - t0:.0f} 秒", json.dumps(res["土台"], ensure_ascii=False), "組に無い", missing, flush=True)

    def taught(item, times, sep=True):
        axis, text, keys, gaps = item
        m = base.clone()
        m.sep = sep
        for _ in range(times):
            m.teach(keys, AK.made_items(S, keys, gaps, () if text.endswith("好きだ") else ("過去",)), gaps, True, 1.0, text=text)
            m.refresh()
        return measure(m, te, surf)

    t1 = time.time()
    res["基準_本体に1回分"] = {TEACH_SET[0][1]: taught(TEACH_SET[0], 1, sep=False)}   # 重さ 1 回分で本体に混ぜる形（I48ak と同じ）
    print("基準 本体に 1 回分", round(time.time() - t1), "秒", json.dumps(res["基準_本体に1回分"], ensure_ascii=False), flush=True)
    for item in TEACH_SET:
        t1 = time.time()
        o = dict(軸=item[0], **taught(item, 1))
        res["文"][item[1]] = o
        print(item[1], round(time.time() - t1), "秒", json.dumps(o, ensure_ascii=False), flush=True)
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    best = sorted(res["文"], key=lambda t: (res["文"][t]["例の順位"], -res["文"][t]["例の点の差"]))[:TOP]
    for text in best:
        item = next(x for x in TEACH_SET if x[1] == text)
        res["回数"][text] = {str(k): (res["文"][text] if k == 1 else taught(item, k)) for k in TIMES}
        print("回数", text, json.dumps(res["回数"][text], ensure_ascii=False), flush=True)
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # I48ak のやり直し（別の板、学び 5 万）: 自作 1・3・10 文と、介入 5,000 文
    res["I48ak_別の板"] = {}
    for outside in (False, True):
        for k in (1, 3, 10):
            m = base.clone()
            m.sep = True
            n = {}
            for a_, b_ in AK.TEACH[:k]:
                keys, gaps = AK.made_sentence(a_, b_)
                for kk, vv in m.teach(keys, AK.made_items(S, keys, gaps), gaps, outside, 1.0).items():
                    n[kk] = n.get(kk, 0) + vv
                m.refresh()
            nm = f"自作・{'候補の外も' if outside else '候補の中だけ'}・{k} 文"
            res["I48ak_別の板"][nm] = dict(数=n, 例=m.examples())
            print(nm, json.dumps(res["I48ak_別の板"][nm], ensure_ascii=False), flush=True)
    pr = tr_all[:-2000][a.train: a.train + a.practice]
    for outside in (False, True):
        t1 = time.time()
        m = base.clone()
        m.sep = True
        n = {}
        for j, s in enumerate(pr):
            xs = AE.items(S, s, "節")
            if xs:
                for kk, vv in m.teach(AK.keys_of(s), xs, G.split(s)[1], outside, 1.0).items():
                    n[kk] = n.get(kk, 0) + vv
            if (j + 1) % AK.REFRESH == 0:
                m.refresh()
        m.refresh()
        nm = f"介入 {len(pr)} 文・{'候補の外も' if outside else '候補の中だけ'}"
        res["I48ak_別の板"][nm] = dict(数=n, 後=m.evaluate(te, surf))
        print(nm, round(time.time() - t1), "秒", json.dumps(res["I48ak_別の板"][nm], ensure_ascii=False), flush=True)
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
