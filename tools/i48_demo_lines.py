"""実演: 自作の内容語の並び（順番つき）を渡して、今の型で間の機能語と文末を埋めた行を出す（持ち主の問いへの確かめ）。

型: 本体（5 万文で学ぶ）と、本体 ＋ 数えの表 ＋ 借りる（I48be と同じ）。
文末の候補の絞り（点の式は変えず、文末の間で選んでよい候補の集合だけを限る）:
  (1) 指定なし
  (2) 過去の形だけ: 詰め物の文字列から終わりの記号（。！？!?、）を除いた物が「た」で終わる物（た・ました・でした・だった・かった・ていた など。たら・たい は入らない）
  (3) 丁寧の形だけ: 同じく記号を除き、終わりの「か・ね・よ」も除いた物が ます・ました・です・でした・ません・ましょう のどれかで終わる物
  あわせて (4) 本体の入口の「印」（機能語の板の意味の印）に「過去」を入れた場合（候補は絞らない）。(1)〜(3) の印は「なし」。
語の表記（活用）: 学びの文から数えた「原形 ＋ 次の詰め物の頭の字 → 表記」（G.Surface）で、いちばん多い表記にする。無い原形は語尾の字の置き換えの数えで作る。
選び方は本体の walk と同じ（今 ＋ R1 ＋ R2 ＋ R3 の和で、上位 K の候補の和集合から 1 つ）。絞る時は、文末の間の候補の集合を、絞った集合に置き換える。
手で置く物: 自作の内容語の並び 20 個、過去・丁寧の形の決め方（上）、印。
python tools/i48_demo_lines.py --conllu ... --content ... --func ... --out out.json [--train 50000]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "intervention"))
import i48ab_long_grain as G  # noqa: E402
import i48ad_lookahead as AD  # noqa: E402
import i48bc_usage_board as BC  # noqa: E402
import i48be_borrow as BE  # noqa: E402
from i48core import model as MD  # noqa: E402

# 自作の内容語の並び（原形, 品詞）
INPUTS = [
    "友達/NOUN 映画/NOUN 見る/VERB", "母/NOUN 台所/NOUN 料理/NOUN 作る/VERB", "昨日/NOUN 雨/NOUN 降る/VERB",
    "猫/NOUN 魚/NOUN 食べる/VERB", "駅/NOUN 電車/NOUN 待つ/VERB", "学生/NOUN 図書館/NOUN 本/NOUN 読む/VERB",
    "父/NOUN 車/NOUN 会社/NOUN 行く/VERB", "子供/NOUN 公園/NOUN 遊ぶ/VERB", "鳥/NOUN 空/NOUN 飛ぶ/VERB",
    "春/NOUN 花/NOUN 咲く/VERB", "朝/NOUN コーヒー/NOUN 飲む/VERB", "先生/NOUN 学生/NOUN 英語/NOUN 教える/VERB",
    "友達/NOUN 手紙/NOUN 書く/VERB", "医者/NOUN 病院/NOUN 働く/VERB", "古い/ADJ 家/NOUN 住む/VERB",
    "部屋/NOUN 窓/NOUN 開ける/VERB", "冬/NOUN 寒い/ADJ", "店/NOUN 新しい/ADJ 服/NOUN 買う/VERB",
    "夜/NOUN 音楽/NOUN 聞く/VERB", "犬/NOUN 庭/NOUN 走る/VERB",
]
PUNCT = "。！？!?、"


def is_past(y):
    s = (y or "").rstrip(PUNCT)
    return s.endswith("た")


def is_polite(y):
    s = re.sub("[かねよ]+$", "", (y or "").rstrip(PUNCT))
    return bool(re.search("(ます|ました|です|でした|ません|ましょう)$", s))


def walk(m, keys, xs, last_allowed=None):
    """本体の walk と同じ選び方。last_allowed があれば、文末の間の候補の集合をそれに置き換える。"""
    fic = m.who.fiction(keys)
    ups = {u for _, u in keys}
    placed, out = [], []
    for x in xs:
        g = m.gap_scores(keys, x, fic)
        r3 = m.gram.score(m.gram.state(ups, placed, x["last"]))
        three = g["R1"] + g["R2"] + r3
        both = g["今"] + three
        ok = G.other_ok(m)
        if not ok:
            three[-1] = both[-1] = -1e9
        if x["last"] and last_allowed is not None:
            cand = set(last_allowed)
        else:
            cand = (set(np.argsort(-g["今"])[:MD.K]) | set(np.argsort(-three)[:MD.K])) - (set() if ok else {m.nc - 1})
        pick = max(cand, key=lambda c: both[c])
        out.append(int(pick))
        placed.append(m.pg.cls[pick])
    return out


def render(m, surf, keys, picks):
    s = ""
    for (l, u), c in zip(keys, picks):
        y = m.pg.cls[c] or ""
        s += surf.form(l, u, y) + y
    return s


def main():
    ap = argparse.ArgumentParser()
    for k in ("conllu", "content", "func", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--train", type=int, default=50000)
    ap.add_argument("--D", type=int, default=8192)
    ap.add_argument("--r1", default="架空・前")
    a = ap.parse_args()
    t0 = time.time()
    sents = [s for s in G.read_conllu(a.conllu) if s["toks"]]
    tr_all = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 != 0]
    hold, tr = tr_all[-2000:], tr_all[:-2000][: a.train]
    del sents
    cb = json.load(open(a.content, encoding="utf-8"))
    func = G.Func(json.load(open(a.func, encoding="utf-8")))
    S = G.AttrSpace(cb, func, a.D, bands=("B", "C", "D", "E", "G"))
    fill, surf = AD.Fill(S), G.Surface()
    for s in tr:
        fill.see(s)
        cont, gaps, _ = G.split(s)
        for (l, f_, u), g in zip(cont, gaps):
            surf.add(l, f_, u, g)
    base = MD.Model(S, fill, a.r1).fit(tr, hold)
    print(f"学び {time.time() - t0:.0f} 秒", flush=True)
    ub = BC.UsageBoard(base, dk=2048).fit(base, tr)
    rows, nf = BE.usage_vectors(ub)
    nbrs = BE.neighbours(ub, rows, nf)
    tb = BE.BorrowTable(base).fit(base, tr, ub, nbrs)
    m_tab = base.clone()
    m_tab.extras.append(tb)
    m_tab.refresh()
    past = [c for c, y in enumerate(base.pg.cls) if y is not None and is_past(y)]
    polite = [c for c, y in enumerate(base.pg.cls) if y is not None and is_polite(y)]
    res = dict(学びの文=len(tr), 過去の形の候補=[base.pg.cls[c] for c in past], 丁寧の形の候補=[base.pg.cls[c] for c in polite], 行=[])
    for line in INPUTS:
        keys = [tuple(w.split("/")) for w in line.split()]
        row = dict(入力="・".join(k[0] for k in keys))
        for mn, m in (("本体", base), ("表＋借りる", m_tab)):
            for cn, allowed, mk in (("指定なし", None, ()), ("過去に絞る", past, ()), ("丁寧に絞る", polite, ()), ("印に過去", None, ("過去",))):
                xs = MD.made_items(S, keys, None, mk=mk)
                row[f"{mn}・{cn}"] = render(m, surf, keys, walk(m, keys, xs, allowed))
        res["行"].append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
