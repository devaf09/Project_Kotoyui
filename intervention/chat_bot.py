"""介入の研究: 試しのチャットボット（I27x2 の行づくりで返事をする）

持ち主（2026-11-08）: 「そしたらいったんチャットボットを作ってみてくれる？」
しくみ（docs/thinking_parts.md の名前）:
  読み手   入力の文を GiNZA で読む（今は GiNZA。いずれ自分の読みに替える）。
  場の板   人の行（新しいほど重い。自分の返事は入れない）と、入力に似た本の行（似ているほど重い）を束ねた板。
           似た行 = 入力の内容語と重なる行（語の珍しさ idf で重みづけ）と、その前後の行。
  汎用板   本の全部から: 述語が枠を取る割合、組の裏づけ、節の期待、て形、述語の流れ、双の裏づけ、問いの語の期待。
  試す・振り返り・反応の板・決め手   I27x2 と同じ（振り返りの回数 --budget、既定 2 = エッジ）。
  入力の述語を「前の行の述語」にして、述語の流れで次の述語を選ぶ。入力と同じ主語・述語の組と、前に自分が言った組は避ける。
本の行は非公開（Web 小説なら _wn）。会話の記録は --log に書く（非公開の置き場へ）。
実行: python intervention/chat_bot.py --cache <解析済み JSON> [--budget 2] [--log <md>] [--say "文" ...]
  --say を並べると、その文を順に入れて返事を出す（試し用）。無ければ対話する（空行で終わり）。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i27m_board_line as M  # noqa: E402
import i27o_board_line2 as O  # noqa: E402
import i27q_book_insert as Q  # noqa: E402
import i27s_fused_generate as S  # noqa: E402
import i27x_chain_clauses as X  # noqa: E402
import i27x2_clause_meaning as X2  # noqa: E402
from board_ops import Board, bundle  # noqa: E402


def to_toks(doc):
    return [dict(i=t.i, text=t.text, lemma=t.lemma_, pos=t.pos_, dep=t.dep_, head=t.head.i) for t in doc]


class Bot:
    def __init__(self, nlp, works, budget=2, k_lines=40, ctx=2, w_conv=1.0, decay=0.5):
        self.nlp, self.budget, self.k_lines, self.ctx, self.w_conv, self.decay = nlp, budget, k_lines, ctx, w_conv, decay
        t0 = time.time()
        Q.prep(works)
        self.works = works
        self.tr, self.roots = O.flow_board(works)
        self.rows = [S.line_rows(w) for w in works]
        self.G = Board()
        for w in works:
            self.G.update(bundle(list(enumerate(S.frame_rows(w)))))
        self.CG = Board()
        for w in works:
            self.CG.update(bundle(list(enumerate(X.clause_rows(w)))))
            self.CG.update(bundle(list(enumerate(X.te_rows(w)))))
        self.nexts = X.index_next(self.CG)
        self.JG = Board()
        for w in works:
            self.JG.update(bundle(list(enumerate(X2.joint_rows(w)))))
        self.propn = {t["lemma"] for w in works for ln in w["lines"] for t in ln["toks"] if t["pos"] in ("PROPN", "PRON")}
        self.propn_only = {t["lemma"] for w in works for ln in w["lines"] for t in ln["toks"] if t["pos"] == "PROPN"}
        # 似た行を探す索引: 内容語 → (作品, 行)
        self.index = defaultdict(list)
        df = Counter()
        for wi, w in enumerate(works):
            for li, ln in enumerate(w["lines"]):
                ls = set(M.content_lemmas(ln["toks"]))
                for x in ls:
                    self.index[x].append((wi, li))
                df.update(ls)
        n = sum(len(w["lines"]) for w in works)
        self.idf = {x: math.log(n / (1 + c)) for x, c in df.items()}
        self.conv = dict(name="会話", lines=[])
        self.load_s = time.time() - t0

    def similar(self, lemmas):
        sc = Counter()
        for x in lemmas:
            for key in self.index.get(x, [])[:5000]:
                sc[key] += self.idf.get(x, 0.0)
        return sc.most_common(self.k_lines)

    def reply(self, text):
        doc = self.nlp(text)
        self.conv["lines"].append(dict(text=text, toks=to_toks(doc), who="人"))
        Q.prep([self.conv])
        crow = S.line_rows(self.conv)
        last = self.conv["lines"][-1]["toks"]
        lem = [x for x in M.content_lemmas(last) if self.idf.get(x, 0) > 0]
        sim = self.similar(lem)
        top = sim[0][1] if sim else 1.0
        rows, text_of = [], {}
        mine = [j for j, ln in enumerate(self.conv["lines"]) if ln["who"] == "人"]
        for r_, j in enumerate(reversed(mine)):           # 人の行（新しいほど重い）。自分の返事は入れない（おうむ返しの輪になる）
            rows.append((self.w_conv * self.decay ** r_, crow[j]))
        text_of.update(self.conv["_text"])
        used = set()
        for (wi, li), s in sim:                           # 似た本の行とその前後
            for d in range(-self.ctx, self.ctx + 1):
                lj = li + d
                if 0 <= lj < len(self.works[wi]["lines"]) and (wi, lj) not in used:
                    used.add((wi, lj))
                    rows.append((s / top / (1 + abs(d)), self.rows[wi][lj]))
            for k, v in self.works[wi]["_text"].items():
                text_of.setdefault(k, v)
        BB = Board()
        for wt, b in rows:
            for k, v in b.items():
                BB[k] += wt * v
        st = S.Site(BB, self.G, Board(), self.propn)
        st.BB_text = text_of
        st.propn_only = self.propn_only
        ch = X.Chain(st, self.CG, Board())
        ch.nexts = self.nexts
        ch.JG, ch.jown = self.JG, Board()
        board = (st.wb, st.form, st.part, text_of)
        prev_p = O.root_pred(last)
        avoid = O.pairs_in(last)
        for ln in self.conv["lines"]:
            if ln["who"] == "ボット":
                avoid |= O.pairs_in(ln["toks"])               # 前に言った組も避ける
        pairs = S.pair_scores(board, avoid, prev_p, self.tr)
        if not pairs:
            out = dict(line="……。", tries=0, breaks=[], sim=len(sim))
        else:
            X.propose = X2.propose2
            (line, cl, chk, d2), tries = X.ring(self.nlp, ch, pairs, self.budget, True)
            out = dict(line=line, tries=tries, breaks=[f"{j + 1}:{k}" for (j, k), v in chk.items() if not v], sim=len(sim))
        rd = self.nlp(out["line"])
        self.conv["lines"].append(dict(text=out["line"], toks=to_toks(rd), who="ボット"))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--budget", type=int, default=2)
    ap.add_argument("--say", nargs="*")
    ap.add_argument("--log")
    a = ap.parse_args()
    import spacy
    nlp = spacy.load("ja_ginza")
    if a.cache.endswith(".gz"):
        import gzip
        works = json.load(gzip.open(a.cache, "rt", encoding="utf-8"))
    else:
        works = json.load(open(a.cache, encoding="utf-8"))
    bot = Bot(nlp, works, budget=a.budget)
    print(f"（読み込み {bot.load_s:.0f} 秒、本の行 {sum(len(w['lines']) for w in works):,}）", flush=True)
    log = ["# チャットボットの記録（非公開）", "", f"- 本: {len(works)} 作品、振り返りの回数 {a.budget}", ""]

    def turn(x):
        t0 = time.time()
        r = bot.reply(x)
        info = f"（試した {r['tries']} 回、似た行 {r['sim']}、崩れ {'・'.join(r['breaks']) or 'なし'}、{time.time() - t0:.1f} 秒）"
        print(f"> {x}\n{r['line']}  {info}", flush=True)
        log.extend([f"- 人: {x}", f"- ボット: **{r['line']}** {info}", ""])

    if a.say:
        for x in a.say:
            turn(x)
    else:
        while True:
            try:
                x = input("> ").strip()
            except EOFError:
                break
            if not x:
                break
            turn(x)
    if a.log:
        os.makedirs(os.path.dirname(os.path.abspath(a.log)), exist_ok=True)
        with open(a.log, "w", encoding="utf-8") as f:
            f.write("\n".join(log))


if __name__ == "__main__":
    main()
