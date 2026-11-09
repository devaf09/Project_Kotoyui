"""介入の研究 実験I35d: チャットボット 3（振り分けのグレイン）

持ち主（2026-11-09）:「違和感や会話の中で指摘があれば、自覚と介入に回す、そういう判断もグレインの学習と判断で出来ればいい」
  「ロジックでも大規模パラメータでもなく、自分で違和感や問題に気づき、自分の会話を見直し、少しずつ変わっていく AI」
chat_bot2.py（I34d）に足したもの:
  振り分けのグレイン（自分の行の側）  作った行の 名詞+助詞+述語 の所ごとに、気づきの器の差を記号にし（i35_route_grain.PieceBoards）、
      相手の指摘の経験から P(指摘 | 記号) を経験の器（I33 の CountGrain）で持つ。指摘を見込む所があれば「介入」:
      その所の名詞を反応の板で暗くして作り直す。気づいた所は自覚の板に残す（I34d と同じ、会話をまたいで持ち越す）。
      I34d の「自分の行と本の行の分かれ方」の決め方の代わり。
  振り分けのグレイン（相手の手がかりの側）  人の行が、ボットの前の行への指摘の形（I35a の手がかりの型）なら、
      1. その経験（指摘された所 → 指摘）を経験の器に足す（会話の記録を経験の器に足していく）。
      2. 回し先を指摘の板（I35a・I35c: 手がかりの型 → 回し先）の、手がかり無しと比べて多く出る回し先にする:
         介入 = 前の返事を、その所を暗くして作り直す／自覚 = 自覚の板に残して相槌で受ける／聞き返す = その語を聞き返す／そのまま = 普通に返す。
経験の器の初めの経験: 相手役（Claude）が自分の行（I27x2・I32・I33b の 100 例）の所に付けた指摘（--pointing、非公開）。
本の行は非公開（Web 小説なら _wn）。会話の記録は --log に書く（非公開の置き場へ）。
実行: python intervention/chat_bot3.py --cache <解析済み JSON> --pointing <指摘の JSON> --route <指摘の板 JSON> [--log <md>] [--say "文" ...]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chat_bot as CB  # noqa: E402
import chat_bot2 as C2  # noqa: E402
import i33_predicate_grain as G  # noqa: E402
import i34_dialogue_board as D  # noqa: E402
import i35_route_grain as R  # noqa: E402


class Bot3(C2.Bot2):
    def __init__(self, nlp, works, pointing=None, route=None, **kw):
        self.pb = R.PieceBoards(works)
        self.grain = G.CountGrain()
        for it in (json.load(open(pointing, encoding="utf-8"))["items"] if pointing else []):
            self.grain.add(self.pb.symbols(it), "指摘" if it["odd"] else "そのまま")
        self.route = json.load(open(route, encoding="utf-8")) if route else {}
        self.negx = None
        super().__init__(nlp, works, **kw)
        self.negx = R.neg_intj(works)
        self.learned = []                                   # 会話で足した経験（記録用）

    # ---- 自分の行の側 ----
    def p_point(self, pc, question=False):
        s = self.grain.scores(self.pb.symbols(pc, question))
        if "指摘" not in s:
            return 0.0
        m = max(s.values())
        z = sum(math.exp(v - m) for v in s.values())
        return math.exp(s["指摘"] - m) / z

    def oddness(self, toks):
        """I34d の物差しの代わり: 行の所のうち、指摘を見込む確からしさがいちばん大きい所。"""
        if not hasattr(self, "grain") or not self.grain.N:
            return super().oddness(toks)
        best = (0.0, ("無", ""), [])
        for pc in self.pb.pieces(toks):
            p = self.p_point(pc)
            if p > best[0]:
                best = (p, ("指摘の見込み", pc["ns"] + pc["c"] + pc["ps"]), [("補語", pc["n"]), ("目的語", pc["n"])])
        return best

    def odd_hist(self, d):
        if not hasattr(self, "grain") or not self.grain.N:
            return super().odd_hist(d)
        return d, 1.0 - d                                   # 指摘 > そのまま なら違和感（P(指摘) > 0.5）

    def learn_self(self, works, self_lines, n=500, seed=0):
        if self_lines:
            return super().learn_self(works, self_lines, n, seed)
        self.h_gen, self.h_book = {}, {}

    # ---- 相手の手がかりの側 ----
    def pick_route(self, kinds):
        base = self.route.get("手がかり無し", {})
        nb = sum(base.values()) or 1
        best = None
        for k in kinds:
            c = self.route.get(k)
            if not c:
                continue
            n = sum(c.values())
            for r in R.ROUTES:
                lift = math.log((c.get(r, 0) + 0.5) / (n + 2)) - math.log((base.get(r, 0) + 0.5) / (nb + 2))
                if best is None or lift > best[0]:
                    best = (lift, r, k)
        return best

    def reply(self, text):
        bots = [ln for ln in self.conv["lines"] if ln["who"] == "ボット"]
        hums = [ln for ln in self.conv["lines"] if ln["who"] == "人"]
        if bots and self.negx is not None:
            toks = CB.to_toks(self.nlp(text))
            a = dict(text="「" + bots[-1]["text"] + "」", toks=bots[-1]["toks"])
            b = dict(text="「" + text + "」", toks=toks)
            cs = R.cues(a, b, self.negx)
            # 自分の行の所（名詞+助詞+述語）や代名詞をそのまま「って」「？」で返されたら、その所への聞き返し
            for pc in self.pb.pieces(a["toks"]):
                for w in (pc["ns"] + pc["c"] + pc["ps"], pc["ns"]):
                    if w + "って" in text or w + "？" in text:
                        cs = [("聞き返し", pc["n"])] + [c for c in cs if c[0] != "短い問い返し"]
                        break
                else:
                    continue
                break
            if cs:
                return self.pointed(text, toks, a, cs, hums[-1]["text"] if hums else None)
        return self._name(super().reply(text))

    def _name(self, out):
        if self.grain.N:
            out["why"] = out["why"].replace("自分の行に出る割合", "P(指摘)").replace("> 本", "> P(そのまま)")
        return out

    def pointed(self, text, toks, a, cs, prev_human):
        pcs = self.pb.pieces(a["toks"])
        hit = {w for _, w in cs if w}
        tgt = [pc for pc in pcs if pc["n"] in hit or pc["p"] in hit]
        if not tgt and pcs:                                 # 指摘された語が分からない（「え？」）: 自分でいちばん怪しいと見る所
            tgt = [max(pcs, key=self.p_point)]
        for pc in tgt:                                      # 1. 経験を足す
            self.grain.add(self.pb.symbols(pc), "指摘")
            self.learned.append(pc["ns"] + pc["c"] + pc["ps"])
            self.aware[("指摘された", pc["ns"] + pc["c"] + pc["ps"])] += 1.0
            for k in (("補語", pc["n"]), ("目的語", pc["n"])):
                self.aware_keys[k] += 1.0
        r = self.pick_route([k for k, _ in cs])
        kinds = "・".join(dict.fromkeys(k for k, _ in cs))
        where = "・".join(pc["ns"] + pc["c"] + pc["ps"] for pc in tgt) or "（所なし）"
        route = r[1] if r else "そのまま"
        head = f"相手の指摘（{kinds}）→ 所「{where}」を経験に足し、自覚の板に残す。回し先 {route}"
        if route == "介入" and prev_human:
            out = self._name(C2.Bot2.reply(self, prev_human))
            out["why"] = head + "（前の人の行に、その所を暗くして返し直す）。" + out["why"]
            return out
        if route == "聞き返す" and tgt:
            t = next((x for x in a["toks"] if x["lemma"] == tgt[0]["n"]), None)
            if t is not None:
                out = dict(line=self.ask_back(t), tries=0, breaks=[], sim=0, why=head)
                return self._log(text, toks, out)
        if route == "自覚":
            h = self.intj("問いでない", "相槌")
            out = dict(line=(h or "") + "。", tries=0, breaks=[], sim=0, why=head + "（相槌で受ける）")
            return self._log(text, toks, out)
        out = self._name(C2.Bot2.reply(self, text))
        out["why"] = head + "。" + out["why"]
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--budget", type=int, default=2)
    ap.add_argument("--say", nargs="*")
    ap.add_argument("--log")
    ap.add_argument("--pointing")
    ap.add_argument("--route")
    ap.add_argument("--dialogue")
    ap.add_argument("--partner", help="相手役の指摘（所 → 指摘あり/なし の JSON）。返事に指摘ありの所があれば「〈名詞〉って？」と聞き返す")
    ap.add_argument("--judge", help="数えるための指摘（--partner と同じ形）。無ければ --partner")
    ap.add_argument("--rounds", type=int, default=1, help="--say を何回くり返すか（会話をまたいで変わるかを見る）")
    ap.add_argument("--self_lines", help="与えると I34d の決め方（自分の行と本の行の分かれ方）で比べる")
    a = ap.parse_args()
    import spacy
    nlp = spacy.load("ja_ginza")
    works = json.load(open(a.cache, encoding="utf-8"))
    bot = Bot3(nlp, works, pointing=None if a.self_lines else a.pointing, route=a.route, budget=a.budget,
               dialogue=a.dialogue, self_lines=a.self_lines)
    print(f"（読み込み {bot.load_s:.0f} 秒）", flush=True)
    log = ["# チャットボット 3 の記録（非公開）", "", f"- 本: {len(works)} 作品、振り返りの回数 {a.budget}、"
           + ("I34d の決め方" if a.self_lines else "振り分けのグレイン"), ""]

    def turn(x):
        t0 = time.time()
        r = bot.reply(x)
        info = f"（{r['why']}、崩れ {'・'.join(r['breaks']) or 'なし'}、{time.time() - t0:.1f} 秒）"
        print(f"> {x}\n{r['line']}  {info}", flush=True)
        log.extend([f"- 人: {x}", f"- ボット: **{r['line']}** {info}", ""])

    partner = {}
    if a.partner:
        for it in json.load(open(a.partner, encoding="utf-8"))["items"]:
            partner[(it["ns"], it["c"], it["ps"])] = it["odd"]
    judge = dict(partner)
    if a.judge:
        for it in json.load(open(a.judge, encoding="utf-8"))["items"]:
            judge[(it["ns"], it["c"], it["ps"])] = it["odd"]
    tally = []
    for rd in range(a.rounds):
        if a.rounds > 1:
            log.extend([f"## {rd + 1} 回目", ""])
        c = dict(返事=0, 指摘ありの所=0, 指摘なしの所=0, 相手役が知らない所=0)
        for x in a.say or []:
            turn(x)
            last = bot.conv["lines"][-1]
            c["返事"] += 1
            pcs = bot.pb.pieces(last["toks"])
            for pc in pcs:
                v = judge.get((pc["ns"], pc["c"], pc["ps"]))
                c["相手役が知らない所" if v is None else ("指摘ありの所" if v else "指摘なしの所")] += 1
            bad = [pc for pc in pcs if partner.get((pc["ns"], pc["c"], pc["ps"]))]
            if bad:
                turn(bad[0]["ns"] + bad[0]["c"] + bad[0]["ps"] + "って？")
        tally.append(c)
        log.extend([f"- 数: {json.dumps(c, ensure_ascii=False)}", ""])
        print(c, flush=True)
    if bot.learned:
        log.extend(["", f"- 会話で足した経験（指摘された所）: {'・'.join(bot.learned)}"])
    if a.log:
        os.makedirs(os.path.dirname(os.path.abspath(a.log)), exist_ok=True)
        with open(a.log, "w", encoding="utf-8") as f:
            f.write("\n".join(log))


if __name__ == "__main__":
    main()
