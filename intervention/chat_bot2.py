"""介入の研究 実験I34b: チャットボット 2（返事の型と、知らないことの気づき）

持ち主（2026-11-09）の 1:
  「短い会話で、問いに関係の無い語板を拾ってきている」
  「初めて話す人の場合、相手の属性語板で知らない部分があれば聞き返していいし、
    自分の語板（ことゆい本体）にない属性なら無いといっていい」
chat_bot.py（I27x2 の行づくり）に足したもの（docs/thinking_parts.md の名前）:
  自分の板   ことゆい本体の語板。("組", 名詞, 属性)。会話で自分について言ったこと・言われて認めたことだけが入る（最初は空）。
  相手の板   人の行から作る（chat_bot.py の場の板の、人の行の部分と同じ）。
  会話の板   隣り合う台詞（I34a）: 問いの型 → 答えの始めの感動詞（肯定で返す時・否定で返す時）、聞き返しの形（名詞のあとの表記）、
             問いの語への「知らない」の返し方（問いの語の問いへの返事で、いちばん多い否定の述語）。
  気づきの器（足したもの、どれも差 |期待 − 観測| の形、手の閾値なし）:
    1. 知らない語: 人が言った名詞は「知っている」と期待される（期待 1）。観測 = 本の板・相手の板での明るさ（飽和つき）の大きい方。
       聞き返す点 |1 − 差| と、そのまま返す点 |0 − 差| を比べ、差のいちばん大きい名詞で聞き返す。
    2. 自分のこと: はい/いいえの問いで主語が相手（2 人称、または主語なし）なら、自分の板の (目的語, 述語) の明るさを観測にし、
       「はい」|1 − 観測| と「いいえ」|0 − 観測| の小さい方で返す。返事は会話の板の感動詞 + 問いの述語の形（相・丁寧は問いに合わせ、
       否定だけ替える。表記は本の述語の表記の板から）。
    3. 問いの語で自分のことを聞かれ、自分の板が空なら、会話の板の「知らない」の返し方で返す。
  それ以外は chat_bot.py と同じ行づくりに、I32c のつなぎ・述部の形・助詞を使う。
本の行は非公開（Web 小説なら _wn）。会話の記録は --log に書く（非公開の置き場へ）。
実行: python intervention/chat_bot2.py --cache <解析済み JSON> [--budget 2] [--log <md>] [--say "文" ...]
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
import chat_bot as CB  # noqa: E402
import i27s_fused_generate as S  # noqa: E402
import i27x_chain_clauses as X  # noqa: E402
import i32_function_layers as L  # noqa: E402
import i34_dialogue_board as D  # noqa: E402
from board_ops import Board, light, react  # noqa: E402

YOU = ("君", "あなた", "お前", "貴方", "あんた", "きみ", "ことゆい")   # 相手（ボット）を指す語。語の一覧（辞書）として持つ
NOUN = ("NOUN", "PROPN")


def _echo_form(rest):
    """聞き返しの形: 名詞のあとの 4 文字まで、最初の ？ で切る。"""
    r = rest[:4]
    for q in ("？", "?"):
        if q in r:
            r = r[: r.index(q) + 1]
    return r.rstrip("。…　 ")


def dialogue_board(works):
    """会話の板: 問いの型ごとの、肯定・否定で返す時の始めの感動詞、聞き返しの形、知らないの返し方。"""
    start = defaultdict(Counter)
    askpat = defaultdict(Counter)
    echo = Counter()
    dunno = Counter()
    for w in works:
        Ls = w["lines"]
        for i in range(len(Ls) - 1):
            if not (D.isq(Ls[i]["text"]) and D.isq(Ls[i + 1]["text"])):
                continue
            qt = D.q_type(Ls[i]["toks"], Ls[i]["text"]).split(":")[0]
            at = Ls[i + 1]["toks"]
            ab = D.body(Ls[i + 1]["text"])
            first = next((t for t in at if t["pos"] not in ("PUNCT", "SYM")), None)
            neg = any(t["lemma"] in ("ない", "ぬ") and t["pos"] == "AUX" for t in at)
            qp = {t["lemma"] for t in Ls[i]["toks"] if t["pos"] in ("VERB", "ADJ")}
            ap_ = {t["lemma"] for t in at if t["pos"] in ("VERB", "ADJ")}
            if first is not None and first["pos"] == "INTJ":
                if qt == "問いでない":
                    start[(qt, "相槌")][first["text"]] += 1
                elif qp & ap_:                                # 問いの述語を繰り返して答えた所だけ（肯定・否定がはっきりする）
                    start[(qt, "否定" if neg else "肯定")][first["text"]] += 1
                elif qt == "問いの語" and neg:
                    start[(qt, "否定")][first["text"]] += 1
            # 問いの語の並び: (述語, 助詞) → 問いの語（「何を読む」「どこで読む」）
            if qt != "問いでない":
                for t in Ls[i]["toks"]:
                    if t["lemma"] in D.ASK and t["pos"] in ("PRON", "NOUN", "ADV"):
                        cs = [k["text"] for k in Ls[i]["toks"] if k["head"] == t["i"] and k["dep"] == "case"]
                        hd = Ls[i]["toks"][t["head"]]
                        if hd["pos"] in ("VERB", "ADJ"):
                            askpat[(hd["lemma"], cs[0] if cs else "")][t["text"]] += 1
                            askpat[("*", cs[0] if cs else "")][t["text"]] += 1
            for t in Ls[i]["toks"]:
                if t["pos"] in NOUN and len(t["text"]) >= 2 and t["text"] in ab:
                    rest = ab.split(t["text"], 1)[1]
                    if rest.startswith(("って", "？", "?")):
                        echo[_echo_form(rest)] += 1
            if qt == "問いの語" and neg:
                kids = L.kids_of(at)
                for t in at:
                    if t["pos"] in ("VERB", "ADJ") and t["dep"] == "ROOT":
                        conn, m, surf = L.layers(kids, t)
                        if m["否定"]:
                            dunno[surf] += 1
    return start, echo, dunno, askpat


def load_board(path):
    """i34c_dialogue_1808.py --merge の板（1,808 作品の台詞から）を、dialogue_board と同じ形にする。"""
    b = json.load(open(path, encoding="utf-8"))
    start, askpat, echo, dunno = defaultdict(Counter), defaultdict(Counter), Counter(b["echo"]), Counter()
    for k, v in b["start"].items():
        qt, kind, x = k.split("|", 2)
        start[(qt, kind)][x] += v
    for k, v in b["askpat"].items():
        p, part, wh = k.split("|", 2)
        askpat[(p, part)][wh] += v
        askpat[("*", part)][wh] += v
    for k, v in b["dunno"].items():
        dunno[k.rsplit("|", 1)[0]] += v
    return start, echo, dunno, askpat


def pred_surfaces(works):
    """述語の表記の板: 原形 → 表記 → 印（言い切りの根から）。"""
    out = defaultdict(Counter)
    marks = {}
    for w in works:
        for ln in w["lines"]:
            T = ln["toks"]
            kids = L.kids_of(T)
            for t in T:
                if t["dep"] == "ROOT" and t["pos"] in L.PRED:
                    conn, m, surf = L.layers(kids, t)
                    if conn == "連用":
                        out[t["lemma"]][surf] += 1
                        marks[surf] = m
    return out, marks


class Bot2(CB.Bot):
    def __init__(self, nlp, works, dialogue=None, self_lines=None, **kw):
        kw_dialogue = dialogue
        super().__init__(nlp, works, **kw)
        t0 = time.time()
        if kw_dialogue:
            self.start, self.echo, self.dunno, self.askpat = load_board(kw_dialogue)
        else:
            self.start, self.echo, self.dunno, self.askpat = dialogue_board(works)
        self.dunno_marks = {}
        self.psurf, self.pmarks = pred_surfaces(works)
        lown = [L.layer_rows(w) for w in works]
        LG = Board()
        for r, _ in lown:
            for b in r:
                LG.update(b)
        idx = L.build_index(LG)
        for w in works:
            for ln in w["lines"]:
                T = ln["toks"]
                kids = L.kids_of(T)
                for t in T:
                    if t["dep"] == "ROOT" and t["pos"] in L.PRED:
                        conn, m, surf = L.layers(kids, t)
                        if conn == "連用":
                            idx["end_marks"][surf] = m
        # 会話の行の印の期待: 本の台詞の印の割合（会話は台詞）
        q = [m for r, ms in lown for qq, m in ms if qq and m is not None]
        ctx = {k: min(max(sum(m[k] for m in q) / max(1, len(q)), 1e-3), 1 - 1e-3) for k in L.MARKS}
        self.layers = L.Layers(LG, Board(), idx, dict(ctx))
        self.base_ctx = ctx
        self.pol = Counter()                               # 相手の丁寧の板（人物の属性）: 人の行の言い切りに丁寧が出た回数
        self.odd_boards(works)
        self._pp_top = {}
        for (p_, _), v in self.pp.items():
            self._pp_top[p_] = max(self._pp_top.get(p_, 0), v)
        self.e_odd = self.expect_oddness(works)
        self.aware = Counter()                             # 自覚の板（気づいた不自然さ）
        self.aware_keys = Board()                          # 自覚の板を反応の板の鍵にしたもの
        self.learn_self(works, self_lines)
        self.me = Board()                                  # 自分の板（最初は空）
        self.you = Board()                                 # 相手の板の名詞の明るさ（人の行から）
        self.load_s += time.time() - t0

    # ---- 気づき 1: 知らない語 ----
    def unknown_noun(self, toks):
        best = None
        for t in toks:
            if t["pos"] not in NOUN or t["lemma"] in YOU:
                continue
            # 本の板では、その名詞が枠に入った行の数。相手の板では、今の行より前に人が言った回数
            seen = max(light(self.G, ("名", t["lemma"]), sat=S.SAT_PAIR), light(self.you, ("名", t["lemma"]), sat=S.SAT_PAIR, minus=self._now))
            d = react(1.0, seen)[0]                       # 知っていると期待される語の差
            ask, keep = react(1.0, d)[0], react(0.0, d)[0]
            if ask < keep and (best is None or d > best[0]):
                best = (d, t)
        return best

    def intj(self, qt, kind):
        """始めの感動詞: その返し方（肯定・否定・相槌）でほかの返し方より多く出る語（割合の比 = 差の大きい語）。"""
        mine = self.start.get((qt, kind), Counter())
        if not mine:
            return ""
        other = Counter()
        for (q2, k2), c in self.start.items():
            if q2 == qt and k2 != kind:
                other.update(c)
        if not other:                                       # 同じ問いの型にほかの返し方が無い（相槌）: ほかの全部と比べる
            for key, c in self.start.items():
                if key != (qt, kind):
                    other.update(c)
        n1, n2 = sum(mine.values()), sum(other.values())
        V = len(set(mine) | set(other))
        sc = {x: math.log((mine[x] + 0.5) / (n1 + 0.5 * V)) - math.log((other[x] + 0.5) / (n2 + 0.5 * V)) + math.log(mine[x] / n1)
              for x in mine}
        return max(sc, key=lambda x: (sc[x], x))

    def _marks(self, toks):
        kids = L.kids_of(toks)
        root = next((t for t in toks if t["dep"] == "ROOT"), None)
        return L.layers(kids, root)[1] if root is not None else {k: False for k in L.MARKS}

    # ---- 気づき 4: 相手の属性で、まだ聞いていない所を聞く ----
    def about_person(self, toks):
        subj = [t for t in toks if t["dep"] in ("nsubj", "dislocated")]
        if any(t["lemma"] in ("私", "僕", "俺", "あたし", "わたし", "自分") for t in subj):
            return True
        return not subj                                     # 主語が無い言い切り = 話し手自身のこと

    def person_question(self, toks):
        """相手の板に入れ、述語の枠でまだ埋まっていない所を、問いの語の並びで聞く。"""
        root = next((t for t in toks if t["dep"] == "ROOT"), None)
        if root is None:
            return None
        for t in toks:
            if t["pos"] in NOUN and t["head"] == root["i"]:
                self.you[("組", t["lemma"], t["dep"] + ":" + root["lemma"])] += 1
        if root["pos"] in L.PRED:
            p = root["lemma"]
        else:                                               # 「学生です」: 相手の板に入れるだけ（名詞の属性は 50 作品では薄い。語板の属性が要る）
            self.you[("組", root["lemma"], "である")] += 1
            return None
        args = [t for t in toks if t["head"] == root["i"] and t["pos"] in NOUN + ("PRON",)]
        filled = {k["text"] for t in args for k in toks if k["head"] == t["i"] and k["dep"] == "case"} if root["pos"] in L.PRED else set()
        if any(t["head"] == root["i"] and t["dep"] == "advcl" and t["pos"] in L.PRED for t in toks):
            filled.add("で")                                # 「歩いて行く」: 述語に係る節（どう・どうやって）が、で・に の枠を埋めている（I35j）
        best = None
        for ro, part in (("される", "を"), ("で・に", "で"), ("で・に", "に")):
            if part in filled or (ro == "で・に" and filled & {"で", "に", "へ"}):
                continue
            e = S.ratio(self.G, Board(), ("有", ro, p), ("出", p))     # その述語がその枠を取る割合（期待）
            wh = self.askpat.get((p, part)) or self.askpat.get(("*", part))
            if not wh:
                continue
            d = react(e, 0.0)[0]                            # 埋まっていない枠の差
            if d <= react(e, 1.0)[0]:                       # 埋まっていなくて当たり前の枠は聞かない（|e − 0| と |e − 1| の比べ）
                continue
            if best is None or d > best[0]:
                best = (d, max(wh, key=lambda x: (wh[x], x)) + part, p)
        head = self.intj("問いでない", "相槌")
        if best is None:                                    # 聞く所が無い: 相槌と、相手の述語を繰り返す（丁寧は自分の期待で）
            want = dict(self._marks(toks))
            pol = self.layers.ctx["丁寧"]
            want["丁寧"] = react(pol, 1.0)[0] < react(pol, 0.0)[0]
            surf = self.pred_form(root["lemma"], want) if root["pos"] in L.PRED else None
            if surf is None:
                surf = L.layers(L.kids_of(toks), root)[2]
            return (head + "、" if head else "") + surf + "。", (0.0, "（繰り返し）", root["lemma"])
        want = self._marks(toks)
        want = dict(want, 否定=False)
        surf = self.pred_form(best[2], want) or best[2]
        tail = "か？" if self.pmarks.get(surf, want)["丁寧"] else "の？"
        return (head + "、" if head else "") + best[1] + surf + tail, best

    def ask_back(self, t):
        form = max(self.echo, key=lambda x: (self.echo[x], x)) if self.echo else "？"
        return t["text"] + form

    # ---- 気づき 2・3: 自分のこと ----
    def about_me(self, toks):
        subj = [t for t in toks if t["dep"] in ("nsubj", "dislocated") and t["pos"] in ("PRON", "PROPN", "NOUN")]
        if any(t["lemma"] in YOU for t in subj):
            return True
        return not any(t["pos"] in ("PRON", "PROPN") for t in subj)   # 人を指す主語が無い → 相手（自分）に聞いている

    def pred_form(self, lemma, want):
        """述語の表記を、印（相・丁寧・否定）が want に合うものから選ぶ。"""
        c = self.psurf.get(lemma, Counter())
        # 印が全部合う表記のうち、足された機能語の少ないもの（同じなら多く出たもの）
        ok = [(-len(s), v, s) for s, v in c.items() if all(self.pmarks[s][k] == want[k] for k in L.MARKS)]
        if ok:
            return max(ok)[2]
        ok = [(-len(s), v, s) for s, v in c.items() if self.pmarks[s]["否定"] == want["否定"] and self.pmarks[s]["た"] == want["た"]]
        return max(ok)[2] if ok else None

    def yes_no_me(self, toks):
        root = next((t for t in toks if t["dep"] == "ROOT" and t["pos"] in L.PRED), None)
        if root is None:
            return None
        kids = L.kids_of(toks)
        _, qm, _ = L.layers(kids, root)
        obj = next((t["lemma"] for t in toks if t["head"] == root["i"] and t["dep"] in ("obj", "nsubj", "obl") and t["pos"] in NOUN
                    and t["lemma"] not in YOU), None)
        seen = light(self.me, ("組", obj, "される:" + root["lemma"]), sat=S.SAT_PAIR) if obj else 0.0
        yes = react(1.0, seen)[0] < react(0.0, seen)[0]
        want = dict(qm)
        want["否定"] = not yes
        surf = self.pred_form(root["lemma"], want)
        if surf is None:
            return None
        head = self.intj("はい/いいえ", "肯定" if yes else "否定")
        return (head + "、" if head else "") + surf + "。", ("はい" if yes else "いいえ"), obj, root["lemma"]

    def reply(self, text):
        doc = self.nlp(text)
        toks = CB.to_toks(doc)
        self._now = Board()
        for t in toks:
            if t["pos"] in NOUN:
                self.you[("名", t["lemma"])] += 1
                self._now[("名", t["lemma"])] += 1
        qt = D.q_type(toks, text).split(":")[0]
        kids = L.kids_of(toks)
        root = next((t for t in toks if t["dep"] == "ROOT"), None)
        if root is not None:
            _, rm, _ = L.layers(kids, root)
            self.pol["n"] += 1
            self.pol["丁寧"] += rm["丁寧"]
        # 相手に合わせる: 言い切りの丁寧の期待 = 相手のこれまでの割合（本の台詞の割合を 1 行ぶんの重みで混ぜる）
        self.layers.ctx["丁寧"] = min(max((self.pol["丁寧"] + self.base_ctx["丁寧"]) / (self.pol["n"] + 1), 1e-3), 1 - 1e-3)
        # 1. 知らない語は聞き返す
        un = self.unknown_noun(toks)
        if un is not None:
            out = dict(line=self.ask_back(un[1]), tries=0, breaks=[], sim=0, why=f"知らない語「{un[1]['text']}」（差 {un[0]:.2f}）を聞き返す")
            return self._log(text, toks, out)
        # 2. 自分のことのはい/いいえ
        if qt == "はい/いいえ" and self.about_me(toks):
            r = self.yes_no_me(toks)
            if r is not None:
                line, yn, obj, pred = r
                out = dict(line=line, tries=0, breaks=[], sim=0, why=f"自分のこと（自分の板に ({obj}, {pred}) が{'ある' if yn == 'はい' else '無い'}）")
                return self._log(text, toks, out)
        # 3. 問いの語で自分のこと、自分の板が空
        if qt == "問いの語" and self.about_me(toks) and not self.me:
            want = self._marks(toks)["丁寧"]
            cands = [(v, f) for f, v in self.dunno.items() if self.pmarks.get(f, {}).get("丁寧", False) == want]
            form = max(cands)[1] if cands else None
            if form:
                head = self.intj("問いの語", "否定")
                out = dict(line=(head + "、" if head else "") + form + "。", tries=0, breaks=[], sim=0, why="自分の板が空なので、知らないと返す")
                return self._log(text, toks, out)
        # 4. 相手が自分のことを言った: 相手の板に入れて、まだ聞いていない所を聞く
        if qt == "問いでない" and self.about_person(toks):
            r = self.person_question(toks)
            if r is not None:
                why = (f"相手の板に入れ、{r[1][2]} の枠の空き（差 {r[1][0]:.2f}）を聞く" if r[1][1] != "（繰り返し）"
                       else "相手の板に入れ、聞く所が無いので相槌と繰り返し")
                out = dict(line=r[0], tries=0, breaks=[], sim=0, why=why)
                return self._log(text, toks, out)
        # 5. それ以外: 行づくり（I32c のつなぎ・形・助詞）
        # 気づき 5: 作った行の中の小さな不自然さ（違和感）。助詞が述語に合うか、修飾を取るはずの名詞が裸か。
        # 違和感があれば、自覚の板に残し（会話をまたいで持ち越す）、その語を暗くして作り直す（介入）。直らなければ相槌で受ける。
        notes = []
        for attempt in range(self.budget + 1):
            L.set_mode("I32")
            X.Chain.L = self.layers
            CB.X2 = _Shim
            seed = Board(self.aware_keys)
            X.Board = lambda *a_, **k_: Board(seed) if not a_ and not k_ else Board(*a_, **k_)   # ring の反応の板に自覚の板を入れる
            try:
                out = super().reply(text)
            finally:
                CB.X2 = _X2
                X.Board = Board
                L.set_mode("I27x2")
            rd = CB.to_toks(self.nlp(out["line"]))
            d, where, keys = self.oddness(rd)
            pg, pb = self.odd_hist(d)
            if pg <= pb:                                    # 本の行に近い: そのまま言う
                break
            self.aware[where] += d
            for k in keys:
                self.aware_keys[k] += d
            notes.append(f"「{out['line']}」→ 違和感「{where[1]}」（{where[0]}、差 {d:.2f}、自分の行に出る割合 {pg:.2f} > 本 {pb:.2f}）")
            del self.conv["lines"][-2:]                     # 言わなかった行と、もう一度足される人の行を外す
        else:
            self.conv["lines"].append(dict(text=text, toks=toks, who="人"))
            head = self.intj("問いでない", "相槌")
            out["line"] = head + "。"
            self.conv["lines"].append(dict(text=out["line"], toks=CB.to_toks(self.nlp(out["line"])), who="ボット"))
            notes.append("直らないので相槌で受ける")
        out["why"] = "行づくり（つなぎ・述部の形つき）" + ("。" + " ／ ".join(notes) if notes else "") + \
            (f"。作り直し {len(notes) - (1 if '直らない' in notes[-1] else 0)} 回" if notes else "")
        return out

    # ---- 気づき 5 の物差し ----
    def odd_boards(self, works):
        """本の行から: (述語, 助詞) の数・述語の数、名詞が修飾を取った数・名詞の数。"""
        self.pp, self.pn, self.mod, self.nn = Counter(), Counter(), Counter(), Counter()
        for w in works:
            for ln in w["lines"]:
                T = ln["toks"]
                kids = L.kids_of(T)
                for t in T:
                    if t["pos"] in NOUN:
                        self.nn[t["lemma"]] += 1
                        if any(k["dep"] in ("acl", "amod", "nmod", "det") and k["i"] < t["i"] for k in kids[t["i"]]):
                            self.mod[t["lemma"]] += 1
                        h = T[t["head"]] if 0 <= t["head"] < len(T) else None
                        cs = [k["text"] for k in kids[t["i"]] if k["dep"] == "case"]
                        if h is not None and h["pos"] in L.PRED and cs:
                            self.pp[(h["lemma"], cs[0])] += 1
                            self.pn[h["lemma"]] += 1

    def oddness(self, toks):
        """行のいちばん不自然な所: 差 = 1 − P(助詞|述語) / 最も多い助詞の割合、または 修飾を取る割合（裸のとき）。"""
        kids = L.kids_of(toks)
        worst = (0.0, ("無", ""), [])
        for t in toks:
            if t["pos"] not in NOUN:
                continue
            n = self.nn.get(t["lemma"], 0)
            if n:
                m = self.mod[t["lemma"]] / n
                has = any(k["dep"] in ("acl", "amod", "nmod", "det") and k["i"] < t["i"] for k in kids[t["i"]])
                if not has and m > worst[0]:
                    worst = (m, ("裸の名詞", t["text"]), [("補語", t["lemma"]), ("目的語", t["lemma"])])
            h = toks[t["head"]] if 0 <= t["head"] < len(toks) else None
            cs = [k["text"] for k in kids[t["i"]] if k["dep"] == "case"]
            if h is not None and h["pos"] in L.PRED and cs and self.pn.get(h["lemma"]):
                top = self._pp_top.get(h["lemma"], 0)
                dd = 1.0 - self.pp.get((h["lemma"], cs[0]), 0) / top if top else 0.0
                if dd > worst[0]:
                    worst = (dd, ("述語に合わない助詞", t["text"] + cs[0] + h["text"]), [("補語", t["lemma"]), ("目的語", t["lemma"])])
        return worst

    BINS = 10

    def _bin(self, d):
        return min(self.BINS - 1, int(d * self.BINS))

    def odd_hist(self, d):
        """不自然さ d が、自分の作った行（self_lines）と本の行で、どれだけ出るか（10 の区切り、0.5 を足して均す）。"""
        if not self.h_gen:
            return 0.0, 1.0
        b = self._bin(d)
        ng, nb = sum(self.h_gen.values()), sum(self.h_book.values())
        return (self.h_gen[b] + 0.5) / (ng + 0.5 * self.BINS), (self.h_book[b] + 0.5) / (nb + 0.5 * self.BINS)

    def learn_self(self, works, self_lines, n=500, seed=0):
        """自分の作った行（前の実験の出力）と本の行の、不自然さの分かれ方を覚える。"""
        import random
        import re
        rng = random.Random(seed)
        lines = [ln for w in works for ln in w["lines"] if ln["toks"]]
        rng.shuffle(lines)
        self.h_book = Counter(self._bin(self.oddness(ln["toks"])[0]) for ln in lines[:n])
        self.h_gen = Counter()
        if self_lines:
            gen = re.findall(r"\*\*(.+?)\*\*", open(self_lines, encoding="utf-8").read())
            gen = list(dict.fromkeys(gen))
            for doc in self.nlp.pipe(gen, batch_size=64):
                self.h_gen[self._bin(self.oddness(CB.to_toks(doc))[0])] += 1

    def expect_oddness(self, works, n=500, seed=0):
        import random
        rng = random.Random(seed)
        lines = [ln for w in works for ln in w["lines"] if ln["toks"]]
        rng.shuffle(lines)
        vals = [self.oddness(ln["toks"])[0] for ln in lines[:n]]
        return sum(vals) / len(vals) if vals else 0.0

    def _log(self, text, toks, out):
        self.conv["lines"].append(dict(text=text, toks=toks, who="人"))
        rd = self.nlp(out["line"])
        self.conv["lines"].append(dict(text=out["line"], toks=CB.to_toks(rd), who="ボット"))
        return out


import i27x2_clause_meaning as _X2  # noqa: E402


class _Shim:
    propose2 = staticmethod(L.propose_layers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--budget", type=int, default=2)
    ap.add_argument("--say", nargs="*")
    ap.add_argument("--log")
    ap.add_argument("--self_lines", help="自分が前に作った行（例の md の **行**）。崩れた行の不自然さの分かれ方を覚える")
    ap.add_argument("--dialogue", help="i34c の会話の板（JSON）。無ければ --cache の作品から数える")
    a = ap.parse_args()
    import spacy
    nlp = spacy.load("ja_ginza")
    if a.cache.endswith(".gz"):
        import gzip
        works = json.load(gzip.open(a.cache, "rt", encoding="utf-8"))
    else:
        works = json.load(open(a.cache, encoding="utf-8"))
    bot = Bot2(nlp, works, budget=a.budget, dialogue=a.dialogue, self_lines=a.self_lines)
    print(f"（読み込み {bot.load_s:.0f} 秒）", flush=True)
    log = ["# チャットボット 2 の記録（非公開）", "", f"- 本: {len(works)} 作品、振り返りの回数 {a.budget}", ""]

    def turn(x):
        t0 = time.time()
        r = bot.reply(x)
        info = f"（{r['why']}、崩れ {'・'.join(r['breaks']) or 'なし'}、{time.time() - t0:.1f} 秒）"
        print(f"> {x}\n{r['line']}  {info}", flush=True)
        log.extend([f"- 人: {x}", f"- ボット: **{r['line']}** {info}", ""])

    for x in a.say or []:
        turn(x)
    if a.log:
        os.makedirs(os.path.dirname(os.path.abspath(a.log)), exist_ok=True)
        with open(a.log, "w", encoding="utf-8") as f:
            f.write("\n".join(log))


if __name__ == "__main__":
    main()
