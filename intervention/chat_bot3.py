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
  今の全部入り（I35l まで）: --trial --uptake <受けの板 JSON> を足す。--say が無ければ対話（空行で終わり）。
"""
from __future__ import annotations

import argparse
import json
import re
import math
from collections import Counter
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chat_bot as CB  # noqa: E402
import chat_bot2 as C2  # noqa: E402
import i33_predicate_grain as G  # noqa: E402
import i34_dialogue_board as D  # noqa: E402
import i32_function_layers as L  # noqa: E402
import i35_route_grain as R  # noqa: E402
import self_board as SB  # noqa: E402
from board_ops import react  # noqa: E402
from i27l_wave_scale import FORMAL2 as FORMAL  # noqa: E402


class Bot3(C2.Bot2):
    def __init__(self, nlp, works, pointing=None, route=None, smooth="全体", attrs=None, use_at=False, **kw):
        self.pb = R.PieceBoards(works)
        self.grain = G.CountGrain(smooth)
        if attrs:
            self.load_attrs(attrs)
        self.use_at = use_at
        for it in (json.load(open(pointing, encoding="utf-8"))["items"] if pointing else []):
            self.grain.add(self.pb.symbols(it) + self.at_sym(it), "指摘" if it["odd"] else "そのまま")
        self.route = json.load(open(route, encoding="utf-8")) if route else {}
        self.negx = None
        super().__init__(nlp, works, **kw)
        self.negx = set(self.route.get("_否定の始めの語") or R.neg_intj(works))   # 1,808 作品の板にあればそれを使う
        self.learned = []                                   # 会話で足した経験（記録用）
        self.answers = []                                   # I35x: 属性の問いへの相手の答え（記録用）
        self.confirmed = []                                 # 言ってみて、指摘されなかった所（記録用）
        self._trials = {}
        self.uptake = None                                  # 受けの板（--uptake）
        self.unresolved = []                                # 試したが、相手が受けていないので分からなかった所

    # ---- 自分の行の側 ----
    def p_point(self, pc, question=False):
        s = self.grain.scores(self.sym(pc, question))
        if "指摘" not in s:
            return 0.0
        m = max(s.values())
        z = sum(math.exp(v - m) for v in s.values())
        return math.exp(s["指摘"] - m) / z

    def sym(self, pc, question=False):
        """所の記号。I35n: その所の名詞と述語の両方を、相手が直前の行（3 行まで）で言っていれば「相手の語」の記号を足す
        （相手の言葉を取り込んだ所。経験に無い記号なので、試しの介入では言ってみて確かめる所になる）。"""
        sy = self.pb.symbols(pc, question) + self.at_sym(pc)
        if self.use_you and hasattr(self, "conv"):
            hum = [ln for ln in self.conv["lines"] if ln["who"] == "人"][-3:]
            words = {t["lemma"] for ln in hum for t in ln["toks"] if t["pos"] in ("NOUN", "PROPN", "VERB", "ADJ")}
            if pc["n"] in words and pc["p"] in words:         # 名詞も述語も相手の語（代名詞は相手と自分で指す人が違うので数えない）
                sy = sy + ["you"]
        return sy

    use_you = False                                         # --you: 相手の語の記号

    def unseen(self, pc, question=False):
        """その所の記号のうち、経験の器がまだ一度も見ていない記号（あれば、見込みは当てにならない）。"""
        return [t for t in self.sym(pc, question) if t not in self.grain.f]

    def _p_of(self, sy):
        s = self.grain.scores(sy)
        if "指摘" not in s:
            return 0.0
        m = max(s.values())
        return math.exp(s["指摘"] - m) / sum(math.exp(v - m) for v in s.values())

    def thin(self, pc, question=False):
        """経験の薄い所: 知らない記号がある（I35g）か、記号を 1 つ外すと見込みが替わり（そのまま になり）、
        その記号の経験の数が、経験の器の記号の経験の数の真ん中より少ない（その 1 つの薄い記号に決めを頼っている）。"""
        sy = self.sym(pc, question)
        if any(t not in self.grain.f for t in sy):
            return True
        cnt = sorted(sum(v.values()) for t, v in self.grain.f.items() if "&" not in t)
        med = cnt[len(cnt) // 2] if cnt else 0
        for t in sy:
            rest = [x for x in sy if x != t]
            if self._p_of(rest) <= 0.5 and sum(self.grain.f[t].values()) < med:
                return True
        return False

    def _ok(self, pc):
        w = pc["ns"] + pc["c"] + pc["ps"]
        if not hasattr(self, "aware"):
            return False
        return self.aware.get(("確かめた", w), 0) > self.aware.get(("指摘された", w), 0)

    def _tried(self, pc):
        return (pc["ns"] + pc["c"] + pc["ps"]) in self.unresolved if hasattr(self, "unresolved") else False

    def _is_q(self, toks):
        return "".join(t["text"] for t in toks).rstrip("。…").endswith(("？", "?"))

    def oddness(self, toks):
        """I34d の物差しの代わり: 行の所のうち、指摘を見込む確からしさがいちばん大きい所。"""
        if not hasattr(self, "grain") or not self.grain.N:
            return super().oddness(toks)
        best = (0.0, ("無", ""), [])
        q = self._is_q(toks)
        line = "".join(t["text"] for t in toks)
        for pc in self.pb.pieces(toks):
            p = self.p_point(pc, q)
            if self._ok(pc):                               # 前に言ってみて通じた所
                continue
            if self.trial_mode and p > self.odd_cut() and not self._tried(pc) and (self.thin(pc, q) if self.trial_mode == "thin" else self.unseen(pc, q)):   # 試しの介入: 知らない記号のある所は、言ってみて相手で確かめる
                self._trials.setdefault(line, []).append((pc, q))
                continue
            if p > best[0]:
                best = (p, ("指摘の見込み", pc["ns"] + pc["c"] + pc["ps"]), [("補語", pc["n"]), ("目的語", pc["n"])])
        return best

    def odd_hist(self, d):
        if not hasattr(self, "grain") or not self.grain.N:
            return super().odd_hist(d)
        c = self.odd_cut()
        return d, 1.0 - d if c <= 0.5 else c                # 指摘 > そのまま なら違和感（P(指摘) > 0.5）。I37f: ふだんの指摘の割合より大きい時だけ

    use_base = False                                        # I37f: 違和感の線を、経験の中の指摘のふだんの割合にする

    def odd_cut(self):
        """I37f: 違和感とみる P(指摘) の線。ふつうは 0.5。use_base では、経験の器の指摘の割合（ふだんの割合）が 0.5 より大きければそれ。
        先生がほぼ毎回何か言う場では、ほとんどの所が P(指摘) > 0.5 になり、全部が違和感になって相槌に落ちる。期待（ふだん）と比べて、それより指摘されやすい所だけを違和感とみる。"""
        if not self.use_base or not self.grain.N:
            return 0.5
        return max(0.5, self.grain.n.get("指摘", 0) / self.grain.N)

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

    asked = None                                            # I35o: 聞き返した語（相手の次の行が、その説明かもしれない）

    def explained(self, text):
        """I35o: 聞き返した語を相手が説明した: 自分の板に「知っている」と、説明の述語の部分を入れ（自覚）、その部分で受ける。"""
        toks = CB.to_toks(self.nlp(text))
        w = self.asked
        self.asked = None
        if not any(t["lemma"] == w for t in toks):
            return None
        root = next((t for t in toks if t["dep"] == "ROOT"), None)
        if root is None or root["lemma"] == w:
            return None
        self.me[("組", w, "される:知る")] += 1                # 聞いて知った
        self.me[("組", w, "である:" + root["lemma"])] += 1
        kids = L.kids_of(toks)
        keep = []                                           # 説明の述語の部分: 根と、根に係る修飾（主題の名詞とその助詞・根の助動詞は外す。修飾の中の助詞は残す）
        for t in toks:
            x, ok = t, False
            while True:
                if x["i"] == root["i"]:
                    ok = True
                    break
                if x["head"] == x["i"] or x["lemma"] == w or x["dep"] in ("nsubj", "dislocated", "aux", "cop", "mark", "punct"):
                    break
                x = toks[x["head"]]
            if ok and t["dep"] not in ("aux", "cop", "mark", "punct") and t["lemma"] != w and not (t["dep"] == "case" and t["head"] == root["i"]):
                keep.append(t["text"])
        part = "".join(keep)
        if not part:
            return None
        head = self.intj("問いでない", "相槌")
        out = dict(line=(head + "、" if head else "") + part + "。", tries=0, breaks=[], sim=0,
                   why=f"聞き返した「{w}」の説明を自分の板に入れ（{w} を知った・{w} は {root['lemma']}）、説明の述語で受ける")
        return self._log(text, toks, out)

    @staticmethod
    def _line_key(line):
        return "".join(ch for ch in line if ch not in "。、！？!?…「」 　")

    def dark_line(self, line):
        """I37a: 行のまるごとの経験。この会話で聞き返された・先生に変と言われた行は言わない（所の無い行「ああ、ならん。」「はい。」も）。
        会話をまたいでは、言った数のうち指摘された割合が、されなかった割合より大きい行だけ（「はい。」はふつう通じる）。"""
        k = self._line_key(line)
        d = self.__dict__
        if k in d.setdefault("conv_dark", set()):
            return True
        said, pt = d.setdefault("line_said", Counter())[k], d.setdefault("line_pointed", Counter())[k]
        return said > 0 and (pt + 0.1) / (said + 1.0) > 0.5

    def mark_line(self, line, w=1.0):
        self.nov_result("指摘")
        k = self._line_key(line)
        self.__dict__.setdefault("line_pointed", Counter())[k] += w
        self.__dict__.setdefault("conv_dark", set()).add(k)

    use_good = False                                        # I37b: 褒めの板と好奇心の点で返事を選ぶ

    def p_lab(self, pc, question, lab):
        s = self.grain.scores(self.sym(pc, question))
        if lab not in s:
            return 0.0
        m = max(s.values())
        return math.exp(s[lab] - m) / sum(math.exp(v - m) for v in s.values())

    def flow(self, toks, line):
        """I37d: つながりの見込み。本の隣り合う台詞で数えた受けの板で、この行が相手の行を受けている見込み P(隣) − 0.5（−0.5〜0.5）。"""
        hums = [ln for ln in self.conv["lines"] if ln["who"] == "人"]
        if self.uptake is None or not hums:
            return 0.0
        a = dict(text="「" + hums[-1]["text"] + "」", toks=hums[-1]["toks"])
        s = R.uptake_score(self.uptake, a, dict(text="「" + line + "」", toks=toks))
        return 1.0 / (1.0 + math.exp(-max(min(s, 30), -30))) - 0.5

    def value(self, toks, line):
        """I37b: 返事の見込みの点 = P(良かった) − P(指摘) + 好奇心（経験の少なさ: 2 値の見込みの事後のばらつきの上限 0.5/√(n+2)）。
        所のある行は所ごとの平均、所の無い行（「はい。」「ああ、ならん。」）は行のまるごとの経験で。"""
        return self._value(toks, line) + (self.flow(toks, line) if self.use_flow else 0.0) + self.novelty(toks)[0]

    use_flow = False
    nov = None                                              # I38a: 新規性の感覚（i38_novelty.Novelty、--novelty）

    def own_prev(self):
        """この会話の自分の前の行（新しい順）。conv の最後の行は、今言おうとしている行なので外す。"""
        return [ln["toks"] for ln in reversed(self.conv["lines"][:-1]) if ln["who"] == "ボット"]

    def novelty(self, toks):
        """I38a: 新規性の点（−0.5〜0.5）と記号ごとの (記号, 観測, 期待)。自分の直近の行で使った語・形・型を、
        本の同じ話し手が繰り返す割合より繰り返すほど減らし、使っていないものほど足す（語の無い行は足さない）。"""
        if self.nov is None:
            return 0.0, []
        return self.nov.score(toks, self.own_prev())

    def _value(self, toks, line):
        pcs = self.pb.pieces(toks)
        if pcs:
            q = self._is_q(toks)
            vs = []
            for pc in pcs:
                n = min((sum(self.grain.f[t].values()) for t in self.sym(pc, q) if t in self.grain.f), default=0)
                if any(t not in self.grain.f for t in self.sym(pc, q)):
                    n = 0
                vs.append(self.p_lab(pc, q, "良") - self.p_point(pc, q) + 0.5 / math.sqrt(n + 2))
            return sum(vs) / len(vs)
        k = self._line_key(line)
        d = self.__dict__
        said = d.setdefault("line_said", Counter())[k]
        g, pt = d.setdefault("line_good", Counter())[k], d.setdefault("line_pointed", Counter())[k]
        return (g + 0.1) / (said + 1.0) - (pt + 0.1) / (said + 1.0) + 0.5 / math.sqrt(said + 2)

    def good(self, toks, line, q=None, w=1.0):
        """I37b: 前の返事が良かった（相手が受けて話を続けた・先生が褒めた）: 所を「良」の経験に、行のまるごとも良かった行に。"""
        q = self._is_q(toks) if q is None else q
        for pc in self.pb.pieces(toks):
            self.grain.add(self.sym(pc, q), "良")
            self.aware[("良かった", pc["ns"] + pc["c"] + pc["ps"])] += w
        self.__dict__.setdefault("line_good", Counter())[self._line_key(line)] += w
        self.nov_result("良")
        if self.sb is not None and self._last_form:
            self.sb.note(self._last_form, "良")

    _last_recall_item = None
    _last_topic = None

    def reply(self, text):
        # I44b: 思い出しで答えたすぐ後に相手が「ううん・違う」と直したら、その項目を弱める（再記憶の逆）
        x = self._last_recall_item
        self._last_recall_item = None
        first = re.split(r"(?<=[。！!？?])", text.strip())[0]
        fixed = text.lstrip("「 ").startswith(("ううん", "違う", "ちがう", "いや", "え、違", "そうじゃなく")) \
            or first.rstrip().endswith(("って？", "って?"))  # I44e: 聞き返し（「履こうって？」）も直しの合図
        if x is not None and fixed:
            x["mentions"] = max(1, x.get("mentions", 1) - 2)
            x["corrected"] = x.get("corrected", 0) + 1
        tw, self._last_topic = self._last_topic, None
        if tw and getattr(self, "mem", None) is not None and hasattr(self.mem, "topic_result"):   # I45b: 話題の答えの当たり外れ
            self.mem.topic_result(tw, not fixed)
        if not self.conv["lines"]:                          # 新しい会話: この会話で指摘された行は空にする
            self.conv_dark = set()
            self.conv_asked = set()
        snap = (list(self.conv["lines"]), list(getattr(self, "trial", [])), getattr(self, "_asked_attr", None), self.asked)
        out = self._reply_dark(text)
        if self.use_good and len(self.conv["lines"]) >= 2 and not out.get("form") and not out.get("recall"):  # I37b（I39a: 自分の板で答えた時は、形の候補どうしで比べ済み）: ほかの返し方（行づくり）とも比べ、見込みの点の大きい方を言う
            v0 = self.value(self.conv["lines"][-1]["toks"], out["line"])
            keep = (out, list(self.conv["lines"]), list(self.trial), getattr(self, "_asked_attr", None), self.asked)
            self.conv["lines"] = list(snap[0])
            self._plain, ex = True, self.explore
            self.explore = True
            try:
                alt = C2.Bot2.reply(self, text)
            finally:
                self._plain, self.explore = False, ex
            alt = self._end_turn(self._name(alt))
            v1 = self.value(self.conv["lines"][-1]["toks"], alt["line"]) if not self.dark_line(alt["line"]) else -9.0
            if v1 > v0 and alt["line"] != out["line"]:
                alt["why"] = f"「{out['line']}」（点 {v0:.2f}）より、行づくりの「{alt['line']}」（点 {v1:.2f}）を言う。" + alt["why"]
                self._asked_attr, self.asked = snap[2], None
                out = alt
            else:
                out, self.conv["lines"], self.trial, self._asked_attr, self.asked = keep
                out["why"] += f"（点 {v0:.2f}、行づくり {v1:.2f}）"
            if self.nov is not None and self.novelty(self.conv["lines"][-1]["toks"])[0] < 0:
                out = self._novel_alt(out)
        return self._said(out)

    def _novel_alt(self, out):
        """I38a: 選んだ返事が自分の直近の行の繰り返し（新規性の点が負）なら、話題の語のまだ出ていない属性を聞く行
        （相手の今の行に名詞が無ければ、名詞のあった前の相手の行の語）も候補にし、点の大きい方を言う。"""
        last = self.conv["lines"][-1]
        v0 = self.value(last["toks"], out["line"])
        saved = getattr(self, "_asked_attr", None)
        for ln in reversed([x for x in self.conv["lines"][:-1] if x["who"] == "人"]):
            r = self.topic_line(ln["toks"], "問いでない")
            if r is None:
                continue
            if self.dark_line(r[0]):
                break
            cand = dict(text=r[0], toks=CB.to_toks(self.nlp(r[0])), who="ボット")
            self.conv["lines"][-1] = cand
            v1 = self.value(cand["toks"], r[0])
            if v1 > v0:
                self.trial, self.asked = [], None
                return dict(line=r[0], tries=0, breaks=[], sim=0,
                            why=f"「{out['line']}」（点 {v0:.2f}）は自分の直近の行の繰り返しなので、{r[1]}（点 {v1:.2f}）")
            self.conv["lines"][-1] = last
            break
        self._asked_attr = saved
        return out

    def _said(self, out):
        if self.sb is not None:                             # I39a: 形の経験（言った数）。結果は相手の次の行・先生の見立てで足す
            self._last_form = out.get("form")
            self._last_pair = out.get("pair")
            self._last_src = out.get("src")
            if self._last_form:
                self.sb.note(self._last_form, "言った")
                self.sb.note_pair(self._last_pair, "言った")
                self.sb.note_src(self._last_src, "言った")
        self.__dict__.setdefault("line_said", Counter())[self._line_key(out["line"])] += 1
        if self.nov is not None and getattr(self, "_asked_attr", None):   # I38a: この会話で聞いた属性（読み手が問いの行の所を取れなくても、同じ属性をまた聞かない）
            self.__dict__.setdefault("conv_asked", set()).add(self._asked_attr)
        if self.nov is not None and self.conv["lines"] and self.conv["lines"][-1]["who"] == "ボット":
            v, det = self.novelty(self.conv["lines"][-1]["toks"])
            for sy, o, _ in det:                            # 自覚の板: この返事で新しく使った・繰り返した記号（会話をまたいで持ち越す）
                self.aware[("繰り返した" if o else "新しかった", sy)] += 1.0
            self._last_nov = v
            b = self.nov_bin(v)
            self.nov_out[b]["言った"] += 1                  # 新規性の点の帯ごとの、言った数・良かった数・指摘された数（重みを学ぶ材料）
            rp = [sy for sy, o, _ in det if o]
            out["why"] += f"（新規性 {v:+.2f}" + ("、繰り返し " + "・".join(rp[:4]) if rp else "") + "）"
        return out

    _last_nov = None

    @staticmethod
    def nov_bin(v):
        return "−" if v < -0.05 else ("+" if v > 0.05 else "0")

    @property
    def nov_out(self):
        return self.__dict__.setdefault("_nov_out", {b: Counter() for b in ("−", "0", "+")})

    def nov_result(self, lab):
        """前の返事の結果（良・指摘）を、その返事の新規性の帯に数える。"""
        if self._last_nov is not None:
            self.nov_out[self.nov_bin(self._last_nov)][lab] += 1

    def _reply_dark(self, text):
        out = self._reply_once(text)
        if self.dark_line(out["line"]) and len(self.conv["lines"]) >= 2 and not out.get("recall"):   # 前に指摘された行（I40e: 思い出した中身は、言い方の指摘で捨てない）をまた言いそう: 行づくりで言い直し、直らなければ見込みの小さい行を言ってみる
            first = out
            del self.conv["lines"][-2:]
            self._plain, ex = True, self.explore
            self.explore = True
            try:
                out = C2.Bot2.reply(self, text)
            finally:
                self._plain, self.explore = False, ex
            out = self._end_turn(self._name(out))
            out["why"] = f"「{first['line']}」は前に指摘された行なので言い直す。" + out["why"]
            if self.dark_line(out["line"]):                 # 言い直しも前に指摘された行: 相手の行の話題の語を聞き返す（言えることが無い時は、相手に話してもらう）
                ns = []
                for ln in reversed(self.conv["lines"][:-1]):    # 相手の行に名詞が無い（「『はい』って？」）: 前の相手の行の話題の語
                    if ln["who"] == "人":
                        ns = [t for t in ln["toks"] if t["pos"] in ("NOUN", "PROPN") and t["lemma"] not in C2.YOU
                              and not self.dark_line(self.ask_back(t))]
                        if ns:
                            break
                if ns:
                    t = max(ns, key=lambda x: (self.idf.get(x["lemma"], 0.0), -x["i"]))
                    line = self.ask_back(t)
                    self.conv["lines"][-1] = dict(text=line, toks=CB.to_toks(self.nlp(line)), who="ボット")
                    out = dict(line=line, tries=0, breaks=[], sim=0,
                               why=out["why"] + f"。それも前に指摘された行なので、相手の話題の語「{t['text']}」を聞き返す")
                    self.trial = []
        return out

    def _reply_once(self, text):
        if self.asked:
            out = self.explained(text)
            if out is not None:
                return self._end_turn(out)
        out = self._reply(text)
        if out["why"].startswith("知らない語「"):
            self.asked = out["why"][len("知らない語「"):].split("」", 1)[0]
        return out

    def _reply(self, text):
        r = self.recall_answer(text)                        # I40h: 覚えているかの問いは、前の返事への指摘の扱いより先に見る
        if r is not None:
            self.trial = []
            return self._end_turn(self._name(r))
        bots = [ln for ln in self.conv["lines"] if ln["who"] == "ボット"]
        hums = [ln for ln in self.conv["lines"] if ln["who"] == "人"]
        if bots and self.negx is not None:
            toks = CB.to_toks(self.nlp(text))
            a = dict(text="「" + bots[-1]["text"] + "」", toks=bots[-1]["toks"])
            b = dict(text="「" + text + "」", toks=toks)
            cs = R.cues(a, b, self.negx)
            if self._is_q(a["toks"]):                       # I35x: はい/いいえの問いへの「ううん」「〜ない」は答え（指摘ではない）
                cs = [c for c in cs if c[0] not in ("否定の始め", "否定の繰り返し")]
                self.answer(a, b)
            # 自分の行の所（名詞+助詞+述語）や代名詞をそのまま「って」「？」で返されたら、その所への聞き返し。
            # 自分の行の一続き（「何を見の」）を返された時も、その中の所への聞き返し（I35o）
            chunk = ""
            for mk in ("って", "？", "?"):
                if mk in text:
                    c0 = text.split(mk, 1)[0].strip("「」『』 　")   # 「天気がなるの」って？ の閉じかっこも外す
                    if len(c0) >= 2 and c0 in bots[-1]["text"]:
                        chunk = c0
                        break
            for pc in self.pb.pieces(a["toks"]):
                for w in (pc["ns"] + pc["c"] + pc["ps"], pc["ns"]):
                    if w + "って" in text or w + "？" in text or (chunk and w in chunk and len(w) >= 2):
                        cs = [("聞き返し", pc["n"])] + [c for c in cs if c[0] != "短い問い返し"]
                        break
                else:
                    continue
                break
            self._chunk = chunk
            if chunk and not cs:                            # I37a: 所の無い行の一続き（「ならん」って？）を返された: 行への聞き返し
                cs = [("聞き返し", None)]
            trial, self.trial = self.trial, []
            self._prev_trial = {pc["ns"] + pc["c"] + pc["ps"] for pc, _ in trial}
            if cs:
                out = self.pointed(text, toks, a, cs, hums[-1]["text"] if hums else None)
                return self._end_turn(out)
            took = self.uptake is not None and R.uptake_score(self.uptake, a, b) > 0
            if self.use_good and took:                     # I37b: 相手が前の返事を受けて話を続けた: 良かった経験
                self.good(a["toks"], bots[-1]["text"])
                trial = []
            if trial and self.uptake is not None and R.uptake_score(self.uptake, a, b) <= 0:
                self.unresolved += [pc["ns"] + pc["c"] + pc["ps"] for pc, _ in trial]
                trial = []                                  # 相手の行が前の返事を受けていない: 通じたかは分からない（経験にしない）
            for pc, q in trial:                             # 言ってみた所が指摘されなかった: そのままの経験に足す
                self.grain.add(self.sym(pc, q), "そのまま")
                self.confirmed.append(pc["ns"] + pc["c"] + pc["ps"])
                self.aware[("確かめた", pc["ns"] + pc["c"] + pc["ps"])] += 1.0   # 自覚の板: 言ってみて通じた所
            if self.quiet_ok:                               # 相手が指摘しなかった: 前の返事の所を「そのまま」の経験に足す
                q = bots[-1]["text"].rstrip("。…").endswith(("？", "?"))
                for pc in self.pb.pieces(a["toks"]):
                    self.grain.add(self.sym(pc, q), "そのまま")
        r = self.recall_answer(text)                        # I40d: 「覚えてる？」には、大事な板・記憶板から
        if r is not None:
            return self._end_turn(self._name(r))
        r = self.self_answer(text)                          # I39a: 自分への問いは、自分の板（プロフィールの「自分のこと」）の所から
        if r is not None:
            return self._end_turn(self._name(r))
        return self._end_turn(self._name(self.checked_reply(text)))

    # ---- I40d: 覚えているかの問いに、大事な板・記憶板で答える ----
    imp = None                                              # memory_board.Important（--recall）
    mem = None                                              # memory_board.Memory
    RECALL_CUE = ("覚えて", "何だったっけ", "なんだったっけ", "だっけ", "言ったっけ")
    ENDS = ("なんだよね", "なんだよ", "なんだ", "んだよね", "んだよ", "んだ", "なの", "だよね", "だよ", "だから覚えといて", "だから覚えておいて", "覚えといて",
            "覚えておいて", "よ", "ね", "の", "だ", "です")

    imp_time = ("明日", "明後日", "来週", "週末", "土曜", "日曜", "誕生日")

    def _restate(self, v):
        v = v.strip("。！!…「」 ")
        segs = [x for x in v.replace("。", "、").split("、") if x]
        tseg = [x for x in segs if any(w in x for w in self.imp_time) or any(ch.isdigit() for ch in x)]
        if tseg:                                            # I40e: 予定は、時の語のある所だけ返す（「ううん、特にないよ。」の前置きを付けない）
            acc = ""                                        # I40h: 時の語のある句から、中身の語が出るところまで（後ろの「仕事の合間に…」は付けない）
            for seg in segs[segs.index(tseg[0]):]:
                acc += ("、" if acc else "") + seg
                rest = acc
                for w in self.imp_time + ("、",):
                    rest = rest.replace(w, "")
                if len([ch for ch in rest if not ch.isdigit()]) >= 3:
                    break
            v = acc.lstrip("はもがを、 ")
        for _ in range(3):
            for e in self.ENDS:
                if v.endswith(e) and len(v) > len(e) + 1:
                    v = v[: -len(e)].rstrip("、 ")
                    break
        v = v.replace("私", "").replace("うちの", "").lstrip("、 ")
        v = v.lstrip("のはもがをに、 ")
        for pre in ("そういえば", "そうそう", "ところで", "あ、", "ねえ"):   # I41h: 頭の前置き
            if v.startswith(pre):
                v = v[len(pre):].lstrip("、 ")                         # I41d: 「私の誕生日」から 私 を外した時に残る「の」
        return v

    TOPIC_CUE = ("何話し", "何を話し", "何の話", "どんな話", "何について話", "なに話し", "何をしゃべ", "何しゃべ")

    def topic_recall(self, text):
        """I45a: 「この前、何の話したか覚えてる？」「さっき何話してたっけ？」に、記憶板の段から話題の名詞で答える。"""
        if getattr(self, "mem", None) is None or self._plain:
            return None
        sents = [x for x in re.split(r"(?<=[。！!？?])", text) if x.strip()]
        q = next((x for x in reversed(sents) if any(c in x for c in self.TOPIC_CUE) and x.rstrip("。…」 ").endswith(("？", "?"))), None)
        if q is None:
            return None
        when = "さっき" if any(w in q for w in ("さっき", "今まで", "ここまで")) else "今日" if "今日" in q else \
            "昨日" if any(w in q for w in ("昨日", "きのう")) else "この前"
        toks = CB.to_toks(self.nlp(text))
        skip = {t["lemma"] for t in toks if t["pos"] in ("NOUN", "PROPN")} | {"話", "さっき", "この前", "前", "昨日", "今日", "前回"}
        if self.imp is not None:
            skip |= {x["value"] for x in self.imp.people.get("相手", []) if x["kind"] == "名前"}
        name, tp = self.mem.topics(when, skip=skip)
        self.conv["lines"].append(dict(text=text, toks=toks, who="人"))
        if tp:
            head = {"さっき": "さっきは", "今日": "今日は", "昨日": "昨日は", "この前": "この前は"}[name]
            line = head + "と".join(w for w, _ in tp) + "の話をした、だよね。"
            why = f"記憶板の「{name}」の段の明るい名詞（{'・'.join(f'{w}×{v:.2g}' for w, v in tp)}）から答える"
            if getattr(self.mem, "rehearse", False):
                self.mem.recall([w for w, _ in tp])
            if name == "さっき":                            # I45b: 次の相手の行で直されたかを見る（I46c: 話し相手は会話をまたいで覚えないので「さっき」だけ）
                self._last_topic = [w for w, _ in tp]
        else:
            line, why = "ごめん、覚えてない。", f"記憶板の「{name}」の段に話題の名詞が無い"
        self.conv["lines"].append(dict(text=line, toks=CB.to_toks(self.nlp(line)), who="ボット"))
        return dict(line=line, tries=0, breaks=[], sim=0, why=why, recall=True, topic=True, recall_value=tp[0][0] if tp else "")

    def recall_answer(self, text):
        r = self.topic_recall(text)
        if r is not None:
            return r
        # I44d: 「私の誕生日はいつか知ってる？」「うちの猫の名前、何だった？」も思い出しの問い（私の・うちの がある時だけ。「〇〇って知ってる？」は物知りの問い）
        if any(x in text for x in ("私の", "うちの")) and any(c in text for c in ("知ってる", "何だった", "なんだった")):
            for c in ("知ってる", "何だった", "なんだった"):
                text = text.replace(c, "覚えてる")
        if self.imp is None or self._plain or not any(c in text for c in self.RECALL_CUE):
            return None
        import re as _re
        qs = [x for x in _re.split(r"(?<=[。！!？?])", text) if x.strip()]
        # I40h: 目印のある文が「？」で終わる問いの時だけ（「覚えてくれたね」「覚えてるね」は問いでない）
        if not any(any(c in x for c in self.RECALL_CUE) and x.rstrip("。…」 ").endswith(("？", "?")) and "覚えてくれ" not in x for x in qs):
            return None
        toks = CB.to_toks(self.nlp(text))
        import re
        # I40f: 問いの種類と語は、思い出す目印（覚えて・だっけ）のある文だけから取る（「楽しみにするよ。ねえ、私の名前覚えてる？」の前置きを見ない）
        sents = [x for x in re.split(r"(?<=[。！!？?])", text) if x.strip()]
        q = next((x for x in reversed(sents) if any(c in x for c in self.RECALL_CUE)), text)
        for pre in ("ねえ、", "ところで、", "ところで", "あ、", "そういえば、", "そうそう、"):   # I40g: 前置きを切る
            q = q.split(pre)[-1]
        text = q
        qt = CB.to_toks(self.nlp(q))
        ws = [t["lemma"] for t in qt if t["pos"] in ("NOUN", "PROPN") and t["lemma"] not in ("私", "名前", "こと", "何")]
        items = self.imp.find("相手", ws) if ws else []
        # I40e: 問いの種類（名前・誰の名前・予定）に合う項目だけを選ぶ（合う項目が無ければ、別の項目を答えずに「覚えてない」）
        # I41d: 「私のこと、みさきって覚えてる？」: 問いの語が名前の項目の名前そのものなら、名前の問い
        names = {x["value"] for x in self.imp.people.get("相手", []) if x["kind"] == "名前"}
        if "名前" not in text and ("私のこと" in text or any(w in names for w in ws)):
            text = text + "（名前）"
        ws = [w for w in ws if w not in names]              # I44d: 「私の名前、みさきって覚えてる？」の みさき は誰の名前でなく、答えの候補
        whose = None
        wh_plan = any(w in text for w in ("どこ", "いつ", "何時", "何曜", "何日"))   # I41c: 「どこに行くか覚えてる？」は予定の問い
        if wh_plan and "名前" not in text and "誕生日" not in text:
            items = [x for x in reversed(self.imp.people.get("相手", [])) if x["kind"] in ("予定", "頼まれごと")
                     and (not ws or any(w in x["value"] for w in ws))] or \
                [x for x in reversed(self.imp.people.get("相手", [])) if x["kind"] == "予定"]
            items.sort(key=lambda x: -x.get("mentions", 1))   # I44e: 同じ問いに合う予定が 2 つあれば、多く言われた（再記憶された）方
        elif "名前" in text:
            whose = next((w for w in ws if w not in ("私", "うち", "僕", "俺", "あたし")), None)
            items = [x for x in reversed(self.imp.people.get("相手", [])) if x["kind"].startswith("名前")
                     and ((f"（{whose}）" in x["kind"]) if whose else x["kind"] == "名前")]
            if not whose:                                   # I40g: 「私の名前」は、誰のでもない名前のうち、いちばん多く言われたもの
                # I41c: 猫・犬などの名前として言われた語（「モモっていう名前の猫」）は外す
                pet = ("猫", "犬", "ペット", "うさぎ", "鳥", "ハムスター")
                others = [y["kind"] + y["value"] for y in self.imp.people.get("相手", []) if y["kind"] != "名前"]
                items = [x for x in items if not any(x["value"] in v and any(p_ in v for p_ in pet) for v in others)] or items
                items.sort(key=lambda x: -x["mentions"])
            if not items and whose:                         # 「猫の名前」を「うちの猫はモモ」のような行で言っていた
                # I41b: 「猫を飼ってる」のように名前の無い行で答えない（「猫は〇〇」「猫の名前は〇〇」の形がある時だけ）
                items = [x for x in self.imp.find("相手", [whose]) if x["kind"] == "名前（" + whose + "）" or
                         (x["kind"] == "相手のこと" and re.search(whose + r"(の名前)?は([^、。！？は]{1,8}?)(だ|です|って|っていう|よ|ね|。|$)", x["value"])
                          and not re.search(whose + r"(の名前)?は[^、。！？]*(飼|好き|いる|可愛|かわい)", x["value"]))]
        elif "予定" in text or any(w in self.imp_time for w in ws):
            items = [x for x in reversed(self.imp.people.get("相手", [])) if x["kind"] in ("予定", "頼まれごと")
                     and (not [w for w in ws if w != "予定"] or any(w in x["value"] for w in ws if w != "予定"))]
            items.sort(key=lambda x: -x.get("mentions", 1))   # I44e
        if not wh_plan and "名前" not in text and "予定" not in text and not any(w in self.imp_time for w in ws):
            items = [x for x in items if not x["kind"].startswith("名前")] if ws else \
                [x for x in reversed(self.imp.people.get("相手", [])) if x["kind"] == "名前"]   # 「私のこと覚えてる？」: 名前で答える
        self.conv["lines"].append(dict(text=sents and "".join(sents) or text, toks=toks, who="人"))
        if items:
            x = items[0]
            if getattr(self.imp, "rehearse", False):
                self.imp.touch(x, getattr(self, "conv_no", 0))
                self._last_recall_item = x
                # I44b: 記憶板は、問いの語でなく、読み出した中身の名詞（歯医者・みさき）で入れ直す
                if getattr(self, "mem", None) is not None:
                    cn = [t.lemma_ for t in self.nlp(x["value"]) if t.pos_ in ("NOUN", "PROPN") and t.lemma_ not in ("私", "こと", "名前")]
                    if cn:
                        self.mem.recall(cn)
            line = (x["value"] if x["kind"].startswith("名前") else self._restate(x["value"])) + "、だよね。"
            why = f"大事な板: {x['kind']}「{x['value'][:30]}」（会話 {'・'.join(map(str, x['convs']))}、{x['mentions']} 回）から答える"
        else:
            # I40f: 記憶板からの「〜の話は覚えてる」は、前置きの語に当たって外れが多かった（I40e の 4 回とも外れ）ので、使わない
            hits = []
            if hits:
                lv, i, k, v = hits[0]
                w = k.split("|")[1]
                line = f"{w}の話は覚えてる。"
                why = f"大事な板に無いが、記憶板の {lv} の段に「{k}」（{v:.2g}）があるので、話したことだけ言う"
            else:
                line = "ごめん、覚えてない。"
                why = "大事な板にも記憶板にも、問いの語が無い"
                if "名前" in text and whose:                # I44c: 「覚えてない」のあとに教えてもらう名前は、聞いた人・物（猫）の名前として入れる
                    self.imp.expect_whose, self.imp.expect_left = whose, 2   # 問いの行と、そのすぐ次の行
        lt = CB.to_toks(self.nlp(line))
        self.conv["lines"].append(dict(text=line, toks=lt, who="ボット"))
        return dict(line=line, tries=0, breaks=[], sim=0, why=why, recall=True, recall_value=items[0]["value"] if items else "")

    # ---- I39a: 自分の板（プロフィールの「自分のこと」）で、自分への問いに答える ----
    sb = None                                               # self_board.SelfBoard（--profile）
    _last_form = None                                       # 前の返事の形（自分の板で答えた時だけ）
    _last_src = None
    _last_pair = None                                       # I39c: 前の返事で使った近さの組（問いの語, 所の語）

    def self_answer(self, text):
        """問いの語・はい/いいえで自分のことを聞かれたら、自分の板から話題に近い所を（近さ × 重みの割合で）選び、
        形の候補（言い切り・説明・これから・問い返し・説明+これから+問い）を返事の点 + 形の経験で比べて言う。
        近い所が無ければ None（今までの返し方に任せる）。"""
        if self.sb is None or self._plain:
            return None
        toks = CB.to_toks(self.nlp(text))
        qt = D.q_type(toks, text).split(":")[0]
        wh = [t for t in toks if t["dep"] in ("nsubj", "dislocated") and t["lemma"] in SB.SKIP and t["lemma"] not in C2.YOU]
        rest = [t for t in toks if t not in wh]             # 「どっちが好き？」の「どっち」は人を指す主語ではない
        if qt not in ("問いの語", "はい/いいえ") or not self.about_me(rest):
            return None
        s = self.sb.pick(toks)
        if s is None:
            return None
        _, v, how, pc = s
        head = ""
        if qt == "はい/いいえ":                              # 問いの述語と同じ述語の所で答える時だけ、はい/いいえ の感動詞を付ける
            qp = {t["lemma"] for t in toks if t["pos"] in ("VERB", "ADJ")}
            if (qp & set(self.sb.lem[pc["p"]])) - SB.LIGHT:
                head = self.intj("はい/いいえ", "否定" if pc["p"].endswith(SB.NEG) else "肯定")
        for t in toks:
            if t["pos"] in C2.NOUN:
                self.you[("名", t["lemma"])] += 1
        self.conv["lines"].append(dict(text=text, toks=toks, who="人"))
        best, tried = None, []
        for f, line, _ in self.sb.forms(pc, head):
            if self.dark_line(line):                        # 前に指摘された行は言わない
                tried.append(f"{f} 指摘された行")
                continue
            lt = CB.to_toks(self.nlp(line))
            self.conv["lines"].append(dict(text=line, toks=lt, who="ボット"))
            sc = self.value(lt, line) + self.sb.form_score(f)
            self.conv["lines"].pop()
            tried.append(f"{f} {sc:.2f}")
            if best is None or sc > best[0]:
                best = (sc, f, line, lt)
        if best is None:
            self.conv["lines"].pop()
            return None
        sc, f, line, lt = best
        self.conv["lines"].append(dict(text=line, toks=lt, who="ボット"))
        why = (f"自分の板: 「{self.sb.surf(pc)}」（{pc['item']}・{pc['time']}、重み {pc['w']:.1f}、近さ {v:.2f}: "
               f"「{how[0]}」と「{how[1]}」が{how[2]}）、形 {f}（{'・'.join(tried)}）")
        return dict(line=line, tries=0, breaks=[], sim=0, why=why, form=f, pair=how[:2], src=["源:" + how[2], "部:" + how[3]])

    def _end_turn(self, out):
        if out.pop("explored", False):                      # I35t: 言ってみた行の所は全部、確かめ中にする
            q = out["line"].rstrip("。…").endswith(("？", "?"))
            self._trials.setdefault(out["line"], []).extend((pc, q) for pc in self.pb.pieces(self.conv["lines"][-1]["toks"]))
        self.trial = self._trials.get(out["line"], [])
        self._trials = {}
        if self.trial:
            out["why"] += "（試し: 「" + "・".join(pc["ns"] + pc["c"] + pc["ps"] for pc, _ in self.trial) + "」は経験の薄い形なので、言ってみて確かめる）"
        return out

    # ---- 介入を、行づくり以外の返し方（聞き返し・自分のこと・相手のことを聞く）にも効かせる ----
    _plain = False
    trial_mode = None                                       # --trial: 試しの介入
    trial = []                                              # 言ってみた（確かめ中の）所
    quiet_ok = False                                        # --quiet_ok: 指摘されなかった返事も経験にする

    def about_person(self, toks):
        return False if self._plain else super().about_person(toks)

    def about_me(self, toks):
        return False if self._plain else super().about_me(toks)

    def unknown_noun(self, toks):
        return None if self._plain else super().unknown_noun(toks)

    # ---- I35q: 相手の話題の名詞について、次に何を言うか（語板の属性） ----
    attrs = None                                            # --attrs: 語板のスレッドの内容語の語板（content_boards.json）
    CASE = {"する": "が", "される": "を"}                     # 語板の C の役 → 助詞（ほかの役は助詞そのもの）

    use_variants = False                                    # I36c: --reader own の時、表記ゆれの板で寄せる
    ROLE = {"が": "する", "は": "する", "も": "する", "を": "される", "に": "に", "へ": "に", "で": "で"}

    def at_sym(self, pc):
        """I35v: 語板の属性の記号。その所（名詞・役・述語）が、本の外の文（語板）でその名詞の属性としてどれだけ出るか。"""
        if not (self.use_at and self.attrs):
            return []
        e, r = self.attrs.get(pc["n"]), self.ROLE.get(pc["c"])
        if e is None or r is None:
            return ["at:" + ("名詞なし" if e is None else "役なし")]
        v = e["attrs"].get("C", {}).get(pc["p"] + "/" + r, 0.0)
        if r == "する":
            v = max(v, e["attrs"].get("B", {}).get("性質:" + pc["p"], 0.0))
        return [f"at:{R._bin(v, (1e-9, 0.01, 0.05))}"]

    def load_attrs(self, path):
        ents = json.load(open(path, encoding="utf-8"))
        self.attrs = {e["name"]: e for e in ents if e["pos"] == "名詞"}
        if self.use_variants:
            import i36_own_reader as OR
            self.variants = OR.variant_pairs(ents)

    def _norm(self, p):
        """語板の述語の書き方（よい・いい・良い）を、読み手の原形にそろえる。"""
        return self._normpos(p)[0]

    def _normpos(self, p):
        c = self.__dict__.setdefault("_normc", {})
        if p not in c:
            d = list(self.nlp(p))
            n, pos = (d[0].norm_, d[0].pos_) if len(d) == 1 else (p, "")
            var = self.__dict__.get("variants")             # I36c: 自分の読みには正規形が無いので、表記ゆれの板で寄せる
            if var:
                n = var.get(n, n)
            c[p] = (n, pos)
        return c[p]

    def said_attrs(self, n, now):
        """会話（人の行・自分の行・今の人の行）で、名詞 n についてもう出た属性: 「述語の原形/役」と「性質:原形」。"""
        got = set()
        for T in [ln["toks"] for ln in self.conv["lines"]] + [now]:
            for t in T:
                if t["lemma"] != n:
                    continue
                h = T[t["head"]] if 0 <= t["head"] < len(T) else None
                cs = [k["text"] for k in T if k["head"] == t["i"] and k["dep"] == "case"]
                if h is not None and h["i"] != t["i"] and h["pos"] in L.PRED:
                    hl = self._norm(h["lemma"])
                    got.add("性質:" + hl)                         # 「雪が多い」「空がきれい」（述語の形容詞）
                    role = ("される" if t["dep"] == "obj" or "を" in cs else
                            next((c for c in cs if c not in ("が", "は", "も", "の")), None) or "する")
                    got.add(hl + "/" + role)
                for k in T:
                    if k["head"] == t["i"] and k["pos"] == "ADJ":
                        got.add("性質:" + self._norm(k["lemma"]))
                got |= {self._norm("ある") + "/する", self._norm("いる") + "/する"}   # 話に出た物は「ある・いる」（聞くと前提を聞き直すことになる）
        if self.nov is not None:
            got |= {k for m, k in self.__dict__.get("conv_asked", ()) if m == n}
        return got

    def taught(self, has_error, odd="", model=""):
        """I35z: 先生役の見立てを、前の返事の所の経験に足す。間違いなら、先生が抜き出した所（無ければ自分でいちばん怪しいと見る所）を指摘に、
        合っていれば全部の所をそのままに。お手本の返事（model）の所もそのままの経験に足す（先生の好みを学ぶことになるので、記録は分ける）。"""
        bots = [ln for ln in self.conv["lines"] if ln["who"] == "ボット"]
        if not bots:
            return []
        toks = bots[-1]["toks"]
        q = self._is_q(toks)
        pcs = self.pb.pieces(toks)
        out = []
        if has_error:
            if self.sb is not None and self._last_form:
                self.sb.note(self._last_form, "指摘")
                self.sb.note_pair(self._last_pair, "指摘")
                self.sb.note_src(self._last_src, "指摘")
            self.sb.note_src(self._last_src, "指摘")
            self.mark_line(bots[-1]["text"])
            out.append(("行", bots[-1]["text"]))
            tgt = [pc for pc in pcs if odd and ((pc["ns"] + pc["c"] + pc["ps"]) in odd or (len(pc["ns"]) >= 2 and pc["ns"] in odd))]
            # I37c: 先生の抜き出した所が、どの所にも当たらない（言い方・つながり・「はい」だけ）なら、行のまるごとの経験だけにする
            # （前は自分でいちばん怪しい所に指摘を足していたが、先生はほぼ毎回何か言うので、普通の所まで P(指摘) 1.00 になった）
            for pc in tgt:
                w = pc["ns"] + pc["c"] + pc["ps"]
                self.grain.add(self.sym(pc, q), "指摘")
                self.aware[("指摘された", w)] += 1.0
                for k in (("補語", pc["n"]), ("目的語", pc["n"])):
                    self.aware_keys[k] += 1.0
                self.learned.append(w)
                out.append(("指摘", w))
        elif self.use_good:                                 # I37b: 先生が褒めた: 良かった経験
            self.good(toks, bots[-1]["text"], q)
            out += [("良", pc["ns"] + pc["c"] + pc["ps"]) for pc in pcs] or [("良", bots[-1]["text"])]
        else:
            for pc in pcs:
                w = pc["ns"] + pc["c"] + pc["ps"]
                self.grain.add(self.sym(pc, q), "そのまま")
                self.aware[("確かめた", w)] += 1.0
                self.confirmed.append(w)
                out.append(("そのまま", w))
        if model:
            mt = CB.to_toks(self.nlp(model))
            for pc in self.pb.pieces(mt):
                self.grain.add(self.sym(pc, self._is_q(mt)), "そのまま")
                out.append(("お手本", pc["ns"] + pc["c"] + pc["ps"]))
        self.trial = []                                     # 先生の見立てで確かめ終わった
        return out

    def answer(self, a, b):
        """I35x: 自分の属性の問いへの相手の答えを相手の板に入れる（はい = 1、いいえ = 0）。"""
        k = self.__dict__.get("_asked_attr")
        self._asked_attr = None
        if not k:
            return
        bt = b["toks"]
        first = next((t for t in bt if t["pos"] not in ("PUNCT", "SYM")), None)
        if first is not None and first["pos"] == "INTJ":   # 始めの感動詞があれば、それで決める（「うん、…なきゃ」は はい）
            neg = first["text"] in self.negx
        else:                                               # 無ければ、最初の言い切りの述語の否定
            root = next((t for t in bt if t["dep"] == "ROOT"), None)
            neg = root is not None and any(t["head"] == root["i"] and t["lemma"] in ("ない", "ぬ") and t["pos"] == "AUX" for t in bt)
        self.you[("属性",) + k] = 0.0 if neg else 1.0
        self.answers.append((k, not neg))

    def topic_line(self, toks, qt):
        """相手の行の名詞（本の板で珍しい語ほど話題とみる）について、語板の属性のうち、会話でまだ出ていないもので
        いちばん期待の大きいもの（属性の割合 × 帯の観測の割合 = 期待、観測 0 との差）を聞く。
        聞くのは一般のこと（相手の行の時制・相に合わせない）なので、述語は言い切りの原形（丁寧だけ相手に合わせる）。"""
        if self._plain or not self.attrs or qt != "問いでない" or self.no_ask:
            return None
        ns = [t for t in toks if t["pos"] == "NOUN" and t["lemma"] in self.attrs and t["lemma"] not in C2.YOU
              and t["lemma"] not in FORMAL]
        if not ns:
            return None
        t = max(ns, key=lambda x: (self.idf.get(x["lemma"], 0.0), -x["i"]))
        e = self.attrs[t["lemma"]]
        got = self.said_attrs(t["lemma"], toks)
        nb, nc = e["n_obs"].get("B", 0) or 0, e["n_obs"].get("C", 0) or 0
        if nb + nc <= 0:
            return None
        pol = self.layers.ctx["丁寧"]
        want = {k: False for k in L.MARKS}
        want["丁寧"] = react(pol, 1.0)[0] < react(pol, 0.0)[0]
        cands = Counter()
        for k, v in e["attrs"].get("C", {}).items():
            if "/" not in k:                                 # 「主語:」「目的語:」（動作の名詞の帯）は述語の属性でない
                continue
            p0, ro = k.split("/", 1)
            p = self._norm(p0)
            q = p0 if p0 in self.psurf else p
            if p + "/" + ro not in got and q in self.psurf:
                cands[(q, self.CASE.get(ro, ro), p + "/" + ro)] += v * nc / (nb + nc)
        for k, v in e["attrs"].get("B", {}).items():
            if not k.startswith("性質:"):
                continue
            p0 = k.split(":", 1)[1]
            p = self._norm(p0)
            q = p0 if p0 in self.psurf else p
            if "性質:" + p not in got and q in self.psurf:          # 言い切りに出る形容詞だけ（「小さな」は出ない）
                cands[(q, "は", "性質:" + p)] += v * nb / (nb + nc)
        seen = set()
        for (p, part, k), d in cands.most_common():
            if k in seen:
                continue
            seen.add(k)
            surf = self.pred_form(p, want)                  # 一般のことを聞くので言い切りの原形（時制・相は相手の行に合わせない）
            if surf:
                break
        else:
            return None
        tail = ("か？" if want["丁寧"] else "の？") if part != "は" else ("か？" if want["丁寧"] else "？")
        head = self.intj("問いでない", "相槌")
        line = (head + "、" if head else "") + t["text"] + part + surf + tail
        self._asked_attr = (t["lemma"], k)
        return line, f"話題の語「{t['lemma']}」の属性で、会話にまだ出ていない「{k}」（期待 {d:.2f}、観測 0）を聞く"

    # ---- I35u: 行づくりの中身の元を、本の似た行から、語板の属性（と会話）に替える ----
    attr_field = False                                      # --attr_field
    no_ask = False                                          # --no_ask: 属性を聞かない（行づくりだけを比べる時）

    def _attr_lines(self, n):
        """名詞 n の属性を、短い行（「雪が降る」「雪は白い」）にして読み手で読んだもの。重み = 属性の割合 × 帯の観測の割合。"""
        c = self.__dict__.setdefault("_alc", {})
        if n in c:
            return c[n]
        e = self.attrs[n]
        nb, nc = e["n_obs"].get("B", 0) or 0, e["n_obs"].get("C", 0) or 0
        out = []
        for k, v in e["attrs"].get("C", {}).items():
            if "/" in k:
                p, ro = k.split("/", 1)
                if p in self.psurf:
                    out.append((v * nc / (nb + nc), n + self.CASE.get(ro, ro) + p))
        for k, v in e["attrs"].get("B", {}).items():
            p = k.split(":", 1)[1]
            if k.startswith("性質:") and p in self.psurf:
                out.append((v * nb / (nb + nc), n + "は" + p))
        texts = [x for _, x in out]
        c[n] = [(w, x, CB.to_toks(d)) for (w, x), d in zip(out, self.nlp.pipe(texts))] if texts else []
        return c[n]

    def extra_lines(self, toks):
        if not (self.attr_field and self.attrs):
            return []
        out = []
        hum = [toks] + [ln["toks"] for ln in reversed(self.conv["lines"]) if ln["who"] == "人"]
        toks = next((T for T in hum if any(t["pos"] == "NOUN" and t["lemma"] in self.attrs and t["lemma"] not in FORMAL for t in T)), toks)
        for t in toks:                                      # 相手の今の行の名詞。無ければ、名詞のあった最後の人の行（話題を続ける）
            if t["pos"] == "NOUN" and t["lemma"] in self.attrs and t["lemma"] not in FORMAL:
                got = self.said_attrs(t["lemma"], [])       # 会話でもう出た属性は入れない（おうむ返しにしない）
                for w, x, tk in self._attr_lines(t["lemma"]):
                    if not (self._attr_keys(t["lemma"], tk) & got):
                        out.append((w, x, tk))
        return out

    def _attr_keys(self, n, tk):
        k = set()
        for t in tk:
            if t["lemma"] == n and 0 <= t["head"] < len(tk) and tk[t["head"]]["i"] != t["i"]:
                h = tk[t["head"]]
                cs = [c["text"] for c in tk if c["head"] == t["i"] and c["dep"] == "case"]
                hl = self._norm(h["lemma"])
                role = ("される" if t["dep"] == "obj" or "を" in cs else
                        next((c for c in cs if c not in ("が", "は", "も", "の")), None) or "する")
                k |= {hl + "/" + role, "性質:" + hl}
        return k

    def checked_reply(self, text):
        """返事を作り、行づくりでない返し方の行にも振り分けのグレインを当てる。指摘を見込む所か、自覚の板で暗い所があれば、
        その返し方をやめて行づくりで返し直す（介入）。"""
        out = C2.Bot2.reply(self, text)
        if not self.grain.N or out["why"].startswith("行づくり"):
            return out
        toks = self.conv["lines"][-1]["toks"]
        q = out["line"].rstrip("。…").endswith(("？", "?"))
        cands = []
        for pc in self.pb.pieces(toks):
            if self._ok(pc):
                continue
            p = self.p_point(pc, q)
            if self.trial_mode and p > self.odd_cut() and not self._tried(pc) and (self.thin(pc, q) if self.trial_mode == "thin" else self.unseen(pc, q)) and not self.aware.get(("指摘された", pc["ns"] + pc["c"] + pc["ps"])):
                self._trials.setdefault(out["line"], []).append((pc, q))
                continue
            cands.append((p, pc))
        worst = max(cands, key=lambda x: x[0], default=(0.0, None))
        dark = [pc for pc in self.pb.pieces(toks) if self.aware.get(("指摘された", pc["ns"] + pc["c"] + pc["ps"]))]
        if worst[0] <= self.odd_cut() and not dark:
            return out
        pc = dark[0] if dark else worst[1]
        del self.conv["lines"][-2:]
        self._plain = True
        try:
            out2 = C2.Bot2.reply(self, text)
        finally:
            self._plain = False
        why = "前に指摘された所" if dark else f"P(指摘) {worst[0]:.2f}"
        out2["why"] = f"「{out['line']}」→ 違和感「{pc['ns'] + pc['c'] + pc['ps']}」（{why}）、行づくりで返し直す。" + out2["why"]
        return out2

    def _name(self, out):
        if self.grain.N:
            out["why"] = out["why"].replace("自分の行に出る割合", "P(指摘)").replace("> 本", "> P(そのまま)")
        return out

    # ---- I46a: 聞き返し（「〜って？」）に、前の人の行に戻らず、聞き返された自分の行を言い直す ----
    use_restate = False
    _restate_head = None

    def restate_head(self):
        """本の台詞で「Xって？」（X は前の行にある）の次の行の始めの語（感動詞・副詞）。多いものから。手で置かない。"""
        if self._restate_head is None:
            c = Counter()
            for wk in self.works:
                L = wk["lines"]
                for i in range(1, len(L) - 1):
                    t = L[i]["text"].strip("「」")
                    m = re.search(r"([^、。「」！？\s]{1,12}?)って", t)
                    if not (m and t.endswith(("？", "?")) and m.group(1) in L[i - 1]["text"] and L[i + 1]["text"].startswith("「")):
                        continue
                    tk = L[i + 1]["toks"]
                    if len(tk) > 2 and tk[0]["text"] == "「" and tk[1]["pos"] in ("INTJ", "ADV") and tk[2]["text"] in ("、", "。", "…", "……"):
                        c[tk[1]["text"]] += 1
            self._restate_head = c
        return self._restate_head.most_common(1)[0][0] if self._restate_head and self._restate_head.most_common(1)[0][1] >= 2 else ""

    def restate(self, a, tgt, prev_human, sure=True):
        """自分の行のうち、聞き返された所を含む文を言い直す。
        問いだった時: 前の人の行の名詞（と助詞）を前に置き、「〜のかなって。」と、何を聞いたかを言う。
        言い切りだった時: 聞き返された所を含む文を、前の人の行の名詞で始めて「〜ってこと。」と言い直す。"""
        own = a["text"].strip("「」")
        sents = [x for x in re.split(r"(?<=[。！!？?])", own) if x.strip()]
        keys = [pc["ps"] for pc in tgt if pc["ps"]] + [pc["ns"] for pc in tgt if pc["ns"]] or \
            ([self._chunk] if getattr(self, "_chunk", "") else [])   # 所の無い行（「行くって？」）は、返された一続き
        sent = next((x for x in sents if any(k and k in x for k in keys)), None)
        if sent is None:
            return None
        st = CB.to_toks(self.nlp(sent))
        while st and (st[0]["pos"] in ("INTJ", "PUNCT") or st[0]["text"] in ("、",)):   # 頭の「はい、」を外す
            st = st[1:]
        body = "".join(t["text"] for t in st).rstrip("。！!？?…")
        if len(body) < 2:
            return None
        ctx = ""
        if prev_human:
            ht = CB.to_toks(self.nlp(prev_human))
            for i, t in enumerate(ht):                      # 前の人の行の、いちばん珍しい名詞とその助詞（猫と・海に）
                if t["pos"] in ("NOUN", "PROPN") and t["lemma"] not in C2.YOU and t["lemma"] not in body and i + 1 < len(ht) \
                        and ht[i + 1]["pos"] == "ADP" and ht[i + 1]["text"] not in ("の", "は"):
                    if not ctx or self.idf.get(t["lemma"], 0.0) > ctx[0]:
                        ctx = (self.idf.get(t["lemma"], 0.0), t["text"], ht[i + 1]["text"])
            if ctx:                                         # 同じ助詞が言い直す文にもある（「映画を、内容を見る」）: 名詞を「の」でつなぐ
                cases = {x["text"] for x in st if x["pos"] == "ADP"}
                ctx = ctx[1] + "の" if ctx[2] in cases else ctx[1] + ctx[2] + "、"
            else:
                ctx = ""
        head = self.restate_head()
        if self._is_q(st) or sent.rstrip().endswith(("？", "?")):
            b = body[:-1] if body.endswith("の") else body
            line = (head + "、" if head else "") + ctx + b + "のかなって。"
            why = "自分の問い「" + sent.strip() + "」を、相手の話（" + (ctx or "なし") + "）を前に置いて言い直す"
        else:
            if not sure:                                    # 言い切りの所が、言ってみた形・指摘の見込みの大きい形: 言い直さず、作り直す（介入）
                return None
            cl = next((x for k in keys for x in body.split("、") if k and k in x), body)   # 聞き返された所（述語を先に）を含む節だけ
            for pc in tgt[:1]:                              # 主語が前の節にある時（「理沙は…、半額になる」）は、主語を前に付ける
                if pc["ns"] and pc["ns"] not in cl and pc["ps"] and pc["ps"] in cl:
                    cl = pc["ns"] + pc["c"] + cl
            body = cl if len(cl) >= 2 else body
            line = (head + "、" if head else "") + ctx + body + "ってこと。"
            why = "自分の行「" + sent.strip() + "」を言い直す"
        if self.dark_line(line):
            return None
        self.mark_line(own)
        return dict(line=line, tries=0, breaks=[], sim=0, why=why + (f"（始めの語「{head}」は本の聞き返しの次の行から）" if head else ""))

    def pointed(self, text, toks, a, cs, prev_human):
        pcs = self.pb.pieces(a["toks"])
        hit = {w for _, w in cs if w}
        tgt = [pc for pc in pcs if pc["n"] in hit or pc["p"] in hit]
        if not tgt and pcs:                                 # 指摘された語が分からない（「え？」）: 自分でいちばん怪しいと見る所
            tgt = [max(pcs, key=self.p_point)]
        qa = self._is_q(a["toks"])
        pre_p = max((self.p_point(pc, qa) for pc in tgt), default=0.0)   # I46a: 経験を足す前の P(指摘)
        if self.sb is not None and self._last_form:          # I39a: 自分の板で答えた形が指摘された
            self.sb.note(self._last_form, "指摘")
            self.sb.note_pair(self._last_pair, "指摘")
            self.sb.note_src(self._last_src, "指摘")
        self.mark_line(a["text"].strip("「」"))              # I37a: 行のまるごとも、指摘された行として残す
        for pc in tgt:                                      # 1. 経験を足す
            self.grain.add(self.sym(pc, qa), "指摘")
            self.learned.append(pc["ns"] + pc["c"] + pc["ps"])
            self.aware[("指摘された", pc["ns"] + pc["c"] + pc["ps"])] += 1.0
            for k in (("補語", pc["n"]), ("目的語", pc["n"])):
                self.aware_keys[k] += 1.0
        if self.use_restate and any(k == "聞き返し" for k, _ in cs):   # I46a: 「〜って？」は、聞き返された自分の行を言い直す
            out = self.restate(a, tgt, prev_human, sure=pre_p <= 0.5 and not any(pc["ns"] + pc["c"] + pc["ps"] in getattr(self, "_prev_trial", ()) for pc in tgt))
            if out is not None:
                out["why"] = f"相手の聞き返し → 所「{'・'.join(pc['ns'] + pc['c'] + pc['ps'] for pc in tgt) or 'なし'}」を経験に足し、" + out["why"]
                return self._log(text, toks, out)
        r = self.pick_route([k for k, _ in cs])
        kinds = "・".join(dict.fromkeys(k for k, _ in cs))
        where = "・".join(pc["ns"] + pc["c"] + pc["ps"] for pc in tgt) or "（所なし、行「" + a["text"].strip("「」") + "」）"
        route = r[1] if r else "そのまま"
        if self.use_restate and any(k == "聞き返し" for k, _ in cs) and route == "そのまま" and prev_human:
            route = "介入"                                   # I46a: 聞き返しの行そのものに、はい/いいえで返さない（言い直せない時は作り直す）
        head = f"相手の指摘（{kinds}）→ 所「{where}」を経験に足し、自覚の板に残す。回し先 {route}"
        if route == "介入" and prev_human:
            out = self._name(self.checked_reply(prev_human))
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
        out = self._name(self.checked_reply(text))
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
    ap.add_argument("--trial", nargs="?", const="unseen", default=None, choices=("unseen", "thin"),
                    help="試しの介入: unseen=知らない記号のある所、thin=それに加えて薄い記号 1 つに頼る見込みの所は、作り直さずに言ってみて相手の次の行で確かめる")
    ap.add_argument("--uptake", help="受けの板（i35_route_grain --uptake）。試しの所は、相手の次の行が前の返事を受けている時だけ通じたとする")
    ap.add_argument("--you", action="store_true", help="相手の語の記号（相手が直前に言った語を使った所）を足す")
    ap.add_argument("--conv_share", type=float, help="場の板の中の、人の行（相手の話題）の割合（例 0.5）")
    ap.add_argument("--attrs", help="語板の属性（content_boards.json）。相手の話題の名詞の、まだ出ていない属性を聞く（I35q）")
    ap.add_argument("--smooth", default="全体", choices=("全体", "事後"), help="経験の器の寄せ方（事後 = I35r: 1 つの経験を何度も数えない）")
    ap.add_argument("--base", action="store_true", help="I37f: 違和感の線を、経験の中の指摘のふだんの割合にする")
    ap.add_argument("--flow", action="store_true", help="I37d: 返事の点に、相手の行を受けている見込み（受けの板）を足す")
    ap.add_argument("--novelty", action="store_true", help="I38a: 返事の点に新規性の感覚（本の同じ話し手の繰り返しの割合との比べ）を足す（--good が要る）")
    ap.add_argument("--good", action="store_true", help="I37b: 褒めの板（相手が受けた・先生が褒めた）と好奇心の点で返事を選ぶ")
    ap.add_argument("--explore", action="store_true", help="I35t: 作り直しが直らない時、相槌で黙らず、いちばん指摘の見込みの小さい行を言ってみて確かめる")
    ap.add_argument("--attr_field", action="store_true", help="I35u: 行づくりの場の板に、本の似た行の代わりに相手の名詞の語板の属性の行を入れる（--attrs が要る）")
    ap.add_argument("--no_ask", action="store_true", help="属性の問い（I35q）を使わない")
    ap.add_argument("--at", action="store_true", help="I35v: 所の記号に語板の属性の記号を足す（--attrs が要る）")
    ap.add_argument("--one_clause", action="store_true", help="I35w: 行づくりで節をつながず、1 つの節で言い切る")
    ap.add_argument("--reader", default="ginza", choices=("ginza", "own"), help="I36: 読み手。own = GiNZA を使わない読み手")
    ap.add_argument("--quiet_ok", action="store_true", help="相手が指摘しなかった返事の所を「そのまま」の経験として足す")
    ap.add_argument("--rounds", type=int, default=1, help="--say を何回くり返すか（会話をまたいで変わるかを見る）")
    ap.add_argument("--profile", help="I39a: プロフィール（assets/characters/*.json）の「自分のこと」から自分の板を始め、自分への問いに答える")
    ap.add_argument("--pair_xp", action="store_true", help="I39c: 近さの組（問いの語, 所の語）で答えて指摘された割合だけ、その組の近さを下げる")
    ap.add_argument("--near_min", type=float, default=0.15, help="I39b: 自分の板で、これより近さの小さい所は選ばない")
    ap.add_argument("--self_attrs", help="I39a: 自分の板の近さに使う語板（content_boards.json）。無ければ --attrs")
    ap.add_argument("--self_lines", help="与えると I34d の決め方（自分の行と本の行の分かれ方）で比べる")
    a = ap.parse_args()
    if a.reader == "ginza":
        import spacy
        nlp = spacy.load("ja_ginza")
    works = json.load(open(a.cache, encoding="utf-8"))
    if a.reader == "own":                                   # I36: GiNZA に頼らない読み手（本の行の板と語板から）
        import i36_own_reader as OR
        nlp = OR.OwnNLP(works, json.load(open(a.attrs, encoding="utf-8")) if a.attrs else None)
    Bot3.use_variants = a.reader == "own"
    bot = Bot3(nlp, works, pointing=None if a.self_lines else a.pointing, route=a.route, budget=a.budget,
               dialogue=a.dialogue, self_lines=a.self_lines, smooth=a.smooth, attrs=a.attrs, use_at=a.at)
    bot.quiet_ok = a.quiet_ok
    bot.explore = a.explore
    bot.attr_field = a.attr_field
    bot.no_ask = a.no_ask
    bot.use_good = a.good
    bot.use_flow = a.flow
    bot.use_base = a.base
    if a.novelty:
        import i38_novelty as NV
        bot.nov = NV.Novelty(NV.novelty_board(works))
    bot.use_chain = not a.one_clause
    bot.use_book = not a.attr_field
    bot.trial_mode = a.trial
    bot.use_you = a.you
    bot.conv_share = a.conv_share
    if a.uptake:
        bot.uptake = json.load(open(a.uptake, encoding="utf-8"))
    if a.profile:
        bot.sb = SB.SelfBoard(nlp, works, a.profile, attrs=SB.load_attrs(a.self_attrs) if a.self_attrs else bot.attrs)
        bot.sb.NEAR_MIN = a.near_min
        bot.sb.use_pair = a.pair_xp
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
    if not a.say:                                           # 対話: 1 行ずつ入れる（空行で終わり）。「って？」「え？」などで指摘すると経験に足す
        print("（話しかけてください。空行で終わり）", flush=True)
        while True:
            try:
                x = input("> ").strip()
            except EOFError:
                break
            if not x:
                break
            turn(x)
        a.rounds = 0
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
            for _ in range(2):                              # 指摘のあとの返事も、相手役は読む（2 回まで）
                if not bad:
                    break
                turn(bad[0]["ns"] + bad[0]["c"] + bad[0]["ps"] + "って？")
                bad = [pc for pc in bot.pb.pieces(bot.conv["lines"][-1]["toks"]) if partner.get((pc["ns"], pc["c"], pc["ps"]))]
        tally.append(c)
        log.extend([f"- 数: {json.dumps(c, ensure_ascii=False)}", ""])
        print(c, flush=True)
    if bot.unresolved:
        log.extend(["", f"- 試したが相手の行が受けていないので分からなかった所: {'・'.join(bot.unresolved)}"])
    if bot.confirmed:
        log.extend(["", f"- 言ってみて指摘されなかった所（そのままの経験に足した）: {'・'.join(bot.confirmed)}"])
    if bot.answers:
        log.extend(["", "- 属性の問いへの相手の答え（相手の板に入れた）: " + "・".join(f"{n} {k} {'はい' if y else 'いいえ'}" for (n, k), y in bot.answers)])
    if bot.learned:
        log.extend(["", f"- 会話で足した経験（指摘された所）: {'・'.join(bot.learned)}"])
    if bot.nov is not None:
        log.extend(["", "- 新規性の帯ごとの 言った・良・指摘: " + json.dumps(bot.nov_out, ensure_ascii=False)])
    if a.log:
        os.makedirs(os.path.dirname(os.path.abspath(a.log)), exist_ok=True)
        with open(a.log, "w", encoding="utf-8") as f:
            f.write("\n".join(log))


if __name__ == "__main__":
    main()
