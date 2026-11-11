"""介入の研究 I40a: 記憶板（行の板 → 話題・会話・日・週の段）を書く（融合の忘れはまだ無し）

持ち主（2026-11-10〜11）:「会話を行にする → 行板作る → 記憶板へ反映」「融合で 秒 → 分 → 時 → 日 → 週」
  置き場は PC の非公開フォルダ（暗号化はクラウドに置く時だけ）。試しでは 1 会話 = 1 時、20 会話 = 1 日（将来は実時間）。
設計: docs/記憶板の設計.md、図 figs/mech_14_memory_board.png（非公開の置き場）。
進める順の 1（I40a）: 行の板を作り、下の段が閉じたら和で上の段を 1 枚作る。
進める順の 2（I40b、--forget）: 上の段に足すのは、期待より明るい鍵だけ（忘れる）。
  観測 = その段での鍵の明るさ 1 − exp(−数 / 飽和)。飽和は段ごとに 2 倍（分 1・時 2・日 4。親の語板の住所で深さごとに 0.5 倍で
  受け継ぐのと同じ形）なので、上の段ほど、何度も出た鍵でないと明るくならない。期待 = ふだんの会話でその段の行の数のうちに出る見込み 1 − exp(−割合 × 行の数)。
  ふだんの割合 = (本の台詞での数 + これまでの行の板での数) / (本の行の数 + これまでの行の数)。
  差が「発見」（期待より明るい）で、|1 − 差| < |0 − 差|（残す方が近い）の鍵だけを上の段に足す。上の段ほど行の数が多いので、
  期待が大きくなり、よく出た所だけが明るく残る。
  持ち主（2026-11-11）「完全に消えなくて OK、多少時間と共に薄れるくらいで OK」: 残す方に入らなかった鍵も消さず、
  上の段には半分の重さ（FADE）で足す。1 度だけ出た鍵は 時 1 → 日 0.5 → 週 0.25 と薄れていき、何度も出た鍵は薄れない。
  行の板: 所 ("組", 名詞, 助詞:述語)、話題の名詞 ("名", 名詞)、誰、時刻、会話、結果（相手の次の行で 良・指摘）
  段: 分 = 話題の名詞が替わるまで、時 = 1 会話、日 = DAY 会話、週 = 7 日
"""
from __future__ import annotations

import json
import re
import os
import time
import math
from collections import Counter

from board_ops import react

LEVELS = ("分", "時", "日", "週")
NOUN = ("NOUN", "PROPN")


def line_board(toks, pieces, who, conv, t=None):
    keys = Counter()
    for pc in pieces:
        keys[f"組|{pc['n']}|{pc['c']}:{pc['p']}"] += 1
    nouns = [x["lemma"] for x in toks if x["pos"] in NOUN]
    for n in nouns:
        keys[f"名|{n}"] += 1
    return dict(who=who, conv=conv, t=t if t is not None else time.time(), keys=dict(keys), nouns=nouns, result=None)


class Memory:
    DAY = 20                                               # 試し: 20 会話 = 1 日
    WEEK = 7

    def __init__(self, path=None):
        self.path = path
        self.lines = []                                     # 行の板（秒）
        self.open = {lv: Counter() for lv in LEVELS}        # 閉じていない段の板
        self.boards = {lv: [] for lv in LEVELS}             # 閉じた段の板: dict(keys, from, to, n)
        self._span = {lv: [None, None, 0] for lv in LEVELS}
        self._topic = set()
        self.n_conv = 0
        self.forget = False
        self.base, self.base_n = Counter(), 0               # 本の台詞での鍵の数と行の数（--forget のふだん）
        self.seen, self.seen_n = Counter(), 0               # これまでの行の板での鍵の数と行の数
        if path and os.path.exists(path):
            self.load(path)

    # ---- 書く ----
    def add(self, toks, pieces, who, conv, lb=None):
        lb = lb or line_board(toks, pieces, who, conv)
        nouns = set(lb["nouns"])
        if who == "人" and nouns and self._topic and not (nouns & self._topic):   # 話題の名詞が替わった: 分の段を閉じる
            self.close("分")
        if who == "人" and nouns:
            self._topic = nouns if not (nouns & self._topic) else (self._topic | nouns)
        self.lines.append(lb)
        self.seen.update(lb["keys"].keys())
        self.seen_n += 1
        self._push("分", lb["keys"], lb["t"])
        return lb

    def result(self, lab):
        """前のボットの行の結果（相手の次の行で 良・指摘）。"""
        for lb in reversed(self.lines):
            if lb["who"] == "ボット":
                lb["result"] = lab
                return

    def _push(self, lv, keys, t, n=1):
        self.open[lv].update(keys)
        sp = self._span[lv]
        sp[0] = t if sp[0] is None else sp[0]
        sp[1] = t
        sp[2] += n

    def close(self, lv):
        """段を閉じて 1 枚にし、上の段に足す（親の語板 = 子の語板の和、と同じ）。"""
        if not self.open[lv]:
            return None
        sp = self._span[lv]
        b = dict(keys=dict(self.open[lv]), start=sp[0], end=sp[1], n=sp[2])
        self.boards[lv].append(b)
        i = LEVELS.index(lv)
        if i + 1 < len(LEVELS):
            up = self.kept(b["keys"], b["n"], 2.0 ** i) if self.forget else b["keys"]
            b["faded"] = self.__dict__.pop("_faded", 0)
            self._push(LEVELS[i + 1], up, b["end"], b["n"])
        self.open[lv] = Counter()
        self._span[lv] = [None, None, 0]
        if lv == "分":
            self._topic = set()
        return b

    def set_base(self, works, pieces):
        """ふだんの会話: 本の台詞の行の板の鍵（pieces = 所を取る関数）。"""
        for wk in works:
            for ln in wk["lines"]:
                ks = line_board(ln["toks"], pieces(ln["toks"]), "", 0, 0)["keys"]
                self.base.update(ks.keys())
                self.base_n += 1

    def expect(self, k, n):
        r = (self.base.get(k, 0) + self.seen.get(k, 0) + 0.5) / (self.base_n + self.seen_n + 1.0)
        return 1.0 - math.exp(-r * n)

    FADE = 0.5                                              # 残す方に入らなかった鍵を、上の段に足す時の重さ（薄れる）

    def kept(self, keys, n, sat=1.0):
        out, faded = {}, 0
        for k, v in keys.items():
            d, kind = react(self.expect(k, n), 1.0 - math.exp(-v / sat))
            if kind == "発見" and abs(1.0 - d) < abs(0.0 - d):
                out[k] = v
            else:
                out[k] = v * self.FADE
                faded += 1
        self._faded = faded
        return out

    def end_conv(self):
        self.close("分")
        self.close("時")
        self.n_conv += 1
        if self.n_conv % self.DAY == 0:
            self.close("日")
            if (self.n_conv // self.DAY) % self.WEEK == 0:
                self.close("週")
        if self.path:
            self.save(self.path)

    # ---- 読む（I40d で返事に使う。今は数えるだけ） ----
    def recall(self, words):
        """新しい段から順に、語のある鍵を探す: [(段, 何番目, 鍵, 数)]。"""
        out = []
        for lv in LEVELS:
            for i in range(len(self.boards[lv]) - 1, -1, -1):
                for k, v in self.boards[lv][i]["keys"].items():
                    if any(f"|{w}|" in k + "|" for w in words):
                        out.append((lv, i, k, v))
        return out

    def summary(self):
        return {lv: dict(枚=len(self.boards[lv]), 鍵=sum(len(b["keys"]) for b in self.boards[lv]),
                         薄れた=sum(b.get("faded", 0) for b in self.boards[lv])) for lv in LEVELS} | dict(行=len(self.lines))

    # ---- 置き場（非公開。PC のフォルダに JSON） ----
    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        d = dict(lines=self.lines, boards=self.boards, open={k: dict(v) for k, v in self.open.items()},
                 span=self._span, n_conv=self.n_conv, day=self.DAY)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False)

    def load(self, path):
        d = json.load(open(path, encoding="utf-8"))
        self.lines, self.boards, self.n_conv = d["lines"], d["boards"], d["n_conv"]
        self.open = {k: Counter(v) for k, v in d["open"].items()}
        self._span = d["span"]


# ---- I40c: 大事な板（相手ごと、薄めない） ----
TIME_W = ("明日", "明後日", "あさって", "来週", "再来週", "今週末", "週末", "来月", "今度", "今夜", "今晩", "午前", "午後",
          "月曜", "火曜", "水曜", "木曜", "金曜", "土曜", "日曜", "誕生日")
ASK_KEEP = ("覚えて", "覚えと", "お願い", "忘れないで", "してほしい", "して欲しい", "教えて")
NAME_CUE = ("名前は", "って呼んで", "と呼んで", "っていう", "って言う", "って名前", "という名前", "です。私", "私は")


def _span(toks, i):
    """i の名詞から、続く名詞・固有名詞をつないだ表記（「言」「結」→「言結」）。"""
    j = i
    while j + 1 < len(toks) and toks[j + 1]["pos"] in NOUN and toks[j + 1]["head"] in (toks[j]["i"], toks[j + 1]["head"]):
        j += 1
    return "".join(t["text"] for t in toks[i:j + 1])


class Important:
    """大事な板: 相手ごとに、名前・予定・頼みごと・相手のこと を、薄めずに置く（初めは目印で入れる。後で学ぶ）。"""

    def __init__(self, path=None):
        self.path = path
        self.people = {}                                    # 相手 → [項目]
        if path and os.path.exists(path):
            self.people = json.load(open(path, encoding="utf-8"))

    def kinds(self, text, toks):
        out = []
        # I40f: 問いの文（「覚えてる？」「〜って？」）は外し、同じ行の事実の文（「来週海に行くんだ。一緒に行く？」の前）だけを見る
        sents = [x for x in re.split(r"(?<=[。！!？?])", text) if x.strip()]
        keep = [x for x in sents if not x.rstrip("。！!…」 ").endswith(("？", "?"))
                and not x.lstrip("うんええそう、 ").startswith(("正解", "当たり", "覚えてくれ"))]   # I40i: 答え合わせの文だけ外す
        if not keep:
            return out
        if len(keep) < len(sents):
            text = "".join(keep)
            toks = [t for t in toks if t["text"] in text]
        if not text.strip("。！!、 あそういえば"):   # 答え合わせの受けは、大事なことではない
            return out
        nouns = [t for t in toks if t["pos"] in NOUN]
        # I41f: 飼っている物を覚えておく（「猫を飼ってるよ」→ 次の行の「モモっていう名前なの」を猫の名前にする）
        if "飼" in text:
            pet = [t["lemma"] for t in nouns if any(k["head"] == t["i"] and k["text"] == "を" for k in toks)
                   or text.split("飼ってる", 1)[-1].startswith(t["text"])]
            if pet:
                self.last_pet = pet[0]
        if any(c in text for c in NAME_CUE):
            nm, whose = [], ""
            if "名前は" in text:
                k = text.index("名前は")
                nm = [text[k + 3:].split("。")[0].split("、")[0].split("だ")[0].split("です")[0].strip("「」 ")]
                pre = text[:k].rstrip("の")                 # 「うちの猫の名前は」の 猫（相手のでなければ、誰の名前かを付ける）
                own = [t for t in toks if t["pos"] in NOUN and pre.endswith(t["text"])]
                if own and own[-1]["lemma"] not in ("私", "僕", "俺", "うち", "あたし"):
                    whose = own[-1]["lemma"]
            for cue in ("っていう", "って言う", "って名前", "という名前", "って呼んで", "と呼んで"):
                if not nm and cue in text and not text.split(cue, 1)[1].startswith(("意味", "こと", "感じ", "話")):   # 「脱ぐっていう意味」は名前でない
                    pre = text.split(cue, 1)[0]
                    w = [t for t in toks if t["pos"] in NOUN + ("PRON",) and pre.endswith(t["text"])]
                    if w and w[-1]["lemma"] not in ("私", "名前", "こと"):
                        nm = [w[-1]["text"]]
                        # I40g: 誰の名前かは「うちの猫」「猫の名前」のように の でつながる名詞か、前の文で飼っている物（「猫を飼ってるよ。モモって言うんだ」）から
                        pre_t = pre
                        for o in ("うちの", "私の", "うちで飼ってる", "飼ってる"):
                            if o in text:
                                aft = text.split(o, 1)[1]
                                own = [t for t in toks if t["pos"] in NOUN and aft.startswith(t["text"])]
                                if own and not whose:
                                    whose = own[0]["lemma"]
                        if not whose and "飼" in text:      # I40i: 「モモっていうの。かわいい猫を飼ってるんだ」のように後ろの文でも
                            own = [t for t in toks if t["pos"] in NOUN and t["i"] != w[-1]["i"] and any(
                                k["head"] == t["i"] and k["text"] == "を" for k in toks)]
                            if own:
                                whose = own[-1]["lemma"]
                        if not whose and "飼" in text and getattr(self, "last_pet", None) and self.last_pet != w[-1]["lemma"]:
                            whose = self.last_pet                # I41f: 前の行で飼っていると言った物
            if not nm and any(c in text for c in NAME_CUE):
                nm = [_span(toks, t["i"]) for t in toks if t["pos"] == "PROPN"]
            if nm and nm[0]:
                out.append(("名前" + (f"（{whose}）" if whose else ""), nm[0]))
        tw = [t["text"] for t in toks if t["text"] in TIME_W or (t["pos"] == "NUM" and any(
            k["text"] in ("時", "日", "月") for k in toks if k["i"] == t["i"] + 1))]
        content = [t for t in toks if t["pos"] in NOUN and t["text"] not in TIME_W and t["lemma"] not in ("こと", "時", "日", "月", "予定", "私", "うち")
                   and t["pos"] != "NUM"]
        if tw and content:                                  # I40h: 時の語のほかに中身の名詞がある時だけ（「うん、明日のこと。」は入れない）
            out.append(("予定", text))
        if any(c in re.sub(r"覚えて(る|くれ|た|い)", "", text) for c in ASK_KEEP) and not text.rstrip("。！!…").endswith(("？", "?")):   # 「覚えてる？」は問い（頼みではない）
            out.append(("頼まれごと", text))
        if not out and any(t["lemma"] in ("私", "僕", "俺", "あたし", "うち") for t in toks) and any(t["lemma"] not in ("私", "僕", "俺", "あたし", "うち") for t in toks if t["pos"] in NOUN + ("PRON",)):
            out.append(("相手のこと", text))
        return out

    def add(self, who, text, toks, pieces, conv):
        ks = self.kinds(text, toks)
        items = self.people.setdefault(who, [])
        got = []
        for kind, val in ks:
            keys = sorted({f"組|{pc['n']}|{pc['c']}:{pc['p']}" for pc in pieces} | {f"名|{t['lemma']}" for t in toks if t["pos"] in NOUN})
            def same(x):                                    # I40f: 同じ予定（鍵の半分以上が同じ）は 1 項目にまとめる
                if x["kind"] != kind:
                    return False
                if x["value"] == val:
                    return True
                a, b = set(keys), set(x["keys"])
                if kind == "予定":                          # I40h: 同じ時の語と同じ中身の名詞があれば同じ予定（言い方が違っても）
                    ta = {w for w in TIME_W if w in val}
                    tb = {w for w in TIME_W if w in x["value"]}
                    na = {k for k in a if k.startswith("名|") and k.split("|")[-1] not in TIME_W}
                    if ta & tb and na & b:
                        return True
                return kind == "予定" and bool(a) and len(a & b) / len(a | b) >= 0.5
            hit = next((x for x in items if same(x)), None)
            if hit:
                hit["mentions"] += 1
                if kind == "予定":
                    hit["value"], hit["keys"] = val, sorted(set(hit["keys"]) | set(keys))
                hit["convs"] = sorted(set(hit["convs"]) | {conv})
            else:
                items.append(dict(kind=kind, value=val, keys=keys, convs=[conv], t=time.time(), mentions=1, done=False))
            got.append((kind, val))
        return got

    def find(self, who, words):
        """相手の大事な板で、語を鍵に持つ項目（新しい順）。"""
        out = []
        for x in reversed(self.people.get(who, [])):
            if any(any(k.endswith("|" + w) or f"|{w}|" in k for k in x["keys"]) or w in x["value"] for w in words):
                out.append(x)
        return out

    def save(self, path=None):
        path = path or self.path
        if not path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.people, f, ensure_ascii=False, indent=1)


# ---- I41a: 大事かどうかを会話から学ぶ（大事さのグレイン） ----
def _bin(v, cuts):
    for i, c in enumerate(cuts):
        if v < c:
            return str(i)
    return str(len(cuts))


LIGHT_N = ("私", "僕", "俺", "あたし", "うち", "君", "あなた", "こと", "もの", "名前", "何", "予定", "今日", "明日", "感じ", "話")


class ImportanceLearner:
    """人の行を記号にし、後で相手がその中身を「覚えてる？」と聞いたら 大事、聞かれないまま AGE 会話たったら ふつう、の経験にする。
    記号は目印そのもの（時の語・名前の言い方・頼み）に加えて、名詞の珍しさ・新しさ・自分のことか・行の長さ。
    どの記号が大事さとつながるかは手で決めず、経験の器（I33 の CountGrain）が数える。"""
    AGE = 3                                                 # この会話の数だけ聞かれなければ ふつう
    W_IMP = 3                                               # I41d: 大事の経験 1 つを何回ぶんに数えるか

    def __init__(self, works=None):
        import i33_predicate_grain as G
        self.grain = G.CountGrain("事後")
        self.pending = []                                   # まだ結果の分からない人の行
        self.df, self.n_lines = Counter(), 0
        for wk in works or []:
            for ln in wk["lines"]:
                self.n_lines += 1
                self.df.update({t["lemma"] for t in ln["toks"] if t["pos"] in NOUN})
        self.seen = Counter()                               # この相手との会話に出た名詞
        self.asked_n = Counter()                            # I41b: 「覚えてる？」で聞かれた名詞
        self.log = []

    def symbols(self, text, toks):
        # I41b: 問いの文（「君は明日何するの？」）は、言った事実の大事さを下げないように記号から外す
        sents = [x for x in re.split(r"(?<=[。！!？?])", text) if x.strip()]
        fact = "".join(x for x in sents if not x.rstrip().endswith(("？", "?"))) or text
        qa = fact != text
        nouns = [t["lemma"] for t in toks if t["pos"] in NOUN]
        fn = [n for n in nouns if n in fact and n not in LIGHT_N]
        text = fact
        sy = []
        # I41b: I41a は言い方（名乗りの形）を学んでいた。中身の名詞そのもの（名詞:歯医者）と、
        # 前に聞かれた名詞を含むか（聞かれた名詞:あり）を記号に足す。
        sy += ["名詞:" + n for n in sorted(set(fn))[:4]]
        sy.append("聞かれた名詞:" + ("あり" if any(self.asked_n.get(n) for n in fn) else "なし"))
        sy.append("時:" + ("あり" if any(w in text for w in TIME_W) else "なし"))
        sy.append("数:" + ("あり" if any(t["pos"] == "NUM" for t in toks) else "なし"))
        sy.append("頼み:" + ("あり" if any(c in text for c in ASK_KEEP) else "なし"))
        sy.append("固有:" + ("あり" if any(t["pos"] == "PROPN" for t in toks) else "なし"))
        sy.append("問い:" + ("あり" if qa else "なし"))
        rare = max((math.log((self.n_lines + 1) / (self.df.get(n, 0) + 1)) for n in nouns), default=0.0)
        sy.append("珍しさ:" + _bin(rare, (4.0, 6.0, 8.0)))
        new = sum(1 for n in nouns if not self.seen.get(n)) / len(nouns) if nouns else 0.0
        sy.append("新しさ:" + _bin(new, (0.34, 0.67, 1.0)))
        return sy, nouns

    MIN_IMP = 8                                             # I41g: 大事の経験がこの数（重みを外した数）たまるまでは見込みを出さない

    def p_imp(self, sy):
        if self.grain.n.get("大事", 0) < self.MIN_IMP * self.W_IMP or not self.grain.n.get("ふつう"):
            return None
        sc = self.grain.scores(sy)
        # I41b: 大事 20 対 ふつう 175 のように偏るので、全体の割合（前の見込み）を外して、記号の比だけで比べる
        N = self.grain.N
        sc = {k: v - math.log(self.grain.n[k] / N) + math.log(0.5) for k, v in sc.items()}
        m = max(sc.values())
        z = sum(math.exp(v - m) for v in sc.values())
        return math.exp(sc.get("大事", -1e9) - m) / z

    def see(self, text, toks, conv, turn):
        sy, nouns = self.symbols(text, toks)
        p = self.p_imp(sy)
        self.pending.append(dict(text=text, sy=sy, nouns=nouns, conv=conv, turn=turn, p=p))
        self.seen.update(nouns)
        return p, sy

    def asked(self, words, value, conv, turn):
        """「覚えてる？」の問い: 前の人の行のうち、問いの語か、思い出した答えの中身を含む行を 大事 の経験にする。"""
        hit = []
        self.asked_n.update(w for w in (words or []) if w not in LIGHT_N)
        for c in list(self.pending):
            if (c["conv"], c["turn"]) >= (conv, turn):
                continue
            if (words and set(words) & set(c["nouns"])) or (value and value in c["text"]):
                for _ in range(self.W_IMP):                 # I41d: 聞かれた 1 回が、ふつうの経験に埋もれないように重く
                    self.grain.add(c["sy"], "大事")
                self.pending.remove(c)
                hit.append(c)
                self.log.append(("大事", c["text"], c["p"]))
        return hit

    def end_conv(self, conv):
        for c in list(self.pending):
            if conv - c["conv"] >= self.AGE:
                self.grain.add(c["sy"], "ふつう")
                self.pending.remove(c)
                self.log.append(("ふつう", c["text"], c["p"]))


def replay(path, works, pieces, forget=True, day=None):
    """I40b: 書いた記憶板（I40a の JSON）の行の板を、もう一度畳み直す（忘れの有無を、話し相手なしで比べる）。"""
    d = json.load(open(path, encoding="utf-8"))
    m = Memory()
    m.forget = forget
    if day:
        m.DAY = day
    if forget:
        m.set_base(works, pieces)
    conv = None
    for lb in d["lines"]:
        if conv is not None and lb["conv"] != conv:
            m.end_conv()
        conv = lb["conv"]
        m.add(None, None, lb["who"], lb["conv"], lb=dict(lb))
    if conv is not None:
        m.end_conv()
    return m


def main():
    import argparse
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import i35_route_grain as R
    ap = argparse.ArgumentParser(description="I40b: 記憶板を薄れありで畳み直し、段ごとの鍵の数と、日・週の板の明るい鍵を出す")
    ap.add_argument("--memory", required=True, help="I40a の記憶板 JSON")
    ap.add_argument("--cache", required=True, help="本の解析済み JSON（ふだんの会話の割合）")
    ap.add_argument("--out", help="畳み直した記憶板の JSON（非公開）")
    ap.add_argument("--top", type=int, default=30)
    a = ap.parse_args()
    works = json.load(open(a.cache, encoding="utf-8"))
    pb = R.PieceBoards(works)
    for fg in (False, True):
        m = replay(a.memory, works, pb.pieces, forget=fg)
        print(("薄れあり" if fg else "薄れなし"), json.dumps(m.summary(), ensure_ascii=False))
        for lv in ("時", "日"):
            if m.boards[lv]:
                b = m.boards[lv][-1]
                top = sorted(b["keys"].items(), key=lambda x: -x[1])[:a.top]
                print(f"  {lv}（最後の 1 枚、鍵 {len(b['keys'])}）:", "・".join(f"{k}×{v}" for k, v in top))
        if fg and a.out:
            m.save(a.out)


if __name__ == "__main__":
    main()
