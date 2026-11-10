"""介入の研究 I39a: 自分の板を、プロフィールの「自分のこと」から始める（自分への問いに答える）

持ち主（2026-11-10）:
  「設定したデータ→出力ではなくて、何かしら関連する内容を間接的に出力できるといいかもね」
  「語板か属性だけたくさん確率で登録しておいて、その時の話題に合わせて会話に入れる」
  「一部の回答は未来形や疑問形、回答で逸らせるかも？ 説明＋未来＋疑問みたいな」
設計: docs/プロフィールと記憶板.md（非公開の置き場）。
  1. 読み込み: assets/characters/*.json の「自分のこと」（項目 → 時 → [名詞, 助詞, 述語, 重み]）を所の一覧にする。
  2. 所を選ぶ: 問いの内容語と、所の語（名詞・述語・項目の名前・時）の近さ。同じ語なら 1（する・ある などの軽い語は 0.3）、
     違う語なら次の大きい方（軽い語どうしは使わない）:
       本: 本の行で一緒に出る度合い（正規化 PMI、0 より大きい時だけ）
       語板（--attrs の content_boards）: 名詞どうしは属性（帯 B・C）の似かた（cos）、名詞と述語は、その名詞の属性に
       その述語が出る割合（いちばん多い属性との比の平方根）。「住む」→「家」、「痛い」→「体」、「猫」→「ペット」。近さ × 重み の割合で 1 つを選ぶ（確率で、話題に合うものを）。
     近い所が 1 つも無い時だけ、今までの返し方（「いや、分からん。」など）に任せる。
  3. 形の候補: 言い切り・説明（同じ項目の所を説明に）・これから・問い返し・説明+これから+問い。
  4. 形を選ぶ: 返事の点（P(良) − P(指摘) + 好奇心 + つながり + 新規性、chat_bot3 の value）に、形ごとの経験
     （その形で言って良かった・指摘された割合）を足して、いちばん大きいもの。形の選び方は手で決めない。
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict

CONTENT = ("NOUN", "PROPN", "VERB", "ADJ", "PRON")
LIGHT = {"する", "ある", "いる", "なる", "ない", "無い", "こと", "もの", "物", "の", "中", "所", "ところ", "私", "君", "みる", "特に",
         "たい", "出る", "行く", "来る", "言う", "思う", "今", "まだ", "いつも", "これから", "これ"}   # 同じ語でも近さを薄くする語
SKIP = {"何", "どこ", "誰", "いつ", "どう", "どれ", "どんな", "なぜ", "どうして", "どの", "どっち", "どちら",
        "君", "あなた", "お前", "貴方", "あんた", "きみ", "ことゆい", "そっち", "そちら"}
NEG = ("ない", "無い", "いない")
W_ITEM, W_TIME = 0.8, 0.5                                 # 項目の名前・時の語で当たった時の近さの掛け目
FORMS = ("言い切り", "説明", "これから", "問い返し", "説明+これから+問い")


def load_attrs(path):
    return {e["name"]: e for e in json.load(open(path, encoding="utf-8")) if e["pos"] == "名詞"}


def load_profile(path):
    d = json.load(open(path, encoding="utf-8"))
    me = d.get("自分のこと") or {}
    out = []
    for item, times in me.items():
        if item.startswith("_") or not isinstance(times, dict):
            continue
        for tm, rows in times.items():
            for r in rows:
                n, c, p, w = r
                out.append(dict(item=item, time=tm, n=n, c=c, p=p, w=float(w)))
    return out


class SelfBoard:
    def __init__(self, nlp, works, profile, seed=0, attrs=None):
        self.nlp = nlp
        self.attrs = attrs or {}                            # 名詞 → 語板（content_boards の 1 項目）
        self.pcs = load_profile(profile)
        self.rng = random.Random(seed)
        lem = {}
        for s in {x for pc in self.pcs for x in (pc["n"], pc["p"], pc["item"], pc["time"])}:
            lem[s] = [t.lemma_ for t in nlp(s) if t.pos_ in CONTENT] or [s]
        self.lem = lem
        # 「…けど」「…ので」の前に だ・な が要る述語（動詞・い形容詞・た の形で終わらないもの: 「好き」「コンピュータの中」「good」）
        self.noun_end = {pc["p"]: not pc["p"].endswith(tuple("いうくぐすつぬぶむるた")) for pc in self.pcs}
        for pc in self.pcs:
            pc["words"] = [(w, 1.0) for w in lem[pc["n"]] + lem[pc["p"]]] + [(w, W_ITEM) for w in lem[pc["item"]]] + \
                [(w, W_TIME) for w in lem[pc["time"]]]
        # 本の行で一緒に出る度合い（所の語についてだけ数える）
        W = {w for pc in self.pcs for w, _ in pc["words"]}
        self.preds = {t["lemma"] for wk in works for ln in wk["lines"] for t in ln["toks"] if t["pos"] in ("VERB", "ADJ")}
        self.df, self.co, self.N = Counter(), defaultdict(Counter), 0
        for wk in works:
            for ln in wk["lines"]:
                s = {t["lemma"] for t in ln["toks"] if t["pos"] in CONTENT}
                if not s:
                    continue
                self.N += 1
                self.df.update(s)
                for w in s & W:
                    self.co[w].update(s)
        self.form_xp = {f: Counter() for f in FORMS}       # 形ごとの経験: 言った・良・指摘

    # ---- 近さ ----
    def npmi(self, q, w):
        c = self.co.get(w, {}).get(q, 0)
        if c < 2 or not self.N:
            return 0.0
        pq, pw, pqw = self.df[q] / self.N, self.df[w] / self.N, c / self.N
        if pqw >= 1.0:
            return 0.0
        return max(0.0, math.log(pqw / (pq * pw)) / -math.log(pqw))

    @property
    def idf(self):
        """語板の属性ごとの珍しさ log(名詞の数 / その属性を持つ名詞の数)。どの名詞にも付く属性（「好き」「言う」）は近さにしない。"""
        if "_idf" not in self.__dict__:
            df = Counter()
            for e in self.attrs.values():
                for b in ("B", "C"):
                    df.update(b + k for k in e["attrs"].get(b, {}) if self._key_ok(b, k))
            N = max(1, len(self.attrs))
            self._idf = {k: math.log(N / v) for k, v in df.items()}
        return self._idf

    def _key_ok(self, b, k):
        """使う属性: 帯 C は「述語/役」、帯 B は「性質:述語」で、その述語が本の言い切りに出るもの（「性質:トム」のような名前は外す）。"""
        if b == "C":
            return "/" in k and k.split("/", 1)[0] in self.preds
        return k.startswith("性質:") and k.split(":", 1)[1] in self.preds

    def _vec(self, w):
        c = self.__dict__.setdefault("_vc", {})
        if w not in c:
            e = self.attrs.get(w)
            v = {}
            if e:
                for b in ("B", "C"):
                    for k, x in e["attrs"].get(b, {}).items():
                        if self._key_ok(b, k):
                            v[b + k] = x * self.idf.get(b + k, 0.0)
            c[w] = (v, math.sqrt(sum(x * x for x in v.values())))
        return c[w]

    def link(self, n, p):
        """名詞 n の語板の属性に、述語 p が出る割合（いちばん多い属性との比の平方根）。"""
        v, _ = self._vec(n)
        m = max(v.values(), default=0.0)
        if not m:
            return 0.0
        x = max([y for k, y in v.items() if k.startswith("C") and k[1:].split("/", 1)[0] == p] + [v.get("B性質:" + p, 0.0)])
        return math.sqrt(x / m)

    def board(self, q, w):
        """語板での近さ: 名詞どうしは属性の cos、名詞と述語は link。"""
        if q in self.attrs and w in self.attrs:
            (a, na), (b, nb) = self._vec(q), self._vec(w)
            if na and nb:
                return sum(x * b.get(k, 0.0) for k, x in a.items()) / (na * nb)
            return 0.0
        if w in self.attrs:                                 # 問いの述語 → 所の名詞（「住む」→「家」）。逆向き（問いの名詞 → 所の述語）は、
            # 「猫」→「好き」のように、どの所の「好き」にも当たってしまうので使わない
            return self.link(w, q)
        return 0.0

    def close(self, qs, pc):
        best, how = 0.0, None
        for q in qs:
            for w, k in pc["words"]:
                if q == w:
                    v = 0.3 if q in LIGHT else 1.0
                else:                                       # 軽い語どうしは、本で一緒に出ても近さにしない（何とでも一緒に出る）
                    v1 = 0.0 if (q in LIGHT or w in LIGHT) else self.npmi(q, w)
                    v2 = 0.0 if (q in LIGHT or w in LIGHT) else self.board(q, w)
                    v, src = (v1, "本で一緒に出る") if v1 >= v2 else (v2, "語板で近い")
                v *= k
                if v > best:
                    best, how = v, (q, w, "同じ語" if q == w else src)
        return best, how

    def near(self, toks):
        qs = {t["lemma"] for t in toks if t["pos"] in CONTENT and t["lemma"] not in SKIP}
        out = []
        for pc in self.pcs:
            v, how = self.close(qs, pc)
            if v > 0:
                out.append((v * pc["w"], v, how, pc))
        return out

    NEAR_MIN, NEAR_SHARE = 0.15, 0.6                      # 近さがこれより小さい所・いちばん近い所の 6 割に届かない所は選ばない

    def pick(self, toks):
        cs = self.near(toks)
        top = max((x[1] for x in cs), default=0.0)
        cs = [x for x in cs if x[1] >= max(self.NEAR_MIN, self.NEAR_SHARE * top)]
        if not cs:
            return None
        s = self.rng.choices(cs, weights=[x[0] for x in cs])[0]
        return s

    # ---- 形 ----
    def surf(self, pc):
        return pc["n"] + pc["c"] + pc["p"]

    def _plain(self, pc):                                  # 「…ので」「…けど」の前の形（名詞で終わる述語には だ・な を足す）
        return self.surf(pc), self.noun_end.get(pc["p"], False)

    def forms(self, pc, head=""):
        same = [x for x in self.pcs if x["item"] == pc["item"] and x is not pc]
        fut = next((x for x in sorted(same, key=lambda x: -x["w"]) if x["time"] == "これから"), None)
        expl = [x for x in sorted(same, key=lambda x: -x["w"]) if x["time"] != "これから"]
        if pc["time"] == "これから":
            fut, base = pc, (expl[0] if expl else None)
        else:
            base = pc
        h = (head + "、") if head else ""
        out = []
        if base is not None:
            out.append(("言い切り", h + self.surf(base) + "。", [base]))
            s, ne = self._plain(base)
            out.append(("問い返し", h + s + ("だ" if ne else "") + "けど、君は？", [base]))
            e = next((x for x in expl if x is not base), None)
            if e is not None:
                es, ene = self._plain(e)
                out.append(("説明", h + es + ("なので" if ene else "ので") + "、" + self.surf(base) + "。", [e, base]))
        if fut is not None:
            out.append(("これから", h + "これから" + self.surf(fut) + "。", [fut]))
            if base is not None:
                s, ne = self._plain(base)
                out.append(("説明+これから+問い", h + s + ("なので" if ne else "ので") + "、" + self.surf(fut) + "。君は？", [base, fut]))
        return out

    def form_score(self, f):
        c = self.form_xp[f]
        return (c["良"] + 0.1) / (c["言った"] + 1.0) - (c["指摘"] + 0.1) / (c["言った"] + 1.0)

    def note(self, f, lab):
        if f in self.form_xp:
            self.form_xp[f][lab] += 1
