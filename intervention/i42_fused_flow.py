"""介入の研究 I42a: 述語の後ろの流れ（助動詞・接続助詞・補助動詞）を、融合の板で文脈から学び、逆引きできるか

持ち主（2026-11-11）:
  「連用形などを合体したもので文の流れを学習、逆引きできるかやってほしい」
  「歩いた → 歩い（連用形）＋た（過去・完了）＝ 融合した板で文脈を学ぶ」
  「歩いている → 歩い（連用形）＋て（接続）→ 融合一時語板（歩いて）＋いる（進行）→ 融合一時語板（歩いている）。ネスト文を渡して学習」
  「今のままだと逆算したときに繋ぎの語とか機能語が出てこない」
形:
  流れ    述語のすぐ後ろに続く 助動詞・接続助詞・補助動詞 の並び（「歩い|て|い|た」なら (て, いる, た)）。
  融合の板 語幹の板（歩く）に流れの 1 つずつを足した一時の板。(歩く) → (歩く+て) → (歩く+て+いる) → (歩く+て+いる+た)。
          一時の板ごとに「次に何が付くか（終わりも含む）」を、文脈の記号と組にして経験の器（I33 の CountGrain）に持つ。
  文脈の記号 文の中の位置（文末・名詞へ・文中・引用）、流れの後ろの語、文の中の時の副詞（もう・まだ・ずっと・昨日 …）、
          前の行の言い切りの流れの始め、台詞か地の文か、述語の品詞。語幹そのものも 1 つの記号（語: 歩く）として入れる。
  逆引き  言いたい語幹と文脈から、融合の板を 1 段ずつ下りて流れを選び（ネスト）、語幹の形を戻す
          （語幹 + 流れの始め → 本で出た表記。無ければ、語の終わりの字と流れの始めの組で、本で出た書き換え）。
比べるもの（10 作品を外して当てる）:
  いつも一番多い流れ／語幹ごとの表／平らな板（流れまるごとを 1 つの答えとして文脈から）／融合の板（1 段ずつ）。
I42b（逆引き）: 言いたい意味の組（過去・進行・否定・丁寧 …、順なし）を渡し、繋ぎの語（て・で）と並びを
  表（意味の組 → 本で多い流れ）／融合の板を意味で縛って 1 段ずつ下りる探し（並びの決まりも本から）で戻す。
実行: python intervention/i42_fused_flow.py --cache <解析済み JSON> [--hold 10] [--out <結果 JSON>]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i33_predicate_grain as G  # noqa: E402

PRED = ("VERB", "ADJ")
FLOW_POS = ("AUX", "SCONJ")
TIME_ADV = ("もう", "まだ", "ずっと", "昨日", "今日", "明日", "今", "いつも", "さっき", "これから", "すでに", "既に", "やっと",
            "ちょうど", "毎日", "よく", "たまに", "前", "先", "後で", "今度")
END = "（終わり）"


def flows(toks):
    """行の述語ごとに (述語の i, 流れ (lemma の並び), 流れの表記の並び, 流れの後ろの i)。"""
    out = []
    used = set()
    for t in toks:
        if t["pos"] not in PRED or t["i"] in used or t["dep"] == "fixed":
            continue
        j = t["i"] + 1
        fl, fs = [], []
        while j < len(toks):
            k = toks[j]
            if k["pos"] in FLOW_POS or (k["pos"] == "VERB" and k["dep"] in ("fixed", "aux")) or (k["pos"] == "ADJ" and k["dep"] == "fixed"):
                fl.append(k["lemma"])
                fs.append(k["text"])
                used.add(j)
                j += 1
                continue
            break
        out.append((t["i"], tuple(fl), tuple(fs), j))
    return out


def context(toks, t, nxt, prev_end, quote):
    sy = []
    k = toks[nxt] if nxt < len(toks) else None
    if k is None or k["pos"] == "PUNCT" and k["text"] in ("。", "」", "！", "？", "!", "?", "…", "』"):
        sy.append("位置:文末")
    elif t["dep"] == "acl" or (k is not None and k["pos"] in ("NOUN", "PRON", "PROPN")):
        sy.append("位置:名詞へ")
    elif k is not None and k["lemma"] in ("と", "って"):
        sy.append("位置:引用")
    else:
        sy.append("位置:文中")
    sy.append("後ろ:" + (k["text"] if k is not None and k["pos"] in ("ADP", "PART", "PUNCT", "CCONJ") else (k["pos"] if k else "無し")))
    adv = sorted({x["lemma"] for x in toks if x["lemma"] in TIME_ADV})
    sy += ["時:" + a for a in adv[:2]] or ["時:無し"]
    sy.append("前の行:" + (prev_end or "無し"))
    sy.append("台詞:" + ("はい" if quote else "いいえ"))
    sy.append("品詞:" + t["pos"])
    sy.append("係り:" + t["dep"])
    sy.append("語:" + t["lemma"])
    return sy


def items(works):
    """作品ごとの [(文脈の記号, 語幹, 流れ, 語幹の表記, 流れの表記)]。"""
    out = []
    for w in works:
        its = []
        prev_end = None
        for ln in w["lines"]:
            T = ln["toks"]
            if not T:
                continue
            quote = ln["text"].startswith(("「", "『"))
            fls = flows(T)
            for i, fl, fs, nxt in fls:
                t = T[i]
                its.append((context(T, t, nxt, prev_end, quote), t["lemma"], fl, t["text"], fs))
            root = next((x for x in fls if T[x[0]]["dep"] == "ROOT"), None)
            prev_end = (root[1][0] if root and root[1] else "言い切り") if root else prev_end
        out.append(its)
    return out


class Flat:
    def __init__(self, train, top=80):
        c = Counter("|".join(fl) for its in train for _, _, fl, _, _ in its)
        self.keep = {k for k, _ in c.most_common(top)}
        self.g = G.CountGrain("全体")
        for its in train:
            for sy, _, fl, _, _ in its:
                k = "|".join(fl)
                self.g.add(sy, k if k in self.keep else "他")

    def predict(self, sy):
        s = self.g.scores(sy)
        s.pop("他", None)
        return tuple(max(s, key=s.get).split("|")) if max(s, key=s.get) else ()


class Fused:
    """融合の板: (語幹 + ここまでの流れ) の一時の板ごとに、自分の経験の器を持ち、次に付くもの（終わりも）を文脈の記号から学ぶ。
    一時の板は「ここまでの流れの最後の 2 つ」で呼ぶ（歩く+て+いる → 板「て+いる」）。経験の少ない板は、最後の 1 つの板、
    それも少なければ全部の板をまとめた器に任せる（親の語板 = 子の和、と同じ形）。"""
    MAX = 6
    MIN = 30

    def __init__(self, train, top=60):
        nx = Counter()
        for its in train:
            for _, _, fl, _, _ in its:
                nx.update(fl)
        self.keep = {k for k, _ in nx.most_common(top)} | {END}
        self.boards = defaultdict(lambda: G.CountGrain("全体"))
        for its in train:
            for sy, _, fl, _, _ in its:
                for d in range(len(fl) + 1):
                    nxt = fl[d] if d < len(fl) else END
                    lab = nxt if nxt in self.keep else "他"
                    for k in self.keys(fl[:d]):
                        self.boards[k].add(sy, lab)

    @staticmethod
    def keys(prefix):
        return ["2:" + "+".join(prefix[-2:]) if prefix else "2:語幹", "1:" + (prefix[-1] if prefix else "語幹"), "全"]

    def board(self, prefix):
        for k in self.keys(prefix):
            b = self.boards.get(k)
            if b is not None and b.N >= self.MIN:
                return b
        return self.boards["全"]

    def predict(self, sy):
        out = []
        for _ in range(self.MAX):
            s = self.board(tuple(out)).scores(sy)
            s.pop("他", None)
            if not s:
                break
            nxt = max(s, key=s.get)
            if nxt == END:
                break
            out.append(nxt)
        return tuple(out)


class FlowSurface:
    """流れの 1 つずつの表記: (その語, 次に付く語) → 本で出た表記（「いる」の次が「た」なら「い」、「ない」の次が「た」なら「なかっ」）。"""

    def __init__(self, train):
        self.t = defaultdict(Counter)
        for its in train:
            for _, _, fl, _, fs in its:
                for i, (l, x) in enumerate(zip(fl, fs)):
                    self.t[(l, fl[i + 1] if i + 1 < len(fl) else END)][x] += 1
                    self.t[(l, "*")][x] += 1

    def form(self, fl):
        out = []
        for i, l in enumerate(fl):
            c = self.t.get((l, fl[i + 1] if i + 1 < len(fl) else END)) or self.t.get((l, "*"))
            out.append(c.most_common(1)[0][0] if c else l)
        return "".join(out)


class Surface:
    """語幹の形を戻す: (語幹, 流れの始め) → 表記。無ければ (語の終わりの字, 流れの始め) → 書き換え（終わりの字を何に替えるか）。"""

    def __init__(self, train):
        self.exact = defaultdict(Counter)
        self.rule = defaultdict(Counter)
        for its in train:
            for _, lem, fl, surf, _ in its:
                f = fl[0] if fl else END
                self.exact[(lem, f)][surf] += 1
                cp = os.path.commonprefix([lem, surf])
                self.rule[(lem[-1:], f)][(lem[len(cp):], surf[len(cp):])] += 1

    def form(self, lem, fl):
        f = fl[0] if fl else END
        if self.exact.get((lem, f)):
            return self.exact[(lem, f)].most_common(1)[0][0]
        r = self.rule.get((lem[-1:], f))
        if r:
            a, b = r.most_common(1)[0][0]
            if lem.endswith(a):
                return lem[: len(lem) - len(a)] + b
        return lem


# I42b: 逆引き。言いたい意味（過去・進行・否定・丁寧 …）を順なしの組で渡し、繋ぎの語（て・で）と並びを融合の板で戻す。
MEAN = {"た": "過去", "だ": "過去・断定", "いる": "進行", "てる": "進行", "おる": "進行", "ない": "否定", "ん": "否定・説明", "ず": "否定",
        "ぬ": "否定", "ます": "丁寧", "です": "丁寧", "くださる": "丁寧", "いたす": "丁寧", "なさる": "丁寧", "れる": "受身・可能",
        "られる": "受身・可能", "せる": "使役", "させる": "使役", "たい": "希望", "ほしい": "希望", "よう": "意志・推量", "う": "意志・推量",
        "だろう": "意志・推量", "そう": "様子", "らしい": "様子", "みたい": "様子", "しまう": "済んでしまう", "ちゃう": "済んでしまう",
        "みる": "試し", "おく": "備え", "とく": "備え", "くる": "向き", "いく": "向き", "くれる": "やりもらい", "もらう": "やりもらい",
        "やる": "やりもらい", "いただく": "やりもらい", "もらえる": "やりもらい", "いただける": "やりもらい", "できる": "可能",
        "ば": "条件", "たら": "条件", "から": "理由", "ので": "理由", "けど": "逆接", "けれど": "逆接", "が": "逆接", "のに": "逆接",
        "ながら": "同時", "つつ": "同時", "の": "説明", "し": "並べ", "と": "条件", "べし": "べき", "ある": "結果"}
GLUE = ("て", "で")


def meaning(fl):
    """流れ → 言いたい意味の組（順なし）。繋ぎの て・で は、後ろにまだ何か付く時は意味に入れない（逆引きで戻すもの）。
    終わりの て・で は「て形」（続ける）として意味に入れる。"""
    out = []
    for i, x in enumerate(fl):
        if x in GLUE:
            if i == len(fl) - 1:
                out.append("て形")
            continue
        out.append(MEAN.get(x, "他:" + x))
    return tuple(sorted(out))


def _logp(s):
    m = max(s.values())
    z = m + math.log(sum(math.exp(v - m) for v in s.values()))
    return {k: v - z for k, v in s.items()}


class Reverse:
    """逆引き。語幹・意味の組・文脈 → 流れ。
    表  意味の組ごとに本で一番多い流れ（語幹ごとの表があればそれ）。
    融合 融合の板を 1 段ずつ下り、各段で「残っている意味のどれかを満たす語」か「繋ぎの語」だけを候補にする。
        意味を全部満たした時だけ終わりを選べる。点 = 各段の板の確からしさ（文脈の記号つき）の和。束（beam）で探す。
        本で見ていない意味の組でも並べられる。"""
    BEAM = 8

    def __init__(self, train, fused):
        self.fused = fused
        self.tab = defaultdict(Counter)
        self.stem = defaultdict(Counter)
        self.by = defaultdict(Counter)        # 意味 → それを満たす語
        for its in train:
            for _, lem, fl, _, _ in its:
                m = meaning(fl)
                self.tab[m][fl] += 1
                self.stem[(lem, m)][fl] += 1
                for i, x in enumerate(fl):
                    if x not in GLUE:
                        self.by[MEAN.get(x, "他:" + x)][x] += 1
        self.by["て形"] = Counter({"て": 1, "で": 1})
        self.prec = Counter()                 # 意味 a が b より前に来た数（並びの決まりを本から）
        for its in train:
            for _, _, fl, _, _ in its:
                ms = [MEAN.get(x, "他:" + x) if x not in GLUE else ("て形" if i == len(fl) - 1 else None) for i, x in enumerate(fl)]
                ms = [x for x in ms if x]
                for i in range(len(ms)):
                    for j in range(i + 1, len(ms)):
                        self.prec[(ms[i], ms[j])] += 1

    def table(self, lem, m):
        c = self.stem.get((lem, m)) or self.tab.get(m)
        return c.most_common(1)[0][0] if c else None

    def search(self, sy, m):
        beam = [(0.0, (), Counter(m))]
        done = []
        for _ in range(Fused.MAX + 2):
            nb = []
            for sc, pre, rest in beam:
                lp = _logp(self.fused.board(pre).scores(sy))
                ok = {}
                for need in rest:
                    for x in self.by.get(need, ()):
                        ok[x] = need
                if not rest:
                    done.append((sc + lp.get(END, -20.0), pre))
                for x, need in ok.items():
                    r2 = rest.copy()
                    r2[need] -= 1
                    if r2[need] <= 0:
                        del r2[need]
                    pen = sum(math.log((self.prec[(need, r)] + 1) / (self.prec[(need, r)] + self.prec[(r, need)] + 2))
                              for r in r2)               # 残りの意味より前に来てよいか
                    nb.append((sc + lp.get(x, lp.get("他", -20.0)) + pen, pre + (x,), r2))
                if rest and pre[-1:] not in (("て",), ("で",)):
                    for g in GLUE:                     # 繋ぎの語は意味を使わずに入れる
                        nb.append((sc + lp.get(g, -20.0), pre + (g,), rest))
            beam = sorted(nb, key=lambda z: -z[0])[: self.BEAM]
            if not beam:
                break
        # 「て形」は終わりの て・で で満たす
        done = [(s, p) for s, p in done if meaning(p) == m]
        return max(done)[1] if done else None


def run(works, hold=10, seed=0):
    rng = random.Random(seed)
    idx = list(range(len(works)))
    rng.shuffle(idx)
    test_i = set(idx[:hold])
    its = items(works)
    train = [x for i, x in enumerate(its) if i not in test_i]
    test = [y for i, x in enumerate(its) if i in test_i for y in x]
    most = Counter(fl for t in train for _, _, fl, _, _ in t).most_common(1)[0][0]
    per = defaultdict(Counter)
    for t in train:
        for _, lem, fl, _, _ in t:
            per[lem][fl] += 1
    flat, fused, surf, fsurf = Flat(train), Fused(train), Surface(train), FlowSurface(train)
    rev = Reverse(train, fused)
    res = Counter()
    ex = []
    for sy, lem, fl, s0, fs in test:
        res["数"] += 1
        res["いつも一番多い流れ"] += fl == most
        res["語幹ごとの表"] += fl == (per[lem].most_common(1)[0][0] if per.get(lem) else most)
        pf = flat.predict(sy)
        res["平らな板"] += pf == fl
        pz = fused.predict(sy)
        res["融合の板"] += pz == fl
        res["融合の板・始めの 1 つ"] += (pz[:1] == fl[:1])
        m = meaning(fl)
        seen = m in rev.tab
        pt = rev.table(lem, m)
        pr = rev.search(sy, m)
        res["逆引き・数"] += 1
        res["逆引き・表"] += pt == fl
        res["逆引き・融合"] += pr == fl
        pc = pt if seen else pr
        res["逆引き・表+融合"] += pc == fl
        res["逆引き・表+融合・形まで"] += pc is not None and surf.form(lem, pc) + fsurf.form(pc) == s0 + "".join(fs)
        res["逆引き・融合・形まで"] += pr is not None and surf.form(lem, pr) + fsurf.form(pr) == s0 + "".join(fs)
        if len(fl) >= 2:
            res["2 つ以上・数"] += 1
            res["2 つ以上・表"] += pt == fl
            res["2 つ以上・融合"] += pr == fl
        if not seen:
            res["見ていない組・数"] += 1
            res["見ていない組・融合"] += pr == fl
        f0 = surf.form(lem, fl)                             # 形を戻す（正しい流れを渡した時の、語幹の表記）
        res["形を戻す・語幹"] += f0 == s0
        res["形を戻す・語幹+流れ"] += (f0 + fsurf.form(fl)) == s0 + "".join(fs)
        if len(ex) < 40 and fl and rng.random() < 0.02:
            ex.append(dict(文脈=[x for x in sy if not x.startswith("語:")], 語幹=lem, 本=s0 + "".join(fs),
                           融合の板="+".join(pz) or "（無し）", 平らな板="+".join(pf) or "（無し）",
                           戻した形=surf.form(lem, pz) + fsurf.form(pz), 意味="・".join(m),
                           逆引き="+".join(pr or ()) or "（無し）",
                           逆引きの形=(surf.form(lem, pr) + fsurf.form(pr)) if pr else "（無し）"))
    n = res["数"]
    acc = {}
    for k, v in res.items():
        if k.endswith("数"):
            continue
        d = res[k.split("・")[0] + "・数"] if k.startswith(("2 つ以上", "見ていない組")) else n
        acc[k] = round(v / max(d, 1), 3)
    acc["2 つ以上の割合"] = round(res["2 つ以上・数"] / n, 3)
    acc["見ていない組の数"] = res["見ていない組・数"]
    return dict(数=n, 当たり=acc, 例=ex)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--hold", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    works = json.load(open(a.cache, encoding="utf-8"))
    r = run(works, a.hold, a.seed)
    print(json.dumps(dict(数=r["数"], 当たり=r["当たり"]), ensure_ascii=False))
    for e in r["例"][:15]:
        print(json.dumps(e, ensure_ascii=False))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(r, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
