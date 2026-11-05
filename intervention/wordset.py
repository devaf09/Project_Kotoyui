"""試験用の語群セット: 読み込みと、試しの組（対の表）・文の生成

持ち主の方針（2026-11-05）: 処理系は、光による一括処理（並べてまとめて判断、A の出力を B の入力に）、融合（劣化あり）、消去、差分検証、逆回しを優先する。
目標は語板の作成と、文章からの改良、最終的に文法を理解する仕組みを光の器（かそれに類するもの）で作ること。
そこで語群セットは 3 つに分ける:
  1. 語の表（words_*.json）: 語の木（親・種類）、属性（完熟・未熟）、語の形と属性の対応（lexicon）、動詞が変える観点。
  2. 文型の表（frames_*.json）: 文の型と、枠ごとの役（主題・主語・目的語・修飾・述語・親）。
  3. 対の表（このプログラムが作る）: 特性ごとの試しの組。
       差分検証 diff: 属性が 1 つの観点だけ違う品種の対（フジ / 紅玉 = 味だけ）。
       融合 fusion:  同じ科の品種の組（重なりの度合い = Jaccard つき）と、別の科の組（比べ）。融合でどれだけ崩れるかを測る。
       消去 erase:   科から品種を 1 つ消したとき、親（和）で残る属性と消える属性。
       逆回し reverse: 属性の組（色・味・形）→ 当てはまる品種。1 つに決まる / 決まらない / 無い。
       連鎖 chain:   熟す → 色が変わる → 味が変わる（A の出力を B の入力に）。熟しても味の変わらない品種（ウメ型）も。
       一括 batch:   同じ文型・同じ観点の問いを並べた束（科ごと）。
       文法 grammar: 最小の対（主語と目的語の入れ替え、語順だけ変える、助詞の付け替え、否定、修飾の色の食い違い、誤った親）。正しい文 / 崩した文と、何が変わったか。
語の表・文型の表・生成物は非公開の置き場（../Project_Kotoyui_private/intervention/wordsets/）に置く（持ち主 2026-11-05: 生成物は公開しない）。
実行: python intervention/wordset.py --words <words_v1.json> --frames <frames_v1.json> [--out <書き出す先>]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
PRIV = os.environ.get("KOTOYUI_PRIVATE_INTERVENTION", os.path.join(HERE, "..", "..", "Project_Kotoyui_private", "intervention"))
WS_DIR = os.path.join(PRIV, "wordsets")
CORE = ("色", "味", "形")                                            # 品種を見分ける観点
REN_COLOR = {"赤": "赤く", "黄": "黄色く", "白": "白く", "茶": "茶色く", "緑": "緑に", "紫": "紫に", "橙": "橙に"}
REN_TASTE = {"甘い": "甘く", "酸っぱい": "酸っぱく", "渋い": "渋く"}


class WordSet:
    def __init__(self, words_path=None, frames_path=None):
        words_path = words_path or os.path.join(WS_DIR, "words_v1.json")
        frames_path = frames_path or os.path.join(WS_DIR, "frames_v1.json")
        W = json.load(open(words_path, encoding="utf-8"))
        self.views, self.lex, self.verb_view = W["views"], W["lexicon"], W["verb_view"]
        self.nodes = {n["name"]: n for n in W["nodes"]}
        self.frames = {f["id"]: f for f in json.load(open(frames_path, encoding="utf-8"))["frames"]}
        self.tok_of = {}                                             # (観点, 値) → 語の形
        for cat in ("形容", "述語"):
            for t, (v, x) in self.lex[cat].items():
                self.tok_of.setdefault((v, x), t)
        self.neg_of = {tuple(vx): t for t, vx in self.lex["述語の否定"].items()}
        self.verb_tok = {v: t for t, v in self.lex["動詞"].items()}
        self.verb_neg = {v: t for t, v in self.lex["動詞の否定"].items()}
        self.paint_tok = {c: t for t, c in self.lex["塗る色"].items()}

    # ---- 木
    def kind(self, k):
        return [n for n, x in self.nodes.items() if x.get("kind") == k]

    def children(self, p):
        return [n for n, x in self.nodes.items() if x.get("parent") == p]

    def ancestors(self, n):
        out = []
        while self.nodes[n].get("parent"):
            n = self.nodes[n]["parent"]
            out.append(n)
        return out

    def varieties(self, fam=None):
        return [n for n in self.kind("品種") if fam is None or self.nodes[n]["parent"] == fam]

    def attrs(self, n, ripe=True):
        x = self.nodes[n]
        a = {v: list(xs) for v, xs in x.get("attrs", {}).items()}
        if not ripe and "unripe" in x:
            a.update({v: list(xs) for v, xs in x["unripe"].items()})
            a["される"] = ["切る"]
        return a

    def aset(self, n, views=CORE, ripe=True):
        a = self.attrs(n, ripe)
        return {(v, x) for v in views for x in a.get(v, [])}

    def parent_mix(self, fam, exclude=()):
        """親 = 子の属性の和（光の融合で明るさを足すのと同じ数え方）。値ごとの割合。"""
        c = Counter()
        for ch in self.varieties(fam):
            if ch in exclude:
                continue
            for v in CORE:
                xs = self.attrs(ch).get(v, [])
                for x in xs:
                    c[(v, x)] += 1 / len(xs)
        out = {}
        for v in CORE:
            tot = sum(c[(v, x)] for x in self.views[v]) or 1
            out[v] = {x: round(c[(v, x)] / tot, 3) for x in self.views[v] if c[(v, x)] > 0}
        return out

    # ---- 対の表
    def diff_pairs(self):
        out = []
        for a, b in itertools.combinations(self.varieties(), 2):
            A, B = self.attrs(a), self.attrs(b)
            dv = [v for v in CORE if sorted(A.get(v, [])) != sorted(B.get(v, []))]
            if len(dv) == 1:
                same = self.nodes[a]["parent"] == self.nodes[b]["parent"]
                out.append(dict(a=a, b=b, view=dv[0], a_val=A[dv[0]], b_val=B[dv[0]], same_family=same))
        return out

    def fusion_groups(self):
        out = []
        fams = self.kind("科")
        fams = [f for f in fams if self.varieties(f)]
        for f in fams:
            for a, b in itertools.combinations(self.varieties(f), 2):
                A, B = self.aset(a), self.aset(b)
                out.append(dict(a=a, b=b, family=f, cross=False, jaccard=round(len(A & B) / len(A | B), 3), shared=sorted(map(":".join, A & B))))
        rng = random.Random(0)
        for f, g in itertools.combinations(fams, 2):                # 別の科の組（比べ）を 2 つずつ
            for _ in range(2):
                a, b = rng.choice(self.varieties(f)), rng.choice(self.varieties(g))
                A, B = self.aset(a), self.aset(b)
                out.append(dict(a=a, b=b, family=f"{f}|{g}", cross=True, jaccard=round(len(A & B) / len(A | B), 3), shared=sorted(map(":".join, A & B))))
        return out

    def erase_sets(self):
        out = []
        for f in self.kind("科"):
            vs = self.varieties(f)
            if len(vs) < 2:
                continue
            full = self.parent_mix(f)
            for e in vs:
                after = self.parent_mix(f, exclude=(e,))
                lost = [f"{v}:{x}" for v in CORE for x in full[v] if x not in after[v]]
                out.append(dict(family=f, erased=e, remaining=[x for x in vs if x != e], lost=lost, parent_before=full, parent_after=after))
        return out

    def reverse_queries(self):
        out = []
        for combo in itertools.product(*[self.views[v] for v in CORE]):
            q = dict(zip(CORE, combo))
            hit = [n for n in self.varieties() if all(q[v] in self.attrs(n).get(v, []) for v in CORE)]
            if not hit:
                continue                                             # 無い組は数が多すぎるので、下で一部だけ足す
            out.append(dict(query=q, answers=hit, kind="unique" if len(hit) == 1 else "ambiguous",
                            families=sorted({self.nodes[h]["parent"] for h in hit})))
        rng = random.Random(1)
        none = [dict(zip(CORE, c)) for c in itertools.product(*[self.views[v] for v in CORE])
                if not any(all(c[i] in self.attrs(n).get(v, []) for i, v in enumerate(CORE)) for n in self.varieties())]
        for q in rng.sample(none, min(len(none), len(out) // 2)):
            out.append(dict(query=q, answers=[], kind="none", families=[]))
        return out

    def chains(self):
        out = []
        for n in self.varieties():
            u, r = self.attrs(n, ripe=False), self.attrs(n, ripe=True)
            out.append(dict(word=n, family=self.nodes[n]["parent"], ripens_change=self.nodes[n]["ripens_change"],
                            steps=[dict(op="熟させる", view="状態", before=u["状態"], after=r["状態"]),
                                   dict(op="→", view="色", before=u["色"], after=r["色"]),
                                   dict(op="→", view="味", before=u["味"], after=r["味"])]))
        return out

    def batches(self):
        """一括処理: 科ごとに、同じ文型・同じ観点の問いを並べた束（答えつき）。"""
        out = []
        for f in self.kind("科"):
            vs = self.varieties(f)
            if not vs:
                continue
            for v in CORE:
                for x in self.views[v]:
                    out.append(dict(family=f, view=v, value=x, items=[dict(word=n, ans=x in self.attrs(n).get(v, [])) for n in vs]))
        return out

    # ---- 文
    def fill(self, fid, slots):
        toks = []
        for t in self.frames[fid]["toks"]:
            toks.append(slots[t[1:-1]] if t.startswith("{") else t)
        return toks

    def sentences(self, n_per=10, seed=0):
        """品種ごとに正しい文を n_per 個（文型をまぜる）。1 行 = {toks, frame, word, roles, attrs}。"""
        rng = random.Random(seed)
        agents = [a for a in self.kind("個体") if "食べる" in self.attrs(a).get("する", [])]
        people = [a for a in agents if self.nodes[a]["parent"] == "人"]
        out = []
        for n in self.varieties():
            fam, a = self.nodes[n]["parent"], self.attrs(n)
            for _ in range(n_per):
                fid = rng.choice(list(self.frames))
                taste = rng.choice(a["味"])
                s = {"品種": n, "科": fam, "味": self.tok_of[("味", taste)], "色": self.tok_of[("色", rng.choice(a["色"]))]}
                if fid == "F4_否定":
                    s["味否定"] = self.neg_of[("味", rng.choice([x for x in self.views["味"] if x not in a["味"]]))]
                if fid in ("F5_他動", "F6_他動否定"):
                    s["生き物"] = rng.choice(agents)
                    s["動詞"], s["動詞否定"] = self.verb_tok["食べる"], self.verb_neg["食べる"]
                if fid in ("F7_塗る", "F8_熟す"):
                    s["生き物"] = rng.choice(people)
                    s["塗る色"] = self.paint_tok[rng.choice(self.views["色"])]
                if fid == "F9_連鎖":
                    s["色"], s["味"] = REN_COLOR[rng.choice(a["色"])], REN_TASTE[taste]
                if fid == "F10_並列":
                    mates = [m for m in self.varieties() if m != n and taste in self.attrs(m)["味"]]
                    s["品種2"] = rng.choice(mates)
                out.append(dict(toks=self.fill(fid, s), frame=fid, word=n, family=fam))
        return out

    def grammar_pairs(self, seed=0, n_per=2):
        """文法の最小の対: 正しい文と、1 か所だけ崩した文。change = 何を変えたか、natural = 崩した文が意味として成り立つか。"""
        rng = random.Random(seed)
        agents = [a for a in self.kind("個体") if "食べる" in self.attrs(a).get("する", [])]
        out = []
        for n in self.varieties():
            fam, a = self.nodes[n]["parent"], self.attrs(n)
            for _ in range(n_per):
                ag = rng.choice(agents)
                ok = self.fill("F5_他動", {"生き物": ag, "品種": n, "動詞": "食べた"})
                out.append(dict(change="主語と目的語の入れ替え", ok=ok, bad=self.fill("F5_他動", {"生き物": n, "品種": ag, "動詞": "食べた"}), natural=False))
                out.append(dict(change="語順の入れ替え（助詞で役は同じ）", ok=ok, bad=[n, "を", ag, "が", "食べた", "。"], natural=True, same_meaning=True))
                out.append(dict(change="助詞の付け替え（役が入れ替わる）", ok=ok, bad=[ag, "を", n, "が", "食べた", "。"], natural=False))
                t = rng.choice(a["味"])
                out.append(dict(change="否定（真偽が反転）", ok=self.fill("F1_属性", {"品種": n, "味": self.tok_of[("味", t)]}),
                                bad=self.fill("F4_否定", {"品種": n, "味否定": self.neg_of[("味", t)]}), natural=True, truth_flip=True))
                wrong_c = rng.choice([c for c in self.views["色"] if c not in a["色"] and ("色", c) in self.tok_of])
                out.append(dict(change="修飾の色の食い違い", ok=self.fill("F2_修飾", {"色": self.tok_of[("色", rng.choice(a["色"]))], "品種": n, "味": self.tok_of[("味", t)]}),
                                bad=self.fill("F2_修飾", {"色": self.tok_of[("色", wrong_c)], "品種": n, "味": self.tok_of[("味", t)]}), natural=False))
                other = rng.choice([f for f in self.kind("科") if f != fam and self.varieties(f)])
                out.append(dict(change="誤った親", ok=self.fill("F3_親子", {"科": fam, "品種": n, "味": self.tok_of[("味", t)]}),
                                bad=self.fill("F3_親子", {"科": other, "品種": n, "味": self.tok_of[("味", t)]}), natural=False))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", default=None)
    ap.add_argument("--frames", default=None)
    ap.add_argument("--out", default=WS_DIR, help="書き出す先（非公開の置き場）")
    ap.add_argument("--n_per", type=int, default=10, help="品種 1 つあたりの文")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ws = WordSet(a.words, a.frames)
    os.makedirs(a.out, exist_ok=True)
    tables = dict(diff=ws.diff_pairs(), fusion=ws.fusion_groups(), erase=ws.erase_sets(), reverse=ws.reverse_queries(),
                  chain=ws.chains(), batch=ws.batches(), grammar=ws.grammar_pairs(a.seed))
    json.dump(tables, open(os.path.join(a.out, "pairs_v1.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    with open(os.path.join(a.out, "sentences_v1.jsonl"), "w", encoding="utf-8") as f:
        for s in ws.sentences(a.n_per, a.seed):
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    summary = {k: len(v) for k, v in tables.items()}
    summary["reverse_kinds"] = dict(Counter(q["kind"] for q in tables["reverse"]))
    summary["diff_views"] = dict(Counter(p["view"] for p in tables["diff"]))
    summary["varieties"] = len(ws.varieties())
    summary["families"] = len([f for f in ws.kind("科") if ws.varieties(f)])
    summary["sentences"] = len(ws.varieties()) * a.n_per
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
