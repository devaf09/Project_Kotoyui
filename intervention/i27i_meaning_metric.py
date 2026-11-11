"""介入の研究 実験I27i（試験の一覧 B8）: 同じ意味で数える物差し

持ち主（2026-11-07）: 「いわゆる国語に完全な正解はないから」。今までの物差しは、答えの語と字（見出し）が同じときだけ正解にしていた。
言い換え（表記ゆれ・近い語）も正解として数え直すと、結果がどれだけ変わるかを見る。格フレームは使わない。
同じ意味の段:
  字      : 見出しが同じ（今までの物差し）。
  正規化  : Sudachi の正規化表記が同じ（表記ゆれ・送り仮名・かな書き）。
  近い語 θ: 正規化で違っても、GiNZA の語ベクトル（ja_ginza に付く 300 次元）の余弦が θ 以上。θ = 0.5 / 0.6 / 0.7。
           語ベクトルは「関わりのある語」も近く出す（兄と弟など）ので、θ ごとに数え直した組の例を非公開の置き場に書き、目で確かめる。
数え直す結果:
  G. I27g（Web 小説）の補い: 補い方ごとの上位 10 候補（--policies、I27g の --dump_policies の出力）。1 位の正解率と、上位 10 に入った割合。
  E. I27e の C: 20 行版の（名詞, 属性:語）の組が、短い版に残っている割合。名詞と語の両方を同じ意味で比べる（属性の種類は同じもの）。
結果: --out（Web 小説を使ったときは _wn を付けた名前にし、非公開の置き場だけに置く）。
実行: python intervention/i27i_meaning_metric.py --policies <wn_policies.json> --tales <tales_v1_parsed.json> --out <jsonl> --pairs <md>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

THETAS = (0.5, 0.6, 0.7)


class Same:
    """2 つの見出しが同じ意味かを段ごとに答える。"""

    def __init__(self):
        import spacy
        from sudachipy import dictionary
        self.nlp = spacy.load("ja_ginza")
        self.tok = dictionary.Dictionary(dict="core").create()
        self._n, self._v = {}, {}
        self.seen = defaultdict(Counter)                 # 段 → 数え直した組（例を書く用）

    def norm(self, w):
        if w not in self._n:
            ms = self.tok.tokenize(w)
            self._n[w] = "".join(m.normalized_form() for m in ms) if ms else w
        return self._n[w]

    def vec(self, w):
        if w not in self._v:
            import numpy as np
            v = None
            for k in (w, self.norm(w)):
                lx = self.nlp.vocab[k]
                if lx.has_vector:
                    v = lx.vector / (np.linalg.norm(lx.vector) + 1e-9)
                    break
            self._v[w] = v
        return self._v[w]

    def level(self, a, b):
        """(段, 余弦)。段 = 字・正規化・余弦・無し（語ベクトルが無い）。"""
        if a == b:
            return ("字", 1.0)
        if len(a) > 1 and len(b) > 1 and self.norm(a) == self.norm(b):       # かな 1 字（し・き など）は正規化が当てにならない
            return ("正規化", 1.0)
        va, vb = self.vec(a), self.vec(b)
        if va is None or vb is None:
            return ("無し", 0.0)
        c = float(va @ vb)
        if c > 0.999:                                    # 字の違う語が同じベクトル = 辞書に無い語が同じ行に落ちた
            return ("無し", 0.0)
        return ("余弦", c)

    def ok(self, a, b, step):
        lv, c = self.level(a, b)
        if step == "字":
            r = lv == "字"
        elif step == "正規化":
            r = lv in ("字", "正規化")
        else:
            r = lv in ("字", "正規化") or (lv == "余弦" and c >= float(step.split()[1]))
        if r and lv != "字":
            self.seen[step][(a, b)] += 1
        return r


STEPS = ["字", "正規化"] + [f"近い語 {t}" for t in THETAS]


def part_g(S, pols):
    tg = pols["targets"]
    out = {}
    for k, lists in pols["lists"].items():
        rec = {}
        for st in STEPS:
            hit = top10 = 0
            for t, cs in zip(tg, lists):
                ans = t[4]
                words = [c for c, _ in cs]
                if words and S.ok(words[0], ans, st):
                    hit += 1
                if any(S.ok(w, ans, st) for w in words):
                    top10 += 1
            rec[st] = dict(acc=hit / len(tg), top10=top10 / len(tg))
        out[k] = rec
    return dict(n=len(tg), by_policy=out)


def part_e(S, tales):
    from i27e_restore import content
    out = {}
    for lv in ("12", "5", "1"):
        rec = {}
        for st in STEPS:
            got = tot = got_p = tot_p = 0
            for t in tales:
                c20, cl = content(t, "20"), content(t, lv)
                by_role = defaultdict(list)
                for k, a in cl:
                    r, w = a.split(":", 1)
                    by_role[r].append((k, w))
                by_role20 = defaultdict(list)
                for k, a in c20:
                    r, w = a.split(":", 1)
                    by_role20[r].append((k, w))
                for k, a in c20:
                    r, w = a.split(":", 1)
                    tot += 1
                    got += any(S.ok(k, k2, st) and S.ok(w, w2, st) for k2, w2 in by_role[r])
                for k, a in cl:
                    r, w = a.split(":", 1)
                    tot_p += 1
                    got_p += any(S.ok(k, k2, st) and S.ok(w, w2, st) for k2, w2 in by_role20[r])
            rec[st] = dict(recall=got / max(1, tot), precision=got_p / max(1, tot_p))
        out[lv] = rec
    return out


def write_pairs(S, path, n=30):
    """段ごとに、字が違うのに同じと数えた語の組（多い順）。非公開の置き場だけに置く。"""
    lines = ["# I27i 同じと数え直した語の組（非公開）", ""]
    for st in STEPS[1:]:
        c = S.seen[st]
        lines += [f"## {st}（{len(c)} 組）", ""]
        for (a, b), v in c.most_common(n):
            lv, cs = S.level(a, b)
            lines.append(f"- {a} ／ {b}　{v} 回　{lv} {cs:.2f}")
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", default=None)
    ap.add_argument("--tales", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", default=None)
    a = ap.parse_args()
    S = Same()
    rec = dict(exp="I27i", tag=a.tag, steps=STEPS)
    if a.policies:
        rec["G"] = part_g(S, json.load(open(a.policies, encoding="utf-8")))
    if a.tales:
        rec["E"] = part_e(S, json.load(open(a.tales, encoding="utf-8")))
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if a.pairs:
        write_pairs(S, a.pairs)


if __name__ == "__main__":
    main()
