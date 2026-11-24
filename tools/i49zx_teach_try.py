"""I49zx: 介入の動作試験（持ち主の言葉「今の実験が一通り終わったら介入の動作試験をしてみようか」から）。

人が教える役になって、介入の板の動きを前後で並べて見るための小さな道具。数えるための道具ではない。
流れ:
  1. 本体（数えて選ぶ）に、材料のテキストを全部聞かせる（--text。リポジトリに入れない材料）。
  2. 問いのファイル（--ask）の問いに答えさせる。教えのファイル（--teach）が無ければ、ここで終わり（1 回目）。
  3. 教えのファイルが有れば、その教えを介入の板に入れてから、同じ問いにもう一度答えさせ、前後を並べる。
     本体の答えと板の答えの両方を出す。板を外した時の答え（= 本体の答え）が 1 回目と同じ事も確かめて出す。
問いのファイル（UTF-8、タブ区切り、1 行 1 問）:  種類 <TAB> が <TAB> を <TAB> どうした    隠す席は「？」。種類は自由な名札（教える・似た問い・関係ない、など）
教えのファイル（UTF-8、タブ区切り）:            問いの番号（1 から） <TAB> 正しい語      出した答えで合っていれば、その語をそのまま書く
候補: 問いのファイルと教えのファイルに出た語のうち、同じ席の語の全部。問いのファイルに「候補 <TAB> 語、語 <TAB> 語、語 <TAB> 語、語」の行を
      置くと、その語も席ごとの候補に足す（区切りは「、」）。
引数: --text 材料（カンマ区切り） --wave 語の波 --content 内容語の板 --ask 問い.tsv [--teach 教え.tsv] --out 結果.md [--dark 0.001]
手で置く物: 暗い側の強さ。上位を 3 つまで見せる事。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, HERE)

import i49zp_text_talks_run as ZP  # noqa: E402
import i49zx_teach_board_run as TR  # noqa: E402

POS = ("NOUN", "NOUN", "VERB")
HIDE = ("？", "?")


def read_ask(path):
    qs = []
    for line in open(path, encoding="utf-8").read().splitlines():
        p = [x.strip() for x in line.split("\t")]
        if len(p) != 4 or sum(x in HIDE for x in p[1:]) != 1:
            continue
        si = next(j for j in range(3) if p[1 + j] in HIDE)
        qs.append((p[0], si, tuple(None if j == si else (p[1 + j], POS[j]) for j in range(3))))
    return qs


def read_extra(path):
    c = [set(), set(), set()]
    for line in open(path, encoding="utf-8").read().splitlines():
        p = [x.strip() for x in line.split("\t")]
        if len(p) == 4 and p[0] == "候補":
            for j in range(3):
                c[j] |= {(w.strip(), POS[j]) for w in p[1 + j].split("、") if w.strip()}
    return c


def read_teach(path):
    out = {}
    for line in open(path, encoding="utf-8").read().splitlines():
        p = [x.strip() for x in line.split("\t")]
        if len(p) == 2 and p[0].isdigit():
            out[int(p[0]) - 1] = p[1]
    return out


def cands_of(qs, teach, extra=None):
    c = [set(x) for x in extra] if extra else [set(), set(), set()]
    for _, si, q in qs:
        for j, w in enumerate(q):
            if w is not None:
                c[j].add(w)
    for i, w in teach.items():
        c[qs[i][1]].add((w, POS[qs[i][1]]))
    return [sorted(x, key=str) for x in c]


def top3(points):
    return "、".join("%s %.2f" % (w[0], v) for w, v in sorted(points.items(), key=lambda kv: -kv[1])[:3])


def show(q, si):
    return " ".join("（？）" if j == si else q[j][0] for j in range(3))


def try_run(body, board, qs, teach, extra=None):
    """→ 行の並び。body は材料をもう聞いた本体。"""
    cands = cands_of(qs, teach, extra)
    rows = []
    for kind, si, q in qs:
        cs = [c for c in cands[si] if c not in {w for w in q if w is not None}]
        o = body.ask(si, q, cs)
        rows.append(dict(種類=kind, 問い=show(q, si), 一回目=None if o["答え"] is None else o["答え"][0], 一回目の段=o["段"], 一回目の上位=top3(o["点"])))
    if not teach:
        return rows
    for i, w in sorted(teach.items()):
        kind, si, q = qs[i]
        tri = tuple((w, POS[si]) if j == si else q[j] for j in range(3))
        ans = rows[i]["一回目"]
        ok = board.teach(si, tri, None if ans is None else (ans, POS[si]))
        rows[i]["教え"] = ("合っている" if ans == w else "違う。正しくは " + w) + ("" if ok else "（属性の無い語が有り、板に入らなかった）")
    for row, (kind, si, q) in zip(rows, qs):
        cs = [c for c in cands[si] if c not in {w for w in q if w is not None}]
        o = body.ask(si, q, cs)
        s = board.points(si, q, cs)
        b = max(s, key=lambda c: (s[c], -cs.index(c))) if s and max(s.values()) > 0 else None
        row["板を外した答え"] = None if o["答え"] is None else o["答え"][0]
        row["外すと一回目と同じ"] = row["板を外した答え"] == row["一回目"]
        row["板の答え"] = None if b is None else b[0]
        row["板の上位"] = top3({c: v for c, v in s.items() if v != 0})
        row["教えた組そのもの"] = board.taught(si, q)
    return rows


def to_md(rows, taught):
    if not taught:
        head = "| 番号 | 種類 | 問い | 答え | 答えた段 | 上位 3 |\n|---|---|---|---|---|---|\n"
        return head + "".join("| %d | %s | %s | %s | %s | %s |\n" % (i + 1, r["種類"], r["問い"], r["一回目"], r["一回目の段"], r["一回目の上位"]) for i, r in enumerate(rows))
    head = "| 番号 | 種類 | 問い | 教える前の答え | 教え | 板の答え | 板の上位 3 | 板を外した答え |\n|---|---|---|---|---|---|---|---|\n"
    return head + "".join("| %d | %s | %s | %s | %s | %s | %s | %s%s |\n" % (
        i + 1, r["種類"], r["問い"], r["一回目"], r.get("教え", ""), r["板の答え"] if r["板の答え"] is not None else "（板は答えない）", r["板の上位"],
        r["板を外した答え"], "" if r["外すと一回目と同じ"] else "（一回目と違う）") for i, r in enumerate(rows))


def main():
    import i49c_read_words as Rw
    import i49m_attr_base as AB
    from i49zx_teach_board import TeachBoard
    ap = argparse.ArgumentParser()
    for k in ("text", "wave", "content", "ask", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--teach")
    ap.add_argument("--dark", type=float, default=1.0 / 1000)
    a = ap.parse_args()
    talks = []
    for f in a.text.split(","):
        talks += ZP.talks_of([l.strip() for l in open(f, encoding="utf-8").read().splitlines()])[0]
    wv = Rw.WordVecs.load(a.wave)

    def wave_of(w):
        v = wv.of(*w)
        return wv.of(w[0], "PROPN") if v is None and w[1] == "NOUN" else v

    full = AB.AttrBase(json.load(open(a.content, encoding="utf-8")))
    body = TR.body_of(wave_of, full.of)
    for tris, _, _ in talks:
        for tri in tris:
            body.learn(tri)
        body.tick()
    qs = read_ask(a.ask)
    teach = read_teach(a.teach) if a.teach else {}
    extra = read_extra(a.ask)
    rows = try_run(body, TeachBoard(full.of, dark=a.dark), qs, teach, extra)
    no_attr = sorted({w[0] for _, _, q in qs for w in q if w is not None and full.of(w) is None} | {w[0] for x in extra for w in x if full.of(w) is None})
    md = to_md(rows, bool(teach)) + ("\n属性の無い語（板の鍵に使えない）: " + "、".join(no_attr) + "\n" if no_attr else "")
    open(a.out, "w", encoding="utf-8").write(md)
    json.dump(rows, open(os.path.splitext(a.out)[0] + ".json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(md)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
