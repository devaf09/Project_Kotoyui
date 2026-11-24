"""I49zx: 介入の動作試験（持ち主の言葉「今の実験が一通り終わったら介入の動作試験をしてみようか」から）。

人が教える役になって、介入の板の動きを前後で並べて見るための小さな道具。数えるための道具ではない。
流れ:
  1. 本体（数えて選ぶ）に、材料のテキストを全部聞かせる（--text。リポジトリに入れない材料）。
  2. 問いのファイル（--ask）の問いに答えさせる。教えのファイル（--teach）が無ければ、ここで終わり（1 回目）。
  3. 教えのファイルが有れば、その教えを介入の板に入れてから、同じ問いにもう一度答えさせ、前後を並べる。
     本体の答えと板の答えの両方を出す。板を外した時の答え（= 本体の答え）が 1 回目と同じ事も確かめて出す。
問いのファイル（UTF-8、タブ区切り、1 行 1 問）:  種類 <TAB> が <TAB> を <TAB> どうした    隠す席は「？」。種類は自由な名札（教える・似た問い・関係ない、など）
教えのファイル（UTF-8、タブ区切り）:            問いの番号（1 から） <TAB> 正しい語      出した答えで合っていれば、その語をそのまま書く
                                              同じ番号を何行も書ける（「この語も合っている」）。出した答えが、その番号の正しい語のどれかなら、暗い側には入れない
                                              正しい語の代わりに「×語」と書くと、「板のこの答えは違う」（I49zz）。その問いの組で、その語を板の止める側に入れる
                                              正しい語の代わりに「遠い」と書くと、「この問いは板の出番ではない」（I49zza）。板がその問いで出している語を全部、出した分だけ止める
候補: 問いのファイルと教えのファイルに出た語のうち、同じ席の語の全部。問いのファイルに「候補 <TAB> 語、語 <TAB> 語、語 <TAB> 語、語」の行を
      置くと、その語も席ごとの候補に足す（区切りは「、」）。
鍵の形（--key、カンマ区切りで並べると、形ごとに同じ問いと教えで表を出す。I49zy）:
  今まで      広げて掛けて、大きさをそろえる（I49zx の初めの形）
  引く        席ごとの「いつもの属性」（材料でその席に出た語の平均）を引いてから、広げて掛ける。大きさはそろえない
  めずらしさ  席ごとの「めずらしさ」（1 − その属性を持つ語の割合）を掛けてから、広げて掛ける。大きさはそろえない
引数: --text 材料（カンマ区切り） --wave 語の波 --content 内容語の板 --ask 問い.tsv [--teach 教え.tsv] --out 結果.md [--dark 0.001] [--key 今まで] [--dim 8192] [--stop 1 か 出した分]
手で置く物: 暗い側の強さ。上位を 3 つまで見せる事。いつもの属性・めずらしさを、材料でその席に出た語（1 語 1 回）から作る事。
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
DENY = ("×", "✕")
FAR = "遠い"


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
            out.setdefault(int(p[0]) - 1, [])
            if p[1] not in out[int(p[0]) - 1]:
                out[int(p[0]) - 1].append(p[1])
    return out


def cands_of(qs, teach, extra=None):
    c = [set(x) for x in extra] if extra else [set(), set(), set()]
    for _, si, q in qs:
        for j, w in enumerate(q):
            if w is not None:
                c[j].add(w)
    for i, ws in teach.items():
        c[qs[i][1]] |= {(w[1:] if w[:1] in DENY else w, POS[qs[i][1]]) for w in ws if w != FAR}
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
    signs = []
    deny = {i: [w[1:] for w in ws if w[:1] in DENY] for i, ws in teach.items()}
    far = sorted(i for i, ws in teach.items() if FAR in ws)
    teach = {i: [w for w in ws if w[:1] not in DENY and w != FAR] for i, ws in teach.items()}
    for i, ws in sorted(teach.items()):
        if not ws:
            continue
        kind, si, q = qs[i]
        if board.pair.sign(si, q) is not None:
            signs.append((si, board.pair.sign(si, q)))
        ans = rows[i]["一回目"]
        ok = True
        for w in ws:
            tri = tuple((w, POS[si]) if j == si else q[j] for j in range(3))
            ok = board.teach(si, tri, None if ans is None or ans in ws else (ans, POS[si])) and ok
        also = [w for w in ws if w != ans]
        rows[i]["教え"] = (("合っている" + ("。" + "、".join(also) + " も合っている" if also else "")) if ans in ws else "違う。正しくは " + "、".join(ws)) \
            + ("" if ok else "（属性の無い語が有り、板に入らなかった）")
    for i, ws in sorted(deny.items()):                     # 「違う」は、合っている教えを全部入れた後に入れる（板が出した分を見るため）
        done = [board.deny(qs[i][1], qs[i][2], (w, POS[qs[i][1]])) for w in ws]
        if ws:
            rows[i]["教え"] = (rows[i]["教え"] + "。" if rows[i].get("教え") else "") + "板の " + "、".join(ws) + " は違う" \
                + ("" if all(done) else "（板はその語を出していなかったので、何も入れていない）")
    for i in far:                                          # 「遠い」も、ほかの教えを全部入れた後に入れる
        n = board.far(qs[i][1], qs[i][2])
        rows[i]["教え"] = (rows[i]["教え"] + "。" if rows[i].get("教え") else "") + "遠い（板の出番ではない。止めた語 %d）" % n
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
        p = board.pair.sign(si, q)
        near = [float((p.conj() * t).sum().real) for sj, t in signs if sj == si] if p is not None else []
        row["教えた組との一番の近さ"] = round(max(near), 4) if near else None
    return rows


def to_md(rows, taught):
    if not taught:
        head = "| 番号 | 種類 | 問い | 答え | 答えた段 | 上位 3 |\n|---|---|---|---|---|---|\n"
        return head + "".join("| %d | %s | %s | %s | %s | %s |\n" % (i + 1, r["種類"], r["問い"], r["一回目"], r["一回目の段"], r["一回目の上位"]) for i, r in enumerate(rows))
    head = "| 番号 | 種類 | 問い | 教える前の答え | 教え | 板の答え | 板の上位 3 | 教えた組との一番の近さ | 板を外した答え |\n|---|---|---|---|---|---|---|---|---|\n"
    return head + "".join("| %d | %s | %s | %s | %s | %s | %s | %s | %s%s |\n" % (
        i + 1, r["種類"], r["問い"], r["一回目"], r.get("教え", ""), r["板の答え"] if r["板の答え"] is not None else "（板は答えない）", r["板の上位"],
        "" if r["教えた組との一番の近さ"] is None else "%.2f" % r["教えた組との一番の近さ"], r["板を外した答え"], "" if r["外すと一回目と同じ"] else "（一回目と違う）") for i, r in enumerate(rows))


def board_of(attr_of, key, seats, dark=1.0 / 1000, dim=8192, stop=1.0):
    import i49zw_dense_pair as D
    from i49zx_teach_board import TeachBoard
    assert key in ("今まで", "引く", "めずらしさ"), key
    if key == "今まで":
        return TeachBoard(attr_of, dim=dim, dark=dark, stop=stop)
    if key == "引く":
        return TeachBoard(attr_of, dim=dim, dark=dark, stop=stop, amp="そのまま", usual=D.usual_of(attr_of, seats))
    return TeachBoard(attr_of, dim=dim, dark=dark, stop=stop, amp="そのまま", rare=D.rare_of(attr_of, seats))


def main():
    import i49c_read_words as Rw
    import i49m_attr_base as AB
    ap = argparse.ArgumentParser()
    for k in ("text", "wave", "content", "ask", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--teach")
    ap.add_argument("--dark", type=float, default=1.0 / 1000)
    ap.add_argument("--key", default="今まで")
    ap.add_argument("--dim", type=int, default=8192)
    ap.add_argument("--stop", default="1")
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
    seats = [[tri[j] for tris, _, _ in talks for tri in tris] for j in range(3)]
    no_attr = sorted({w[0] for _, _, q in qs for w in q if w is not None and full.of(w) is None} | {w[0] for x in extra for w in x if full.of(w) is None})
    md, res = "", {}
    for key in a.key.split(","):
        rows = try_run(body, board_of(full.of, key, seats, a.dark, a.dim, a.stop if a.stop == "出した分" else float(a.stop)), qs, teach, extra)
        md += "鍵の形: %s\n\n" % key + to_md(rows, bool(teach)) + "\n"
        res[key] = rows
    md += "属性の無い語（板の鍵に使えない）: " + "、".join(no_attr) + "\n" if no_attr else ""
    open(a.out, "w", encoding="utf-8").write(md)
    json.dump(res, open(os.path.splitext(a.out)[0] + ".json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(md)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
