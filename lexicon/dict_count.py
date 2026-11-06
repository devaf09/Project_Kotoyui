"""無料の辞書を同じ形の表にそろえ、語数と属性の種類の数を数える（語板を作る前の見積もり）

持ち主（2026-11-05）: 「文法を理解するために単語と文字について先に進めたい。網羅的に理解するため、無料の辞書をいくつかダウンロードし、
語板作成前に単語数と属性数を網羅するために数える。ダウンロードしたデータはその後も使うので保存する」。
辞書（どれも無料で再配布できる。取ってきた元は README_dict.md に書く）:
  ipadic     : mecab-ipadic 2.7.0（npm mecab-ipadic-seed の CSV。この版は UTF-8）。品詞 4 段 + 活用型・活用形。
  unidic     : UniDic 2.1.2（PyPI unidic-lite の sys.dic。MeCab の辞書の形を直接読む）。品詞 4 段 + 活用 + 語彙素・語種。
  sudachi    : SudachiDict core（PyPI sudachidict-core、SudachiPy の entries() で全部の見出しを読む）。品詞 6 段。
  jmdict     : JMdict / JMnedict / KANJIDIC2（PyPI jamdict-data の SQLite）。見出し・読み・意味・品詞タグ・分野。漢字（文字）の表も。
出す物（非公開の置き場 ../Project_Kotoyui_private/data/dict/）:
  lex_<辞書>.tsv.gz : 1 行 = 1 見出し。surface, lemma, reading, pos1..pos6, ctype, cform, extra（そろえた形）
  counts.json       : 辞書ごとの見出し数・異なり語数（表記・見出し語）・品詞の組の数・品詞 1 段目ごとの数・活用の種類・重なり
実行: python lexicon/dict_count.py --src <展開した辞書の置き場> --out <非公開の置き場>
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import os
import sqlite3
import struct
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
PRIV = os.environ.get("KOTOYUI_PRIVATE", os.path.join(HERE, "..", "..", "Project_Kotoyui_private"))
COLS = ["surface", "lemma", "reading", "pos1", "pos2", "pos3", "pos4", "pos5", "pos6", "ctype", "cform", "extra"]


def norm(x):
    return "" if x in (None, "*") else x


def read_ipadic(src):
    for f in sorted(glob.glob(os.path.join(src, "package", "lib", "dict", "*.csv"))):
        with open(f, encoding="utf-8") as fh:
            for r in csv.reader(fh):
                # 表層形,左,右,コスト,品詞,品詞細分類1,2,3,活用型,活用形,原形,読み,発音
                r += [""] * (13 - len(r))
                yield dict(surface=r[0], lemma=norm(r[10]), reading=norm(r[11]), pos1=norm(r[4]), pos2=norm(r[5]), pos3=norm(r[6]),
                           pos4=norm(r[7]), pos5="", pos6="", ctype=norm(r[8]), cform=norm(r[9]), extra=os.path.basename(f))


def read_mecab_sysdic(path):
    """MeCab の sys.dic: 見出し（トークン）ごとの素性の文字列を取り出す（表層形は素性の orth を使う）。"""
    with open(path, "rb") as fh:
        b = fh.read()
    magic, ver, typ, lexsize, lsize, rsize, dsize, tsize, fsize, _ = struct.unpack("<10I", b[:40])
    off_tok = 72 + dsize
    off_feat = off_tok + tsize
    feats = b[off_feat:off_feat + fsize]
    for i in range(lexsize):
        lc, rc, posid, wcost, fo, comp = struct.unpack("<HHHhII", b[off_tok + 16 * i: off_tok + 16 * (i + 1)])
        end = feats.index(b"\0", fo)
        yield feats[fo:end].decode("utf-8")


def read_unidic(src):
    # UniDic 2.1.2 の素性: pos1,pos2,pos3,pos4,cType,cForm,lForm,lemma,orth,pron,orthBase,pronBase,goshu,iType,iForm,fType,fForm,...
    for feat in read_mecab_sysdic(os.path.join(src, "unidic-lite-1.0.8", "unidic_lite", "dicdir", "sys.dic")):
        r = next(csv.reader([feat]))
        r += [""] * (17 - len(r))
        yield dict(surface=norm(r[8]), lemma=norm(r[7]), reading=norm(r[6]), pos1=norm(r[0]), pos2=norm(r[1]), pos3=norm(r[2]), pos4=norm(r[3]),
                   pos5="", pos6="", ctype=norm(r[4]), cform=norm(r[5]), extra=f"語種:{norm(r[12])}")


def read_sudachi(src):
    from sudachipy import Dictionary
    d = Dictionary(dict=os.path.join(src, "sudachi", "sudachidict_core", "resources", "system.dic"))
    for m in d.entries():
        p = list(m.part_of_speech()) + [""] * 6
        yield dict(surface=m.surface(), lemma=m.dictionary_form(), reading=m.reading_form(), pos1=norm(p[0]), pos2=norm(p[1]), pos3=norm(p[2]),
                   pos4=norm(p[3]), pos5="", pos6="", ctype=norm(p[4]), cform=norm(p[5]), extra=f"正規化:{m.normalized_form()}")


def read_jmdict(src):
    """JMdict: 1 行 = 1 見出し（表記 × 意味のまとまり）ではなく、1 行 = 1 表記（漢字があれば漢字、無ければかな）。品詞は最初の意味のタグ。"""
    db = sqlite3.connect(os.path.join(src, "jamdict_data-1.5", "jamdict_data", "jamdict.db"))
    pos = {}
    for idseq, t in db.execute("SELECT Sense.idseq, pos.text FROM pos JOIN Sense ON pos.sid = Sense.ID ORDER BY Sense.ID"):
        pos.setdefault(idseq, [])
        if t not in pos[idseq]:
            pos[idseq].append(t)
    kana = {}
    for idseq, t in db.execute("SELECT idseq, text FROM Kana ORDER BY ID"):
        kana.setdefault(idseq, t)
    seen = set()
    for idseq, t in db.execute("SELECT idseq, text FROM Kanji ORDER BY ID"):
        seen.add(idseq)
        p = pos.get(idseq, []) + [""] * 6
        yield dict(surface=t, lemma=t, reading=kana.get(idseq, ""), pos1=p[0], pos2=p[1], pos3=p[2], pos4=p[3], pos5="", pos6="", ctype="", cform="", extra=f"idseq:{idseq}")
    for idseq, t in db.execute("SELECT idseq, text FROM Kana ORDER BY ID"):
        if idseq in seen:
            continue
        p = pos.get(idseq, []) + [""] * 6
        yield dict(surface=t, lemma=t, reading=t, pos1=p[0], pos2=p[1], pos3=p[2], pos4=p[3], pos5="", pos6="", ctype="", cform="", extra=f"idseq:{idseq}")


def jmdict_extra(src):
    db = sqlite3.connect(os.path.join(src, "jamdict_data-1.5", "jamdict_data", "jamdict.db"))
    q = lambda s: db.execute(s).fetchone()[0]
    return dict(
        見出し=q("SELECT COUNT(*) FROM Entry"), 漢字表記=q("SELECT COUNT(*) FROM Kanji"), かな表記=q("SELECT COUNT(*) FROM Kana"),
        意味のまとまり=q("SELECT COUNT(*) FROM Sense"), 英訳=q("SELECT COUNT(*) FROM SenseGloss"),
        品詞タグの種類=q("SELECT COUNT(DISTINCT text) FROM pos"), 分野タグの種類=q("SELECT COUNT(DISTINCT text) FROM field"),
        その他タグの種類=q("SELECT COUNT(DISTINCT text) FROM misc"), 方言タグの種類=q("SELECT COUNT(DISTINCT text) FROM dialect"),
        品詞タグ=dict(Counter(t for (t,) in db.execute("SELECT text FROM pos")).most_common()),
        固有名詞の見出し_JMnedict=q("SELECT COUNT(*) FROM NEEntry"),
        漢字_KANJIDIC2=q("SELECT COUNT(*) FROM character"),
        漢字の部首の種類=q("SELECT COUNT(DISTINCT value) FROM radical") if db.execute("SELECT COUNT(*) FROM radical").fetchone()[0] else 0,
        漢字の読み=q("SELECT COUNT(*) FROM reading"), 漢字の意味_英=q("SELECT COUNT(*) FROM meaning WHERE m_lang = '' OR m_lang IS NULL"),
        漢字_学年つき=q("SELECT COUNT(*) FROM character WHERE grade IS NOT NULL AND grade != ''"),
    )


READERS = dict(ipadic=read_ipadic, unidic=read_unidic, sudachi=read_sudachi, jmdict=read_jmdict)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="展開した辞書の置き場")
    ap.add_argument("--out", default=os.path.join(PRIV, "data", "dict"))
    ap.add_argument("--only", nargs="*", default=list(READERS))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    counts, surf, lem = {}, {}, {}
    for name in a.only:
        rows = 0
        S, L, P, P1, CT, CF = set(), set(), Counter(), Counter(), set(), set()
        with gzip.open(os.path.join(a.out, f"lex_{name}.tsv.gz"), "wt", encoding="utf-8") as fh:
            fh.write("\t".join(COLS) + "\n")
            for r in READERS[name](a.src):
                fh.write("\t".join(str(r[c]).replace("\t", " ") for c in COLS) + "\n")
                rows += 1
                S.add(r["surface"])
                L.add((r["lemma"] or r["surface"], r["pos1"]))
                P[tuple(r[c] for c in ("pos1", "pos2", "pos3", "pos4"))] += 1
                P1[r["pos1"]] += 1
                if r["ctype"]:
                    CT.add(r["ctype"])
                if r["cform"]:
                    CF.add(r["cform"])
        surf[name], lem[name] = S, {l for l, _ in L}
        counts[name] = dict(見出しの行=rows, 異なり表記=len(S), 異なり見出し語_品詞1段目つき=len(L), 品詞の組_4段=len(P), 品詞1段目の種類=len(P1),
                            活用型の種類=len(CT), 活用形の種類=len(CF), 品詞1段目ごとの行=dict(P1.most_common()),
                            品詞の組=[dict(pos="-".join(x for x in k if x), n=v) for k, v in P.most_common()])
        print(name, rows, len(S), flush=True)
    if "jmdict" in a.only:
        counts["jmdict"]["詳しく"] = jmdict_extra(a.src)
    names = list(surf)
    counts["重なり_異なり表記"] = {f"{x}&{y}": len(surf[x] & surf[y]) for i, x in enumerate(names) for y in names[i + 1:]}
    counts["全部の辞書の和_異なり表記"] = len(set().union(*surf.values()))
    counts["全部の辞書に共通_異なり表記"] = len(set.intersection(*surf.values()))
    json.dump(counts, open(os.path.join(a.out, "counts.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
