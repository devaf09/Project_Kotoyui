"""特別な語板（品詞語板・機能語語板）の属性の表を作る（第 1 版: 辞書だけ）

持ち主（2026-11-05）: 「文法上重要な語板を作る。現段階では品詞、機能語を優先して抽出し、品詞語板、機能語語板という特別な語板を作る」。
語板 = 1 つの物 = 1 台の光の器、出口の明るさ = 属性の割合（介入の研究と同じ）。ここでは学ばせる前の「属性の割合」を辞書から数える。
  品詞語板: 品詞（UniDic の 1〜2 段目）ごとに 1 枚。属性 = 自立/付属、活用する/しない、活用型のまとまり、活用形、語種、書字（ひらがな・カタカナ・漢字…）、長さ。
  機能語語板: 機能語（助詞・助動詞・補助動詞・補助形容詞・形式名詞・指示詞・接続詞・フィラー）ごとに 1 枚。
    属性 = 品詞、機能（手の表: 格・主題・取り立て・接続・終助・準体・コピュラ・テンス・アスペクト・ヴォイス・否定・丁寧・願望・推量・伝聞…）、
           格の意味（が = 主格 など。主な格助詞だけ手の表）、活用型のまとまり、活用形、文語か、語種、長さ。
  親（機能のまとまり）: 子の割合の和で作る（光の融合と同じ）ので、ここでは子だけを出す。
文の中での振る舞い（前に付く語の品詞・活用形、係り方）は、UD Japanese-GSD を読めるようになったら第 2 版で足す。
出す物（非公開の置き場 ../Project_Kotoyui_private/data/boards/）: pos_boards.json、func_boards.json（1 枚 = {name, attrs: {観点: {値: 割合}}, n}）
実行: python lexicon/special_boards.py --dict <lex_*.tsv.gz の置き場> --out <書き出す先>
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PRIV = os.environ.get("KOTOYUI_PRIVATE", os.path.join(HERE, "..", "..", "Project_Kotoyui_private"))
FUNC_POS1 = {"助詞", "助動詞"}
DEPENDENT = {"助詞", "助動詞", "接尾辞"}                          # 付属語（接尾辞は自立しないので付属に入れる）

# 機能の手の表（助動詞・補助動詞などの主なもの。UniDic の見出し語の形）
AUX_FUNC = {
    "だ": "コピュラ", "です": "コピュラ", "じゃ": "コピュラ", "や": "コピュラ", "どす": "コピュラ", "ざます": "コピュラ", "ざんす": "コピュラ",
    "なり-断定": "コピュラ", "たり-断定": "コピュラ",
    "た": "テンス", "き": "テンス", "けり": "テンス", "つ": "テンス", "ぬ": "テンス", "たり-完了": "テンス", "り": "テンス",
    "てる": "アスペクト", "とる": "アスペクト", "ちゃう": "アスペクト", "ちまう": "アスペクト", "とく": "アスペクト", "てく": "アスペクト", "たる": "アスペクト",
    "れる": "ヴォイス", "られる": "ヴォイス", "せる": "ヴォイス", "させる": "ヴォイス", "しめる": "ヴォイス", "す": "ヴォイス", "さす": "ヴォイス",
    "ない": "否定", "ず": "否定", "へん": "否定", "ん": "否定", "なんだ": "否定", "じ": "否定", "まじ": "否定",
    "ます": "丁寧", "んす": "丁寧", "やす": "丁寧",
    "たい": "願望", "たがる": "願望",
    "う": "推量", "よう": "推量", "む": "推量", "けむ": "推量", "らむ": "推量", "べし": "推量", "まい": "推量", "らしい": "推量", "らし": "推量",
    "べらなり": "推量", "ごとし": "比況", "なり-伝聞": "伝聞",
    "しゃる": "尊敬", "はる": "尊敬", "てらっしゃる": "尊敬", "よる": "尊敬", "よらす": "尊敬", "とらす": "尊敬", "やがる": "卑罵", "ちゃる": "授受", "たげる": "授受",
    "つう": "引用", "ずら": "推量",
}
CASE_MEANING = {   # 主な格助詞・係助詞・副助詞の意味（手の表）
    "が": "主格", "を": "対格", "に": "着点・相手", "で": "場所・手段", "と": "相手・引用", "から": "起点", "へ": "方向", "より": "比較・起点",
    "の": "連体", "まで": "終点", "は": "主題", "も": "同類", "こそ": "強調", "だけ": "限定", "しか": "限定（否定と共起）", "さえ": "極端", "すら": "極端",
    "ばかり": "限定", "など": "例示", "くらい": "程度", "ほど": "程度", "ずつ": "配分", "なんか": "例示",
}
AUX_VERB_FUNC = {   # 補助動詞（て形のあと）の機能
    "いる": "アスペクト", "ある": "アスペクト", "しまう": "アスペクト", "おく": "アスペクト", "いく": "アスペクト", "くる": "アスペクト",
    "みる": "試行", "あげる": "授受", "やる": "授受", "くれる": "授受", "もらう": "授受", "いただく": "授受", "くださる": "授受", "下さる": "授受",
    "差し上げる": "授受", "頂く": "授受", "おる": "アスペクト", "いらっしゃる": "尊敬", "始める": "アスペクト", "終わる": "アスペクト", "続ける": "アスペクト",
    "出す": "アスペクト", "かける": "アスペクト", "過ぎる": "程度", "合う": "相互", "直す": "反復",
}
KOSOADO = {f"{k}{s}": (k, s) for s in ("れ", "の", "こ", "ちら", "っち", "う", "んな", "いつ", "なた") for k in ("こ", "そ", "あ", "ど")}
KOSOADO = {("あそこ" if w == "あこ" else ("ああ" if w == "あう" else w)): v for w, v in KOSOADO.items()}   # あ系の例外（あそこ・ああ）


def hira(s):
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)


def script_of(s):
    if re.fullmatch(r"[ぁ-ゖー]+", s):
        return "ひらがな"
    if re.fullmatch(r"[ァ-ヺー]+", s):
        return "カタカナ"
    if re.search(r"[一-鿿々]", s):
        return "漢字を含む"
    if re.fullmatch(r"[A-Za-zＡ-Ｚａ-ｚ]+", s):
        return "ローマ字"
    return "記号・その他"


def ctype_group(ct):
    if not ct:
        return "活用しない"
    if ct.startswith("文語"):
        return "文語"
    for g in ("五段", "上一段", "下一段", "サ行変格", "カ行変格", "形容詞", "助動詞"):
        if ct.startswith(g):
            return g
    for g, to in (("一段", "上一段・下一段"), ("サ変", "サ行変格"), ("カ変", "カ行変格"), ("特殊", "助動詞"), ("不変化", "活用しない")):
        if ct.startswith(g):                                         # IPAdic の活用型の名前
            return to
    return "その他の活用"


IPA_CFORM = (("基本形", "終止形"), ("未然ウ接続", "意志推量形"), ("未然", "未然形"), ("連用", "連用形"), ("仮定", "仮定形"), ("命令", "命令形"),
             ("体言接続", "連体形"), ("ガル接続", "語幹"), ("文語基本形", "終止形"), ("音便基本形", "終止形"))


def cform_head(cf):
    """活用形を 1 段目にそろえる（UniDic の「連用形-促音便」→ 連用形、IPAdic の「連用タ接続」→ 連用形）。"""
    if not cf:
        return "なし"
    head = cf.split("-")[0]
    for k, to in IPA_CFORM:
        if head.startswith(k):
            return to
    return head


def length_bin(s):
    n = len(s)
    return "1字" if n == 1 else ("2字" if n == 2 else ("3字" if n == 3 else "4字以上"))


def load(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        cols = next(f).rstrip("\n").split("\t")
        for line in f:
            yield dict(zip(cols, line.rstrip("\n").split("\t")))


def norm_attrs(c):
    out = {}
    for v, cnt in c.items():
        tot = sum(cnt.values())
        out[v] = {x: round(n / tot, 4) for x, n in cnt.most_common()}
    return out


def pos_boards(uni):
    boards = defaultdict(lambda: defaultdict(Counter))
    lemmas = defaultdict(set)
    for r in uni:
        key = r["pos1"] + ("-" + r["pos2"] if r["pos1"] in ("名詞", "助詞", "接尾辞", "補助記号", "動詞", "形容詞", "形状詞") and r["pos2"] else "")
        A = boards[key]
        A["自立"]["付属語" if r["pos1"] in DEPENDENT else "自立語"] += 1
        A["活用"]["する" if r["ctype"] else "しない"] += 1
        A["活用型"][ctype_group(r["ctype"])] += 1
        A["活用形"][cform_head(r["cform"])] += 1
        A["語種"][r["extra"].replace("語種:", "") or "不明"] += 1
        A["書字"][script_of(r["surface"])] += 1
        A["長さ"][length_bin(r["surface"])] += 1
        lemmas[key].add(r["lemma"] or r["surface"])
    return [dict(name=k, kind="品詞", n_rows=sum(boards[k]["自立"].values()), n_lemmas=len(lemmas[k]), attrs=norm_attrs(boards[k]))
            for k in sorted(boards, key=lambda k: -len(lemmas[k]))]


def func_boards(uni_rows, ipa_rows):
    boards = defaultdict(lambda: defaultdict(Counter))
    info = {}

    def add(name, r, func, src):
        A = boards[name]
        A["品詞"]["-".join(x for x in (r["pos1"], r["pos2"], r["pos3"]) if x)] += 1
        A["機能"][func] += 1
        if func == "指示詞":
            k, sfx = KOSOADO[name.split("|")[0]]
            A["距離"][{"こ": "近称", "そ": "中称", "あ": "遠称", "ど": "不定称"}[k]] += 1
            A["系列"][{"れ": "もの", "の": "連体", "こ": "場所", "ちら": "方向", "っち": "方向（くだけた）", "う": "様子", "んな": "連体（様子）", "いつ": "人（くだけた）", "なた": "人"}[sfx]] += 1
        if name.split("|")[0] in CASE_MEANING and r["pos1"] == "助詞":
            A["意味"][CASE_MEANING[name.split("|")[0]]] += 1
        A["活用型"][ctype_group(r["ctype"])] += 1
        A["活用形"][cform_head(r["cform"])] += 1
        A["文語"]["文語" if r["ctype"].startswith("文語") else "口語"] += 1
        A["長さ"][length_bin(name.split("|")[0].split("-")[0])] += 1
        info.setdefault(name, dict(src=set(), func=func))["src"].add(src)

    for r in uni_rows:
        lem = r["lemma"] or r["surface"]
        p1, p2 = r["pos1"], r["pos2"]
        func = None
        if p1 == "助詞":
            func = {"格助詞": "格", "係助詞": "取り立て", "副助詞": "取り立て", "接続助詞": "接続", "終助詞": "終助", "準体助詞": "準体"}.get(p2)
            if lem == "は":
                func = "主題"
            name = f"{lem}|{p2}"                                     # 同じ形でも品詞が違えば別の語板（から = 格助詞 / 接続助詞）
        elif p1 == "助動詞" or (p1 in ("形状詞", "名詞") and p2 == "助動詞語幹"):
            func, name = AUX_FUNC.get(lem, "その他の助動詞"), f"{lem}|助動詞"
            if p2 == "助動詞語幹":
                func = {"そう-様態": "様態", "みたい": "比況", "様": "比況", "そう-伝聞": "伝聞"}.get(lem, func)
        elif p1 == "動詞" and p2 == "非自立可能":
            yomi = hira(r["reading"]) or lem                         # 補助動詞は読み（ひらがな）で 1 枚にまとめる（仕舞う / しまう）
            func, name = AUX_VERB_FUNC.get(yomi, "補助動詞（その他）"), f"{yomi}|補助動詞"
        elif p1 == "形容詞" and p2 == "非自立可能":
            func, name = "補助形容詞", f"{lem}|補助形容詞"
        elif p1 == "接続詞":
            func, name = "接続詞", f"{lem}|接続詞"
        elif p1 == "感動詞" and p2 == "フィラー":
            func, name = "フィラー", f"{lem}|フィラー"
        elif p1 in ("代名詞", "連体詞", "副詞", "形状詞") and r["surface"] in KOSOADO:
            func, name = "指示詞", f"{r['surface']}|指示詞"
        if func:
            add(name, r, func, "unidic")
    base_yomi = {r["lemma"]: hira(r["reading"]) for r in ipa_rows if r["pos2"] == "非自立" and r["cform"] == "基本形"}
    for r in ipa_rows:                                               # 形式名詞・補助動詞・補助形容詞は IPAdic の「非自立」も使う
        lem = r["lemma"] or r["surface"]
        if r["pos1"] == "名詞" and r["pos2"] == "非自立":
            add(f"{lem}|形式名詞", r, "形式名詞", "ipadic")
        elif r["pos1"] == "動詞" and r["pos2"] == "非自立":
            yomi = base_yomi.get(lem) or hira(lem)                     # IPAdic の読みは活用した形なので、原形の行の読みを使う
            add(f"{yomi}|補助動詞", r, AUX_VERB_FUNC.get(yomi, "補助動詞（その他）"), "ipadic")
        elif r["pos1"] == "形容詞" and r["pos2"] == "非自立":
            add(f"{lem}|補助形容詞", r, "補助形容詞", "ipadic")
    out = []
    for k in sorted(boards):
        out.append(dict(name=k, word=k.split("|")[0], kind=k.split("|")[1], func=info[k]["func"], src=sorted(info[k]["src"]),
                        n_rows=sum(boards[k]["機能"].values()), attrs=norm_attrs(boards[k])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dict", default=os.path.join(PRIV, "data", "dict"))
    ap.add_argument("--out", default=os.path.join(PRIV, "data", "boards"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    uni = list(load(os.path.join(a.dict, "lex_unidic.tsv.gz")))
    ipa = list(load(os.path.join(a.dict, "lex_ipadic.tsv.gz")))
    P = pos_boards(uni)
    F = func_boards(uni, ipa)
    json.dump(P, open(os.path.join(a.out, "pos_boards.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(F, open(os.path.join(a.out, "func_boards.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("品詞語板", len(P), "機能語語板", len(F))
    print("機能ごとの数", dict(Counter(b["func"] for b in F).most_common()))


if __name__ == "__main__":
    main()
