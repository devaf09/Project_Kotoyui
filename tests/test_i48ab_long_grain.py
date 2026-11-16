"""I48ab（長いグレイン: 文 → 要点 → 文）の確かめ（小さな手作りの語板。GiNZA を使わない）。"""
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "intervention"))
import i48ab_long_grain as G  # noqa: E402

CB = [dict(name="犬", pos="名詞", freq=10, attrs=dict(D={"クラス:動物": 1.0}, B={"性質:白い": 0.5, "性質:小さい": 0.5})),
      dict(name="猫", pos="名詞", freq=9, attrs=dict(D={"クラス:動物": 1.0}, B={"性質:黒い": 1.0})),
      dict(name="友達", pos="名詞", freq=8, attrs=dict(D={"クラス:人": 1.0}, B={"性質:仲良い": 1.0})),
      dict(name="先生", pos="名詞", freq=7, attrs=dict(D={"クラス:人": 1.0}, B={"性質:えらい": 1.0})),
      dict(name="散歩", pos="名詞", freq=6, attrs=dict(D={"クラス:行ない": 1.0})),
      dict(name="行く", pos="動詞", freq=5, attrs=dict(C={"主語:人": 1.0}))]
FB = [dict(word="た", attrs={"品詞": {"助動詞": 1.0}, "機能": {"テンス": 1.0}}),
      dict(word="の", attrs={"品詞": {"助詞-格助詞": 1.0}, "機能": {"格": 1.0}}),
      dict(word="と", attrs={"品詞": {"助詞-格助詞": 1.0}, "機能": {"格": 1.0}})]


def t(form, lemma, upos, xpos="", dep="obl"):
    return dict(form=form, lemma=lemma, upos=upos, xpos=xpos, dep=dep)


def sent(n):
    return dict(id="x", text="", toks=[t(n, n, "NOUN"), t("の" if n in ("犬", "猫") else "と", "の" if n in ("犬", "猫") else "と", "ADP", "助詞-格助詞", "case"),
                                      t("散歩", "散歩", "NOUN"), t("に", "に", "ADP", "助詞-格助詞", "case"), t("行っ", "行く", "VERB", "動詞-一般", "root"),
                                      t("た", "た", "AUX", "助動詞", "aux"), t("。", "。", "PUNCT", "補助記号-句点", "punct")])


def test_split_and_marks():
    s = sent("犬")
    cont, gaps, head = G.split(s)
    assert [c[0] for c in cont] == ["犬", "散歩", "行く"] and gaps == ["の", "に", "た。"] and head == ""
    assert G.Func(FB).marks(s["toks"]) == {"過去"}


def test_encode_read_shift():
    S = G.AttrSpace(CB, G.Func(FB), 1024)
    toks = sent("犬")["toks"]
    X = S.encode(toks)
    lex = [dict(lemma=w["name"], upos="NOUN" if w["pos"] == "名詞" else "VERB", xpos="", form=w["name"]) for w in CB]
    M = np.stack([S.dense(*S.tok(x)) for x in lex])
    R = S.read(X, 0)
    assert lex[int(np.argmax(M @ R))]["lemma"] == "犬"          # 位置 0 のシフトを戻すと、犬の属性がいちばん近い


def test_pair_grain_generalizes_by_class():
    """学びに「先生」が無くても、同じクラスの「友達」の経験から「と」を選ぶ（語の表は、無い語ではいちばん多い詰め物に落ちる）。"""
    S = G.AttrSpace(CB, G.Func(FB), 512, bands=("D",))
    rows, cnt = [], Counter()
    for n, y in (("犬", "の"), ("猫", "の"), ("友達", "と")) * 3 + (("犬", "の"),):
        f, v = S.pair(S.attrs(n, "NOUN"), S.attrs("散歩", "NOUN"))
        rows.append((f, v, y))
        cnt[y] += 1
    pg = G.PairGrain(512).fit(iter(rows), cnt)
    f, v = S.pair(S.attrs("先生", "NOUN"), S.attrs("散歩", "NOUN"))
    assert pg.predict(f, v)[0] == "と"
    wt = G.WordTable().fit([("犬", "散歩", "の"), ("猫", "散歩", "の"), ("友達", "散歩", "と")])
    assert wt.predict("先生", "散歩") == "の"


def test_surface():
    s = G.Surface()
    s.add("行く", "行っ", "VERB", "た。")
    s.add("書く", "書い", "VERB", "た。")
    assert s.form("行く", "VERB", "た。") == "行っ"
    assert s.form("聞く", "VERB", "た。") in ("聞っ", "聞い")         # 無い原形は、最後の字と次の頭からの書き換え
