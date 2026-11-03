"""青空文庫の会話文の読み込み — 会話文・話者の一人称の類・続けて並ぶ会話の列（M1 から）

データ: data/aozora/（SN2-Parts-2D が青空文庫から取得した 1,000 作品を、持ち主の許可を得て写したもの、2026-10-24）。
        著作権の切れた作品（分類 913・新字新仮名、作品・作者とも著作権フラグなし）。青空文庫の「収録ファイルの取り扱い規準」により自由に利用できる。
出どころ: 会話文の取り出し方（clean・「」の正規表現）・一人称の表（FIRST）・12 類へのまとめ方（GROUP）・作品の分け方は、
          SN2-Parts-2D の experiments/text/aozoradata.py と exp71_wig.py（同じ著作権者の MIT）と同じにした。SN2 の実験72 の数字と比べるため。
足したもの: 行の並びを残し、地の文をはさまずに続く会話文の行を 1 つの「列」にまとめる（話者のラベルを広げる見積もり、est_label_spread.py）。

使い方:
    import aozora
    d = aozora.load()          # 1 回目は MeCab で一人称を判定してキャッシュ（data/cache/dialogue.json）を作る（数分）
    d["works"][0]["runs"]      # 作品ごとの会話の列: [[{"u": 会話文, "fp": 一人称の類 or None}, ...], ...]
    d["test_works"]            # テストに回す作品の番号（作品の 15%）
"""
from __future__ import annotations

import json
import os
import random
import re
import zipfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_ROOT = os.environ.get("KOTOYUI_DATA", os.path.join(ROOT, "..", "Project_Kotoyui_private", "data"))   # 会話のデータは非公開の置き場（2026-11-03 持ち主の依頼、git の外）
DATA = os.path.join(DATA_ROOT, "aozora")
CACHE = os.path.join(DATA_ROOT, "cache", "dialogue.json")
# 作品の組: "1000" = 最初の 1,000 作品（picked.json、SN2 の実験72・見積もり E1〜E4 と同じ）、"all" = 取得した全部（picked_all.json、2026-10-25 に 4,477 作品）
SETS = {"1000": ("picked.json", CACHE), "all": ("picked_all.json", os.path.join(DATA_ROOT, "cache", "dialogue_all.json"))}

# 一人称（表記ゆれをまとめる）。SN2-Parts-2D の aozoradata.py と同じ
FIRST = {
    "私": "私", "わたし": "わたし", "わたくし": "わたくし", "あたし": "あたし", "あたい": "あたい", "あっし": "あっし",
    "わっし": "わっし", "わし": "わし", "儂": "わし", "僕": "僕", "ぼく": "僕", "俺": "俺", "おれ": "俺", "おいら": "おいら",
    "おら": "おら", "己": "俺", "拙者": "拙者", "それがし": "それがし", "某": "それがし", "身共": "身共", "手前": "手前",
    "吾輩": "吾輩", "我輩": "吾輩", "わらわ": "わらわ", "妾": "わらわ", "小生": "小生", "自分": "自分", "うち": "うち",
    "我々": "我々", "われわれ": "我々", "私たち": "私たち", "僕ら": "僕ら", "俺たち": "俺たち", "わちき": "わちき", "朕": "朕",
}
# 12 類へのまとめ方。SN2-Parts-2D の exp71_wig.py と同じ（ここにない一人称は使わない）
GROUP = {"私": "私", "わたし": "私", "わたくし": "わたくし", "あたし": "あたし", "あたい": "あたし", "僕": "僕", "僕ら": "僕", "俺": "俺",
         "おいら": "おいら", "おら": "おいら", "俺たち": "俺", "わし": "わし", "拙者": "武士", "身共": "武士", "それがし": "武士", "吾輩": "わし",
         "わらわ": "わらわ", "あっし": "あっし", "わっし": "あっし", "わちき": "わちき", "我々": "我々", "私たち": "我々"}
CLASSES = ["私", "僕", "俺", "わし", "あたし", "あっし", "わたくし", "我々", "武士", "おいら", "わらわ", "わちき"]
_UTT = re.compile(r"「([^「」]{2,200})」")


def clean(text):
    """ルビ《》・注記［＃…］・前後の解説を除いた本文。"""
    text = re.sub(r"《[^》]*》", "", text)
    text = re.sub(r"［＃[^］]*］", "", text)
    text = text.replace("｜", "")
    parts = re.split(r"-{20,}", text)
    body = parts[2] if len(parts) >= 3 else text
    return body.split("底本：")[0]


def runs(text):
    """会話文を、地の文をはさまずに続く列にまとめる。前の会話文の「」の後から次の会話文の「までが空白・改行だけなら同じ列、
    地の文が 1 文字でもあれば新しい列にする。会話文は本文全体に「」の正規表現を当てて取る（行をまたぐ会話文も取れる。SN2 の utterances と同じ並び）。"""
    out, cur, last = [], [], 0
    for m in _UTT.finditer(text):
        if cur and text[last:m.start()].strip():
            out.append(cur)
            cur = []
        cur.append(m.group(1).strip())
        last = m.end()
    if cur:
        out.append(cur)
    return out


_tagger = None


def first_persons(u):
    """会話文の中の一人称。MeCab（unidic）で代名詞と判定されたものだけ。"""
    global _tagger
    if _tagger is None:
        import fugashi
        _tagger = fugashi.Tagger()
    return [FIRST[w.surface] for w in _tagger(u) if w.surface in FIRST and w.feature.pos1 == "代名詞"]


def label(u):
    """会話文の一人称（表記をまとめたもの）。1 種類だけのときだけ。なし・2 種類以上は None。12 類へのまとめは labeled() で行う。"""
    found = set(first_persons(u))
    return found.pop() if len(found) == 1 else None


def mask(u):
    """会話文の中の一人称を「＊」で隠す（長いものから）。"""
    for p in sorted(FIRST, key=len, reverse=True):
        u = u.replace(p, "＊")
    return u


def build(wset="1000"):
    """全部の作品を読み、会話の列と一人称の類を作ってキャッシュに保存する。"""
    listfile, cache = SETS[wset]
    meta = json.load(open(os.path.join(DATA, listfile), encoding="utf-8"))
    works = []
    for m in meta:
        fn = os.path.join(DATA, "raw", f"{m['id']}.zip")
        if not os.path.exists(fn):
            continue
        z = zipfile.ZipFile(fn)
        txt = [n for n in z.namelist() if n.endswith(".txt")]
        if not txt:
            continue
        text = clean(z.read(txt[0]).decode("cp932", errors="ignore"))
        rs = [[{"u": u, "fp": label(u)} for u in r] for r in runs(text)]
        works.append({"id": m["id"], "author": m.get("author", ""), "runs": rs})
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(works, f, ensure_ascii=False)
    return works


def load(wset="1000"):
    """キャッシュがあれば読み、なければ作る。作品の 15% をテストに回す（SN2 の exp71_wig.py と同じ乱数・同じ分け方）。"""
    cache = SETS[wset][1]
    works = json.load(open(cache, encoding="utf-8")) if os.path.exists(cache) else build(wset)
    ids = list(range(len(works)))
    random.Random(0).shuffle(ids)
    test = set(ids[:30] if len(works) <= 150 else ids[:len(works) * 15 // 100])
    return {"works": works, "test_works": test}


def cls(fp, sn2=False):
    """一人称 → 類。12 類にまとまらないもの（自分・うち・手前・小生・朕）は None。
    sn2=True: SN2-Parts-2D の exp71_wig.py と同じく、まとまらないものもそのまま 1 類として残す（SN2 の実験72 の件数 19,326 / 4,101 に合わせるとき）。"""
    if fp is None:
        return None
    return GROUP.get(fp, fp if sn2 else None)


def labeled(d, split, sn2=False):
    """一人称の類がついた会話文 [(作品の番号, 隠した会話文, 類)]。split は "train" か "test"。"""
    te = d["test_works"]
    return [(wi, mask(x["u"]), cls(x["fp"], sn2)) for wi, w in enumerate(d["works"]) if (wi in te) == (split == "test")
            for r in w["runs"] for x in r if cls(x["fp"], sn2)]
