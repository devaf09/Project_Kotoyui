"""青空文庫の作品（新字新仮名・著作権の切れたもの）に GiNZA で係り受けを付け、文書の CoNLL-U で書き出す（語板 v2.2）

parse_docs.py と同じ形（1 作品 = 1 文書、「# newdoc id = aozora-<作品ID>」）。
入力: 青空文庫のテキスト（.txt か、.txt を 1 つ含む .zip。Shift_JIS か UTF-8）を含むフォルダと、作品の一覧
      （list_person_all_extended_utf8.csv など。列「作品ID」「文字遣い種別」「作品著作権フラグ」を使う）。
      一覧があれば、文字遣い種別 = 新字新仮名 かつ 作品著作権フラグ = なし の作品だけを使う。一覧が無ければ旧仮名の目安（ゐ・ゑ）で外す。
前処理: 冒頭の説明（-------- で挟まれた記号の説明）と末尾の「底本：」以降を落とし、ルビ《》・ルビの始め｜・注記［＃…］・外字の注を消す。
        文は 。！？ と改行で切り、4〜120 文字の文だけ使う。
実行: python lexicon/parse_aozora.py --src <青空文庫のフォルダ> --meta <作品の一覧.csv> --out aozora_ginza.conllu --max_sents 300000 [--n_process 4]
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import random
import re
import zipfile

import spacy

RUBY = re.compile(r"《[^》]*》")
NOTE = re.compile(r"［＃[^］]*］")
GAIJI = re.compile(r"※［[^］]*］")
SPLIT = re.compile(r"(?<=[。！？])|\n")
ID_IN_NAME = re.compile(r"^(\d+)_")   # 青空文庫のファイル名は「作品ID_ruby_番号」（例 789_ruby_5639.zip、5_ruby_21311.txt）


def decode(b):
    for enc in ("utf-8-sig", "cp932"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            pass
    return b.decode("cp932", errors="ignore")


def clean(text):
    text = text.replace("\r\n", "\n")
    parts = text.split("-------------------------------------------------------")
    if len(parts) >= 3:                                              # 冒頭の記号の説明を落とす
        text = parts[2]
    text = text.split("\n底本：")[0]
    text = GAIJI.sub("", text)
    text = RUBY.sub("", text).replace("｜", "")
    text = NOTE.sub("", text)
    return text


def read_text(path):
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".txt")]
            if not names:
                return None
            return decode(z.read(names[0]))
    return decode(open(path, "rb").read())


def load_meta(path):
    ok = set()
    if not path:
        return None
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r.get("文字遣い種別", "") == "新字新仮名" and r.get("作品著作権フラグ", "なし") == "なし":
                ok.add(r["作品ID"].lstrip("0"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--meta", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_sents", type=int, default=300000)
    ap.add_argument("--max_per_work", type=int, default=2000, help="1 作品から使う文の上限（長い作品に偏らないように）")
    ap.add_argument("--n_process", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ok = load_meta(a.meta)
    files = []
    for d, _, fs in os.walk(a.src):
        for f in fs:
            if f.lower().endswith((".txt", ".zip")):
                files.append(os.path.join(d, f))
    random.Random(a.seed).shuffle(files)
    rows, seen, n_doc, skipped = [], set(), 0, dict(一覧に無い=0, 旧仮名=0, 読めない=0)
    for p in files:
        if len(rows) >= a.max_sents:
            break
        m = ID_IN_NAME.match(os.path.basename(p))
        wid = (m.group(1).lstrip("0") or "0") if m else os.path.basename(p)
        if ok is not None and wid not in ok:
            skipped["一覧に無い"] += 1
            continue
        if wid in seen:
            continue
        try:
            t = read_text(p)
        except Exception:
            t = None
        if not t:
            skipped["読めない"] += 1
            continue
        t = clean(t)
        if ok is None and (t.count("ゐ") + t.count("ゑ")) > 5:
            skipped["旧仮名"] += 1
            continue
        ss = [s.strip().replace("　", "") for s in SPLIT.split(t)]
        ss = [s for s in ss if 4 <= len(s) <= 120]
        if len(ss) < 2:
            continue
        seen.add(wid)
        did = f"aozora-{wid}"
        rows += [(did, k, s) for k, s in enumerate(ss[:min(a.max_per_work, a.max_sents - len(rows))])]
        n_doc += 1
    print("作品", n_doc, "文", len(rows), "外した", skipped, flush=True)
    nlp = spacy.load("ja_ginza")
    with open(a.out, "w", encoding="utf-8") as f:
        for (did, k, text), doc in zip(rows, nlp.pipe((r[2] for r in rows), batch_size=200, n_process=a.n_process)):
            if k == 0:
                f.write(f"# newdoc id = {did}\n")
            f.write(f"# sent_id = {did}-{k}\n# text = {text}\n")
            for t in doc:
                head = 0 if t.dep_ == "ROOT" else t.head.i + 1
                ne = f"NE={t.ent_type_}" if t.ent_type_ else "_"
                f.write(f"{t.i + 1}\t{t.text}\t{t.lemma_}\t{t.pos_}\t{t.tag_}\t_\t{head}\t{t.dep_.lower() if t.dep_ != 'ROOT' else 'root'}\t_\t{ne}\n")
            f.write("\n")


if __name__ == "__main__":
    main()
